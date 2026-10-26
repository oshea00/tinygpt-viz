"""Local server: a WebSocket bridge to the model worker, plus the built UI.

    uv run python -m core.server --config bpe-small
    then open http://localhost:8765
"""

import argparse
import asyncio
import json
import mimetypes
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .config import DEFAULT_PROFILE, PROFILES, DeviceError, TrainConfig, pick_device, profile
from . import hub
from .data import load_data
from .scheduler import Message, ModelWorker
from .train import Trainer

UI_DIST = Path(__file__).resolve().parent.parent / "ui" / "dist"
DOCS = Path(__file__).resolve().parent.parent / "docs"
MAX_QUEUE = 64     # a client this far behind gets traces dropped, never metrics


def create_app(trainer: Trainer, **worker_kw) -> FastAPI:
    """worker_kw go to ModelWorker, e.g. training=False for a model that only runs."""
    clients: set[asyncio.Queue[Message]] = set()
    state: dict = {}

    def broadcast(msg: Message) -> None:
        for q in clients:
            if isinstance(msg, bytes) and q.qsize() > MAX_QUEUE:
                continue
            q.put_nowait(msg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loop = asyncio.get_running_loop()
        worker = ModelWorker(trainer, lambda m: loop.call_soon_threadsafe(broadcast, m), **worker_kw)
        state["worker"] = worker
        worker.start()
        yield
        worker.stop()
        worker.join(timeout=5)

    app = FastAPI(lifespan=lifespan)

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        q: asyncio.Queue[Message] = asyncio.Queue()
        clients.add(q)
        worker: ModelWorker = state["worker"]
        worker.sent_weights_version = -1          # a new client needs the weights
        worker.submit({"type": "hello"})
        worker.submit({"type": "run", "prompt": worker.probe})

        async def sender():
            while True:
                msg = await q.get()
                if isinstance(msg, bytes):
                    await ws.send_bytes(msg)
                else:
                    await ws.send_text(json.dumps(msg))

        task = asyncio.create_task(sender())
        try:
            while True:
                worker.submit(json.loads(await ws.receive_text()))
        except WebSocketDisconnect:
            pass
        finally:
            clients.discard(q)
            task.cancel()

    # The guide, for the in-app help's links. Sent as plain text so browsers show the Markdown
    # instead of downloading it.
    mimetypes.add_type("text/plain; charset=utf-8", ".md")
    if DOCS.exists():
        app.mount("/docs", StaticFiles(directory=DOCS), name="docs")
    if UI_DIST.exists():
        app.mount("/", StaticFiles(directory=UI_DIST, html=True), name="ui")
    else:
        @app.get("/")
        def no_ui():
            return HTMLResponse("<p>UI not built: run <code>npm run build</code> in <code>ui/</code>, "
                                "or use the Vite dev server (<code>npm run dev</code>).</p>")
    return app


def main() -> None:
    import uvicorn

    p = argparse.ArgumentParser()
    p.add_argument("--config", default=DEFAULT_PROFILE, choices=list(PROFILES))
    p.add_argument("--device", default=None)
    p.add_argument("--fresh", action="store_true", help="start from random weights, not the latest checkpoint")
    p.add_argument("--host", default="127.0.0.1",
                   help="interface to bind; 0.0.0.0 serves other machines (there is no authentication)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--train", action="store_true",
                   help="enable training: the training panel, the training step tab, and checkpoint replay")
    args = p.parse_args()

    cfg = profile(args.config)
    try:
        device = pick_device(args.device)
    except DeviceError as e:
        raise SystemExit(f"error: {e}")
    if not args.fresh and not hub.has_checkpoints(cfg):
        hub.pull(cfg)
    trainer = Trainer(cfg, TrainConfig(), load_data(need_train=args.train), device)
    if not args.fresh and trainer.checkpoints():
        trainer.load_checkpoint()
    print(f"{cfg.name} at step {trainer.step_num} on {trainer.device}, training {'on' if args.train else 'off'}: "
          f"http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0', '::') else args.host}:{args.port}")
    # No per-message compression: deflating megabytes of float32 costs more than sending them locally.
    uvicorn.run(create_app(trainer, training=args.train), host=args.host, port=args.port, log_level="warning",
                ws_per_message_deflate=False)


if __name__ == "__main__":
    main()
