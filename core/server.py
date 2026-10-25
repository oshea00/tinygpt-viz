"""Local server: a WebSocket bridge to the model worker, plus the built UI.

    uv run python -m core.server --config default
    then open http://localhost:8765
"""

import argparse
import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .config import PROFILES, TrainConfig, pick_device
from .data import CharData
from .scheduler import Message, ModelWorker
from .train import Trainer

UI_DIST = Path(__file__).resolve().parent.parent / "ui" / "dist"
MAX_QUEUE = 64     # a client this far behind gets traces dropped, never metrics


def create_app(trainer: Trainer, **worker_kw) -> FastAPI:
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
    p.add_argument("--config", default="default", choices=[k for k, v in PROFILES.items() if v.tokenizer == "char"])
    p.add_argument("--device", default=None)
    p.add_argument("--fresh", action="store_true", help="start from random weights, not the latest checkpoint")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()

    trainer = Trainer(PROFILES[args.config], TrainConfig(), CharData(), pick_device(args.device))
    if not args.fresh and trainer.checkpoints():
        trainer.load_checkpoint()
    print(f"{args.config} at step {trainer.step_num} on {trainer.device}: http://localhost:{args.port}")
    # No per-message compression: deflating megabytes of float32 costs more than sending them locally.
    uvicorn.run(create_app(trainer), host="127.0.0.1", port=args.port, log_level="warning",
                ws_per_message_deflate=False)


if __name__ == "__main__":
    main()
