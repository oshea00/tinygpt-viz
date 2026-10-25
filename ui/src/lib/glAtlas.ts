// Draws the atlas: one texture per tile, each a quad placed at its atlas position, so pan
// and zoom only change uniforms. Tiles off-screen are skipped. Falls back to a 2D canvas
// (drawImage with smoothing off) where WebGL2 is unavailable.

export interface View {
  scale: number;   // screen pixels per atlas pixel
  tx: number;      // screen position of the atlas origin
  ty: number;
}

export interface TilePixels {
  x: number;       // atlas position
  y: number;
  width: number;
  height: number;
  rgba: Uint8ClampedArray;
}

export interface AtlasRenderer {
  setTiles(tiles: TilePixels[]): void;
  draw(view: View, cssWidth: number, cssHeight: number): void;
  readonly kind: "webgl2" | "2d";
}

const visible = (t: TilePixels, v: View, w: number, h: number) =>
  v.tx + (t.x + t.width) * v.scale > 0 && v.tx + t.x * v.scale < w && v.ty + (t.y + t.height) * v.scale > 0 && v.ty + t.y * v.scale < h;

const VS = `#version 300 es
in vec2 corner;
uniform vec4 rect;      // tile x, y, width, height in atlas px
uniform vec3 view;      // scale, tx, ty (css px)
uniform vec2 canvas;    // canvas size in css px
out vec2 uv;
void main() {
  uv = corner;
  vec2 p = (rect.xy + corner * rect.zw) * view.x + view.yz;
  gl_Position = vec4(p / canvas * 2.0 - 1.0, 0.0, 1.0);
  gl_Position.y = -gl_Position.y;
}`;

const FS = `#version 300 es
precision mediump float;
in vec2 uv;
uniform sampler2D tex;
out vec4 color;
void main() { color = texture(tex, uv); }`;

function webgl(canvas: HTMLCanvasElement): AtlasRenderer | null {
  const gl = canvas.getContext("webgl2", { premultipliedAlpha: false, antialias: false });
  if (!gl) return null;
  const compile = (type: number, src: string) => {
    const s = gl.createShader(type)!;
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? "shader");
    return s;
  };
  const prog = gl.createProgram()!;
  gl.attachShader(prog, compile(gl.VERTEX_SHADER, VS));
  gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FS));
  gl.linkProgram(prog);
  gl.useProgram(prog);
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(prog, "corner");
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  const u = (n: string) => gl.getUniformLocation(prog, n);
  const uRect = u("rect"), uView = u("view"), uCanvas = u("canvas");

  let tiles: (TilePixels & { tex: WebGLTexture })[] = [];
  return {
    kind: "webgl2",
    setTiles(next) {
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      // Reuse textures where the tile at this index has the same size (the usual case: every
      // trace re-colours the same layout); otherwise allocate.
      const out: typeof tiles = [];
      next.forEach((t, i) => {
        const old = tiles[i];
        let tex: WebGLTexture;
        const data = new Uint8Array(t.rgba.buffer, t.rgba.byteOffset, t.rgba.byteLength);
        if (old && old.width === t.width && old.height === t.height) {
          tex = old.tex;
          gl.bindTexture(gl.TEXTURE_2D, tex);
          gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, t.width, t.height, gl.RGBA, gl.UNSIGNED_BYTE, data);
        } else {
          if (old) gl.deleteTexture(old.tex);
          tex = gl.createTexture()!;
          gl.bindTexture(gl.TEXTURE_2D, tex);
          // Nearest sampling: zoomed in, every number is a crisp square.
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, t.width, t.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
        }
        out.push({ ...t, tex });
      });
      for (const old of tiles.slice(next.length)) gl.deleteTexture(old.tex);
      tiles = out;
    },
    draw(view, w, h) {
      const dpr = window.devicePixelRatio || 1;
      if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
      }
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.uniform3f(uView, view.scale, view.tx, view.ty);
      gl.uniform2f(uCanvas, w, h);
      for (const t of tiles) {
        if (!visible(t, view, w, h)) continue;
        gl.bindTexture(gl.TEXTURE_2D, t.tex);
        gl.uniform4f(uRect, t.x, t.y, t.width, t.height);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      }
    },
  };
}

function canvas2d(canvas: HTMLCanvasElement): AtlasRenderer {
  const ctx = canvas.getContext("2d")!;
  let tiles: (TilePixels & { img: HTMLCanvasElement })[] = [];
  return {
    kind: "2d",
    setTiles(next) {
      tiles = next.map((t, i) => {
        const img = tiles[i]?.img ?? document.createElement("canvas");
        img.width = t.width;
        img.height = t.height;
        img.getContext("2d")!.putImageData(new ImageData(new Uint8ClampedArray(t.rgba), t.width, t.height), 0, 0);
        return { ...t, img };
      });
    },
    draw(view, w, h) {
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      ctx.imageSmoothingEnabled = false;
      for (const t of tiles) {
        if (visible(t, view, w, h)) ctx.drawImage(t.img, view.tx + t.x * view.scale, view.ty + t.y * view.scale, t.width * view.scale, t.height * view.scale);
      }
    },
  };
}

export function createRenderer(canvas: HTMLCanvasElement): AtlasRenderer {
  try {
    return webgl(canvas) ?? canvas2d(canvas);
  } catch {
    return canvas2d(canvas);
  }
}
