// Draws the atlas bitmap as one texture, so pan and zoom only change two uniforms.
// Falls back to a 2D canvas (drawImage with smoothing off) where WebGL2 is unavailable.

export interface View {
  scale: number;   // screen pixels per atlas pixel
  tx: number;      // screen position of the atlas origin
  ty: number;
}

export interface AtlasRenderer {
  setImage(rgba: Uint8ClampedArray, width: number, height: number): void;
  draw(view: View, cssWidth: number, cssHeight: number): void;
  readonly kind: "webgl2" | "2d";
}

const VS = `#version 300 es
in vec2 corner;
uniform vec2 size;      // atlas size in atlas px
uniform vec3 view;      // scale, tx, ty (css px)
uniform vec2 canvas;    // canvas size in css px
out vec2 uv;
void main() {
  uv = corner;
  vec2 p = corner * size * view.x + view.yz;
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
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(prog, "corner");
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  // Nearest sampling: zoomed in, every number is a crisp square.
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  const u = (n: string) => gl.getUniformLocation(prog, n);
  let size = [1, 1];
  return {
    kind: "webgl2",
    setImage(rgba, width, height) {
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(rgba.buffer, rgba.byteOffset, rgba.byteLength));
      size = [width, height];
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
      gl.uniform2f(u("size"), size[0], size[1]);
      gl.uniform3f(u("view"), view.scale, view.tx, view.ty);
      gl.uniform2f(u("canvas"), w, h);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    },
  };
}

function canvas2d(canvas: HTMLCanvasElement): AtlasRenderer {
  const ctx = canvas.getContext("2d")!;
  const img = document.createElement("canvas");
  return {
    kind: "2d",
    setImage(rgba, width, height) {
      img.width = width;
      img.height = height;
      img.getContext("2d")!.putImageData(new ImageData(new Uint8ClampedArray(rgba), width, height), 0, 0);
    },
    draw(view, w, h) {
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(img, view.tx, view.ty, img.width * view.scale, img.height * view.scale);
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
