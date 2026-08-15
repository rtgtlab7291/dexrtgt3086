# pyright: reportMissingImports=false
import asyncio
import base64
import io
import json
import os
import subprocess
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import tyro
import zmq
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, Response
from PIL import Image


@dataclass
class RequestState:
    image_rgb: np.ndarray
    points: list = field(default_factory=list)
    box: Optional[Tuple[float, float, float, float]] = None
    mask: Optional[np.ndarray] = None
    done: threading.Event = field(default_factory=threading.Event)


current: Optional[RequestState] = None
loop: Optional[asyncio.AbstractEventLoop] = None
ws_conn: Optional[WebSocket] = None
predictor: Any = None
last_image_id: Optional[int] = None
INDEX_HTML = (Path(__file__).parent / "index.html").read_text()
APP_JSX = (Path(__file__).parent / "app.jsx").read_text()
EXAMPLE_IMG = np.array(Image.open(Path(__file__).parent / "example.jpg").convert("RGB"))


def pick_free_gpu() -> int:
    out = (
        subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        .decode()
        .strip()
        .splitlines()
    )
    return max(range(len(out)), key=lambda i: int(out[i]))


def encode_png_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


MASK_RGBA = np.array([130, 200, 230, 140], dtype=np.uint8)


def encode_mask_b64(mask: np.ndarray) -> str:
    overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
    overlay[mask > 0] = MASK_RGBA
    return encode_png_b64(overlay)


def run_predict() -> np.ndarray:
    import torch

    global last_image_id
    state = current
    assert state is not None
    img_id = id(state.image_rgb)
    if img_id != last_image_id:
        predictor.set_image(state.image_rgb)
        last_image_id = img_id
    if not state.points and state.box is None:
        empty = np.zeros(state.image_rgb.shape[:2], dtype=np.uint8)
        state.mask = empty
        return empty
    pts = np.array([(x, y) for x, y, _ in state.points], dtype=np.float32) if state.points else None
    label = np.array([lb for _, _, lb in state.points], dtype=np.int32) if state.points else None
    box = np.array(state.box, dtype=np.float32) if state.box is not None else None
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        masks, _, _ = predictor.predict(point_coords=pts, point_labels=label, box=box, multimask_output=False)
    mask = (masks[0] > 0).astype(np.uint8)
    state.mask = mask
    return mask


def build_app(zmq_port: int) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global loop
        loop = asyncio.get_running_loop()
        threading.Thread(target=zmq_loop, args=(zmq_port,), daemon=True).start()
        yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return INDEX_HTML

    @app.get("/app.jsx")
    async def app_jsx() -> Response:
        return Response(APP_JSX, media_type="application/javascript")

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        global ws_conn, current, last_image_id
        if ws_conn is not None:
            await ws.close(code=1008)
            return
        await ws.accept()
        ws_conn = ws
        try:
            state = current
            if state is not None and not state.done.is_set():
                await ws.send_json(
                    {
                        "type": "image",
                        "w": int(state.image_rgb.shape[1]),
                        "h": int(state.image_rgb.shape[0]),
                        "image_b64": encode_png_b64(state.image_rgb),
                    }
                )
                if state.mask is not None and state.mask.any():
                    await ws.send_json({"type": "preview", "mask_b64": encode_mask_b64(state.mask)})
            while True:
                event = await ws.receive()
                if event["type"] == "websocket.disconnect":
                    break
                msg = json.loads(event["text"])
                t = msg["type"]
                if t == "load_example":
                    img = EXAMPLE_IMG.copy()
                    current = RequestState(image_rgb=img)
                    last_image_id = None
                    await ws.send_json(
                        {
                            "type": "image",
                            "w": int(img.shape[1]),
                            "h": int(img.shape[0]),
                            "image_b64": encode_png_b64(img),
                        }
                    )
                    continue
                st = current
                if st is None or st.done.is_set():
                    continue
                if t == "set_state":
                    st.points = [(float(p["x"]), float(p["y"]), int(p["positive"])) for p in msg["points"]]
                    b = msg.get("box")
                    st.box = (float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])) if b else None
                elif t == "submit":
                    if st.mask is None:
                        st.mask = np.zeros(st.image_rgb.shape[:2], dtype=np.uint8)
                    st.done.set()
                    continue
                mask = await asyncio.get_running_loop().run_in_executor(None, run_predict)
                await ws.send_json({"type": "preview", "mask_b64": encode_mask_b64(mask)})
        finally:
            ws_conn = None

    return app


def zmq_loop(zmq_port: int):
    global current, last_image_id
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://*:{zmq_port}")
    print(f"[sam2] ZMQ REP bound on tcp://*:{zmq_port}", flush=True)
    while True:
        header_bytes, body = sock.recv_multipart()
        meta = json.loads(header_bytes.decode())
        shape = tuple(meta["shape"])
        image = np.frombuffer(body, dtype=np.dtype(meta["dtype"])).reshape(shape).copy()
        state = RequestState(image_rgb=image)
        current = state
        last_image_id = None
        if ws_conn is not None and loop is not None:

            async def push(img: np.ndarray = image):
                if ws_conn is None:
                    return
                await ws_conn.send_json(
                    {
                        "type": "image",
                        "w": int(img.shape[1]),
                        "h": int(img.shape[0]),
                        "image_b64": encode_png_b64(img),
                    }
                )

            asyncio.run_coroutine_threadsafe(push(), loop)
        state.done.wait()
        mask = state.mask
        assert mask is not None
        reply_header = json.dumps({"dtype": "uint8", "shape": list(mask.shape)}).encode()
        sock.send_multipart([reply_header, mask.tobytes()])
        current = None


def main(port: int = 8000, zmq_port: int = 5555, model: str = "facebook/sam2.1-hiera-base-plus"):
    gpu = pick_free_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"[sam2] picked GPU {gpu}", flush=True)

    from sam2.sam2_image_predictor import SAM2ImagePredictor

    global predictor
    predictor = SAM2ImagePredictor.from_pretrained(model, device="cuda")
    print(f"[sam2] loaded {model}", flush=True)

    app = build_app(zmq_port)

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    tyro.cli(main)
