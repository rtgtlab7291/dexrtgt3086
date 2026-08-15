# pyright: reportMissingImports=false
import json

import numpy as np


def segment(image: np.ndarray, address: str = "tcp://localhost:5555", timeout_s: float = 3600.0) -> np.ndarray:
    """Send `image` (uint8 HxWx3, RGB) to a running SAM2 annotation server and
    block until the human annotator clicks Submit in the browser. Returns the final
    binary mask as a uint8 HxW numpy array.

    Wire protocol (two-frame ZMQ multipart, both directions):
      frame 0: JSON header `{"dtype": str, "shape": [...]}`
      frame 1: raw `ndarray.tobytes()` payload

    Example (requires a running server and a human in the browser)::

        mask = segment(image)
    """
    import zmq

    assert image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3
    image = np.ascontiguousarray(image)
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
    sock.connect(address)
    header = json.dumps({"dtype": "uint8", "shape": list(image.shape)}).encode()
    sock.send_multipart([header, image.tobytes()])
    reply_header, reply_body = sock.recv_multipart()
    sock.close()
    meta = json.loads(reply_header.decode())
    mask = np.frombuffer(reply_body, dtype=np.dtype(meta["dtype"])).reshape(meta["shape"])
    return np.ascontiguousarray(mask)
