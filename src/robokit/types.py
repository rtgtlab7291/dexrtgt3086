from typing import Union

import numpy as np


try:
    import torch  # torch is optional
except ImportError:
    pass

try:
    import warp as wp  # warp is optional
except ImportError:
    pass

ArrayLike = Union[np.ndarray, "torch.Tensor", "wp.array"]
