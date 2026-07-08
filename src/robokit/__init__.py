import logging
from dataclasses import dataclass

import rich
from rich.logging import RichHandler


# ----------------------------- Logging -----------------------------
rich.reconfigure(log_path=False)

logger = logging.getLogger("robokit")
logger.propagate = False
logger.setLevel("INFO")
if not logger.handlers:
    logger.addHandler(RichHandler(console=rich.get_console(), show_path=False, log_time_format="[%X]"))


# ----------------------------- Config -----------------------------
@dataclass
class Config:
    """Configuration for robokit."""

    enable_torch_jit: bool = True
    """Whether to enable torch jit."""


CONFIG = Config()

__version__ = "0.1.0"

__all__ = ["CONFIG", "__version__"]
