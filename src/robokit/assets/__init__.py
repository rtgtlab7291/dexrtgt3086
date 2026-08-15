"""Robokit asset access: HuggingFace fetch + local symlink.

Assets live in the HuggingFace dataset `rtgtlab7291/robokit-assets` (top-level
`robots/ body_models/ motions/ objects/`). Two on-disk roles under `~/.robokit/`:

- `cache/`: read-only content-addressed snapshots via `fetch`
  (`snapshot_download(cache_dir=...)`). Per-commit, blob-deduplicated. What code consumes.
- `hf_repo/`: a writable `git` / `git-xet` clone of the dataset for authoring; add,
  commit and push with plain git. Not created here.

Consume assets through the per-asset modules `robokit.assets.<kind>.<name>` which expose
static path symbols (e.g. `robokit.assets.robots.hands.sharpa_hand.URDF_PATH`); a name is valid
iff the symbol resolves, so IDEs, pyright and greps can check it. `link` points
`<cwd>/robokit_assets` at either the `hf_repo` clone (live dev edits) or a pinned snapshot.

Default revision is `main`; set `ROBOKIT_ASSETS_REVISION` to pin a specific commit.
"""

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import List, Optional


logger = logging.getLogger("robokit")

HF_REPO_ID = "rtgtlab7291/robokit-assets"
ROBOKIT_HOME = Path.home() / ".robokit"
CACHE_DIR = ROBOKIT_HOME / "cache"
HF_REPO_DIR = ROBOKIT_HOME / "hf_repo"
REVISION = os.environ.get("ROBOKIT_ASSETS_REVISION", "main")


def fetch(patterns: Optional[List[str]] = None, revision: str = REVISION, force: bool = False) -> Path:
    """Download `patterns` (default: the whole dataset) at `revision` into `CACHE_DIR`.

    Materializes **real files** (not symlinks) under `CACHE_DIR/<revision>/` and returns that
    directory. Real files preserve extensions, so loaders that canonicalize a path before
    choosing a reader by extension (e.g. SAPIEN mesh loading) work. Overlapping `patterns` at
    the same `revision` accumulate into one directory.

    Local-first: the first successful download of a `patterns` set drops a marker under
    `CACHE_DIR/<revision>/.done/`; later calls with the same set skip `snapshot_download`
    (no HF network round-trip) and return immediately. Pass `force=True` (or set
    `ROBOKIT_ASSETS_FORCE`) to ignore the marker and re-download.

    Example:
        >>> d = fetch(["robots/collision_spheres/g1/**"])                          # doctest: +SKIP
        >>> spheres = d / "robots/collision_spheres/g1/collision_spheres.yaml"     # doctest: +SKIP
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

    local = CACHE_DIR / revision
    key = hashlib.sha1("\n".join(sorted(patterns) if patterns else ["(all)"]).encode()).hexdigest()
    marker = local / ".done" / key
    if marker.exists() and not force and not os.environ.get("ROBOKIT_ASSETS_FORCE"):
        return local
    logger.debug(f"Fetching {HF_REPO_ID}@{revision} {patterns if patterns else '(all)'} into {local}")
    for attempt in range(3):
        try:
            snapshot_download(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                revision=revision,
                allow_patterns=patterns,
                local_dir=str(local),
            )
            break
        except (HfHubHTTPError, LocalEntryNotFoundError) as e:
            # HF quota is 1000 api requests / 5 min; concurrent CI jobs can trip it. The 429
            # may arrive wrapped in a LocalEntryNotFoundError whose message (hub <1.0) only
            # mentions the network in general, so walk the cause chain looking for it.
            # Already-downloaded files are skipped on retry, so waiting out the window is cheap.
            is_429 = False
            cause: Optional[BaseException] = e
            while cause is not None and not is_429:
                status = getattr(getattr(cause, "response", None), "status_code", None)
                is_429 = status == 429 or "429" in str(cause)
                cause = cause.__cause__
            if attempt == 2 or not is_429:
                raise
            logger.warning("HuggingFace 429 rate limit; retrying in 100 s")
            time.sleep(100)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    return local


def link(target: Optional[Path] = None) -> Path:
    """Symlink `target` (default `<cwd>/robokit_assets`) to the `~/.robokit/hf_repo` git clone.

    Full `git clone` of the dataset on first use (LFS blobs materialized, pushable;
    `cd assets && git status` works). Idempotent.

    Example:
        >>> link()                    # doctest: +SKIP
    """
    target = target if target is not None else Path.cwd() / "robokit_assets"
    if not HF_REPO_DIR.is_dir():
        import subprocess

        from huggingface_hub import get_token

        token = get_token()
        auth = f"hf:{token}@" if token else ""
        subprocess.run(
            ["git", "clone", f"https://{auth}huggingface.co/datasets/{HF_REPO_ID}", str(HF_REPO_DIR)],
            check=True,
        )
    if target.is_symlink():
        if target.resolve() == HF_REPO_DIR.resolve():
            logger.info(f"Symlink {target} already points to {HF_REPO_DIR}")
            return target
        target.unlink()  # repoint a stale symlink (e.g. the pre-refactor ~/.robokit/assets)
    target.symlink_to(HF_REPO_DIR)
    logger.info(f"Created symlink {target} -> {HF_REPO_DIR}")
    return target
