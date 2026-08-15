"""Kinematics utilities for hand retargeting."""

from typing import Literal, Sequence

import numpy as np

from robokit.helpers.hand_retargeting.config import HandSpec


def build_retargeting_pairs(
    spec: HandSpec,
    selected_names: Sequence[str],
    pair_mode: Literal["direct", "all"],
    include_root: bool,
) -> np.ndarray:
    """Build ordered pairs from named target chains.

    Args:
        spec: Named hand topology.
        selected_names: Target names retained by the solver.
        pair_mode: Adjacent chain pairs or every non-self pair.
        include_root: Whether to connect the root to every retained target.

    Returns:
        Pair indices shaped `[num_pairs, 2]`.

    Example:
        >>> from types import SimpleNamespace
        >>> spec = SimpleNamespace(
        ...     target_chains=(("root", "middle", "tip"),),
        ...     root_target_name="root",
        ... )
        >>> build_retargeting_pairs(spec, ("root", "middle", "tip"), "direct", True).tolist()
        [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]]
    """
    if pair_mode == "all":
        num_targets = len(selected_names)
        return np.asarray(
            [(i, j) for i in range(num_targets) for j in range(num_targets) if i != j],
            dtype=np.int32,
        ).reshape((-1, 2))

    selected_index = {name: index for index, name in enumerate(selected_names)}
    pairs = set()
    for chain in spec.target_chains:
        indices = [selected_index[name] for name in chain if name in selected_index]
        for origin, target in zip(indices, indices[1:]):
            pairs.add((origin, target))
            pairs.add((target, origin))
    if include_root:
        root = selected_index[spec.root_target_name]
        for target in range(len(selected_names)):
            if target != root:
                pairs.add((root, target))
                pairs.add((target, root))
    return np.asarray(sorted(pairs), dtype=np.int32).reshape((-1, 2))


__all__ = ["build_retargeting_pairs"]
