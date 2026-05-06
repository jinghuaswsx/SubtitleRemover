"""VACE profile registry.

Only RTX 4070 Ti Super profiles are registered; this server doesn't deploy
on RTX 3060. See docs/vace_integration_plan.md §5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class VACEProfile:
    name: str
    model_name: str          # e.g. "vace-1.3B"
    size: str                # Wan2.1 SIZE_CONFIGS key, e.g. "832*480"
    frame_num: int           # must be 4n+1, max 81
    sample_steps: int
    offload_model: bool
    t5_cpu: bool
    chunk_seconds: float
    max_long_edge: int
    max_short_edge: int


PROFILES: Dict[str, VACEProfile] = {
    "rtx4070tis_fast": VACEProfile(
        name="rtx4070tis_fast",
        model_name="vace-1.3B",
        size="832*480",
        frame_num=41,
        sample_steps=20,
        offload_model=False,
        t5_cpu=False,
        chunk_seconds=2.7,
        max_long_edge=832,
        max_short_edge=480,
    ),
    "rtx4070tis_balanced": VACEProfile(
        name="rtx4070tis_balanced",
        model_name="vace-1.3B",
        size="832*480",
        frame_num=81,
        sample_steps=25,
        offload_model=False,
        t5_cpu=False,
        chunk_seconds=4.8,
        max_long_edge=832,
        max_short_edge=480,
    ),
    # vace-1.3B SUPPORTED_SIZES is only ('480*832', '832*480'); 720p needs vace-14B
    # which doesn't fit on 16 GB. "quality" stays at 832*480 with more sample steps.
    "rtx4070tis_quality": VACEProfile(
        name="rtx4070tis_quality",
        model_name="vace-1.3B",
        size="832*480",
        frame_num=81,
        sample_steps=30,
        offload_model=False,
        t5_cpu=False,
        chunk_seconds=4.8,
        max_long_edge=832,
        max_short_edge=480,
    ),
}


def get_profile(name: str) -> VACEProfile:
    if name not in PROFILES:
        raise KeyError(f"unknown VACE profile: {name}")
    return PROFILES[name]


_FALLBACK_CHAIN: Dict[str, Optional[str]] = {
    "rtx4070tis_quality": "rtx4070tis_balanced",
    "rtx4070tis_balanced": "rtx4070tis_fast",
    "rtx4070tis_fast": None,
}


def fallback_profile(p: VACEProfile) -> Optional[VACEProfile]:
    """Return the next OOM-fallback profile, or None at chain end."""
    nxt = _FALLBACK_CHAIN.get(p.name)
    if nxt is None:
        return None
    return PROFILES[nxt]
