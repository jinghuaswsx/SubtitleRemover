"""Cross-process GPU lock for SubtitleRemover (Linux fcntl).

See docs/vace_integration_plan.md §6.3.
"""

from .file_lock import cross_process_gpu_lock

__all__ = ["cross_process_gpu_lock"]
