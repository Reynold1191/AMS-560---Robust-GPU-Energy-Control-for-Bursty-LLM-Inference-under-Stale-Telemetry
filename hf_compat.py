"""
hf_compat.py
Import this before `transformers`.
"""

import logging


def _disable_blocked_psutil() -> None:
    """
    On this laptop, Windows Application Control (Smart App Control) blocks
    psutil's compiled DLL. transformers only checks that the package is
    installed, then crashes importing it. psutil is optional there (a CPU
    memory report in continuous batching), so if the import fails we tell
    transformers it is unavailable instead of weakening the OS policy.
    """
    try:
        import psutil  # noqa: F401
    except ImportError as exc:
        import transformers.utils as tu
        import transformers.utils.import_utils as iu

        def _unavailable() -> bool:
            return False

        iu.is_psutil_available = _unavailable
        tu.is_psutil_available = _unavailable
        logging.getLogger(__name__).warning(
            "psutil cannot be imported (%s); running transformers without it.", exc)


_disable_blocked_psutil()
