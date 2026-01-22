"""
Time utility functions for elapsed time calculations.
"""

import time
from typing import Optional

from src.frontend.i18n import t


def get_elapsed_tips(
    start_time: float,
    end_time: Optional[float] = None,
    /,
    lang: str = "zh",
) -> str:
    """
    Get elapsed time message.

    Args:
        start_time: Start timestamp.
        end_time: End timestamp (defaults to current time).
        lang: Language code.

    Returns:
        Formatted elapsed time message.
    """
    end_time = end_time or time.time()
    elapsed_time = end_time - start_time
    return t("time_elapse", lang, elapsed_time)
