"""IPC type published by the wakeword daemon."""

import numpy as np

from bbos import realtime


@realtime(ms=1000)
def wakeword_state():
    """Whether the wake phrase was detected since the prior publication."""
    return [("active", np.bool_)]