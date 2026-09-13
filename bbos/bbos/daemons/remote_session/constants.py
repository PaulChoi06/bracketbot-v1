"""Config for the remote_session daemon."""

from pathlib import Path

from bbos import register


def _api_key() -> str:
    """Empty when the key is absent or unreadable; only uploads need it."""
    try:
        return Path("/etc/BB_API_KEY").read_text().strip()
    except OSError:
        return ""


# ============================================================================
# Configs
# ============================================================================
@register
class remote_session:
    """Tunables for the remote_session daemon, read once at startup."""

    bb_api_key: str = _api_key()
    bb_api_url: str = "https://api.bracketbot.com"
    poll_interval_s: float = 3.0
    http_timeout_s: float = 10.0
    fps: int = 30
    home_duration_s: float = 2.5
    drive_speed_scale: float = 0.7   # Scales every WHEEL_VEL_COMBOS entry.
