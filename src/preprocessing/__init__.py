import os

from dotenv import load_dotenv


def _get_alpaca_credentials() -> tuple[str, str]:
    """Read Alpaca market data credentials from the environment."""
    load_dotenv()
    api_key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
    if not api_key or not secret_key:
        raise ValueError(
            "Data requires ALPACA_API_KEY/ALPACA_SECRET_KEY or "
            "APCA_API_KEY_ID/APCA_API_SECRET_KEY."
        )
    return api_key, secret_key
