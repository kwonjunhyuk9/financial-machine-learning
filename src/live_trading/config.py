from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DB_PATH = PROJECT_ROOT / "data/trading_state/live_trading.db"


@dataclass(frozen=True)
class LiveTradingConfig:
    """Runtime settings for Kraken spot trading.

    Args:
        api_key: Kraken API key.
        api_secret: Kraken API secret.
        symbol: CCXT trading pair, such as ``"BTC/USD"``.
        order_size: Default base-asset order size.
        dry_run: Whether to simulate orders instead of submitting them.
        max_order_size: Maximum allowed base-asset amount per order.
        max_position: Maximum allowed absolute base-asset position.
        state_db_path: SQLite path for orders and executions.
    """

    api_key: str
    api_secret: str
    symbol: str
    order_size: float
    dry_run: bool
    max_order_size: float
    max_position: float
    state_db_path: Path = DEFAULT_STATE_DB_PATH

    def __post_init__(self) -> None:
        """Validate configuration values created outside environment loading.

        Raises:
            ValueError: If symbol, order sizing, or state path is invalid.
        """
        if "/" not in self.symbol:
            raise ValueError("symbol must use CCXT pair notation, such as 'BTC/USD'.")
        if self.order_size <= 0 or self.max_order_size <= 0 or self.max_position <= 0:
            raise ValueError(
                "order_size, max_order_size, and max_position must be positive."
            )
        if self.order_size > self.max_order_size:
            raise ValueError("order_size cannot exceed max_order_size.")


def load_live_trading_config() -> LiveTradingConfig:
    """Load and validate live-trading settings from environment variables.

    Returns:
        A validated immutable live-trading configuration.

    Raises:
        ValueError: If required credentials or numeric settings are invalid.
    """
    load_dotenv()
    api_key = os.getenv("KRAKEN_SPOT_API_KEY", "")
    api_secret = os.getenv("KRAKEN_SPOT_API_SECRET", "")
    symbol = os.getenv("KRAKEN_SYMBOL", "BTC/USD")
    order_size = _get_positive_float("KRAKEN_ORDER_SIZE", "0.001")
    max_order_size = _get_positive_float("KRAKEN_MAX_ORDER_SIZE", str(order_size))
    max_position = _get_positive_float("KRAKEN_MAX_POSITION", str(max_order_size))
    dry_run = _get_bool("KRAKEN_DRY_RUN", "true")
    state_db_path = Path(os.getenv("KRAKEN_STATE_DB_PATH", DEFAULT_STATE_DB_PATH))

    if not dry_run and (not api_key or not api_secret):
        raise ValueError(
            "KRAKEN_SPOT_API_KEY and KRAKEN_SPOT_API_SECRET "
            "are required for live trading."
        )
    if "/" not in symbol:
        raise ValueError(
            "KRAKEN_SYMBOL must use CCXT pair notation, such as 'BTC/USD'."
        )
    if order_size > max_order_size:
        raise ValueError("KRAKEN_ORDER_SIZE cannot exceed KRAKEN_MAX_ORDER_SIZE.")

    return LiveTradingConfig(
        api_key=api_key,
        api_secret=api_secret,
        symbol=symbol,
        order_size=order_size,
        dry_run=dry_run,
        max_order_size=max_order_size,
        max_position=max_position,
        state_db_path=state_db_path,
    )


def _get_positive_float(name: str, default: str) -> float:
    """Read one positive floating-point environment setting."""
    try:
        value = float(os.getenv(name, default))
    except ValueError as error:
        raise ValueError(f"{name} must be a positive number.") from error
    if value <= 0:
        raise ValueError(f"{name} must be a positive number.")
    return value


def _get_bool(name: str, default: str) -> bool:
    """Read one boolean environment setting with explicit accepted values."""
    value = os.getenv(name, default).lower()
    if value in {"1", "true", "yes"}:
        return True
    if value in {"0", "false", "no"}:
        return False
    raise ValueError(f"{name} must be true or false.")
