from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DB_PATH = PROJECT_ROOT / "data/trading_state/live_trading.db"
DEFAULT_SYMBOL = "BTC/USD"
DEFAULT_ORDER_SIZE = 0.001
DEFAULT_MAX_ORDER_SIZE = 0.001
DEFAULT_MAX_POSITION = 0.001
DEFAULT_DRY_RUN = True


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


def load_live_trading_config(
        *,
        symbol: str = DEFAULT_SYMBOL,
        order_size: float = DEFAULT_ORDER_SIZE,
        max_order_size: float = DEFAULT_MAX_ORDER_SIZE,
        max_position: float = DEFAULT_MAX_POSITION,
        dry_run: bool = DEFAULT_DRY_RUN,
        state_db_path: Path = DEFAULT_STATE_DB_PATH,
) -> LiveTradingConfig:
    """Load Kraken credentials and assemble explicit live-trading settings.

    Returns:
        A validated immutable live-trading configuration.

    Raises:
        ValueError: If live-mode credentials or explicit settings are invalid.
    """
    load_dotenv()
    api_key = os.getenv("KRAKEN_SPOT_API_KEY", "")
    api_secret = os.getenv("KRAKEN_SPOT_API_SECRET", "")

    if not dry_run and (not api_key or not api_secret):
        raise ValueError(
            "KRAKEN_SPOT_API_KEY and KRAKEN_SPOT_API_SECRET "
            "are required for live trading."
        )
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
