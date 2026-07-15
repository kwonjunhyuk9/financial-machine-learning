from __future__ import annotations

from collections.abc import Callable
from time import sleep
from typing import Any

from loguru import logger

from src.live_trading.config import LiveTradingConfig
from src.live_trading.kraken_client import KrakenClient
from src.live_trading.order_manager import OrderManager
from src.live_trading.risk_manager import RiskManager


class LiveRunner:
    """Run signals through risk checks and order submission.

    Args:
        config: Validated live-trading configuration.
        client: Kraken client used for balances and prices.
        order_manager: SQLite-backed order manager.
        risk_manager: Pre-trade risk validator.
    """

    def __init__(
            self,
            config: LiveTradingConfig,
            client: KrakenClient,
            order_manager: OrderManager,
            risk_manager: RiskManager,
    ):
        """Store the explicit runtime dependencies.

        Args:
            config: Validated live-trading configuration.
            client: Kraken client used for balances and prices.
            order_manager: SQLite-backed order manager.
            risk_manager: Pre-trade risk validator.
        """
        self.config = config
        self.client = client
        self.order_manager = order_manager
        self.risk_manager = risk_manager

    def run_once(self, signal: dict[str, Any]) -> dict[str, Any]:
        """Validate and submit one normalized trading signal.

        Args:
            signal: Mapping with ``side`` and optional ``amount``, ``order_type``,
                and ``price``.

        Returns:
            The recorded order response.

        Raises:
            ValueError: If the signal cannot be validated against runtime constraints.
        """
        side = signal.get("side")
        amount = float(signal.get("amount", self.config.order_size))
        order_type = signal.get("order_type", "market")
        price = signal.get("price") or self.client.fetch_market_price(
            self.config.symbol
        )
        current_position = self.order_manager.get_position(self.config.symbol)
        available_cash = (
            float("inf")
            if self.config.dry_run
            else _available_quote_cash(self.client.fetch_balance(), self.config.symbol)
        )
        self.risk_manager.validate_order(
            side,
            amount,
            float(price),
            current_position,
            available_cash,
        )
        order = self.order_manager.submit_order(
            self.config.symbol,
            side,
            amount,
            order_type,
            float(price) if order_type == "limit" else None,
        )
        logger.info("Processed {} signal for {} {}.", side, amount, self.config.symbol)
        return order

    def run(
            self,
            signal_generator: Callable[[], dict[str, Any]],
            interval_seconds: float,
            max_iterations: int | None = None,
    ) -> None:
        """Run a bounded or continuous live-trading loop.

        Args:
            signal_generator: Callable returning one normalized signal per iteration.
            interval_seconds: Delay between iterations in seconds.
            max_iterations: Optional finite iteration count for controlled runs.

        Raises:
            ValueError: If the polling interval or iteration count is invalid.
        """
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive.")
        if max_iterations is not None and max_iterations <= 0:
            raise ValueError("max_iterations must be positive when provided.")
        iteration = 0
        while max_iterations is None or iteration < max_iterations:
            self.run_once(signal_generator())
            iteration += 1
            if max_iterations is None or iteration < max_iterations:
                sleep(interval_seconds)


def _available_quote_cash(balance: dict[str, Any], symbol: str) -> float:
    """Extract available quote-currency cash from a CCXT balance response."""
    quote_currency = symbol.split("/", maxsplit=1)[1]
    free_balance = balance.get("free", {}).get(quote_currency, 0.0)
    return float(free_balance or 0.0)
