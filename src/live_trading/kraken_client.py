from __future__ import annotations

from typing import Any

import ccxt

from src.live_trading.config import LiveTradingConfig


class KrakenClient:
    """Expose the Kraken operations needed by the live-trading runtime.

    Args:
        config: Validated live-trading configuration.
        exchange: Optional injected CCXT-compatible exchange for tests.
    """

    def __init__(self, config: LiveTradingConfig, exchange: Any | None = None):
        """Create a Kraken client without performing a network request.

        Args:
            config: Validated live-trading configuration.
            exchange: Optional injected CCXT-compatible exchange for tests.
        """
        self.config = config
        self.exchange = exchange or ccxt.kraken(
            {
                "apiKey": config.api_key,
                "secret": config.api_secret,
                "enableRateLimit": True,
            }
        )

    def fetch_balance(self) -> dict[str, Any]:
        """Fetch the Kraken account balance.

        Returns:
            The normalized CCXT balance response.
        """
        return self.exchange.fetch_balance()

    def fetch_market_price(self, symbol: str) -> float:
        """Fetch the latest traded price for a symbol.

        Args:
            symbol: CCXT trading pair.

        Returns:
            The latest traded price.

        Raises:
            ValueError: If Kraken returns no usable last price.
        """
        price = self.exchange.fetch_ticker(symbol).get("last")
        if price is None or price <= 0:
            raise ValueError(f"Kraken returned no valid last price for {symbol}.")
        return float(price)

    def fetch_order(self, order_id: str, symbol: str) -> dict[str, Any]:
        """Fetch an order by exchange identifier.

        Args:
            order_id: Exchange order identifier.
            symbol: CCXT trading pair.

        Returns:
            The normalized CCXT order response.
        """
        return self.exchange.fetch_order(order_id, symbol)

    def create_order(
            self,
            symbol: str,
            side: str,
            amount: float,
            order_type: str = "market",
            price: float | None = None,
    ) -> dict[str, Any]:
        """Submit a Kraken market or limit order.

        Args:
            symbol: CCXT trading pair.
            side: ``"buy"`` or ``"sell"``.
            amount: Base-asset quantity.
            order_type: ``"market"`` or ``"limit"``.
            price: Required limit price for limit orders.

        Returns:
            The normalized CCXT order response.

        Raises:
            ValueError: If the order parameters are invalid.
        """
        if side not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'.")
        if order_type not in {"market", "limit"}:
            raise ValueError("order_type must be 'market' or 'limit'.")
        if amount <= 0:
            raise ValueError("amount must be positive.")
        if order_type == "limit" and (price is None or price <= 0):
            raise ValueError("limit orders require a positive price.")
        return self.exchange.create_order(symbol, order_type, side, amount, price)

    def cancel_order(self, order_id: str, symbol: str) -> dict[str, Any]:
        """Cancel an open Kraken order.

        Args:
            order_id: Exchange order identifier.
            symbol: CCXT trading pair.

        Returns:
            The normalized CCXT cancellation response.
        """
        return self.exchange.cancel_order(order_id, symbol)
