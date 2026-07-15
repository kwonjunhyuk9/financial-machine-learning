from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from loguru import logger

from src.live_trading.kraken_client import KrakenClient


class OrderManager:
    """Submit orders and persist orders and executions in SQLite.

    Args:
        client: Kraken client used for live exchange actions.
        state_db_path: SQLite path for orders and executions.
        dry_run: Whether to simulate orders instead of submitting them.
    """

    def __init__(self, client: KrakenClient, state_db_path: Path, dry_run: bool):
        """Initialize the SQLite schema for the trading state store.

        Args:
            client: Kraken client used for live exchange actions.
            state_db_path: SQLite path for orders and executions.
            dry_run: Whether to simulate orders instead of submitting them.
        """
        self.client = client
        self.state_db_path = Path(state_db_path)
        self.dry_run = dry_run
        self.state_db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_store()

    def submit_order(
            self,
            symbol: str,
            side: str,
            amount: float,
            order_type: str = "market",
            price: float | None = None,
    ) -> dict[str, Any]:
        """Submit or simulate an order and record it in the state store.

        Args:
            symbol: CCXT trading pair.
            side: ``"buy"`` or ``"sell"``.
            amount: Base-asset quantity.
            order_type: ``"market"`` or ``"limit"``.
            price: Limit price when applicable.

        Returns:
            A normalized order record.

        Raises:
            ValueError: If the order parameters are invalid.
        """
        _validate_order(side, amount, order_type, price)
        if self.dry_run:
            order = {
                "id": f"dry-run-{uuid4()}",
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "type": order_type,
                "price": price,
                "status": "simulated",
                "filled": 0.0,
            }
        else:
            order = self.client.create_order(symbol, side, amount, order_type, price)
        self._record_order(order)
        self._record_filled_execution(order)
        logger.info(
            "Recorded {} {} order {} for {} {}.",
            order_type,
            side,
            order["id"],
            amount,
            symbol,
        )
        return order

    def cancel_order(self, order_id: str, symbol: str) -> dict[str, Any]:
        """Cancel an order and record its latest status.

        Args:
            order_id: Exchange order identifier.
            symbol: CCXT trading pair.

        Returns:
            The cancellation response or a simulated cancellation record.
        """
        if self.dry_run:
            order = {
                "id": order_id,
                "symbol": symbol,
                "status": "canceled",
                "filled": 0.0,
            }
        else:
            order = self.client.cancel_order(order_id, symbol)
        self._record_order(order)
        return order

    def sync_order(self, order_id: str, symbol: str) -> dict[str, Any]:
        """Fetch, persist, and return the latest order status.

        Args:
            order_id: Exchange order identifier.
            symbol: CCXT trading pair.

        Returns:
            The latest normalized order response.
        """
        order = self.client.fetch_order(order_id, symbol)
        self._record_order(order)
        self._record_filled_execution(order)
        return order

    def record_execution(
            self,
            order_id: str,
            symbol: str,
            side: str,
            amount: float,
            price: float,
    ) -> None:
        """Persist a confirmed execution for position calculations.

        Args:
            order_id: Associated order identifier.
            symbol: CCXT trading pair.
            side: Executed ``"buy"`` or ``"sell"`` side.
            amount: Filled base-asset amount.
            price: Execution price.

        Raises:
            ValueError: If the execution direction, amount, or price is invalid.
        """
        if side not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'.")
        if amount <= 0 or price <= 0:
            raise ValueError("amount and price must be positive.")
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO executions VALUES (?, ?, ?, ?, ?, ?, ?)",
                (str(uuid4()), order_id, symbol, side, amount, price, _utc_now()),
            )

    def get_position(self, symbol: str) -> float:
        """Return net base-asset position from recorded executions.

        Args:
            symbol: CCXT trading pair.

        Returns:
            Net bought minus sold quantity.
        """
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COALESCE("
                "SUM(CASE side WHEN 'buy' THEN amount ELSE -amount END), 0) "
                "FROM executions WHERE symbol = ?",
                (symbol,),
            ).fetchone()
        return float(row[0])

    def _initialize_store(self) -> None:
        """Create the state-store tables when they do not exist."""
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS orders ("
                "order_id TEXT PRIMARY KEY, symbol TEXT, side TEXT, amount REAL, "
                "order_type TEXT, price REAL, status TEXT, filled REAL, "
                "updated_at TEXT)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS executions ("
                "execution_id TEXT PRIMARY KEY, order_id TEXT, symbol TEXT, side TEXT, "
                "amount REAL, price REAL, executed_at TEXT)"
            )

    def _record_order(self, order: dict[str, Any]) -> None:
        """Upsert a normalized order response into SQLite."""
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    order["id"],
                    order.get("symbol"),
                    order.get("side"),
                    order.get("amount"),
                    order.get("type"),
                    order.get("price"),
                    order.get("status"),
                    order.get("filled", 0.0), _utc_now(),
                ),
            )

    def _record_filled_execution(self, order: dict[str, Any]) -> None:
        """Persist the aggregate filled quantity reported by a CCXT order."""
        filled = float(order.get("filled") or 0.0)
        price = order.get("average") or order.get("price")
        if filled <= 0 or price is None or not order.get("side"):
            return
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM executions WHERE order_id = ?",
                (order["id"],),
            )
        self.record_execution(
            order_id=order["id"],
            symbol=order["symbol"],
            side=order["side"],
            amount=filled,
            price=float(price),
        )

    def _connect(self) -> sqlite3.Connection:
        """Open a SQLite connection with transactional context-manager support."""
        return sqlite3.connect(self.state_db_path)


def _utc_now() -> str:
    """Return the current UTC time in ISO format."""
    return datetime.now(UTC).isoformat()


def _validate_order(
        side: str,
        amount: float,
        order_type: str,
        price: float | None,
) -> None:
    """Validate parameters shared by live and dry-run order submission."""
    if side not in {"buy", "sell"}:
        raise ValueError("side must be 'buy' or 'sell'.")
    if amount <= 0:
        raise ValueError("amount must be positive.")
    if order_type not in {"market", "limit"}:
        raise ValueError("order_type must be 'market' or 'limit'.")
    if order_type == "limit" and (price is None or price <= 0):
        raise ValueError("limit orders require a positive price.")
