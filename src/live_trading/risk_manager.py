from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskManager:
    """Validate orders against cash, position, and size limits.

    Args:
        max_order_size: Maximum allowed base-asset amount per order.
        max_position: Maximum allowed absolute base-asset position.
    """

    max_order_size: float
    max_position: float

    def __post_init__(self) -> None:
        """Validate immutable risk limits.

        Raises:
            ValueError: If either configured risk limit is non-positive.
        """
        if self.max_order_size <= 0 or self.max_position <= 0:
            raise ValueError("max_order_size and max_position must be positive.")

    def validate_order(
            self,
            side: str,
            amount: float,
            price: float,
            current_position: float,
            available_cash: float,
    ) -> None:
        """Raise when a proposed order exceeds configured risk limits.

        Args:
            side: ``"buy"`` or ``"sell"``.
            amount: Base-asset quantity.
            price: Current or limit price.
            current_position: Current base-asset position from the state store.
            available_cash: Available quote-currency cash from Kraken.

        Raises:
            ValueError: If order direction, size, position, or cash is invalid.
        """
        if side not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'.")
        if amount <= 0 or price <= 0:
            raise ValueError("amount and price must be positive.")
        if amount > self.max_order_size:
            raise ValueError("order amount exceeds max_order_size.")
        position_change = amount if side == "buy" else -amount
        if abs(current_position + position_change) > self.max_position:
            raise ValueError("order would exceed max_position.")
        if side == "buy" and amount * price > available_cash:
            raise ValueError("order cost exceeds available cash.")
