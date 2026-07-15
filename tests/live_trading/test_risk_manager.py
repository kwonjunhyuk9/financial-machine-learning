import pytest

from src.live_trading.risk_manager import RiskManager


def test_risk_manager_allows_order_inside_limits():
    manager = RiskManager(max_order_size=1.0, max_position=2.0)

    assert manager.validate_order("buy", 0.5, 100.0, 0.0, 100.0) is None


def test_risk_manager_blocks_cash_position_and_size_breaches():
    manager = RiskManager(max_order_size=1.0, max_position=2.0)

    with pytest.raises(ValueError, match="max_order_size"):
        manager.validate_order("buy", 1.1, 100.0, 0.0, 200.0)
    with pytest.raises(ValueError, match="max_position"):
        manager.validate_order("buy", 0.5, 100.0, 1.8, 200.0)
    with pytest.raises(ValueError, match="available cash"):
        manager.validate_order("buy", 0.5, 100.0, 0.0, 10.0)


def test_risk_manager_requires_positive_limits():
    with pytest.raises(ValueError, match="positive"):
        RiskManager(max_order_size=0.0, max_position=1.0)
