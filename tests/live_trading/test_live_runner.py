from src.live_trading.config import LiveTradingConfig
from src.live_trading.live_runner import LiveRunner
from src.live_trading.risk_manager import RiskManager


class FakeClient:
    def fetch_market_price(self, symbol):
        return 100.0

    def fetch_balance(self):
        return {"free": {"USD": 1000.0}}


class FakeOrderManager:
    def __init__(self):
        self.order = None

    def get_position(self, symbol):
        return 0.0

    def submit_order(self, symbol, side, amount, order_type, price):
        self.order = {"id": "dry-run-1", "symbol": symbol, "side": side, "amount": amount, "type": order_type, "price": price}
        return self.order


def test_run_once_routes_signal_through_risk_and_order_manager(tmp_path):
    config = LiveTradingConfig("", "", "BTC/USD", 0.1, True, 1.0, 2.0, tmp_path / "state.db")
    manager = FakeOrderManager()
    runner = LiveRunner(config, FakeClient(), manager, RiskManager(1.0, 2.0))

    order = runner.run_once({"side": "buy"})

    assert order["side"] == "buy"
    assert order["amount"] == 0.1


def test_run_rejects_non_positive_interval(tmp_path):
    config = LiveTradingConfig("", "", "BTC/USD", 0.1, True, 1.0, 2.0, tmp_path / "state.db")
    runner = LiveRunner(config, FakeClient(), FakeOrderManager(), RiskManager(1.0, 2.0))

    import pytest

    with pytest.raises(ValueError, match="interval_seconds"):
        runner.run(lambda: {"side": "buy"}, interval_seconds=0, max_iterations=1)
