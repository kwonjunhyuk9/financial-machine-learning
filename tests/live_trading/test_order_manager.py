from src.live_trading.order_manager import OrderManager


class FakeClient:
    def create_order(self, symbol, side, amount, order_type, price):
        return {
            "id": "order-1",
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "type": order_type,
            "price": price,
            "status": "open",
            "filled": 0.0,
        }


class FilledClient(FakeClient):
    def create_order(self, symbol, side, amount, order_type, price):
        order = super().create_order(symbol, side, amount, order_type, price)
        order["filled"] = amount
        order["status"] = "closed"
        order["price"] = 100.0
        return order


def test_dry_run_orders_and_executions_are_persisted(tmp_path):
    manager = OrderManager(FakeClient(), tmp_path / "state.db", dry_run=True)

    order = manager.submit_order("BTC/USD", "buy", 0.25)
    manager.record_execution(order["id"], "BTC/USD", "buy", 0.25, 100.0)

    assert order["status"] == "simulated"
    assert manager.get_position("BTC/USD") == 0.25


def test_live_orders_delegate_to_kraken_client(tmp_path):
    manager = OrderManager(FakeClient(), tmp_path / "state.db", dry_run=False)

    order = manager.submit_order("BTC/USD", "sell", 0.1, "limit", 101.0)

    assert order["id"] == "order-1"


def test_filled_live_order_updates_the_stored_position(tmp_path):
    manager = OrderManager(FilledClient(), tmp_path / "state.db", dry_run=False)

    manager.submit_order("BTC/USD", "buy", 0.1)

    assert manager.get_position("BTC/USD") == 0.1


def test_dry_run_orders_validate_the_same_inputs_as_live_orders(tmp_path):
    import pytest

    manager = OrderManager(FakeClient(), tmp_path / "state.db", dry_run=True)

    with pytest.raises(ValueError, match="side"):
        manager.submit_order("BTC/USD", "hold", 0.1)
