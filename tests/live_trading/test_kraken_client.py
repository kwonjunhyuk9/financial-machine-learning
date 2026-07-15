import pytest

from src.live_trading.config import LiveTradingConfig
from src.live_trading.kraken_client import KrakenClient


class FakeExchange:
    def fetch_ticker(self, symbol):
        return {"last": 100.0}

    def create_order(self, symbol, order_type, side, amount, price):
        return {"id": "order-1", "symbol": symbol, "type": order_type, "side": side, "amount": amount, "price": price}


def test_client_fetches_price_and_submits_normalized_order(tmp_path):
    settings = LiveTradingConfig("key", "secret", "BTC/USD", 0.1, True, 1.0, 2.0, tmp_path / "state.db")
    client = KrakenClient(settings, FakeExchange())

    assert client.fetch_market_price("BTC/USD") == 100.0
    assert client.create_order("BTC/USD", "buy", 0.1)["id"] == "order-1"


def test_client_rejects_invalid_limit_order(tmp_path):
    settings = LiveTradingConfig("key", "secret", "BTC/USD", 0.1, True, 1.0, 2.0, tmp_path / "state.db")
    client = KrakenClient(settings, FakeExchange())

    with pytest.raises(ValueError, match="limit orders"):
        client.create_order("BTC/USD", "buy", 0.1, "limit")
