import pytest

from src.live_trading import config


def test_load_config_uses_dry_run_defaults(monkeypatch):
    monkeypatch.setattr(config, "load_dotenv", lambda: None)
    monkeypatch.delenv("KRAKEN_SPOT_API_KEY", raising=False)
    monkeypatch.delenv("KRAKEN_SPOT_API_SECRET", raising=False)

    settings = config.load_live_trading_config()

    assert settings.dry_run is True
    assert settings.symbol == "BTC/USD"
    assert settings.state_db_path == config.DEFAULT_STATE_DB_PATH


def test_load_config_requires_credentials_for_live_mode(monkeypatch):
    monkeypatch.setattr(config, "load_dotenv", lambda: None)
    monkeypatch.delenv("KRAKEN_SPOT_API_KEY", raising=False)
    monkeypatch.delenv("KRAKEN_SPOT_API_SECRET", raising=False)

    with pytest.raises(ValueError, match="KRAKEN_SPOT_API_KEY"):
        config.load_live_trading_config(dry_run=False)


def test_load_config_accepts_explicit_non_secret_settings(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "load_dotenv", lambda: None)
    settings = config.load_live_trading_config(
        symbol="ETH/USD",
        order_size=0.1,
        max_order_size=0.2,
        max_position=1.0,
        state_db_path=tmp_path / "state.db",
    )

    assert settings.symbol == "ETH/USD"
    assert settings.state_db_path == tmp_path / "state.db"
