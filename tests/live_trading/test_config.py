import pytest

from src.live_trading import config


def test_load_config_uses_dry_run_defaults(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "load_dotenv", lambda: None)
    monkeypatch.setenv("KRAKEN_STATE_DB_PATH", str(tmp_path / "state.db"))
    monkeypatch.delenv("KRAKEN_SPOT_API_KEY", raising=False)
    monkeypatch.delenv("KRAKEN_SPOT_API_SECRET", raising=False)

    settings = config.load_live_trading_config()

    assert settings.dry_run is True
    assert settings.symbol == "BTC/USD"
    assert settings.state_db_path == tmp_path / "state.db"


def test_load_config_requires_credentials_for_live_mode(monkeypatch):
    monkeypatch.setattr(config, "load_dotenv", lambda: None)
    monkeypatch.setenv("KRAKEN_DRY_RUN", "false")
    monkeypatch.delenv("KRAKEN_SPOT_API_KEY", raising=False)
    monkeypatch.delenv("KRAKEN_SPOT_API_SECRET", raising=False)

    with pytest.raises(ValueError, match="KRAKEN_SPOT_API_KEY"):
        config.load_live_trading_config()


def test_load_config_rejects_invalid_dry_run_flag(monkeypatch):
    monkeypatch.setattr(config, "load_dotenv", lambda: None)
    monkeypatch.setenv("KRAKEN_DRY_RUN", "sometimes")

    with pytest.raises(ValueError, match="KRAKEN_DRY_RUN"):
        config.load_live_trading_config()
