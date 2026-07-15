def test_config_module_imports():
    import src.live_trading.config

    assert src.live_trading.config.__name__ == "src.live_trading.config"
