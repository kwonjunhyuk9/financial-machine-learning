def test_kraken_client_module_imports():
    import src.live_trading.kraken_client

    assert src.live_trading.kraken_client.__name__ == "src.live_trading.kraken_client"
