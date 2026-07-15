def test_live_runner_module_imports():
    import src.live_trading.live_runner

    assert src.live_trading.live_runner.__name__ == "src.live_trading.live_runner"
