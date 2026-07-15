def test_order_manager_module_imports():
    import src.live_trading.order_manager

    assert src.live_trading.order_manager.__name__ == "src.live_trading.order_manager"
