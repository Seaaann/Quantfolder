import numpy as np
import pandas as pd

from grid_trading import GridConfig, plot_grid_backtest, run_grid_backtest


if __name__ == "__main__":
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="D")

    # 构造一个震荡价格序列，适合演示网格策略
    base = 100
    trend = np.linspace(-2, 2, n)
    wave = 6 * np.sin(np.linspace(0, 8 * np.pi, n))
    noise = np.random.normal(0, 1.2, n)
    prices = pd.Series(base + trend + wave + noise, index=dates)

    cfg = GridConfig(
        lower_price=88,
        upper_price=112,
        grid_count=12,
        capital=100_000,
        fee_bps=5,
        base_position_ratio=0.5,
    )

    result = run_grid_backtest(prices, cfg)
    print("metrics:")
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}")

    output = plot_grid_backtest(result, out_file="artifacts/grid_backtest_result.png")
    print(f"saved plot: {output}")
