# Quantfolder

这个仓库目前包含一些策略回测脚本。新增并完善了一个**网格交易回测**模块。 

## 安装依赖

```bash
pip install -r requirements.txt
```

## 模块说明

- 文件：`单一选股策略/grid_trading.py`
- 核心接口：`run_grid_backtest(prices, cfg)`
- 可视化接口：`plot_grid_backtest(result, out_file)`

## 快速示例

```python
import pandas as pd
from 单一选股策略.grid_trading import GridConfig, run_grid_backtest, plot_grid_backtest

prices = pd.Series(
    [100, 98, 95, 97, 102, 105, 103, 99, 96, 101],
    index=pd.date_range("2024-01-01", periods=10, freq="D")
)

cfg = GridConfig(
    lower_price=90,
    upper_price=110,
    grid_count=10,
    capital=100000,
    fee_bps=5,
    base_position_ratio=0.5,
)

result = run_grid_backtest(prices, cfg)
print(result["metrics"])
plot_grid_backtest(result, out_file="artifacts/result.png")
```

## 一键演示（含收益可视化）

```bash
python 单一选股策略/grid_demo.py
```

运行后会输出回测指标并生成图片：`artifacts/grid_backtest_result.png`。

## 策略规则（当前实现）

- 在 `[lower_price, upper_price]` 之间均匀划分网格区间；
- 价格下穿网格：按固定名义金额买入一档；
- 价格上穿网格：按固定名义金额卖出一档；
- 每档名义金额为 `capital / grid_count`；
- 支持手续费（`fee_bps`）。
