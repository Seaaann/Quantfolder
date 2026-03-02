from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class GridConfig:
    lower_price: float
    upper_price: float
    grid_count: int = 10
    capital: float = 100_000.0
    fee_bps: float = 5.0
    base_position_ratio: float = 0.5


def run_grid_backtest(prices: pd.Series, cfg: GridConfig) -> dict:
    """
    现金-现货网格回测（单资产）：
    - 将 [lower_price, upper_price] 均分成 grid_count 个网格区间；
    - 价格下穿网格线：买入 1 份；上穿网格线：卖出 1 份；
    - 每份名义金额固定为 capital / grid_count。
    """
    if cfg.grid_count < 2:
        raise ValueError("grid_count 至少为 2")
    if cfg.lower_price <= 0 or cfg.upper_price <= cfg.lower_price:
        raise ValueError("价格区间参数非法")
    if not (0 <= cfg.base_position_ratio <= 1):
        raise ValueError("base_position_ratio 需在 [0,1]")

    px = prices.dropna().astype(float)
    if px.empty:
        raise ValueError("prices 不能为空")

    # grid_count 个区间 => grid_count+1 条网格线
    grid_lines = np.linspace(cfg.lower_price, cfg.upper_price, cfg.grid_count + 1)
    slot_notional = cfg.capital / cfg.grid_count

    init_price = float(px.iloc[0])
    init_asset_val = cfg.capital * cfg.base_position_ratio
    asset_qty = init_asset_val / init_price
    cash = cfg.capital - init_asset_val

    def get_grid_idx(price: float) -> int:
        idx = int(np.searchsorted(grid_lines, price, side="right") - 1)
        return int(np.clip(idx, 0, len(grid_lines) - 2))

    prev_idx = get_grid_idx(init_price)
    records = []
    trades = []

    for dt, price in px.items():
        cur_idx = get_grid_idx(float(price))
        crossed = cur_idx - prev_idx

        if crossed != 0:
            step = int(np.sign(crossed))
            for _ in range(abs(crossed)):
                trade_px = float(price)
                fee = slot_notional * (cfg.fee_bps / 1e4)

                if step < 0:
                    buy_qty = slot_notional / trade_px
                    total_cost = slot_notional + fee
                    if cash >= total_cost:
                        cash -= total_cost
                        asset_qty += buy_qty
                        trades.append(
                            {
                                "date": dt,
                                "side": "BUY",
                                "price": trade_px,
                                "qty": buy_qty,
                                "notional": slot_notional,
                                "fee": fee,
                            }
                        )
                else:
                    sell_qty = slot_notional / trade_px
                    if asset_qty >= sell_qty:
                        cash += slot_notional - fee
                        asset_qty -= sell_qty
                        trades.append(
                            {
                                "date": dt,
                                "side": "SELL",
                                "price": trade_px,
                                "qty": sell_qty,
                                "notional": slot_notional,
                                "fee": fee,
                            }
                        )

        equity = cash + asset_qty * float(price)
        records.append(
            {
                "date": dt,
                "price": float(price),
                "cash": cash,
                "asset_qty": asset_qty,
                "equity": equity,
                "grid_idx": cur_idx,
            }
        )
        prev_idx = cur_idx

    equity_curve = pd.DataFrame(records).set_index("date")
    returns = equity_curve["equity"].pct_change().fillna(0.0)
    maxdd = (equity_curve["equity"] / equity_curve["equity"].cummax() - 1).min()

    metrics = {
        "final_equity": float(equity_curve["equity"].iloc[-1]),
        "total_return": float(equity_curve["equity"].iloc[-1] / cfg.capital - 1),
        "max_drawdown": float(maxdd),
        "trade_count": len(trades),
        "total_fees": float(sum(t["fee"] for t in trades)),
        "buy_count": int(sum(t["side"] == "BUY" for t in trades)),
        "sell_count": int(sum(t["side"] == "SELL" for t in trades)),
        "volatility": float(returns.std() * np.sqrt(252)) if len(returns) > 1 else 0.0,
    }

    return {
        "equity_curve": equity_curve,
        "trades": pd.DataFrame(trades),
        "metrics": metrics,
        "grid_lines": grid_lines,
    }


def plot_grid_backtest(result: dict, out_file: str | None = None) -> str | None:
    """绘制价格与净值曲线，并可选择保存图片。"""
    equity_curve = result["equity_curve"]
    trades = result["trades"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(equity_curve.index, equity_curve["price"], label="Price", color="#1f77b4")
    if not trades.empty:
        buy_mask = trades["side"] == "BUY"
        sell_mask = trades["side"] == "SELL"
        axes[0].scatter(trades.loc[buy_mask, "date"], trades.loc[buy_mask, "price"], label="BUY", marker="^", s=50)
        axes[0].scatter(trades.loc[sell_mask, "date"], trades.loc[sell_mask, "price"], label="SELL", marker="v", s=50)
    axes[0].set_title("Grid Trading: Price & Trades")
    axes[0].set_ylabel("Price")
    axes[0].legend(loc="best")

    axes[1].plot(equity_curve.index, equity_curve["equity"], label="Equity", color="#2ca02c")
    axes[1].set_title("Equity Curve")
    axes[1].set_ylabel("Equity")
    axes[1].set_xlabel("Date")
    axes[1].legend(loc="best")

    plt.tight_layout()

    if out_file:
        out_path = Path(out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        return str(out_path)

    plt.show()
    return None
