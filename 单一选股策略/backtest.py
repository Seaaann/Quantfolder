import pandas as pd
import numpy as np
from math import sqrt

def backtest(
    open_px: pd.DataFrame,         # DataFrame[date × ticker] 开盘价
    close_px: pd.DataFrame,        # DataFrame[date × ticker] 收盘价
    universe: list,                # 股票池
    start: str, end: str,          # 回测区间（'YYYY-MM-DD'）
    strategy_fn,                   # def strategy_fn(signal_date, data_view) -> pd.Series(weights)
    rebalance_rule: str = "W-FRI", # 调仓频率（如 "W-FRI","M","D"）
    fee_bps: float = 5.0,          # 手续费（名义成交额的 bps）
    slip_bps: float = 10.0,        # 滑点（bps；买加卖减）
    initial_capital: float = 1_000_000.0,
    allow_short: bool = False,     # 是否允许做空
    leverage_limit: float = 1.0,   # sum(|w|) ≤ leverage_limit
    max_weight: float = 0.2,       # 单票绝对权重上限
    round_lot: int = 1,            # 最小交易单位（股）
    min_trade_value: float = 0.0   # 小额阈值（名义 < 此值则忽略）
):
    # —— 数据对齐 ——
    open_px  = open_px.loc[start:end, universe].copy()
    close_px = close_px.loc[start:end, universe].copy()
    assert (open_px.index.equals(close_px.index)) and (open_px.columns.equals(close_px.columns)), "open/close 不一致"
    cal = close_px.index

    # —— 调仓日（按频率取周五/月底等；跳首个以避免前视） ——
    rebal_dates = pd.Series(index=cal).resample(rebalance_rule).last().index
    rebal_dates = [d for d in rebal_dates if d in cal][1:]

    cash = initial_capital
    pos  = pd.Series(0.0, index=universe)   # 股数
    equity = pd.Series(index=cal, dtype=float)
    trades, holds = [], []

    for r in rebal_dates:
        # 信号可见日：r 的前一交易日
        i = cal.searchsorted(r)
        if i == 0: 
            continue
        signal_date = cal[i-1]

        # 给策略的视图：只到 signal_date 为止
        view = {"open": open_px.loc[:signal_date],
                "close": close_px.loc[:signal_date],
                "calendar": cal[cal <= signal_date]}

        # —— 策略输出目标权重（index=ticker, 值∈[-1,1]） ——
        w = strategy_fn(signal_date, view).reindex(universe).fillna(0.0)
        if not allow_short: 
            w = w.clip(lower=0.0)
        w = w.clip(lower=-max_weight, upper=max_weight)
        L1 = w.abs().sum()
        if L1 > 0:
            w *= min(1.0, leverage_limit / L1)   # 杠杆归一

        holds.append({"signal_date": signal_date, "rebalance_date": r, "weights": w[w!=0].to_dict()})

        # 成交日：r 当天（用开盘成交；若严格避免前视可改为下一交易日开盘）
        px_open = open_px.loc[r]
        tradable = px_open.dropna().index
        w = w.where(w.index.isin(tradable), 0.0)

        port_val = cash + (pos * px_open.fillna(method="ffill").fillna(0)).sum()
        tgt_val  = w * port_val
        cur_val  = pos * px_open
        dval     = (tgt_val - cur_val).fillna(0.0)

        # 滑点价格
        exec_px = px_open * (1 + np.sign(dval) * slip_bps/1e4)

        # 换算股数，四舍五入到最小单位
        sh = (dval / exec_px).replace([np.inf, -np.inf], 0).fillna(0.0)
        sh = (np.sign(sh) * (np.floor(np.abs(sh)/round_lot) * round_lot)).astype(float)

        # 忽略太小的交易
        tiny = (sh.abs() * exec_px) < min_trade_value
        sh[tiny] = 0.0

        notional = (sh.abs() * exec_px).sum()
        fee = notional * (fee_bps/1e4)

        cash -= (sh * exec_px).sum() + fee
        pos  = (pos + sh).astype(float)

        trades.append({"date": r, "notional": float(notional), "fees": float(fee),
                       "cash_after": float(cash), "fills": sh[sh!=0].to_dict()})

        # 从 r 到下一个调仓日前一天（含 r）每日估值（收盘）
        next_r = rebal_dates[rebal_dates.index(r)+1] if r != rebal_dates[-1] else cal[-1]
        stop = (cal[cal <= next_r].max() - pd.tseries.offsets.BDay(1)) if r != rebal_dates[-1] else next_r
        days = cal[(cal >= r) & (cal <= stop)]
        for d in days:
            equity.loc[d] = cash + (pos * close_px.loc[d].fillna(method="ffill").fillna(0)).sum()

    if pd.isna(equity.iloc[0]): 
        equity.iloc[0] = initial_capital
    equity = equity.ffill().astype(float)
    r = equity.pct_change().fillna(0)

    def maxdd(s): return (s/s.cummax() - 1).min()
    metrics = {
        "CAGR": (equity.iloc[-1]/equity.iloc[0])**(252/len(equity)) - 1,
        "Vol_Ann": r.std()*sqrt(252),
        "Sharpe_Ann": (r.mean()/r.std())*sqrt(252) if r.std()>0 else np.nan,
        "MaxDD": maxdd(equity),
        "Rebalances": len(trades),
        "Fees_$": sum(t["fees"] for t in trades),
        "Turnover_$": sum(t["notional"] for t in trades),
    }
    return {"equity": equity, "returns": r, "metrics": metrics, "trades": trades, "holdings": holds}
