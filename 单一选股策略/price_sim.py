import numpy as np
import pandas as pd

def simulate_prices(
    n_names: int = 10,
    bdays: int = 252,
    seed: int = 42,
    model: str = "gbm",                 # 'gbm' | 'gbm_jumps' | 'sv'（随机波动）
    mu_bounds = (-0.05, 0.15),          # 年化漂移（均匀采样区间）
    sigma_bounds = (0.15, 0.35),        # 年化波动（均匀采样区间）
    equicorr: float = 0.3,              # 相关性（等相关矩阵的 rho）
    jump_intensity: float = 0.0,        # 跳跃强度（年化到达率 λ; model='gbm_jumps' 时生效）
    jump_mean: float = -0.02,           # 跳幅均值（对数收益空间）
    jump_std: float = 0.05,             # 跳幅标准差
    sv_mean_rev: float = 4.0,           # 随机波动：log-variance 均值回复速度（越大回复越快）
    sv_vol_of_vol: float = 0.30,        # 随机波动：波动的波动（放大日内 sigma 的随机性）
    overnight_scale: float = 0.5,       # 开盘“隔夜噪声”的相对日波动比例
    price0_bounds = (50.0, 150.0)       # 初始价格采样区间
):
    """
    生成两张对齐的价格表：
      - close: DataFrame[date × ticker]
      - open_: DataFrame[date × ticker]（用前一日收盘 * 隔夜噪声构造）

    可选模型：
      - 'gbm'：几何布朗运动（带相关）
      - 'gbm_jumps'：GBM + 泊松跳跃
      - 'sv'：随机波动（对 log-variance 做 OU/均值回复）
    """
    rng = np.random.default_rng(seed)
    tickers = [f"S{i}" for i in range(1, n_names + 1)]

    # 交易日历
    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.tseries.offsets.BDay(bdays)
    calendar = pd.bdate_range(start=start_date, end=end_date)
    T = len(calendar)
    dt = 1.0 / 252.0

    # 资产参数
    mu = rng.uniform(mu_bounds[0], mu_bounds[1], size=n_names)        # 年化漂移
    sigma_ann = rng.uniform(sigma_bounds[0], sigma_bounds[1], size=n_names)  # 年化波动
    sigma_day = sigma_ann * np.sqrt(dt)

    # 等相关矩阵的 Cholesky
    rho = equicorr
    cov = (1 - rho) * np.eye(n_names) + rho * np.ones((n_names, n_names))
    L = np.linalg.cholesky(cov)

    # 标准正态因子（相关化）
    Z = rng.standard_normal(size=(T, n_names)) @ L

    # 随机波动（可选）：对 log-variance 做 OU，得到 time-varying sigma_t
    if model == "sv":
        # 基于各资产的日波动为均值，围绕其对数做 OU
        log_var0 = np.log(sigma_day**2 + 1e-12)
        log_var = np.empty((T, n_names))
        log_var[0, :] = log_var0
        kappa = sv_mean_rev               # 均值回复强度
        eta = sv_vol_of_vol               # 波动的波动
        for t in range(1, T):
            # d ln(v) = -kappa*(ln(v) - ln(v0))*dt + eta*sqrt(dt)*N(0,1)
            noise = rng.standard_normal(n_names) * np.sqrt(dt)
            log_var[t, :] = (log_var[t-1, :]
                             - kappa * (log_var[t-1, :] - log_var0) * dt
                             + eta * noise)
        sigma_day_t = np.sqrt(np.exp(log_var))  # T × n
    else:
        sigma_day_t = np.tile(sigma_day, (T, 1))  # 常数波动

    # 跳跃项（可选）
    if model == "gbm_jumps" and jump_intensity > 0:
        # 时间步长内的到达概率
        lam_dt = jump_intensity * dt
        N_jump = rng.binomial(1, lam_dt, size=(T, n_names))           # 0/1 是否发生跳跃
        J = rng.normal(jump_mean, jump_std, size=(T, n_names)) * N_jump
    else:
        J = np.zeros((T, n_names))

    # 日对数收益：mu*dt + sigma_t * Z_t + 跳跃
    drift_day = (mu - 0.5 * (sigma_ann**2)) * dt      # 更贴近连续 GBM（含 Itô 修正）
    r = drift_day + sigma_day_t * Z + J               # T × n

    # 价格路径
    s0 = rng.uniform(price0_bounds[0], price0_bounds[1], size=n_names)
    close_path = s0 * np.exp(np.cumsum(r, axis=0))

    close = pd.DataFrame(close_path, index=calendar, columns=tickers)

    # 构造开盘价：用“前日收盘 × 隔夜噪声”
    # 隔夜波动设为 overnight_scale * 当日波动
    overnight_eps = rng.standard_normal((T, n_names))
    overnight_sig = overnight_scale * sigma_day_t
    overnight_r = overnight_sig * overnight_eps

    open_ = close.shift(1) * np.exp(pd.DataFrame(overnight_r, index=calendar, columns=tickers))
    # 第一日 open 没有前收，设为首日收盘加微小扰动
    open_.iloc[0, :] = close.iloc[0, :] * np.exp(pd.Series(overnight_r[0, :], index=tickers))

    return open_, close, tickers, calendar



