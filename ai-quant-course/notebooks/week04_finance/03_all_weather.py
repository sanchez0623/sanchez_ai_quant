# -*- coding: utf-8 -*-
"""
第4周-Part3: 全天候(All Weather)资产配置
=======================================
桥水基金达利奥的经典策略, 用Python模拟验证
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def _pick_gold_date_column(df: pd.DataFrame) -> str:
    """自动识别黄金日期列。"""
    candidates = ["日期", "交易时间", "交易日期", "时间", "date", "datetime"]
    for col in df.columns:
        if str(col).strip() in candidates:
            return col

    best_col = None
    best_valid = 0
    for col in df.columns:
        parsed = pd.to_datetime(df[col], errors="coerce")
        valid = parsed.notna().sum()
        if valid > best_valid:
            best_valid = valid
            best_col = col

    if best_col is None or best_valid == 0:
        raise ValueError(f"未识别到可用日期列，当前列: {list(df.columns)}")
    return best_col


def _pick_gold_price_column(df: pd.DataFrame, date_col: str = None) -> str:
    """优先按常见列名选择黄金价格列，失败则回退到第一个可转数值列。"""
    preferred = [
        "晚盘价", "收盘", "收盘价", "今收盘", "价格", "金价", "AU9999", "AU99.99", "基准价"
    ]
    for col in df.columns:
        col_name = str(col).strip()
        if any(key in col_name for key in preferred):
            return col

    for col in df.columns:
        if date_col is not None and col == date_col:
            continue
        if str(col).strip() in {"日期", "交易时间", "交易日期", "时间", "date", "datetime"}:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().sum() > 0:
            return col

    raise ValueError("未识别到可用黄金价格列")


def _fetch_etf_close_sina(symbol: str, start_date: str) -> pd.Series:
    """获取ETF日线收盘价（Sina源，稳定性较好）。"""
    df = ak.fund_etf_hist_sina(symbol=symbol)
    if df is None or df.empty:
        raise ValueError(f"ETF {symbol} 返回空数据")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").set_index("date")
    start_ts = pd.to_datetime(start_date)
    df = df[df.index >= start_ts]

    if "close" not in df.columns:
        raise ValueError(f"ETF {symbol} 未找到close列，当前列: {list(df.columns)}")

    return pd.to_numeric(df["close"], errors="coerce")


def simulate_portfolio(
    weights: dict,
    portfolio_name: str = "My Portfolio",
    start_date: str = "20180101",
    initial_capital: float = 100_000,
    rebalance_freq: str = "QE",
):
    """
    模拟资产配置组合的历史表现

    参数:
        weights: 配置比例, 如 {"stock_hs300": 0.3, "bond_cn": 0.4, "gold": 0.15, "commodity_cn": 0.1, "cash_mm": 0.05}
                 所有权重之和必须为1
        portfolio_name: 组合名称
        start_date:     起始日期
        initial_capital: 初始资金
        rebalance_freq:  再平衡频率
                         "Q" = 每季度再平衡
                         "M" = 每月再平衡
                         "Y" = 每年再平衡

    术语:
      再平衡(Rebalance): 定期把组合比例调回目标比例.
        例: 目标是股票60%债券40%. 股票涨了变成70%:30%,
        就卖掉一些股票买入债券, 恢复60:40.
        作用: 强制"高抛低吸", 控制风险.

    返回:
        DataFrame, 包含每日净值和回撤
    """
    total_weight_input = sum(weights.values())
    if total_weight_input <= 0:
        raise ValueError("权重之和必须大于0")
    normalized_weights = {k: v / total_weight_input for k, v in weights.items()}
    if abs(total_weight_input - 1.0) >= 0.01:
        print(f"  {portfolio_name}: 输入权重和={total_weight_input:.2f}，已自动归一化为1.00")

    # 获取各资产数据
    asset_prices = {}

    if "stock_hs300" in normalized_weights:
        try:
            df = ak.stock_zh_index_daily(symbol="sh000300")
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            asset_prices["stock_hs300"] = df["close"]
        except Exception as e:
            print(f"  沪深300获取失败: {e}")

    if "gold" in normalized_weights:
        try:
            try:
                df = ak.spot_golden_benchmark_sge(
                    start_date=start_date.replace("-", ""),
                    end_date="20261231"
                )
            except TypeError:
                df = ak.spot_golden_benchmark_sge()

            df.columns = [c.strip() for c in df.columns]

            date_col = _pick_gold_date_column(df)
            df["date"] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date").set_index("date")

            price_col = _pick_gold_price_column(df, date_col=date_col)
            asset_prices["gold"] = pd.to_numeric(df[price_col], errors="coerce")
        except Exception as e:
            print(f"  黄金获取失败: {e}")

    if "bond_cn" in normalized_weights:
        try:
            # 10年国债ETF（511010）
            asset_prices["bond_cn"] = _fetch_etf_close_sina("sh511010", start_date)
        except Exception as e:
            print(f"  债券ETF获取失败: {e}")
            try:
                # 兜底：上证国债指数
                df = ak.stock_zh_index_daily(symbol="sh000012")
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["date"]).set_index("date")
                df = df[df.index >= pd.to_datetime(start_date)]
                asset_prices["bond_cn"] = pd.to_numeric(df["close"], errors="coerce")
                print("  债券使用兜底数据源: sh000012(上证国债指数)")
            except Exception as e2:
                print(f"  债券获取失败: {e2}")

    if "cash_mm" in normalized_weights:
        try:
            # 货币ETF（511880）
            asset_prices["cash_mm"] = _fetch_etf_close_sina("sh511880", start_date)
        except Exception as e:
            print(f"  货币ETF获取失败: {e}")

    if "commodity_cn" in normalized_weights:
        try:
            # 商品ETF（510170）
            asset_prices["commodity_cn"] = _fetch_etf_close_sina("sh510170", start_date)
        except Exception as e:
            print(f"  商品ETF获取失败: {e}")
            try:
                # 兜底：有色ETF（159980）
                asset_prices["commodity_cn"] = _fetch_etf_close_sina("sz159980", start_date)
                print("  商品使用兜底数据源: sz159980(有色ETF)")
            except Exception as e2:
                print(f"  商品获取失败: {e2}")

    prices_df = pd.DataFrame(asset_prices).dropna(how="all")
    prices_df = prices_df[prices_df.index >= start_date]
    prices_df = prices_df.ffill().dropna()

    if prices_df.empty or len(prices_df) < 10:
        print(f"  {portfolio_name}: 数据不足, 跳过模拟")
        return None

    # 计算各资产日收益率
    returns_df = prices_df.pct_change().fillna(0)

    # 模拟组合净值(考虑再平衡)
    nav = [initial_capital]  # 净值序列
    current_weights = {k: v for k, v in normalized_weights.items() if k in returns_df.columns}

    # 归一化权重(只保留有数据的资产)
    total_w = sum(current_weights.values())
    current_weights = {k: v / total_w for k, v in current_weights.items()}

    # 标记再平衡日期
    rebalance_dates = set(
        returns_df.resample(rebalance_freq).last().index
    )

    for i in range(1, len(returns_df)):
        date = returns_df.index[i]
        daily_return = sum(
            current_weights[asset] * returns_df.iloc[i][asset]
            for asset in current_weights
        )
        new_nav = nav[-1] * (1 + daily_return)
        nav.append(new_nav)

        # 再平衡: 恢复目标权重
        if date in rebalance_dates:
            current_weights = {k: v / total_w for k, v in normalized_weights.items() if k in returns_df.columns}

    # 构建结果DataFrame
    result = pd.DataFrame({
        "nav": nav,
        "date": returns_df.index,
    }).set_index("date")

    result["daily_return"] = result["nav"].pct_change()
    result["cumulative_return"] = result["nav"] / initial_capital - 1
    result["drawdown"] = result["nav"] / result["nav"].cummax() - 1

    return result


def compare_portfolios(
    portfolios: dict,
    start_date: str = "20180101",
    initial_capital: float = 100_000,
    risk_free_rate: float = 0.025,
):
    """
    对比多个资产配置方案

    参数:
        portfolios: {
            "方案名": {"asset1": weight1, "asset2": weight2, ...},
            ...
        }
        start_date:      起始日期
        initial_capital:  初始资金
        risk_free_rate:   无风险利率(年化)
    """
    print("=" * 60)
    print("  资产配置方案对比")
    print("=" * 60)

    results = {}
    stats_list = []

    for name, weights in portfolios.items():
        print(f"\n  模拟 [{name}]...")
        weight_str = " + ".join([f"{k}:{v:.0%}" for k, v in weights.items()])
        print(f"    配比: {weight_str}")

        nav_df = simulate_portfolio(
            weights=weights,
            portfolio_name=name,
            start_date=start_date,
            initial_capital=initial_capital,
        )

        if nav_df is None:
            continue

        results[name] = nav_df

        # 计算绩效指标
        daily_ret = nav_df["daily_return"].dropna()
        trading_days = len(daily_ret)
        cum_ret = nav_df["cumulative_return"].iloc[-1]
        ann_ret = (1 + cum_ret) ** (252 / trading_days) - 1
        ann_vol = daily_ret.std() * np.sqrt(252)
        sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else 0
        max_dd = nav_df["drawdown"].min()
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

        stats_list.append({
            "方案": name,
            "累计收益率": f"{cum_ret:.2%}",
            "年化收益率": f"{ann_ret:.2%}",
            "年化波动率": f"{ann_vol:.2%}",
            "夏普比率": f"{sharpe:.2f}",
            "最大回撤": f"{max_dd:.2%}",
            "卡尔玛比率": f"{calmar:.2f}",
            "终值(元)": f"{nav_df['nav'].iloc[-1]:,.0f}",
        })

    if not stats_list:
        print("  所有方案模拟失败")
        return

    stats_df = pd.DataFrame(stats_list)
    print(f"\n  绩效对比({start_date[:4]}年至今, 初始资金{initial_capital:,.0f}元):")
    print(stats_df.to_string(index=False))

    # ---- 可视化 ----
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFD93D", "#96CEB4"]

    # 上图: 净值走势
    ax = axes[0]
    for i, (name, nav_df) in enumerate(results.items()):
        ax.plot(nav_df.index, nav_df["nav"] / initial_capital,
                label=name, linewidth=1.5, color=colors[i % len(colors)])
    ax.set_title("净值走势对比(1=初始本金)", fontsize=13, fontweight="bold")
    ax.set_ylabel("净值倍数")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color="black", linewidth=0.5, linestyle="--")

    # 下图: 回撤对比
    ax = axes[1]
    for i, (name, nav_df) in enumerate(results.items()):
        ax.fill_between(nav_df.index, nav_df["drawdown"] * 100,
                        alpha=0.2, color=colors[i % len(colors)])
        ax.plot(nav_df.index, nav_df["drawdown"] * 100,
                label=name, linewidth=1, color=colors[i % len(colors)])
    ax.set_title("回撤走势对比(越接近0越好)", fontsize=13, fontweight="bold")
    ax.set_ylabel("回撤(%)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("portfolio_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n  图表已保存: portfolio_comparison.png")


# ================================================================
# 执行对比 -- 修改这里的参数即可测试不同方案!
# ================================================================

compare_portfolios(
    portfolios={
        "全仓股票(激进)": {
            "stock_hs300": 1.0,
        },
        "等权基准(25/25/25/25)": {
            "stock_hs300": 0.20,
            "bond_cn": 0.20,
            "gold": 0.20,
            "commodity_cn": 0.20,
            "cash_mm": 0.20,
        },
        "股6债2金1现1(平衡)": {
            "stock_hs300": 0.6,
            "bond_cn": 0.2,
            "gold": 0.1,
            "cash_mm": 0.1,
        },
        "全天候(稳健)": {
            "stock_hs300": 0.30,
            "bond_cn": 0.40,
            "gold": 0.15,
            "cash_mm": 0.15,
        },
        # 桥水原版(简化): 30%股票 + 55%债券 + 7.5%黄金 + 7.5%商品
        # 说明: 经典原版会进一步细分债券久期，这里用bond_cn单桶代理。
        "桥水原版全天候(简化)": {
            "stock_hs300": 0.30,
            "bond_cn": 0.55,
            "gold": 0.075,
            "commodity_cn": 0.075,
        },
    },
    start_date="20180101",
    initial_capital=100_000,
    risk_free_rate=0.025,
)