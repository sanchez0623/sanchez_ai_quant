# -*- coding: utf-8 -*-
"""
第4周-Part2: 资产类别分析 -- 用数据看清风险与收益
================================================
股票/债券/黄金/现金, 到底谁赚得多? 谁风险大?
不要猜, 用数据说话.
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
        col_name = str(col).strip()
        if col_name in candidates:
            return col

    # 兜底：选第一个可被解析为日期且非空比例较高的列
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


def fetch_asset_data(start_date="20180101"):
    """
    获取多类资产的历史数据, 用于横向对比

    返回: DataFrame, 列为各资产的每日收盘价, 行为日期

    术语:
      沪深300: A股最大300只股票的指数, 代表股市整体
      中证国债: 中国国债指数, 代表债券市场
      黄金: 以人民币计价的黄金价格
      货币基金: 类似余额宝, 代表"现金"的收益
    """
    assets = {}

    # 1. 股票 -- 沪深300指数
    print("  获取沪深300指数...")
    try:
        df = ak.stock_zh_index_daily(symbol="sh000300")
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df = df[df.index >= start_date]
        assets["stock_hs300"] = df["close"]
        print(f"    成功, {len(df)} 条数据")
    except Exception as e:
        print(f"    失败({e})")

    # 2. 黄金
    print("  获取黄金价格...")
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

        start_ts = pd.to_datetime(start_date)
        df = df[df.index >= start_ts]

        price_col = _pick_gold_price_column(df, date_col=date_col)
        assets["gold"] = pd.to_numeric(df[price_col], errors="coerce")
        print(f"    黄金字段识别: 日期列={date_col}, 价格列={price_col}")
        print(f"    成功, {len(df)} 条数据")
    except Exception as e:
        print(f"    黄金数据: {e}, 跳过")

    # 合并所有资产
    result = pd.DataFrame(assets).dropna(how="all")
    result = result.ffill().dropna()
    return result


def analyze_assets(
    start_date="20180101",
    risk_free_rate=0.025,
):
    """
    多资产对比分析

    参数:
        start_date:     分析起始日期
        risk_free_rate: 无风险利率(年化), 用于计算夏普比率

    术语:
      年化收益率(Annualized Return): 把任意时段的收益率折算成"每年赚多少"
        公式: (1 + 累计收益率) ^ (252/交易天数) - 1
      年化波动率(Annualized Volatility): 日收益率标准差 * sqrt(252)
        衡量价格的波动剧烈程度, 越高风险越大
      夏普比率(Sharpe Ratio): (年化收益 - 无风险利率) / 年化波动
        每承担1单位风险, 获得多少超额收益. >1优秀, >2非常优秀
      最大回撤(Max Drawdown): 从最高点跌到最低点的最大幅度
        衡量"最坏情况你会亏多少"
      卡尔玛比率(Calmar Ratio): 年化收益 / 最大回撤
        衡量"收益能不能补偿你承受的最大亏损". >1较好, >2优秀
    """
    print("=" * 60)
    print("  多资产对比分析")
    print("=" * 60)

    df = fetch_asset_data(start_date)
    if df.empty:
        print("  数据获取失败, 无法分析")
        return

    # 中文名称映射
    name_map = {
        "stock_hs300": "沪深300(股票)",
        "gold": "黄金",
    }

    # 计算各资产的统计指标
    returns_df = df.pct_change().dropna()
    trading_days = len(returns_df)

    stats = []
    for col in df.columns:
        daily_ret = returns_df[col]
        cum_ret = (1 + daily_ret).prod() - 1
        ann_ret = (1 + cum_ret) ** (252 / trading_days) - 1
        ann_vol = daily_ret.std() * np.sqrt(252)
        sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else 0

        # 最大回撤
        cummax = (1 + daily_ret).cumprod().cummax()
        drawdown = ((1 + daily_ret).cumprod() / cummax - 1)
        max_dd = drawdown.min()

        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

        stats.append({
            "asset": name_map.get(col, col),
            "asset_en": col,
            "cumulative_return": cum_ret,         # 累计收益率
            "annualized_return": ann_ret,          # 年化收益率
            "annualized_volatility": ann_vol,      # 年化波动率
            "sharpe_ratio": sharpe,                # 夏普比率
            "max_drawdown": max_dd,                # 最大回撤
            "calmar_ratio": calmar,                # 卡尔玛比率
        })

    stats_df = pd.DataFrame(stats)

    # 打印统计表
    print(f"\n  分析区间: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"  交易天数: {trading_days}")
    print(f"  无风险利率: {risk_free_rate:.2%}\n")

    display_df = stats_df[["asset"]].copy()
    display_df["累计收益率"] = stats_df["cumulative_return"].apply(lambda x: f"{x:.2%}")
    display_df["年化收益率"] = stats_df["annualized_return"].apply(lambda x: f"{x:.2%}")
    display_df["年化波动率"] = stats_df["annualized_volatility"].apply(lambda x: f"{x:.2%}")
    display_df["夏普比率"] = stats_df["sharpe_ratio"].apply(lambda x: f"{x:.2f}")
    display_df["最大回撤"] = stats_df["max_drawdown"].apply(lambda x: f"{x:.2%}")
    display_df["卡尔玛比率"] = stats_df["calmar_ratio"].apply(lambda x: f"{x:.2f}")
    display_df.columns = [
        "资产", "累计收益率", "年化收益率", "年化波动率(风险)",
        "夏普比率(风险调整收益)", "最大回撤(最大亏损)", "卡尔玛比率(收益/回撤)"
    ]
    print(display_df.to_string(index=False))

    # ---- 可视化 ----
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 左上: 累计收益率走势
    ax = axes[0, 0]
    colors = ["#FF6B6B", "#FFD93D", "#4ECDC4", "#45B7D1", "#96CEB4"]
    for i, col in enumerate(df.columns):
        cum = (1 + returns_df[col]).cumprod()
        ax.plot(cum.index, (cum - 1) * 100,
                label=name_map.get(col, col),
                linewidth=1.5, color=colors[i % len(colors)])
    ax.set_title("累计收益率对比(%)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")

    # 右上: 风险收益散点图
    ax = axes[0, 1]
    for i, row in stats_df.iterrows():
        ax.scatter(row["annualized_volatility"] * 100, row["annualized_return"] * 100,
                   s=200, color=colors[i % len(colors)], zorder=5, edgecolors="black")
        ax.annotate(name_map.get(row["asset_en"], row["asset"]),
                    xy=(row["annualized_volatility"] * 100, row["annualized_return"] * 100),
                    xytext=(8, 8), textcoords="offset points", fontsize=10)
    ax.set_xlabel("年化波动率(风险) %", fontsize=11)
    ax.set_ylabel("年化收益率 %", fontsize=11)
    ax.set_title("风险-收益散点图", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # 左下: 回撤走势
    ax = axes[1, 0]
    for i, col in enumerate(df.columns):
        cum = (1 + returns_df[col]).cumprod()
        dd = (cum / cum.cummax() - 1) * 100
        ax.fill_between(dd.index, dd, alpha=0.3, color=colors[i % len(colors)])
        ax.plot(dd.index, dd, label=name_map.get(col, col),
                linewidth=1, color=colors[i % len(colors)])
    ax.set_title("回撤走势(越接近0越好)", fontsize=13, fontweight="bold")
    ax.set_ylabel("回撤(%)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 右下: 相关性热力图
    ax = axes[1, 1]
    corr = returns_df.rename(columns=name_map).corr()
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, fontsize=10)
    ax.set_yticklabels(corr.columns, fontsize=10)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=12, fontweight="bold")
    ax.set_title("相关性矩阵(越接近-1分散效果越好)", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle("多资产对比分析", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig("asset_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n  图表已保存: asset_comparison.png")

    return stats_df


# 执行分析 -- 参数可以随时修改!
stats = analyze_assets(start_date="20160101", risk_free_rate=0.025)