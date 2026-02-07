"""
第2周-Part3：Matplotlib 投资可视化
==================================
不是教你画图，而是教你画"有用的"投资分析图
"""

# 先设置路径，再导入
# import notebooks.week02_python_ds.path_setup  # noqa: F401  (或直接执行00_path_setup.py的内容)

# 更实用的写法：直接内联
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# ✅ 从公共模块导入，不再重复定义
from quant_core.data import fetch_stock
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 设置全局中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 图表1：专业K线分析图（价格+均线+成交量+MACD布局）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_stock_analysis(df: pd.DataFrame, name: str, code: str):
    """
    绘制专业级股票分析图
    包含：价格+均线 / 成交量 / 收益率分布 / 滚动波动率
    """
    # 计算指标
    df = df.copy()
    df["sma5"]  = df["close"].rolling(5).mean()
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma60"] = df["close"].rolling(60).mean()
    df["daily_return"] = df["close"].pct_change()
    df["volatility"] = df["daily_return"].rolling(20).std() * np.sqrt(252)

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(4, 4, hspace=0.35, wspace=0.3,
                           height_ratios=[3, 1, 1.2, 1.2])

    # ---- 主图：价格与均线 ----
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df.index, df["close"], label="收盘价", linewidth=1.5, color="#333")
    ax1.plot(df.index, df["sma5"],  label="MA5",  linewidth=1, color="#FF6B6B", alpha=0.8)
    ax1.plot(df.index, df["sma20"], label="MA20", linewidth=1, color="#4ECDC4", alpha=0.8)
    ax1.plot(df.index, df["sma60"], label="MA60", linewidth=1, color="#45B7D1", alpha=0.8)

    ax1.fill_between(df.index, df["sma5"], df["sma20"],
                     where=(df["sma5"] > df["sma20"]),
                     alpha=0.08, color="red")
    ax1.fill_between(df.index, df["sma5"], df["sma20"],
                     where=(df["sma5"] <= df["sma20"]),
                     alpha=0.08, color="green")

    ax1.set_title(f"{name}({code}) 技术分析看板", fontsize=16, fontweight="bold", pad=15)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel("价格 (元)")

    # ---- 成交量 ----
    ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
    colors_vol = ["#FF4444" if c >= o else "#00AA00"
                  for c, o in zip(df["close"], df["open"])]
    ax2.bar(df.index, df["volume"] / 10000, color=colors_vol, alpha=0.6, width=1)
    ax2.set_ylabel("成交量(万手)")
    ax2.grid(True, alpha=0.3)

    # ---- 收益率分布直方图 ----
    ax3 = fig.add_subplot(gs[2, :2])
    returns = df["daily_return"].dropna()
    ax3.hist(returns * 100, bins=50, color="#5B86E5", alpha=0.7, edgecolor="white")
    ax3.axvline(x=0, color="black", linewidth=1, linestyle="--")
    ax3.axvline(x=returns.mean() * 100, color="red", linewidth=1.5,
                linestyle="--", label=f"均值: {returns.mean():.3%}")
    ax3.set_title("日收益率分布", fontsize=12)
    ax3.set_xlabel("日收益率 (%)")
    ax3.set_ylabel("频次")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # ---- 滚动波动率 ----
    ax4 = fig.add_subplot(gs[2, 2:])
    ax4.fill_between(df.index, df["volatility"] * 100, alpha=0.3, color="#FF6B6B")
    ax4.plot(df.index, df["volatility"] * 100, color="#FF6B6B", linewidth=1)
    ax4.set_title("20日滚动年化波动率", fontsize=12)
    ax4.set_ylabel("波动率 (%)")
    ax4.grid(True, alpha=0.3)

    # ---- 关键统计指标面板 ----
    ax5 = fig.add_subplot(gs[3, :])
    ax5.axis("off")

    cum_return = (1 + returns).prod() - 1
    annual_return = (1 + cum_return) ** (252 / len(returns)) - 1
    max_drawdown = ((df["close"] / df["close"].cummax()) - 1).min()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

    stats_text = (
        f"┃ 累计收益: {cum_return:.2%}  "
        f"┃ 年化收益: {annual_return:.2%}  "
        f"┃ 最大回撤: {max_drawdown:.2%}  "
        f"┃ Sharpe比率: {sharpe:.2f}  "
        f"┃ 年化波动: {returns.std() * np.sqrt(252):.2%}  "
        f"┃ 交易天数: {len(returns)} ┃"
    )
    ax5.text(0.5, 0.5, stats_text, transform=ax5.transAxes,
             fontsize=11, ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.8", facecolor="#F0F4FF", edgecolor="#5B86E5", alpha=0.9))

    plt.savefig(f"{code}_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ 图表已保存为 {code}_analysis.png")


# 生成茅台分析图
maotai = fetch_stock("600519")
plot_stock_analysis(maotai, "贵州茅台", "600519")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 图表2：多股票对比图（投资组合视角）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_portfolio_comparison(stocks: dict):
    """
    多股票对比分析图

    参数：
        stocks: {"股票名": "代码", ...}
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 获取所有股票数据
    all_data = {}
    for name, code in stocks.items():
        df = fetch_stock(code)
        df["daily_return"] = df["close"].pct_change()
        df["cumulative"] = (1 + df["daily_return"]).cumprod()
        all_data[name] = df

    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8"]

    # ---- 左上：累计收益率对比 ----
    ax = axes[0, 0]
    for i, (name, df) in enumerate(all_data.items()):
        ax.plot(df.index, (df["cumulative"] - 1) * 100,
                label=name, linewidth=1.5, color=colors[i])
    ax.set_title("累计收益率对比 (%)", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")

    # ---- 右上：滚动20日波动率对比 ----
    ax = axes[0, 1]
    for i, (name, df) in enumerate(all_data.items()):
        vol = df["daily_return"].rolling(20).std() * np.sqrt(252) * 100
        ax.plot(df.index, vol, label=name, linewidth=1, color=colors[i])
    ax.set_title("20日滚动年化波动率对比 (%)", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- 左下：收益率分布对比 ----
    ax = axes[1, 0]
    for i, (name, df) in enumerate(all_data.items()):
        ax.hist(df["daily_return"].dropna() * 100, bins=40,
                alpha=0.5, label=name, color=colors[i], edgecolor="white")
    ax.set_title("日收益率分布对比", fontsize=13, fontweight="bold")
    ax.set_xlabel("日收益率 (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- 右下：相关性热力图 ----
    ax = axes[1, 1]
    returns_df = pd.DataFrame({
        name: df["daily_return"] for name, df in all_data.items()
    }).dropna()
    corr = returns_df.corr()

    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, fontsize=10)
    ax.set_yticklabels(corr.columns, fontsize=10)

    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=12, fontweight="bold")

    ax.set_title("收益率相关性矩阵", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("投资组合对比分析", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("portfolio_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ 图表已保存为 portfolio_comparison.png")


# 对比三只不同行业的股票
plot_portfolio_comparison({
    "恩捷股份": "002812",   # 电池
    "万华化学": "600309",     # 化工
    "农业银行": "601288",   # 金融
})