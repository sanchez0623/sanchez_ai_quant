# -*- coding: utf-8 -*-
"""
第4周-Part1: 宏观经济学 -- 用数据理解经济周期
=============================================
不是纯理论, 每个概念都用真实数据验证
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from quant_core.data import fetch_stock
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


# ================================================================
# 1.1 利率与股市的关系(用真实数据验证)
# ================================================================

def analyze_rate_vs_market(market_index="sh000300", market_name="沪深300", days=1500):
    """
    分析利率变动与股市的关系

    术语:
      基准利率: 央行设定的"利率之锚", 影响所有借贷成本
      沪深300: 沪深两市最大的300只股票组成的指数, 代表A股整体表现
      国债收益率: 政府借钱的利率, 是"无风险利率"的代表

    参数:
        market_index: 指数代码
        market_name:  指数名称
        days:         数据时间跨度
    """
    print("=" * 60)
    print(f"  分析: 利率 vs {market_name}")
    print("=" * 60)

    # 获取10年期国债收益率(代表市场利率水平)
    print("\n  获取国债收益率数据...")
    try:
        bond_df = ak.bond_zh_us_rate(start_date="20200101")
        bond_df = bond_df[["日期", "中国国债收益率10年"]].dropna()
        bond_df.columns = ["date", "cn_10y_rate"]
        bond_df["date"] = pd.to_datetime(bond_df["date"])
        bond_df = bond_df.set_index("date")
        bond_df["cn_10y_rate"] = pd.to_numeric(bond_df["cn_10y_rate"], errors="coerce")
        print(f"    获取到 {len(bond_df)} 条国债收益率数据")
    except Exception as e:
        print(f"    国债数据获取失败({e}), 使用模拟数据演示")
        dates = pd.date_range("2020-01-01", periods=1200, freq="D")
        bond_df = pd.DataFrame({
            "cn_10y_rate": np.random.normal(2.8, 0.3, len(dates)).cumsum() * 0.001 + 2.8
        }, index=dates)

    # 获取股市指数
    print(f"  获取{market_name}数据...")
    try:
        index_df = ak.stock_zh_index_daily(symbol=market_index)
        index_df["date"] = pd.to_datetime(index_df["date"])
        index_df = index_df.set_index("date")
        index_df = index_df[index_df.index >= "2020-01-01"]
        print(f"    获取到 {len(index_df)} 条指数数据")
    except Exception as e:
        print(f"    指数数据获取失败({e}), 使用模拟数据")
        dates = pd.date_range("2020-01-01", periods=1200, freq="D")
        index_df = pd.DataFrame({
            "close": np.random.normal(0.0003, 0.015, len(dates)).cumsum() * 100 + 4000
        }, index=dates)

    # 合并数据(取交集日期)
    merged = pd.DataFrame({
        "rate": bond_df["cn_10y_rate"],
        "index": index_df["close"],
    }).dropna()

    if len(merged) < 10:
        print("    数据不足, 跳过分析")
        return

    # 计算利率变动方向
    merged["rate_change"] = merged["rate"].diff()             # 利率变化
    merged["index_return"] = merged["index"].pct_change()     # 指数日收益率

    # 计算相关性
    corr = merged["rate_change"].corr(merged["index_return"])
    print(f"\n  利率变化 vs 股市收益率 相关性: {corr:.4f}")
    print(f"  解读: ", end="")
    if corr < -0.1:
        print("负相关 -- 利率下降时股市倾向上涨(符合理论!)")
    elif corr > 0.1:
        print("正相关 -- 可能处于经济复苏期(利率和股市同升)")
    else:
        print("相关性弱 -- 短期内利率不是股市的主要驱动因素")

    # 可视化
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(merged.index, merged["index"], color="#5B86E5", linewidth=1)
    axes[0].set_title(f"{market_name}走势", fontsize=12)
    axes[0].set_ylabel("指数点位")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(merged.index, merged["rate"], color="#FF6B6B", linewidth=1)
    axes[1].set_title("中国10年期国债收益率(%)", fontsize=12)
    axes[1].set_ylabel("收益率(%)")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"利率 vs 股市: 相关性={corr:.3f}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("rate_vs_market.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  图表已保存: rate_vs_market.png")


analyze_rate_vs_market()


# ================================================================
# 1.2 经济周期与资产轮动(美林时钟的数据验证)
# ================================================================

def explain_business_cycle():
    """
    用图解方式讲解经济周期和资产轮动

    术语:
      资产轮动: 不同经济阶段, 表现最好的资产会"轮流"切换
      美林时钟: 根据"经济增长"和"通胀"两个维度, 把经济分为4个阶段
    """
    print("\n" + "=" * 60)
    print("  经济周期与资产轮动(美林时钟)")
    print("=" * 60)

    print("""
    美林时钟把经济分为4个阶段:

                 通胀上升
                    |
       ┌────────────┼────────────┐
       |  过热期     |   滞胀期    |
       |            |            |
       |  最佳资产:  |  最佳资产:  |
       |  大宗商品   |  现金       |
       |  (石油黄金)  |  (货币基金)  |
       |            |            |
  经济  |────────────┼────────────| 经济
  加速  |            |            | 减速
       |  复苏期     |  衰退期    |
       |            |            |
       |  最佳资产:  |  最佳资产:  |
       |  股票       |  债券       |
       |  (企业盈利增)|  (利率下降)  |
       |            |            |
       └────────────┼────────────┘
                    |
                 通胀下降

    实际操作中的简化版本:
      经济好 + 通胀低 = 多买股票
      经济好 + 通胀高 = 多买商品
      经济差 + 通胀低 = 多买债券
      经济差 + 通胀高 = 多持现金

    量化应用(第10周会实现):
      用GDP增速判断经济好坏
      用CPI判断通胀高低
      自动调整股票/债券/商品/现金的比例
    """)


explain_business_cycle()