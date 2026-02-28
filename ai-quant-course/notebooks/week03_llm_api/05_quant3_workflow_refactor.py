# -*- coding: utf-8 -*-
"""
第3周-Part5: 完整量化3.0智能分析工作流 (增强版)
=============================================
重点改进：
- 支持动态修改分析股票的代码、时间范围、起始资金
- 核心逻辑封装成复用性强的函数
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from quant_core.data import fetch_stock
from quant_core.ai import DeepSeekClient, StockAnalyzer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


# ================================================================
# 动态分析与可扩展性支持
# ================================================================

def analyze_stock_workflow(
    stock_code: str = "600519",                  # 股票代码，默认贵州茅台
    stock_name: str = "贵州茅台",               # 股票名称
    days: int = 120,                            # 数据时长 (天)，默认最近120天
    initial_capital: float = 100_000,           # 初始资金，默认10万元
    ai_client: DeepSeekClient = None            # 自定义AI客户端 (可选)
):
    """
    量化3.0数据+AI分析工作流 (动态参数化)

    参数：
        stock_code (str):    股票代码
        stock_name (str):    股票名称
        days (int):          获取数据的时间跨度（天数）
        initial_capital (float): 初始资金，用于策略回测
        ai_client (DeepSeekClient): 自定义AI客户端（可选）

    返回���
        dict，包含完整分析结果
    """
    # 如果未提供自定义AI客户端，则使用默认
    if not ai_client:
        analyzer = StockAnalyzer()
    else:
        analyzer = StockAnalyzer(api_key=ai_client.api_key)

    # --- 步骤1: 数据获取与指标计算 ---
    df = fetch_stock(stock_code, days=days)
    df["sma5"] = df["close"].rolling(5).mean()
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma60"] = df["close"].rolling(60).mean()
    df["daily_return"] = df["close"].pct_change()
    latest = df.iloc[-1]

    # --- 步骤2: 调用AI模型分析 ---
    print("\n准备调用AI分析...")
    analysis = analyzer.analyze(stock_code, stock_name)

    # 提取AI分析结果
    trend = analysis.get("trend", "sideways")
    trend_cn = analysis.get("trend_cn", "无明显趋势")

    risk_cn = analysis.get("risk_cn", "未知")
    confidence = float(analysis.get("confidence", 0))
    support = analysis.get("support", "N/A")
    resistance = analysis.get("resistance", "N/A")

    # --- 步骤3: 自动生成决策信号 ---
    if trend == "uptrend" and confidence > 0.6:
        signal = "BUY"
        signal_cn = "买入信号"
        signal_color = "#FF4040"
    elif trend == "downtrend" and confidence > 0.6:
        signal = "SELL"
        signal_cn = "卖出信号"
        signal_color = "#40FF40"
    else:
        signal = "HOLD"
        signal_cn = "观望信号"
        signal_color = "#FFD700"

    # 打印分析结果
    print("\nAI分析结果:")
    print(f"   股票名称: {stock_name} ({stock_code})")
    print(f"   趋势: {trend_cn}")
    print(f"   支撑位: {support}")
    print(f"   压力位: {resistance}")
    print(f"   信号: {signal_cn}")
    print(f"   风险评级: {risk_cn}")
    print(f"   AI置信度: {confidence:.2%}")

    # 返回结果
    return {
        "dataframe": df,
        "trend": trend,
        "trend_cn": trend_cn,
        "risk_cn": risk_cn,
        "confidence": confidence,
        "signal": signal,
        "signal_cn": signal_cn,
        "signal_color": signal_color,
        "support": support,
        "resistance": resistance,
        "capital": initial_capital,
        "analysis": analysis,
    }


def plot_analysis_results(result: dict, stock_name: str, stock_code: str):
    """
    可视化分析结果 (自动加入AI标注和决策信号)

    参数：
        result: `analyze_stock_workflow` 输出的结果字典
        stock_name: 股票名称
        stock_code: 股票代码
    """
    df = result["dataframe"]
    latest = df.iloc[-1]

    fig = plt.figure(figsize=(18, 14))

    # ---- 上图: 价格+均线+信号 ----
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(df.index, df["close"], label="收盘价", linewidth=1.5, color="#333333")
    ax1.plot(df.index, df["sma5"], label="MA5(5日均线)", linewidth=1, color="#FF6B6B", alpha=0.8)
    ax1.plot(df.index, df["sma20"], label="MA20(20日均线)", linewidth=1, color="#4ECDC4", alpha=0.8)

    # 标注支撑位/压力位 (如果AI有输出)
    try:
        support = float(result["support"])
        ax1.axhline(y=support, color="#00AA00", linestyle="--", alpha=0.6)
        ax1.text(df.index[0], support, " 支撑 ", va="bottom", color="#00AA00")
    except ValueError:
        pass

    ax1.annotate(
        result["signal_cn"],
        xy=(df.index[-1], latest["close"]),
        xycoords="data",
        xytext=(-30, 40),
        textcoords="offset points",
        arrowprops=dict(facecolor=result["signal_color"], shrink=0.05),
        fontsize=9,
    )

    ax1.set_title(f"{stock_name} ({stock_code}) - 技术分析")
    ax1.grid()

    print("\n正在展示图表...")
    plt.show()


# ================================================================
# 用户测试入口 - 提供动态修改参数
# ================================================================

if __name__ == "__main__":
    # 从用户输入获取动态参数
    stock_code = input("请输入股票代码(default=600519 贵州茅台): ") or "600519"
    stock_name = input(f"请输入股票名称(如 贵州茅台): ") or "贵州茅台"
    days = int(input("请输入分析时间跨度(单位:天，default=120): ") or 120)
    initial_capital = float(input("请输入初始资金(默认10万): ") or 100_000)

    # 执行量化3.0分析工作流
    result = analyze_stock_workflow(
        stock_code=stock_code,
        stock_name=stock_name,
        days=days,
        initial_capital=initial_capital
    )

    # 可视化结果
    plot_analysis_results(result, stock_name, stock_code)

    # 总结
    print("\n完成! 可以更换股票和测试参数继续运行!")