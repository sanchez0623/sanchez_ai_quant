# -*- coding: utf-8 -*-
"""
第3周-Part5: 量化3.0完整工作流
==============================
数据获取 -> 指标计算 -> AI分析 -> 自动决策 -> 可视化
全流程打通!
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
# 流程1: 一键分析(1行代码搞定完整流程)
# ================================================================

print("=" * 60)
print("  流程1: 一键AI股票分析")
print("=" * 60)

analyzer = StockAnalyzer()

# 一行代码: 获取数据 -> 计算指标 -> AI分析 -> 结构化输出
result = analyzer.analyze("600519", "贵州茅台")

print("\n  AI分析结果(JSON):")
print(json.dumps(result, indent=2, ensure_ascii=False))

# 程序自动提取AI的结论用于决策
trend = result.get("trend", "unknown")
trend_cn = result.get("trend_cn", trend)
confidence = float(result.get("confidence", 0))
risk = result.get("risk_level", "unknown")
risk_cn = result.get("risk_cn", risk)
strength = result.get("strength", "N/A")
ai_support = result.get("support", "N/A")
ai_resistance = result.get("resistance", "N/A")

print("\n  程序自动提取的决策信息:")
print(f"    趋势判断:   {trend_cn}")
print(f"    趋势强度:   {strength}/10")
print(f"    支撑位:     {ai_support}")
print(f"    压力位:     {ai_resistance}")
print(f"    风险等级:   {risk_cn}")
print(f"    置信度:     {confidence}")


# ================================================================
# 流程2: 多股票对比 + AI排名
# ================================================================

print("\n" + "=" * 60)
print("  流程2: 多股票AI对比分析")
print("=" * 60)

stocks = {
    "贵州茅台": "600519",     # 消费
    "比亚迪": "002594",       # 新能源
    "招商银行": "600036",     # 金融
}

comparison_result = analyzer.compare(stocks)
print("\n  AI对比分析结果:")
print(json.dumps(comparison_result, indent=2, ensure_ascii=False))


# ================================================================
# 流程3: 数据 + AI + 可视化(完整量化3.0管线)
# ================================================================

print("\n" + "=" * 60)
print("  流程3: 完整量化3.0管线(含可视化)")
print("=" * 60)

# --- 第1步: 获取数据并计算指标 ---
df = fetch_stock("600519", days=120)
df["sma5"] = df["close"].rolling(5).mean()
df["sma20"] = df["close"].rolling(20).mean()
df["sma60"] = df["close"].rolling(60).mean()
df["daily_return"] = df["close"].pct_change()
df["volatility_20d"] = df["daily_return"].rolling(20).std() * np.sqrt(252)
df["cumulative_return"] = (1 + df["daily_return"]).cumprod() - 1

latest = df.iloc[-1]

# --- 第2步: AI分析 ---
print("\n  正在调用AI分析...")
ai_result = analyzer.analyze("600519", "贵州茅台")

ai_trend = ai_result.get("trend", "sideways")
ai_trend_cn = ai_result.get("trend_cn", "N/A")
ai_risk_cn = ai_result.get("risk_cn", "N/A")
ai_outlook = ai_result.get("outlook", ai_result.get("short_term_outlook", "N/A"))
ai_confidence = float(ai_result.get("confidence", 0))
ai_support = ai_result.get("support", "N/A")
ai_resistance = ai_result.get("resistance", "N/A")

# --- 第3步: 自动决策逻辑 ---
#
# 根据AI返回的趋势和置信度, 自动生成交易信号
#
# 术语说明:
#   信号(Signal):    系统建议的交易动作
#   置信度(Confidence): AI对自己判断的把握程度, 0~1之间
#   上涨趋势(Uptrend):  价格整体向上���动
#   下跌趋势(Downtrend): 价格整体向下运动
#   横盘震荡(Sideways):  价格在一个区间内波动

if ai_trend == "uptrend" and ai_confidence >= 0.6:
    signal = "BUY"
    signal_cn = "买入信号"
    signal_color = "#FF4444"
elif ai_trend == "downtrend" and ai_confidence >= 0.6:
    signal = "SELL"
    signal_cn = "卖出信号"
    signal_color = "#00AA00"
else:
    signal = "HOLD"
    signal_cn = "观望等待"
    signal_color = "#FFB347"

print(f"\n  自动决策结果: {signal_cn}")
print(f"    AI趋势:  {ai_trend_cn}")
print(f"    AI置信度: {ai_confidence}")
print(f"    AI风险:  {ai_risk_cn}")


# --- 第4步: 可视化 ---
print("\n  正在生成分析图表...")

fig = plt.figure(figsize=(18, 14))

# ---- 上图: 价格 + 均线 + AI标注 ----
ax1 = fig.add_subplot(3, 1, 1)
ax1.plot(df.index, df["close"], label="收盘价", linewidth=1.5, color="#333333")
ax1.plot(df.index, df["sma5"],  label="MA5(5日均线)",  linewidth=1, color="#FF6B6B", alpha=0.8)
ax1.plot(df.index, df["sma20"], label="MA20(20日均线)", linewidth=1, color="#4ECDC4", alpha=0.8)
ax1.plot(df.index, df["sma60"], label="MA60(60日均线)", linewidth=1, color="#45B7D1", alpha=0.8)

# 多头/空头区域填充
ax1.fill_between(df.index, df["sma5"], df["sma20"],
                 where=(df["sma5"] > df["sma20"]),
                 alpha=0.08, color="red")
ax1.fill_between(df.index, df["sma5"], df["sma20"],
                 where=(df["sma5"] <= df["sma20"]),
                 alpha=0.08, color="green")

# 标注AI给出的支撑位和压力位
try:
    support_val = float(ai_support)
    ax1.axhline(y=support_val, color="#00AA00", linewidth=1, linestyle="--", alpha=0.7)
    ax1.text(df.index[2], support_val, f"  AI支撑位: {support_val:.0f}",
             fontsize=9, color="#00AA00", va="bottom")
except (ValueError, TypeError):
    pass

try:
    resistance_val = float(ai_resistance)
    ax1.axhline(y=resistance_val, color="#FF4444", linewidth=1, linestyle="--", alpha=0.7)
    ax1.text(df.index[2], resistance_val, f"  AI压力位: {resistance_val:.0f}",
             fontsize=9, color="#FF4444", va="top")
except (ValueError, TypeError):
    pass

# 在最新一天标注交易信号
ax1.scatter(df.index[-1], latest["close"], color=signal_color, s=150, zorder=5, edgecolors="black")
ax1.annotate(
    f"{signal_cn}\n({ai_trend_cn})\n置信度: {ai_confidence:.0%}",
    xy=(df.index[-1], latest["close"]),
    xytext=(30, 30), textcoords="offset points",
    fontsize=9, fontweight="bold", color=signal_color,
    arrowprops=dict(arrowstyle="->", color=signal_color, lw=1.5),
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=signal_color, alpha=0.9),
)

ax1.set_title("贵州茅台(600519) - 量化3.0 AI分析看板", fontsize=14, fontweight="bold")
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_ylabel("价格(元)")


# ---- 中图: 成交量 ----
ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
colors_vol = ["#FF4444" if c >= o else "#00AA00"
              for c, o in zip(df["close"], df["open"])]
ax2.bar(df.index, df["volume"] / 10000, color=colors_vol, alpha=0.6, width=1)
ax2.set_ylabel("成交量(万手)")
ax2.set_title("成交量", fontsize=12)
ax2.grid(True, alpha=0.3)


# ---- 下图: 滚动波动率 + AI风险标注 ----
ax3 = fig.add_subplot(3, 1, 3, sharex=ax1)
ax3.fill_between(df.index, df["volatility_20d"] * 100, alpha=0.3, color="#FF6B6B")
ax3.plot(df.index, df["volatility_20d"] * 100, color="#FF6B6B", linewidth=1)
ax3.set_ylabel("年化波动率(%)")
ax3.set_title("20日滚动年化波动率", fontsize=12)
ax3.grid(True, alpha=0.3)

# 标注AI风险等级
risk_lower = str(risk).lower()
if "high" in risk_lower:
    risk_bg_color = "#F8D7DA"
elif "medium" in risk_lower:
    risk_bg_color = "#FFF3CD"
else:
    risk_bg_color = "#D4EDDA"

ax3.text(
    0.98, 0.92,
    f"AI风险评级: {ai_risk_cn}",
    transform=ax3.transAxes, fontsize=11, fontweight="bold",
    ha="right", va="top",
    bbox=dict(boxstyle="round,pad=0.5", facecolor=risk_bg_color, edgecolor="#333", alpha=0.9),
)


# ---- 底部AI摘要面板 ----
fig.subplots_adjust(bottom=0.15)
summary_text = (
    f"AI分析摘要  |  "
    f"趋势: {ai_trend_cn}  |  "
    f"信号: {signal_cn}  |  "
    f"支撑位: {ai_support}  |  "
    f"压力位: {ai_resistance}  |  "
    f"风险: {ai_risk_cn}  |  "
    f"置信度: {ai_confidence:.0%}  |  "
    f"展望: {ai_outlook}"
)

fig.text(
    0.5, 0.02, summary_text,
    ha="center", va="center", fontsize=10,
    bbox=dict(boxstyle="round,pad=0.6", facecolor="#EEF2FF", edgecolor="#5B86E5", alpha=0.95),
)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("quant3_workflow.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n  图表已保存: quant3_workflow.png")


# ================================================================
# 流程4: AI生成量化策略代码
# ================================================================

print("\n" + "=" * 60)
print("  流程4: AI自动生成策略代码")
print("=" * 60)

code_client = DeepSeekClient(model="r1")
code_client.set_system_prompt(
    "你是一位Python量化开发专家。"
    "生成的代码必须可以直接运行, 使用akshare获取数据(不用yfinance), "
    "包含详细的中文注释。"
)

strategy_prompt = f"""
基于以下AI分析结果, 生成一个简单的交易策略回测代码:

AI分析:
- 股票: 贵州茅台(600519)
- 趋势: {ai_trend_cn}
- 信号: {signal}
- 支撑位: {ai_support}
- 压力位: {ai_resistance}

要求:
1. 用akshare获取600519最近2年的日线数据
2. 策略规则: 收盘价>MA20 且 MA5>MA20时买入; MA5<MA20时卖出
3. 初始资金1000万元, 每次全仓操作
4. 计算: 累计收益率、年化收益率、最大回撤、Sharpe比率
5. 画出策略净值曲线 vs 买入持有基准的对比图
6. 最后打印一个汇总表格

请只输出完整可运行的Python代码。
"""

print("\n  正在让AI生成策略代码...")
strategy_code = code_client.chat(strategy_prompt, temperature=0.1)
print("\n  AI生成的策略代码:")
print("-" * 60)
print(strategy_code)
print("-" * 60)

print("""
  注意事项:
    AI生成的代码是"初稿", 你需要:
    1. 仔细检查代码逻辑是否合理
    2. 实际运行看是否报错
    3. 验证回测结果是否符合预期
    这就是"人机结合": AI写初版, 你审核优化
""")


# ================================================================
# 流程5: API费用追踪
# ================================================================

print("\n" + "=" * 60)
print("  流程5: 本次API费用追踪")
print("=" * 60)

# 估算本次实战的API费用
calls = [
    ("流程1: 单股票分析",          500,  1000),
    ("流程2: 多股票对比(3只)",     1500, 2000),
    ("流程3: 完整管线分析",        800,  1500),
    ("流程4: 策略代码生成",        400,  3000),
]

total_cost = 0.0
print(f"\n  {'调用场景':<30} {'输入Token':<12} {'输出Token':<12} {'费用(元)':<10}")
print("  " + "-" * 70)

for name, input_t, output_t in calls:
    # DeepSeek-V3定价: 输入 1元/百万Token, 输出 2元/百万Token
    cost = input_t * (1.0 / 1_000_000) + output_t * (2.0 / 1_000_000)
    total_cost += cost
    print(f"  {name:<30} {input_t:<12} {output_t:<12} {cost:.6f}")

print("  " + "-" * 70)
print(f"  {'合计':<30} {'':12} {'':12} {total_cost:.6f}")

print(f"""
  费用总结:
    本次实战总费用:  约 {total_cost:.4f} 元!
""")


# ================================================================
# 第3周总结
# ================================================================

print("""
================================================================
    第3周 量化3.0工作流 -- 全部完成!
================================================================

  本周你掌握了:

  流程1: 一键AI股票分析
         fetch_stock -> 计算指标 -> AI分析 -> JSON结构化输出

  流程2: 多股票AI对比
         同时分析3只股票 -> AI排名 -> 结构化对比

  流程3: 完整管线 + 可视化
         数据 + AI + 自动决策 + 专业图表(含AI标注)

  流程4: AI自动生成策略代码
         用自然语言描述策略 -> AI写出Python回测代码

  流程5: API费用追踪
         了解API定价, 控制学习成本

  核心认知:
    量化3.0 = 传统量化引擎 + AI智能层
    AI负责: 语义理解、文本分析、代码生成
    程序负责: 精确计算、数据处理、执行交易
    人负责: 策略设计、风险判断、最终决策

  下周预告(第4周):
    金融通识与资产配置 -- 建立投资决策框架
    学习宏观经济学、资产类型、全天候配置方法

================================================================
""")