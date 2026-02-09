"""
第3周-Part4：结构化输出 — 让AI的回答变成程序能用的数据
====================================================
这是"量化3.0"的核心技术：AI分析结果 → JSON → 程序自动处理
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from quant_core.ai import DeepSeekClient
from quant_core.data import fetch_stock
import pandas as pd
import json

client = DeepSeekClient(model="r1")
client.set_system_prompt("""
你是一位量化分析师。所有回答必须使用严格的JSON格式。
不要在JSON外面添加任何解释文字。
所有数值保留4位小数，所有评分使用1-10的整数。
""")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 场景1：单只股票分析 → 结构化输出
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("📊 场景1：单只股票结构化分析")
print("=" * 60)

df = fetch_stock("600519", days=60)
df["daily_return"] = df["close"].pct_change()
latest = df.iloc[-1]

prompt = f"""
分析以下股票数据，以JSON格式返回分析结果。

数据：
- 股票：贵州茅台(600519)
- 最新收盘价: {latest['close']:.2f}
- 5日均线: {df['close'].rolling(5).mean().iloc[-1]:.2f}
- 20日均线: {df['close'].rolling(20).mean().iloc[-1]:.2f}
- 近5日平均涨跌幅: {df['daily_return'].tail(5).mean():.4%}
- 近20日波动率: {df['daily_return'].tail(20).std():.4%}
- 近5日平均换手率: {df['turnover'].tail(5).mean():.2f}%

请返回如下JSON格式：
{{
    "stock_code": "600519",
    "stock_name": "贵州茅台",
    "analysis_date": "分析日期",
    "trend": "uptrend/downtrend/sideways之一",
    "trend_cn": "趋势的中文描述",
    "strength_score": "趋势强度1-10分",
    "support_level": "支撑位（数字）",
    "resistance_level": "压力位（数字）",
    "volatility_level": "high/medium/low之一",
    "volatility_cn": "波动水平中文描述",
    "short_term_outlook": "3-5日展望的中文描述",
    "risk_factors": ["风险因子1", "风险因子2"],
    "confidence": "分析置信度0.0-1.0"
}}
"""

result = client.chat_json(prompt)
print(f"\n🤖 AI返回的JSON：")
print(json.dumps(result, indent=2, ensure_ascii=False))

# 关键：程序可以直接使用这个结果！
print(f"\n📌 程序自动提取：")
print(f"   趋势判断: {result.get('trend_cn', 'N/A')}")
print(f"   趋势强度: {result.get('strength_score', 'N/A')}/10")
print(f"   支撑位:   {result.get('support_level', 'N/A')}")
print(f"   压力位:   {result.get('resistance_level', 'N/A')}")
print(f"   置信度:   {result.get('confidence', 'N/A')}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 场景2：批量股票筛选 — AI做初筛，程序做决策
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n" + "=" * 60)
print("📊 场景2：批量股票AI初筛")
print("=" * 60)

stocks = {
    "600519": "贵州茅台",
    "002594": "比亚迪",
    "600036": "招商银行",
}

all_results = []

for code, name in stocks.items():
    print(f"\n  正在分析 {name}({code})...")

    df = fetch_stock(code, days=60)
    df["daily_return"] = df["close"].pct_change()
    latest = df.iloc[-1]

    prompt = f"""
    快速评估此股票，返回JSON：
    - 股票：{name}({code})
    - 最新价: {latest['close']:.2f}
    - 近20日涨跌幅: {df['close'].pct_change(20).iloc[-1]:.2%}
    - 近20日波动率: {df['daily_return'].tail(20).std():.4%}
    - 近5日均换手率: {df['turnover'].tail(5).mean():.2f}%

    返回格式：
    {{
        "code": "{code}",
        "name": "{name}",
        "score": "综合评分1-10",
        "trend": "uptrend/downtrend/sideways",
        "trend_cn": "中文趋势",
        "risk_level": "high/medium/low",
        "risk_level_cn": "中文风险等级",
        "one_line_summary": "一句话总结，中文"
    }}
    """

    result = client.chat_json(prompt)
    all_results.append(result)

# 汇总成DataFrame
summary_df = pd.DataFrame(all_results)
print(f"\n📊 AI批量筛选结果：")
print(summary_df.to_string(index=False))

# 按评分排序
if "score" in summary_df.columns:
    summary_df["score"] = pd.to_numeric(summary_df["score"], errors="coerce")
    summary_df = summary_df.sort_values("score", ascending=False)
    print(f"\n🏆 按AI评分排名：")
    for _, row in summary_df.iterrows():
        print(f"   {row.get('name', 'N/A')}: {row.get('score', 'N/A')}分 - {row.get('one_line_summary', 'N/A')}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 场景3：AI生成量化策略代码
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n" + "=" * 60)
print("📊 场景3：让AI帮你写量化策略代码")
print("=" * 60)

code_client = DeepSeekClient(model="r1")
code_client.set_system_prompt("""
你是一位Python量化开发专家。
生成的代码必须：
1. 可以直接运行（不需要额外修改）
2. 包含详细的中文注释
3. 使用akshare获取数据（不用yfinance）
4. 包含完整的函数定义和调用示例
""")

strategy_prompt = """
请帮我写一个简单的"双均线策略"回测代码：

策略规则：
- 当5日均线上穿20日均线时：买入信号
- 当5日均线下穿20日均线时：卖出信号

要求：
1. 用akshare获取贵州茅台最近2年的日线数据
2. 计算买卖���号
3. 模拟交易：初始资金10万元，每次全仓买入/卖出
4. 计算最终收益率，并与"买入持有"策略对比
5. 画出净值曲线对比图

请直接输出完整可运行的Python代码。
"""

code_answer = code_client.chat(strategy_prompt, temperature=0.1)
print(f"\n🤖 AI生成的策略代码：\n")
print(code_answer)

print("""
💡 提示：
   AI生成的代码不一定100%正确，你需要：
   1. 检查代码逻辑是否合理
   2. 实际运行看是否报错
   3. 验证回测结果是否符合预期
   这就是"人机结合"——AI写初版，你审核优化
""")