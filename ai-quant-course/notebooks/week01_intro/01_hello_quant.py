"""
第1周作业：你的第一个 AI 量化程序
=================================
目标：体验"数据获取 → AI分析 → 可视化"的完整流程
"""
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第0步：安装依赖（首次运行）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# pip install akshare pandas matplotlib

import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第1步：获取A股数据
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fetch_stock_data(stock_code: str, stock_name: str, days: int = 365) -> pd.DataFrame:
    """
    使用AkShare获取A股历史行情数据

    参数：
        stock_code: 股票代码，如 "600519"（不需要加 .SS/.SZ 后缀）
        stock_name: 股票名称，用于打印信息
        days: 获取最近多少天的数据

    返回：
        DataFrame，包含日期、开盘价、收盘价、最高价、最低价、成交量等
    """
    print(f"📊 正在获取 {stock_name}({stock_code}) 的历史数据...")

    # 计算起止日期
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")

    # AkShare 获取A股日线数据
    # ak.stock_zh_a_hist() 是最常用的A股行情接口
    df = ak.stock_zh_a_hist(
        symbol=stock_code,
        period="daily",          # 日线（也支持 "weekly", "monthly"）
        start_date=start_date,
        end_date=end_date,
        adjust="qfq",            # 前复权（推荐用于技术分析）
    )

    # 规范化列名（方便后续统一处理）
    df = df.rename(columns={
        "日期": "Date",
        "开盘": "Open",
        "收盘": "Close",
        "最高": "High",
        "最低": "Low",
        "成交量": "Volume",
        "成交额": "Amount",
        "振幅": "Amplitude",
        "涨跌幅": "Change_Pct",
        "涨跌额": "Change_Amt",
        "换手率": "Turnover_Rate",
    })

    # 将日期设为索引
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")

    print(f"✅ 获取成功！共 {len(df)} 个交易日的数据")
    print(f"   时间范围: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")

    return df


# 获取贵州茅台数据
stock_code = "600519"
stock_name = "贵州茅台"
df = fetch_stock_data(stock_code, stock_name)

print(f"\n最近5天数据：")
print(df.tail())

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第2步：技术指标计算
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 计算移动平均线
df["SMA5"] = df["Close"].rolling(window=5).mean()
df["SMA20"] = df["Close"].rolling(window=20).mean()

# 计算每日收益率（AkShare已经提供了涨跌幅，但我们也自己算一下验证）
df["Daily_Return"] = df["Close"].pct_change()

# 判断均线状态
latest = df.iloc[-1]
sma_status = "金叉（看多）🟢" if latest["SMA5"] > latest["SMA20"] else "死叉（看空）🔴"

print(f"\n📈 {stock_name} 基础统计：")
print(f"  最新收盘价:      {latest['Close']:.2f}")
print(f"  5日均线:         {latest['SMA5']:.2f}")
print(f"  20日均线:        {latest['SMA20']:.2f}")
print(f"  均线状态:        {sma_status}")
print(f"  年度最高价:      {df['High'].max():.2f}")
print(f"  年度最低价:      {df['Low'].min():.2f}")
print(f"  平均日收益率:    {df['Daily_Return'].mean():.4%}")
print(f"  日收益率标准差:  {df['Daily_Return'].std():.4%}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第3步：可视化
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 设置中文字体（根据你的系统选择）
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                         gridspec_kw={"height_ratios": [3, 1, 1]})

# ---- 上图：价格 + 均线 ----
axes[0].plot(df.index, df["Close"], label="收盘价", linewidth=1.5, color="#333333")
axes[0].plot(df.index, df["SMA5"], label="5日均线", linewidth=1, color="#FF6B6B", alpha=0.8)
axes[0].plot(df.index, df["SMA20"], label="20日均线", linewidth=1, color="#4ECDC4", alpha=0.8)

# 标注金叉/死叉区域
axes[0].fill_between(df.index, df["SMA5"], df["SMA20"],
                     where=(df["SMA5"] > df["SMA20"]),
                     alpha=0.1, color="red", label="多头区间")
axes[0].fill_between(df.index, df["SMA5"], df["SMA20"],
                     where=(df["SMA5"] <= df["SMA20"]),
                     alpha=0.1, color="green", label="空头区间")

axes[0].set_title(f"{stock_name}({stock_code}) 价格走势与均线", fontsize=14, fontweight="bold")
axes[0].legend(loc="upper left")
axes[0].grid(True, alpha=0.3)

# ---- 中图：成交量 ----
colors = ["#FF4444" if row["Close"] >= row["Open"] else "#00AA00"
          for _, row in df.iterrows()]
axes[1].bar(df.index, df["Volume"] / 10000, color=colors, alpha=0.6, width=1)
axes[1].set_title("成交量（万手）", fontsize=12)
axes[1].grid(True, alpha=0.3)

# ---- 下图：日收益率 ----
colors_ret = ["#FF4444" if x >= 0 else "#00AA00"
              for x in df["Daily_Return"].fillna(0)]
axes[2].bar(df.index, df["Daily_Return"] * 100, color=colors_ret, alpha=0.6, width=1)
axes[2].set_title("日涨跌幅（%）", fontsize=12)
axes[2].axhline(y=0, color="black", linewidth=0.5)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("maotai_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n✅ 图表已保存为 maotai_analysis.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第4步：AI辅助分析（Prompt构造）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

analysis_prompt = f"""
你是一位资深量化分析师，拥有10年A股市场经验。
请基于以下数据，对{stock_name}({stock_code})进行短期技术分析。

## 数据摘要（截至 {df.index[-1].strftime('%Y-%m-%d')}）
- 最新收盘价: {latest['Close']:.2f}
- 5日均线: {latest['SMA5']:.2f}
- 20日均线: {latest['SMA20']:.2f}
- 均线关系: {sma_status}
- 最近5日平均涨跌幅: {df['Daily_Return'].tail(5).mean():.4%}
- 最近20日波动率(标准差): {df['Daily_Return'].tail(20).std():.4%}
- 最近5日平均换手率: {df['Turnover_Rate'].tail(5).mean():.2f}%

## 请输出（JSON格式）
{{
  "trend": "上涨/震荡/下跌",
  "support_level": "关键支撑位",
  "resistance_level": "关键压力位",
  "short_term_outlook": "3-5日展望",
  "action": "买入/持有/卖出/观望",
  "confidence": "0.0-1.0的置信度",
  "risk_warning": "主要风险提示"
}}
"""

print("\n🤖 为AI准备的分析Prompt：")
print("=" * 60)
print(analysis_prompt)
print("=" * 60)
print("\n💡 第3周学完DeepSeek API后，就可以把这段Prompt发给AI获取分析结果了！")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第5步：AkShare 更多能力展示（课程后续会用到）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n" + "=" * 60)
print("📚 AkShare 能力预览（后续课程会逐步用到）")
print("=" * 60)

# 预览1：获取实时行情（第16周）
print("\n🔹 实时行情（第16周：交易系统模块）")
try:
    spot_df = ak.stock_zh_a_spot_em()
    print(f"   当前A股共 {len(spot_df)} 只股票有实时行情")
    print(f"   示例：{spot_df[['代码','名称','最新价','涨跌幅']].head(3).to_string(index=False)}")
except Exception as e:
    print(f"   （非交易时间，跳过实时行情: {e}）")

# 预览2：财务指标（第9周：基本面分析）
print("\n🔹 财务指标（第9周：价值投资量化）")
try:
    fin_df = ak.stock_financial_analysis_indicator(symbol=stock_code)
    print(f"   获取到 {len(fin_df)} 期财务指标数据")
    print(f"   包含: ROE, 毛利率, 净利率, 资产负债率等")
except Exception as e:
    print(f"   （财务数据获取示例: {e}）")

# 预览3：基金数据（第13-14周）
print("\n🔹 基金数据（第13-14周：基金量化分析）")
try:
    fund_df = ak.fund_open_fund_rank_em(symbol="全部")
    print(f"   当前共 {len(fund_df)} 只开放式基金")
except Exception as e:
    print(f"   （基金数据获取示例: {e}）")

# 预览4：指数数据（第13周）
print("\n🔹 指数数据（第13周：指数基金配置）")
try:
    index_df = ak.stock_zh_index_daily(symbol="sh000300")  # 沪深300
    print(f"   沪深300指数历史数据共 {len(index_df)} 个交易日")
except Exception as e:
    print(f"   （指数数据获取示例: {e}）")


print("""
╔══════════════════════════════════════════════════╗
║         🎉 第1周 Hello World (AkShare版) 完成！    ║
╠══════════════════════════════════════════════════╣
║                                                    ║
║  ✅ AkShare 将作为我们整个课程的数据基础设施：        ║
║                                                    ║
║  第2周:  stock_zh_a_hist     → 股票历史行情         ║
║  第5周:  financial_indicator → 财务数据              ║
║  第8周:  stock_zh_a_hist     → 技术指标计算          ║
║  第9周:  financial_analysis  → 基本面指标            ║
║  第13周: fund_open_fund      → 基金数据              ║
║  第14周: fund_etf / fund_nav → 基金净值              ║
║  第16周: stock_zh_a_spot_em  → 实时行情              ║
║                                                    ║
║  下周预告（第2周）：                                 ║
║  深入Pandas数据分析 + Scikit-Learn股票预测            ║
║                                                    ║
╚══════════════════════════════════════════════════╝
""")