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

# 获取三只股票数据用于学习
print("📊 获取学习用数据...")
maotai = fetch_stock("600519")   # 贵州茅台（消费）
byd    = fetch_stock("002594")   # 比亚迪  （新能源）
zhaohang = fetch_stock("600036") # 招商银行 （金融）

print(f"  贵州茅台: {len(maotai)} 天")
print(f"  比亚迪:   {len(byd)} 天")
print(f"  招商银行: {len(zhaohang)} 天")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2.2 DataFrame 基础操作：像看财报一样看数据
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n" + "=" * 60)
print("📖 2.2 DataFrame 基础操作")
print("=" * 60)

# --- 查看数据结构 ---
print("\n🔹 数据形状：")
print(f"  maotai.shape = {maotai.shape}")  # (行数, 列数)
print(f"  意味着：{maotai.shape[0]}个交易日，每天{maotai.shape[1]}个指标")

print("\n🔹 数据类型：")
print(maotai.dtypes)
# 关键认知：close/open/high/low 是 float64（浮点数）
# volume 是 int64（整数），这些类型影响后续计算

print("\n🔹 基础统计（一行代码看全貌）：")
print(maotai[["open", "close", "high", "low", "volume"]].describe())
# describe() 是你快速了解数据分布的好朋友
# count: 数据量    mean: 均值    std: 标准差
# min/25%/50%/75%/max: 分位数


# --- 数据切片：精确定位你想看的数据 ---
print("\n🔹 数据切片技巧：")

# 取最近10个交易日
print("\n最近10天收盘价：")
print(maotai["close"].tail(10))

# 按日期范围取
print("\n2025年3月数据：")
mar_data = maotai.loc["2025-03":"2025-03"]
# print(type(maotai.index))
# print(maotai.index[:5])
print(f"  共 {len(mar_data)} 个交易日")
print(mar_data[["close", "volume", "change_pct"]].head())

# 条件筛选：找出涨幅超过3%的日子
big_up_days = maotai[maotai["change_pct"] > 3]
print(f"\n过去一年涨幅超过3%的天数: {len(big_up_days)}")
print(big_up_days[["close", "change_pct", "volume"]].head())

# 条件筛选：找出放量上涨的日子（涨幅>2% 且 换手率>均值的1.5倍）
avg_turnover = maotai["turnover"].mean()
strong_days = maotai[
    (maotai["change_pct"] > 2) &
    (maotai["turnover"] > avg_turnover * 1.5)
]
print(f"\n放量上涨天数（涨>2% + 换手>均值1.5倍）: {len(strong_days)}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2.3 Pandas核心技能：量化中最常用的操作
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n" + "=" * 60)
print("📖 2.3 量化中最常用的Pandas操作")
print("=" * 60)

# --- 技能1：rolling()  滑动窗口 ---
# 量化中用得最多的操作之一！均线、波动率、滚动收益都要用
print("\n🔹 技能1：滑动窗口 rolling()")

maotai["sma5"]  = maotai["close"].rolling(window=5).mean()    # 5日均线
maotai["sma10"] = maotai["close"].rolling(window=10).mean()   # 10日均线
maotai["sma20"] = maotai["close"].rolling(window=20).mean()   # 20日均线
maotai["sma60"] = maotai["close"].rolling(window=60).mean()   # 60日均线

# 滚动波动率（20日）
maotai["volatility_20"] = maotai["close"].pct_change().rolling(window=20).std() * np.sqrt(252)
# np.sqrt(252) 是年化因子，252个交易日

# 滚动最大值/最小值（用于计算支撑位/压力位）
maotai["high_20"] = maotai["high"].rolling(window=20).max()   # 20日最高
maotai["low_20"]  = maotai["low"].rolling(window=20).min()    # 20日最低

print("  ✅ 已计算: 4条均线 + 20日波动率 + 20日高低点")
print(maotai[["close", "sma5", "sma20", "sma60", "volatility_20"]].tail())

# --- 技能2：pct_change()  收益率计算 ---
print("\n🔹 技能2：收益率计算 pct_change()")

maotai["daily_return"] = maotai["close"].pct_change()          # 日收益率
maotai["return_5d"]    = maotai["close"].pct_change(periods=5)  # 5日收益率
maotai["return_20d"]   = maotai["close"].pct_change(periods=20) # 20日（月）收益率

# 累计收益率（如果第一天投了1块钱，现在值多少）
maotai["cumulative_return"] = (1 + maotai["daily_return"]).cumprod() - 1

print(f"  过去一年累计收益率: {maotai['cumulative_return'].iloc[-1]:.2%}")
print(f"  最大单日涨幅: {maotai['daily_return'].max():.2%}")
print(f"  最大单日跌幅: {maotai['daily_return'].min():.2%}")

# --- 技能3：shift()  时间偏移 ---
# 量化中非常重要！用来构造"昨天的数据"、"上周的数据"作为特征
print("\n🔹 技能3：时间偏移 shift()")

maotai["prev_close"] = maotai["close"].shift(1)     # 昨日收盘价
maotai["prev_volume"] = maotai["volume"].shift(1)    # 昨日成交量
maotai["next_return"] = maotai["daily_return"].shift(-1)  # 明日收益率（预测目标！）

# 这是机器学习的关键：用"今天的特征"预测"明天的收益"
print("  shift(1)  = 往后移1天 = 获取昨天的值")
print("  shift(-1) = 往前移1天 = 获取明天的值（作为预测目标）")

# --- 技能4：resample()  时间重采样 ---
# 把日线数据聚合成周线/月线
print("\n🔹 技能4：时间重采样 resample()")

monthly = maotai["close"].resample("ME").agg(["first", "last", "max", "min"])
monthly.columns = ["月开盘", "月收盘", "月最高", "月最低"]
monthly["月涨跌幅"] = (monthly["月收盘"] / monthly["月开盘"] - 1) * 100

print("  月线数据（最近5个月）：")
print(monthly.tail().round(2))

# --- 技能5：rank() 和 qcut()  排名与分组 ---
# 量化选股必备：在一堆股票中排名
print("\n🔹 技能5：排名与分组")

# 假设我们有三只股票的最近20日收益率
comparison = pd.DataFrame({
    "茅台": maotai["return_20d"].iloc[-1:].values,
    "比亚迪": byd["close"].pct_change(20).iloc[-1:].values,
    "招行": zhaohang["close"].pct_change(20).iloc[-1:].values,
}, index=["20日收益率"])

print("  三只股票近20日收益对比：")
print(f"  {comparison.round(4).to_string()}")
print(f"  排名: {comparison.iloc[0].rank(ascending=False).to_dict()}")

# --- 技能6：多股票DataFrame合并 ---
print("\n🔹 技能6：多股票数据合并")

# 方法：把多只股票的收盘价合并到一个DataFrame中
close_df = pd.DataFrame({
    "茅台": maotai["close"],
    "比亚迪": byd["close"],
    "招行": zhaohang["close"],
})

# 计算相关性矩阵（非常重要！用于投资组合分析）
returns_df = close_df.pct_change().dropna()
corr_matrix = returns_df.corr()

print("  三只股票日收益率相关性矩阵：")
print(corr_matrix.round(3))
print("\n  💡 相关性解读：")
print("     1.0 = 完全同涨同跌")
print("     0.0 = 完全无关")
print("    -1.0 = 完全反向（一个涨另一个跌）")
print("     分散投资应选相关性低的股票！")


print("\n✅ Pandas 6大核心技能掌握完毕！")
print("   rolling / pct_change / shift / resample / rank / merge")
print("   这6个操作覆盖了量化中80%的数据处理需求")