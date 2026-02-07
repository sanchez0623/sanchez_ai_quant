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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 问题1：缺失值处理
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("🔧 问题1：缺失值处理")
print("=" * 60)

df = fetch_stock("600519")

# 实际场景中，停牌日、数据源问题都可能导致缺失
# 我们先人为制造一些缺失来演示处理方法
df_dirty = df.copy()
np.random.seed(42)
mask = np.random.random(len(df_dirty)) < 0.03  # 随机3%的数据变成NaN
df_dirty.loc[mask, "close"] = np.nan
df_dirty.loc[mask, "volume"] = np.nan

print(f"\n原始数据缺失情况：")
print(df_dirty.isnull().sum())

# --- 方法1：前向填充（最常用！用昨天的价格填今天的缺失）---
df_ffill = df_dirty.copy()
df_ffill["close"] = df_ffill["close"].ffill()
print(f"\n方法1 - 前向填充 ffill(): 适合价格数据")
print(f"  逻辑：停牌日价格 = 最后一个交易日的收盘价")
print(f"  填充后缺失: {df_ffill['close'].isnull().sum()}")

# --- 方法2：线性插值（适合连续数据）---
df_interp = df_dirty.copy()
df_interp["close"] = df_interp["close"].interpolate(method="linear")
print(f"\n方法2 - 线性插值 interpolate(): 适合平滑过渡")
print(f"  逻辑：缺失值 = 前后两个有效值的线性中间值")
print(f"  填充后缺失: {df_interp['close'].isnull().sum()}")

# --- 方法3：成交量用0填充（停牌日确实没有成交）---
df_ffill["volume"] = df_ffill["volume"].fillna(0)
print(f"\n方法3 - 零值填充 fillna(0): 适合成交量")
print(f"  逻辑：停牌日成交量确实为0")

# --- 最佳实践：不同列用不同策略 ---
print("\n💡 最佳实践总结：")
print("   价格类(open/close/high/low) → ffill() 前向填充")
print("   成交量(volume)              → fillna(0) 零值填充")
print("   收益率(return)              → fillna(0) 或 dropna()")
print("   技术指标                     → 重新计算（不要填充！）")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 问题2：异常值检测与处理
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n" + "=" * 60)
print("🔧 问题2：异常值检测")
print("=" * 60)

df = fetch_stock("002812")
df["daily_return"] = df["close"].pct_change()

# --- 方法1：统计方法（3σ原则）---
mean = df["daily_return"].mean()
std = df["daily_return"].std()
upper = mean + 3 * std
lower = mean - 3 * std

outliers_3sigma = df[
    (df["daily_return"] > upper) |
    (df["daily_return"] < lower)
]

print(f"\n方法1 - 3σ原则:")
print(f"  均值: {mean:.4%}, 标准差: {std:.4%}")
print(f"  正常范围: [{lower:.4%}, {upper:.4%}]")
print(f"  异常值数量: {len(outliers_3sigma)}")
if len(outliers_3sigma) > 0:
    print(f"  异常日期及涨跌幅：")
    for date, row in outliers_3sigma.iterrows():
        print(f"    {date.strftime('%Y-%m-%d')}: {row['daily_return']:.2%}")

# --- 方法2：IQR方法（更稳健）---
Q1 = df["daily_return"].quantile(0.25)
Q3 = df["daily_return"].quantile(0.75)
IQR = Q3 - Q1
lower_iqr = Q1 - 1.5 * IQR
upper_iqr = Q3 + 1.5 * IQR

outliers_iqr = df[
    (df["daily_return"] > upper_iqr) |
    (df["daily_return"] < lower_iqr)
]

print(f"\n方法2 - IQR方法:")
print(f"  Q1={Q1:.4%}, Q3={Q3:.4%}, IQR={IQR:.4%}")
print(f"  正常范围: [{lower_iqr:.4%}, {upper_iqr:.4%}]")
print(f"  异常值数量: {len(outliers_iqr)}")

# --- 方法3：A股特有——涨跌停检测 ---
limit_up   = df[df["change_pct"] >= 9.9]   # 涨停（主板10%，这里用9.9%）
limit_down = df[df["change_pct"] <= -9.9]  # 跌停

print(f"\n方法3 - 涨跌停检测（A股特有）:")
print(f"  涨停天数: {len(limit_up)}")
print(f"  跌停天数: {len(limit_down)}")

# --- 处理策略 ---
print("\n💡 异常值处理策略：")
print("   ⚠️  不要轻易删除异常值！在金融数据中，异常值可能是重要信号")
print("   策略1: 标记但保留（添加一列 is_outlier）")
print("   策略2: Winsorize 缩尾处理（将极值拉到边界）")
print("   策略3: 在回测中特殊处理（涨停无法买入、跌停无法卖出）")

# Winsorize 示例
df["return_winsorized"] = df["daily_return"].clip(lower=lower_iqr, upper=upper_iqr)
print(f"\n  Winsorize前 最大值: {df['daily_return'].max():.4%}")
print(f"  Winsorize后 最大值: {df['return_winsorized'].max():.4%}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 问题3：数据对齐（多股票场景）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n" + "=" * 60)
print("🔧 问题3：多股票数据对齐")
print("=" * 60)

# 不同股票可能有不同的交易日（新股上市晚、停牌等）
stock1 = fetch_stock("600519")  # 茅台
stock2 = fetch_stock("688802")  # 沐曦股份

close_df = pd.DataFrame({
    "茅台": stock1["close"],
    "沐曦股份": stock2["close"],
})

print(f"\n合并前：")
print(f"  茅台交易日: {len(stock1)}")
print(f"  沐曦股份交易日: {len(stock2)}")
print(f"  合并后行数: {len(close_df)}")
print(f"  含缺失值的行: {close_df.isnull().any(axis=1).sum()}")

# 处理方式：取交集（两只股票都有数据的日子）
close_aligned = close_df.dropna()
print(f"\n对齐后（取交集）: {len(close_aligned)} 个共同交易日")

print("\n✅ 数据清洗三大问题全部掌握！")