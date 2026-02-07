"""
第2周-Part4：Scikit-Learn 股票预测实战
=======================================
目标：用机器学习预测股票第二天涨还是跌
这是你的第一个ML模型，重要的是理解流程，而非追求准确率
"""

# 先设置路径，再导入
# import notebooks.week02_python_ds.path_setup  # noqa: F401  (或直接执行00_path_setup.py的内容)

# 更实用的写法：直接内联
import datetime
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# ✅ 从公共模块导入，不再重复定义
from quant_core.data import fetch_stock
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第1步：特征工程（最重要的环节！）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    特征工程：从原始行情数据中提取有预测价值的特征

    这是整个ML流程中最重要的环节
    特征的质量直接决定模型的上限

    特征分为4大类：
    1. 价格特征：均线位置、价格动量
    2. 成交量特征：量价关系
    3. 波动特征：波动率变化
    4. 时间特征：星期几效应
    """
    df = df.copy()

    # ===== 类别1：价格类特征 =====

    # 多周期收益率（动量因子）
    for period in [1, 2, 3, 5, 10, 20]:
        df[f"return_{period}d"] = df["close"].pct_change(period)

    # 均线偏离度（当前价格偏离均线的程度）
    for window in [5, 10, 20, 60]:
        sma = df["close"].rolling(window).mean()
        df[f"sma{window}_bias"] = (df["close"] - sma) / sma

    # 均线多头/空头排列
    df["sma5"]  = df["close"].rolling(5).mean()
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma60"] = df["close"].rolling(60).mean()
    df["golden_cross"] = (df["sma5"] > df["sma20"]).astype(int)  # 金叉=1

    # 价格在N日高低点的位置
    for window in [10, 20]:
        highest = df["high"].rolling(window).max()
        lowest  = df["low"].rolling(window).min()
        df[f"price_position_{window}d"] = (df["close"] - lowest) / (highest - lowest + 1e-8)

    # ===== 类别2：成交量特征 =====

    # 成交量变化率
    df["volume_change"] = df["volume"].pct_change()

    # 量比（今日成交量 / 5日平均成交量）
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(5).mean()

    # 换手率相对水平
    df["turnover_ratio"] = df["turnover"] / df["turnover"].rolling(20).mean()

    # ===== 类别3：波动特征 =====

    # N日波动率
    for window in [5, 10, 20]:
        df[f"volatility_{window}d"] = df["close"].pct_change().rolling(window).std()

    # 振幅（日内波动）
    df["intraday_range"] = (df["high"] - df["low"]) / df["open"]

    # ===== 类别4：时间特征 =====

    df["weekday"] = df.index.weekday          # 0=周一, 4=周五
    df["month"]   = df.index.month            # 1-12月
    df["is_month_start"] = (df.index.day <= 5).astype(int)  # 月初效应
    df["is_month_end"]   = (df.index.day >= 25).astype(int) # 月末效应

    # ===== 目标变量：明天涨还是跌 =====
    df["next_return"] = df["close"].pct_change().shift(-1)   # 明日收益率
    df["target"] = (df["next_return"] > 0).astype(int)       # 1=涨, 0=跌

    return df


# 获取数据并创建特征
print("📊 正在准备数据...")
df = fetch_stock("600519", days=800)
df = create_features(df)

# 查看创建了多少特征
feature_cols = [col for col in df.columns
                if col not in ["open", "close", "high", "low", "volume", "amount",
                               "change_pct", "turnover", "sma5", "sma20", "sma60",
                               "next_return", "target"]]
print(f"✅ 共创建 {len(feature_cols)} 个特征")
print(f"   价格类: return_*, sma*_bias, golden_cross, price_position_*")
print(f"   成交量: volume_change, volume_ratio, turnover_ratio")
print(f"   波动类: volatility_*, intraday_range")
print(f"   时间类: weekday, month, is_month_*")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第2步：准备训练数据
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 删除含NaN的行（均线计算前60天会产生NaN）
df_clean = df.dropna()
print(f"\n清洗后数据量: {len(df_clean)} 个交易日")

X = df_clean[feature_cols]
y = df_clean["target"]

print(f"目标变量分布：")
print(f"  涨的天数: {y.sum()} ({y.mean():.1%})")
print(f"  跌的天数: {len(y) - y.sum()} ({1 - y.mean():.1%})")

# ⚠️ 关键：时间序列不能随机划分！必须按时间顺序
# 前80%训练，后20%测试（模拟真实场景：用历史预测未来）
split_idx = int(len(X) * 0.8)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"\n数据划分（时间顺序！不是随机划分！）：")
print(f"  训练集: {X_train.index[0].strftime('%Y-%m-%d')} ~ {X_train.index[-1].strftime('%Y-%m-%d')} ({len(X_train)}天)")
print(f"  测试集: {X_test.index[0].strftime('%Y-%m-%d')} ~ {X_test.index[-1].strftime('%Y-%m-%d')} ({len(X_test)}天)")

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第3步：训练多个模型并对比
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n" + "=" * 60)
print("🤖 训练模型...")
print("=" * 60)

models = {
    "逻辑回归": LogisticRegression(random_state=42, max_iter=1000),
    "随机森林": RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
    ),
    "梯度提升": GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
    ),
}

results = {}

for name, model in models.items():
    print(f"\n📌 训练 {name}...")

    # 训练
    if name == "逻辑回归":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    # 评估
    accuracy = accuracy_score(y_test, y_pred)

    # 交叉验证（更可靠的评估）
    if name == "逻辑回归":
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)

    results[name] = {
        "model": model,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "accuracy": accuracy,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
    }

    print(f"  测试集准确率: {accuracy:.4f}")
    print(f"  交叉验证: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# 最佳模型
best_name = max(results, key=lambda x: results[x]["accuracy"])
print(f"\n🏆 最佳模型: {best_name}，准确率: {results[best_name]['accuracy']:.4f}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第4步：模型分析（理解模型在做什么）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n" + "=" * 60)
print("🔍 模型深度分析")
print("=" * 60)

# --- 特征重要性（随机森林）---
rf_model = results["随机森林"]["model"]
importance = pd.Series(rf_model.feature_importances_, index=feature_cols)
importance = importance.sort_values(ascending=False)

print("\n📊 随机森林 — Top 10 重要特征：")
for i, (feat, imp) in enumerate(importance.head(10).items(), 1):
    bar = "█" * int(imp * 200)
    print(f"  {i:2d}. {feat:25s} {imp:.4f} {bar}")

# --- 分类报告 ---
best_pred = results[best_name]["y_pred"]
print(f"\n📊 {best_name} — 详细分类报告：")
print(classification_report(y_test, best_pred, target_names=["跌", "涨"]))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第5步：可视化分析结果（完整版）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 左上：模型准确率对比
# ━━━━━━━━━━━━━━━━━━��━━━━━━━━━━━━
ax = axes[0, 0]
names = list(results.keys())
accs = [results[n]["accuracy"] for n in names]
cv_means = [results[n]["cv_mean"] for n in names]
x_pos = np.arange(len(names))
width = 0.35

bars1 = ax.bar(x_pos - width/2, accs, width, label="测试集准确率", color="#5B86E5")
bars2 = ax.bar(x_pos + width/2, cv_means, width, label="交叉验证均值", color="#36D1DC")
ax.set_ylabel("准确率")
ax.set_title("模型准确率对比", fontsize=13, fontweight="bold")
ax.set_xticks(x_pos)
ax.set_xticklabels(names)
ax.legend()
ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="随机猜测基准(50%)")
ax.set_ylim(0.4, 0.65)
ax.grid(True, alpha=0.3, axis="y")

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.003,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.003,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 右上：特征重要性 Top 10
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ax = axes[0, 1]
top10 = importance.head(10)

# 特征名中英文映射（方便非金融人士理解）
FEATURE_CN = {
    "return_1d":          "1日收益率",
    "return_2d":          "2日收益率",
    "return_3d":          "3日收益率",
    "return_5d":          "5日收益率",
    "return_10d":         "10日收益率",
    "return_20d":         "20日收益率",
    "sma5_bias":          "5日均线偏离度",
    "sma10_bias":         "10日均线偏离度",
    "sma20_bias":         "20日均线偏离度",
    "sma60_bias":         "60日均线偏离度",
    "golden_cross":       "金叉信号(MA5>MA20)",
    "price_position_10d": "10日价格位置",
    "price_position_20d": "20日价格位置",
    "volume_change":      "成交量变化率",
    "volume_ratio":       "量比(今日/5日均量)",
    "turnover_ratio":     "换手率相对水平",
    "volatility_5d":      "5日波动率",
    "volatility_10d":     "10日波动率",
    "volatility_20d":     "20日波动率",
    "intraday_range":     "日内振幅",
    "weekday":            "星期几",
    "month":              "月份",
    "is_month_start":     "是否月初",
    "is_month_end":       "是否月末",
}

# 生成带中文的标签
labels = [f"{FEATURE_CN.get(feat, feat)}\n({feat})" for feat in top10.index]

bars = ax.barh(range(len(top10)), top10.values, color="#5B86E5", alpha=0.8)
ax.set_yticks(range(len(top10)))
ax.set_yticklabels(labels, fontsize=9)
ax.invert_yaxis()  # 最重要的在最上面
ax.set_xlabel("重要性分数")
ax.set_title("随机森林 — Top 10 重要特征", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3, axis="x")

# 在条形上标注数值
for i, (val, bar) in enumerate(zip(top10.values, bars)):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', ha='left', va='center', fontsize=9)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 左下：混淆矩阵
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 📖 术语：混淆矩阵 (Confusion Matrix)
#    展示模型预测结果 vs 真实结果的对比表格：
#    ┌─────────────┬──────────┬──────────┐
#    │             │ 预测=跌   │ 预测=涨   │
#    ├─────────────┼──────────┼──────────┤
#    │ 实际=跌      │ TN(正确) │ FP(误判) ��
#    │ 实际=涨      │ FN(漏判) │ TP(正确) │
#    └─────────────┴──────────┴──────────┘
#    TN=True Negative  正确预测了跌
#    TP=True Positive  正确预测了涨
#    FP=False Positive 实际跌但预测涨（亏钱！）
#    FN=False Negative 实际涨但预测跌（踏空）

ax = axes[1, 0]
best_pred = results[best_name]["y_pred"]
cm = confusion_matrix(y_test, best_pred)

im = ax.imshow(cm, cmap="Blues", alpha=0.8)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["预测=跌", "预测=涨"], fontsize=11)
ax.set_yticklabels(["实际=跌", "实际=涨"], fontsize=11)
ax.set_xlabel("模型预测", fontsize=11)
ax.set_ylabel("实际涨跌", fontsize=11)
ax.set_title(f"{best_name} — 混淆矩阵", fontsize=13, fontweight="bold")

# 在格子里标注数值和含义
labels_cm = [
    [f"TN\n正确预测跌\n{cm[0,0]}次", f"FP\n误判为涨\n{cm[0,1]}次"],
    [f"FN\n漏判了涨\n{cm[1,0]}次",   f"TP\n正确预测涨\n{cm[1,1]}次"],
]
for i in range(2):
    for j in range(2):
        ax.text(j, i, labels_cm[i][j], ha="center", va="center",
                fontsize=10, fontweight="bold",
                color="white" if cm[i, j] > cm.max() / 2 else "black")

fig.colorbar(im, ax=ax, shrink=0.8)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 右下：预测概率分布
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 📖 术语：预测概率 (Predict Probability)
#    模型不仅输出"涨/跌"的结论，还输出"有多大把握"。
#    例如：概率=0.8 意味着模型认为有80%的可能性会涨。
#    概率接近0.5说明模型很不确定（像抛硬币），
#    概率接近0或1说明模型很有把握。

ax = axes[1, 1]
best_prob = results[best_name]["y_prob"]

# 按实际涨跌分组画直方图
prob_up   = best_prob[y_test.values == 1]  # 实际涨的那些天，模型给的概率
prob_down = best_prob[y_test.values == 0]  # 实际跌的那些天，模型给的概率

ax.hist(prob_down, bins=30, alpha=0.6, color="#00AA00", label=f"实际=跌 ({len(prob_down)}天)", edgecolor="white")
ax.hist(prob_up,   bins=30, alpha=0.6, color="#FF4444", label=f"实际=涨 ({len(prob_up)}天)", edgecolor="white")
ax.axvline(x=0.5, color="black", linewidth=1.5, linestyle="--", label="决策边界(0.5)")
ax.set_xlabel("模型预测'涨'的概率", fontsize=11)
ax.set_ylabel("天数", fontsize=11)
ax.set_title(f"{best_name} — 预测概率分布", fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 添加解读文字
ax.text(0.25, ax.get_ylim()[1] * 0.9, "← 模型认为会跌", ha="center", fontsize=9, color="#00AA00")
ax.text(0.75, ax.get_ylim()[1] * 0.9, "模型认为会涨 →", ha="center", fontsize=9, color="#FF4444")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 保存并展示
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fig.suptitle(f"贵州茅台(600519) 股票预测模型分析报告", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("prediction_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ 可视化报告已保存为 prediction_analysis.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 最终总结输出
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print(f"""
╔══════════════════════════════════════════════════════╗
║           📊 股票预测模型 — 最终报告                   ║
╠══════════════════════════════════════════════════════╣
║                                                        ║
║  🏆 最佳模型: {best_name:10s}                            ║
║  📈 测试集准确率: {results[best_name]['accuracy']:.2%}                          ║
║  📊 交叉验证: {results[best_name]['cv_mean']:.2%} ± {results[best_name]['cv_std']:.2%}                    ║
║                                                        ║
║  📌 模型对比:                                           ║""")

for name, res in results.items():
    flag = " 👑" if name == best_name else "   "
    print(f"║    {name:6s}: 测试={res['accuracy']:.2%}  CV={res['cv_mean']:.2%}{flag}          ║")

print(f"""║                                                        ║
║  💡 关键发现:                                           ║
║    • Top特征多为短期收益率和均线偏离度                      ║
║    • 说明模型主要在捕捉"短期动量"效应                      ║
║    • 准确率略高于50%，需要配合风控才能盈利                  ║
║                                                        ║
║  ⚠️  注意:                                              ║
║    • 这是教学演示，不要直接用于实盘交易                     ║
║    • 第10周会构建更完整的回测框架来验证策略                  ║
║                                                        ║
╚══════════════════════════════════════════════════════╝
""")