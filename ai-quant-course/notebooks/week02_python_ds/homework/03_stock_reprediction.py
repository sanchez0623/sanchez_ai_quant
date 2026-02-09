"""
Week02 Homework: Re-train with >1% target
第2周作业：将预测目标改为“涨幅>1%”并对比结果
"""

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from quant_core.data.fetcher import fetch_stock


# Configure fonts for Chinese display.
# 配置中文显示字体。
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def create_features(df: pd.DataFrame, target_threshold: float) -> pd.DataFrame:
	"""
	Feature engineering with configurable target threshold.
	带可配置目标阈值的特征工程。
	"""
	df = df.copy()

	# Price features: multi-period returns.
	# 价格特征：多周期收益率。
	for period in [1, 2, 3, 5, 10, 20]:
		df[f"return_{period}d"] = df["close"].pct_change(period)

	# Price features: moving average bias.
	# 价格特征：均线偏离度。
	for window in [5, 10, 20, 60]:
		sma = df["close"].rolling(window).mean()
		df[f"sma{window}_bias"] = (df["close"] - sma) / sma

	# Trend features: moving average relationships.
	# 趋势特征：均线关系。
	df["sma5"] = df["close"].rolling(5).mean()
	df["sma20"] = df["close"].rolling(20).mean()
	df["sma60"] = df["close"].rolling(60).mean()
	df["golden_cross"] = (df["sma5"] > df["sma20"]).astype(int)

	# Price position within recent highs/lows.
	# 价格在近期高低点中的位置。
	for window in [10, 20]:
		highest = df["high"].rolling(window).max()
		lowest = df["low"].rolling(window).min()
		df[f"price_position_{window}d"] = (df["close"] - lowest) / (highest - lowest + 1e-8)

	# Volume features.
	# 成交量特征。
	df["volume_change"] = df["volume"].pct_change()
	df["volume_ratio"] = df["volume"] / df["volume"].rolling(5).mean()
	df["turnover_ratio"] = df["turnover"] / df["turnover"].rolling(20).mean()

	# Volatility features.
	# 波动特征。
	for window in [5, 10, 20]:
		df[f"volatility_{window}d"] = df["close"].pct_change().rolling(window).std()
	df["intraday_range"] = (df["high"] - df["low"]) / df["open"]

	# Time features.
	# 时间特征。
	df["weekday"] = df.index.weekday
	df["month"] = df.index.month
	df["is_month_start"] = (df.index.day <= 5).astype(int)
	df["is_month_end"] = (df.index.day >= 25).astype(int)

	# Target: next-day return exceeding threshold.
	# 目标：次日涨幅是否超过阈值。
	df["next_return"] = df["close"].pct_change().shift(-1)
	df["target"] = (df["next_return"] > target_threshold).astype(int)

	return df


def prepare_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, list[str]]:
	"""
	Prepare X/y and feature columns.
	准备X/y与特征列列表。
	"""
	df_clean = df.dropna()
	feature_cols = [
		col for col in df_clean.columns
		if col not in [
			"open", "close", "high", "low", "volume", "amount",
			"change_pct", "turnover", "sma5", "sma20", "sma60",
			"next_return", "target",
		]
	]
	X = df_clean[feature_cols]
	y = df_clean["target"]
	return X, y, feature_cols


def train_and_eval(X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict[str, float]]:
	"""
	Train multiple models and return accuracy summary.
	训练多个模型并返回准确率汇总。
	"""
	split_idx = int(len(X) * 0.8)
	X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
	y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.transform(X_test)

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
		if name == "逻辑回归":
			model.fit(X_train_scaled, y_train)
			y_pred = model.predict(X_test_scaled)
			y_prob = model.predict_proba(X_test_scaled)[:, 1]
			cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
		else:
			model.fit(X_train, y_train)
			y_pred = model.predict(X_test)
			y_prob = model.predict_proba(X_test)[:, 1]
			cv_scores = cross_val_score(model, X_train, y_train, cv=5)

		accuracy = accuracy_score(y_test, y_pred)
		precision, recall, f1, _ = precision_recall_fscore_support(
			y_test, y_pred, average="binary", zero_division=0
		)
		try:
			roc_auc = roc_auc_score(y_test, y_prob)
		except ValueError:
			roc_auc = float("nan")
		results[name] = {
			"accuracy": accuracy,
			"precision": precision,
			"recall": recall,
			"f1": f1,
			"roc_auc": roc_auc,
			"cv_mean": cv_scores.mean(),
			"cv_std": cv_scores.std(),
		}

	return results


def print_summary(
	title: str,
	results: Dict[str, Dict[str, float]],
	y: pd.Series,
) -> None:
	"""
	Print a compact summary.
	打印简洁汇总。
	"""
	print("\n" + "=" * 60)
	print(title)
	print("=" * 60)
	# Class distribution.
	# 类别分布。
	pos_rate = y.mean()
	neg_rate = 1 - pos_rate
	print(f"Class distribution / 类别分布: pos={pos_rate:.2%}, neg={neg_rate:.2%}")
	for name, res in results.items():
		print(
			f"{name}: accuracy={res['accuracy']:.4f}, "
			f"precision={res['precision']:.4f}, recall={res['recall']:.4f}, "
			f"f1={res['f1']:.4f}, roc_auc={res['roc_auc']:.4f}, "
			f"cv={res['cv_mean']:.4f} ± {res['cv_std']:.4f}"
		)


def plot_comparison(
	results_a: Dict[str, Dict[str, float]],
	results_b: Dict[str, Dict[str, float]],
	y_a: pd.Series,
	y_b: pd.Series,
) -> None:
	"""
	Visualize metric comparison between Case A and Case B.
	可视化对比Case A与Case B的指标表现。
	"""
	metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
	metric_titles = {
		"accuracy": "Accuracy / 准确率",
		"precision": "Precision / 精确率",
		"recall": "Recall / 召回率",
		"f1": "F1 / F1分数",
		"roc_auc": "ROC-AUC / ROC-AUC",
	}

	models = list(results_a.keys())
	case_labels = ["Case A / 次日涨跌", "Case B / 次日涨幅>1%"]
	case_results = [results_a, results_b]

	fig, axes = plt.subplots(2, 3, figsize=(16, 9))
	axes = axes.flatten()

	for idx, metric in enumerate(metrics):
		ax = axes[idx]
		x_pos = np.arange(len(models))
		width = 0.35
		values_a = [results_a[m][metric] for m in models]
		values_b = [results_b[m][metric] for m in models]

		ax.bar(x_pos - width / 2, values_a, width, label=case_labels[0], color="#5B86E5")
		ax.bar(x_pos + width / 2, values_b, width, label=case_labels[1], color="#36D1DC")
		ax.set_title(metric_titles[metric])
		ax.set_xticks(x_pos)
		ax.set_xticklabels(models)
		ax.set_ylim(0, 1)
		ax.grid(True, alpha=0.3, axis="y")
		ax.legend(loc="upper right", fontsize=8, frameon=False)

		for i, val in enumerate(values_a):
			ax.text(i - width / 2, val + 0.01, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
		for i, val in enumerate(values_b):
			ax.text(i + width / 2, val + 0.01, f"{val:.2f}", ha="center", va="bottom", fontsize=8)

	# Class distribution summary panel.
	# 类别分布汇总面板。
	ax = axes[-1]
	pos_a = y_a.mean()
	pos_b = y_b.mean()
	neg_a = 1 - pos_a
	neg_b = 1 - pos_b
	labels = ["Case A", "Case B"]
	pos_vals = [pos_a, pos_b]
	neg_vals = [neg_a, neg_b]
	ax.bar(labels, pos_vals, label="Pos / 正类", color="#FF8C42")
	ax.bar(labels, neg_vals, bottom=pos_vals, label="Neg / 负类", color="#6C8EBF")
	ax.set_title("Class Distribution / 类别分布")
	ax.set_ylim(0, 1)
	ax.grid(True, alpha=0.3, axis="y")
	ax.legend(loc="upper right", fontsize=8, frameon=False)
	for i, (p, n) in enumerate(zip(pos_vals, neg_vals)):
		ax.text(i, p / 2, f"{p:.1%}", ha="center", va="center", color="white", fontsize=9)
		ax.text(i, p + n / 2, f"{n:.1%}", ha="center", va="center", color="white", fontsize=9)

	fig.suptitle("Case A vs Case B Metric Comparison / 指标对比", fontsize=14)
	fig.tight_layout(rect=[0, 0.02, 1, 0.95])
	plt.show()


def main() -> None:
	# Load stock data.
	# 加载股票数据。
	df = fetch_stock("600519", days=800)

	# Case A: target is next-day up/down (>0).
	# 情况A：目标为次日涨跌（>0）。
	df_updown = create_features(df, target_threshold=0.0)
	X_a, y_a, _ = prepare_xy(df_updown)
	results_a = train_and_eval(X_a, y_a)

	# Case B: target is next-day gain > 1%.
	# 情况B：目标为次日涨幅 > 1%。
	df_up1 = create_features(df, target_threshold=0.01)
	X_b, y_b, _ = prepare_xy(df_up1)
	results_b = train_and_eval(X_b, y_b)

	# Compare results.
	# 对比结果。
	print_summary("Case A: Next-day return > 0 / 次日涨跌", results_a, y_a)
	print_summary("Case B: Next-day return > 1% / 次日涨幅>1%", results_b, y_b)
	plot_comparison(results_a, results_b, y_a, y_b)


if __name__ == "__main__":
	main()
