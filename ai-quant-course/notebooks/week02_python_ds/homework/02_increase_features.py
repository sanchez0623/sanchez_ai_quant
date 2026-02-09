"""
Week02 Homework: Feature Expansion for Stock Prediction
第2周作业：股票预测特征扩展
"""

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from quant_core.data.fetcher import fetch_stock


def create_features_base(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Base feature set from week02/04.
	来自week02/04的基础特征集合。
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

	# Target: next-day direction.
	# 目标：次日涨跌方向。
	df["next_return"] = df["close"].pct_change().shift(-1)
	df["target"] = (df["next_return"] > 0).astype(int)

	return df


def create_features_plus(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Base features plus 3 new useful signals.
	基础特征 + 3个新增特征。
	"""
	df = create_features_base(df)

	# New feature 1: RSI(14).
	# 新特征1：RSI(14)。
	delta = df["close"].diff()
	gain = delta.clip(lower=0).rolling(14).mean()
	loss = (-delta.clip(upper=0)).rolling(14).mean()
	rs = gain / (loss + 1e-8)
	df["rsi_14"] = 100 - (100 / (1 + rs))

	# New feature 2: MACD histogram.
	# 新特征2：MACD柱状图。
	ema12 = df["close"].ewm(span=12, adjust=False).mean()
	ema26 = df["close"].ewm(span=26, adjust=False).mean()
	macd = ema12 - ema26
	signal = macd.ewm(span=9, adjust=False).mean()
	df["macd_hist"] = macd - signal

	# New feature 3: Bollinger Band width.
	# 新特征3：布林带宽度。
	sma20 = df["close"].rolling(20).mean()
	std20 = df["close"].rolling(20).std()
	upper = sma20 + 2 * std20
	lower = sma20 - 2 * std20
	df["bb_width"] = (upper - lower) / (sma20 + 1e-8)

	return df


def prepare_xy(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
	"""
	Build X and y with NaNs removed.
	构建X和y并去除NaN。
	"""
	df_clean = df.dropna()
	X = df_clean[feature_cols]
	y = df_clean["target"]
	return X, y


def evaluate_accuracy(X: pd.DataFrame, y: pd.Series) -> float:
	"""
	Evaluate accuracy with time-ordered split and scaling.
	使用时间顺序切分并标准化来评估准确率。
	"""
	split_idx = int(len(X) * 0.8)
	X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
	y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.transform(X_test)

	model = LogisticRegression(random_state=42, max_iter=1000)
	model.fit(X_train_scaled, y_train)
	y_pred = model.predict(X_test_scaled)

	return accuracy_score(y_test, y_pred)


def main() -> None:
	# Load stock data.
	# 加载股票数据。
	df = fetch_stock("600519", days=800)

	# Base features.
	# 基础特征。
	df_base = create_features_base(df)
	feature_cols_base = [
		col for col in df_base.columns
		if col not in [
			"open", "close", "high", "low", "volume", "amount",
			"change_pct", "turnover", "sma5", "sma20", "sma60",
			"next_return", "target",
		]
	]
	X_base, y_base = prepare_xy(df_base, feature_cols_base)
	acc_base = evaluate_accuracy(X_base, y_base)

	# Enhanced features.
	# 增强特征。
	df_plus = create_features_plus(df)
	feature_cols_plus = [
		col for col in df_plus.columns
		if col not in [
			"open", "close", "high", "low", "volume", "amount",
			"change_pct", "turnover", "sma5", "sma20", "sma60",
			"next_return", "target",
		]
	]
	X_plus, y_plus = prepare_xy(df_plus, feature_cols_plus)
	acc_plus = evaluate_accuracy(X_plus, y_plus)

	# Report comparison.
	# 输出对比结果。
	print("=== Accuracy Comparison / 准确率对比 ===")
	print(f"Base features: {acc_base:.4f}")
	print(f"Enhanced features (+RSI/MACD/BB): {acc_plus:.4f}")
	print(f"Delta: {acc_plus - acc_base:+.4f}")


if __name__ == "__main__":
	main()
