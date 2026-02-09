import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

# Ensure project root is on sys.path so local modules can be imported.
# 确保项目根目录在sys.path中，便于导入本地模块。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from quant_core.data.fetcher import fetch_stock


@dataclass
class StockSpec:
	symbol: str
	name: str
	sector: str


def load_stocks(
	stocks: List[StockSpec],
	years: int = 2,
	cache_dir: str = ".cache/stock",
	min_interval_seconds: float = 1.5,
) -> Dict[str, pd.DataFrame]:
	# Fetch and normalize daily data for each stock.
	# 拉取并标准化每只股票的日线数据。
	data = {}
	for spec in stocks:
		df = fetch_stock(
			spec.symbol,
			days=365 * years,
			adjust="qfq",
			cache_dir=cache_dir,
			cache_ttl_seconds=3600,
			min_interval_seconds=min_interval_seconds,
			sources=["eastmoney", "tencent"],
		)
		df = df.sort_index()
		data[spec.symbol] = df
	return data


def build_comparison_frame(
	data: Dict[str, pd.DataFrame],
	stocks: List[StockSpec],
) -> pd.DataFrame:
	# Align close prices by date to compare performance.
	# 按日期对齐收盘价，便于对比表现。
	closes = []
	for spec in stocks:
		series = data[spec.symbol]["close"].rename(spec.name)
		closes.append(series)
	df = pd.concat(closes, axis=1).dropna()
	return df


def plot_comparison(df: pd.DataFrame, stocks: List[StockSpec]) -> None:
	# Normalize prices to show relative performance.
	# 归一化价格，用于展示相对表现。
	normalized = df / df.iloc[0]

	fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

	axes[0].plot(df.index, df)
	axes[0].set_title("Close Price Comparison / 收盘价对比")
	axes[0].set_ylabel("Price / 价格")
	axes[0].grid(True, alpha=0.3)
	axes[0].legend([s.name for s in stocks], loc="best")

	axes[1].plot(normalized.index, normalized)
	axes[1].set_title("Normalized Performance (Start=1.0) / 归一化表现(起点=1.0)")
	axes[1].set_ylabel("Normalized Price / 归一化价格")
	axes[1].grid(True, alpha=0.3)

	plt.tight_layout()
	plt.show()


def main() -> None:
	# Define stocks from different sectors for comparison.
	# 定义不同行业的股票用于对比分析。
	stocks = [
		StockSpec(symbol="600519", name="Kweichow Moutai", sector="Consumer Staples"),
		StockSpec(symbol="600036", name="China Merchants Bank", sector="Financials"),
		StockSpec(symbol="300750", name="CATL", sector="Industrials"),
	]

	# Load data, build comparison table, and plot.
	# 加载数据、构建对比表并绘图。
	data = load_stocks(stocks, years=2)
	comparison = build_comparison_frame(data, stocks)
	plot_comparison(comparison, stocks)


if __name__ == "__main__":
	main()
