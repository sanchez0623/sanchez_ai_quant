import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path for local imports.
# 确保项目根目录在sys.path中，便于导入本地模块。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

# Use StockAnalyzer from analyzer.py as requested.
# 按要求使用analyzer.py里的StockAnalyzer。
from quant_core.ai.analyzer import StockAnalyzer


def main() -> None:
	# Define sector leaders (consumption/new energy/finance/tech/pharma).
	# 定义行业龙头（消费/新能源/金融/科技/医药）。
	stocks = {
		"消费": {"name": "贵州茅台", "code": "600519"},
		"新能源": {"name": "宁德时代", "code": "300750"},
		"金融": {"name": "招商银行", "code": "600036"},
		"科技": {"name": "中芯国际", "code": "688981"},
		"医药": {"name": "恒瑞医药", "code": "600276"},
	}

	# Initialize analyzer client.
	# 初始化分析器。
	analyzer = StockAnalyzer(model="r1")

	# Collect analysis results.
	# 收集分析结果。
	rows = []
	for sector, info in stocks.items():
		result = analyzer.analyze(info["code"], info["name"], days=60)
		trend = (
			result.get("trend")
			or result.get("trend_label")
			or result.get("trend_rating")
			or "unknown"
		)
		risk = (
			result.get("risk")
			or result.get("risk_rating")
			or result.get("risk_level")
			or "unknown"
		)

		rows.append({
			"sector": sector,
			"name": info["name"],
			"code": info["code"],
			"trend": trend,
			"risk_rating": risk,
		})

	# Summarize into a DataFrame for comparison.
	# 汇总成DataFrame便于对比。
	df = pd.DataFrame(rows)
	print("=== Trend & Risk Comparison / 趋势与风险评级对比 ===")
	print(df.to_string(index=False))


if __name__ == "__main__":
	main()
