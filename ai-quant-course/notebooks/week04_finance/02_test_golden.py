# -*- coding: utf-8 -*-
"""
测试 AkShare 黄金基准接口: spot_golden_benchmark_sge
目标:
1) 验证接口可用
2) 兼容不同版本参数签名
3) 输出最近10天黄金数据
"""

import pandas as pd
import akshare as ak


def _pick_price_column(df: pd.DataFrame) -> str:
	"""优先按常见列名选择价格列，失败则回退到第一个可转数值列。"""
	preferred = [
		"晚盘价", "收盘", "收盘价", "今收盘", "价格", "金价", "AU9999", "AU99.99", "基准价"
	]
	for col in df.columns:
		col_name = str(col).strip()
		if any(key in col_name for key in preferred):
			return col

	for col in df.columns:
		if str(col).strip() in {"日期", "date"}:
			continue
		series = pd.to_numeric(df[col], errors="coerce")
		if series.notna().sum() > 0:
			return col

	raise ValueError("未识别到可用价格列")


def fetch_gold_data(start_date: str = "20250101", end_date: str = "20261231") -> pd.DataFrame:
	"""兼容不同 AkShare 版本的黄金接口调用方式。"""
	try:
		df = ak.spot_golden_benchmark_sge(start_date=start_date, end_date=end_date)
	except TypeError:
		df = ak.spot_golden_benchmark_sge()

	if df is None or df.empty:
		raise ValueError("接口返回空数据")

	df.columns = [str(c).strip() for c in df.columns]

	date_candidates = ["日期", "交易时间", "时间", "date", "datetime"]
	date_col = next((c for c in date_candidates if c in df.columns), None)
	if date_col is None:
		raise ValueError(f"未找到日期列，当前列: {list(df.columns)}")

	df["date"] = pd.to_datetime(df[date_col], errors="coerce")
	df = df.dropna(subset=["date"]).sort_values("date")

	price_col = _pick_price_column(df)
	df["price"] = pd.to_numeric(df[price_col], errors="coerce")
	df = df.dropna(subset=["price"])

	start_ts = pd.to_datetime(start_date)
	end_ts = pd.to_datetime(end_date)
	df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]

	return df[["date", "price"]]


def main() -> None:
	print("=" * 60)
	print("测试接口: ak.spot_golden_benchmark_sge")
	print("=" * 60)

	try:
		df = fetch_gold_data(start_date="20240101", end_date="20261231")
		print(f"接口调用成功，共获取 {len(df)} 条数据")
		print("\n最近10天数据:")
		recent = df.tail(10).copy()
		recent["date"] = recent["date"].dt.strftime("%Y-%m-%d")
		recent["price"] = recent["price"].round(2)
		print(recent.to_string(index=False))
	except Exception as e:
		print(f"接口测试失败: {e}")


if __name__ == "__main__":
	main()

