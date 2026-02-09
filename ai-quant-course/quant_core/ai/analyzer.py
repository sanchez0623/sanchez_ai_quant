"""
AI分析器
========
封装"获取数据 → 构建Prompt → 调用AI → 解析结果"的完整流程
让调用方一行代码就能获取AI分析结果
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import pandas as pd
from quant_core.data import fetch_stock
from quant_core.ai.llm_client import DeepSeekClient
from quant_core.ai.prompts import QuantPrompts


class StockAnalyzer:
    """
    股票AI分析器

    使用方式：
        from quant_core.ai.analyzer import StockAnalyzer
        analyzer = StockAnalyzer()
        result = analyzer.analyze("600519", "贵州茅台")
    """

    def __init__(self, api_key: str = None, model: str = "r1"):
        self.client = DeepSeekClient(api_key=api_key, model=model)
        self.client.set_system_prompt(
            "你是一位专业的量化分析师。所有回答必须使用严格的JSON格式。"
        )

    def analyze(self, code: str, name: str, days: int = 60) -> dict:
        """
        一键分析股票：获取数据 → AI分析 → 返回结构化结果

        参数：
            code: 股票代码
            name: 股票名称
            days: 分析用的历史数据天数

        返回：
            dict，包含AI的结构化分析结果
        """
        # 1. 获取数据
        df = fetch_stock(code, days=days)
        df["daily_return"] = df["close"].pct_change()
        latest = df.iloc[-1]

        # 2. 构建数据字典
        data = {
            "name": name,
            "code": code,
            "close": latest["close"],
            "sma5": df["close"].rolling(5).mean().iloc[-1],
            "sma20": df["close"].rolling(20).mean().iloc[-1],
            "return_5d": df["daily_return"].tail(5).mean(),
            "volatility_20d": df["daily_return"].tail(20).std(),
            "turnover_5d": df["turnover"].tail(5).mean(),
        }

        # 3. 调用AI
        prompt = QuantPrompts.stock_technical_analysis(data)
        result = self.client.chat_json(prompt)

        # 4. 补充原始数据
        result["raw_data"] = data
        result["analysis_date"] = df.index[-1].strftime("%Y-%m-%d")

        return result

    def compare(self, stocks: dict, days: int = 60) -> dict:
        """
        多股票对比分析

        参数：
            stocks: {"贵州茅台": "600519", "比亚迪": "002594", ...}
        """
        stocks_data = []
        for name, code in stocks.items():
            df = fetch_stock(code, days=days)
            df["daily_return"] = df["close"].pct_change()
            stocks_data.append({
                "name": name,
                "code": code,
                "return_20d": df["close"].pct_change(20).iloc[-1],
                "volatility": df["daily_return"].tail(20).std(),
            })

        prompt = QuantPrompts.stock_comparison(stocks_data)
        return self.client.chat_json(prompt)