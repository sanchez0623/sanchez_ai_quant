"""
Prompt模板库
============
所有量化分析的Prompt模板集中管理
后续课程（第9周基本面、第12周选股等）都会往这里添加新模板
"""


class QuantPrompts:
    """
    量化分析Prompt模板库

    使用方式：
        from quant_core.ai.prompts import QuantPrompts
        prompt = QuantPrompts.stock_technical_analysis(data_dict)
    """

    @staticmethod
    def stock_technical_analysis(data: dict) -> str:
        """
        技术面分析Prompt

        data需包含: name, code, close, sma5, sma20, return_5d, volatility_20d, turnover_5d
        """
        return f"""
你是一位资深量化分析师。请基于以下数据进行技术面分析，以JSON格式返回。

## 数据
- 股票: {data['name']}({data['code']})
- 最新收盘价: {data['close']:.2f}
- 5日均线(MA5): {data['sma5']:.2f}
- 20日均线(MA20): {data['sma20']:.2f}
- 近5日平均涨跌幅: {data['return_5d']:.4%}
- 近20日波动率: {data['volatility_20d']:.4%}
- 近5日平均换手率: {data['turnover_5d']:.2f}%

## 返回格式（严格JSON）
{{
    "trend": "uptrend/downtrend/sideways",
    "trend_cn": "上涨趋势/下跌趋势/横盘震荡",
    "strength": 1到10的整数,
    "support": 支撑位数字,
    "resistance": 压力位数字,
    "risk_level": "high/medium/low",
    "risk_cn": "高风险/中等风险/低风险",
    "outlook": "短期展望中文描述",
    "confidence": 0到1的小数
}}
"""

    @staticmethod
    def stock_comparison(stocks_data: list) -> str:
        """
        多股票对比分析Prompt

        stocks_data: [{"name": ..., "code": ..., "return_20d": ..., ...}, ...]
        """
        stocks_text = ""
        for s in stocks_data:
            stocks_text += f"- {s['name']}({s['code']}): 20日涨跌幅={s['return_20d']:.2%}, 波动率={s['volatility']:.4%}\n"

        return f"""
你是一位资深量化分析师。请对比以下股票，以JSON格式返回排名和分析。

## 股票数据
{stocks_text}

## 返回格式（严格JSON）
{{
    "ranking": [
        {{"code": "代码", "name": "名称", "rank": 排名, "score": 评分, "reason_cn": "中文理由"}}
    ],
    "diversification_advice_cn": "分散投资建议，中文",
    "overall_market_view_cn": "整体市场观点，中文"
}}
"""

    @staticmethod
    def news_sentiment(news_text: str, stock_name: str) -> str:
        """
        新闻情绪分析Prompt（第18周AI Agent会大量使用）
        """
        return f"""
你是一位金融舆情分析师。请分析以下新闻对{stock_name}的影响。

## 新闻内容
{news_text}

## 返回格式（严格JSON）
{{
    "sentiment": "positive/negative/neutral",
    "sentiment_cn": "利好/利空/中性",
    "impact_score": -10到10的整数（正数=利好，负数=利��），
    "impact_duration": "short_term/medium_term/long_term",
    "impact_duration_cn": "短期影响/中期影响/长期影响",
    "key_points_cn": ["关键点1", "关键点2"],
    "confidence": 0到1的小数
}}
"""