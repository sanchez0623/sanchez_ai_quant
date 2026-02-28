import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from quant_core.ai.prompts import QuantPrompts
from quant_core.ai.llm_client import DeepSeekClient

text = "财联社2月11日电，网易发布2025年Q4及全年财报。财报显示，2025年网易业绩稳健，Q4营收275亿元，全年总营收1126亿元。全年营业利润358亿元，同比增长21%；Q4营业利润83亿元，同比增长6%。年研发投入达177亿元，连续六年研发投入突破百亿。游戏及相关增值服务净收入921亿元，其中在线游戏净收入达896亿元，同比增长11%。"
prompt = QuantPrompts.earnings_report_analysis(text, stock_name="网易")
client = DeepSeekClient()
result = client.chat_json(prompt)
print(result)