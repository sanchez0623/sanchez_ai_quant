
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from quant_core.ai import DeepSeekClient

client = DeepSeekClient(model="r1")
history = []

step1 = client.chat_with_history(history, "分析贵州茅台")
history = step1["history"]

step2 = client.chat_with_history(history, "和比亚迪比呢?")
print(step2["reply"])