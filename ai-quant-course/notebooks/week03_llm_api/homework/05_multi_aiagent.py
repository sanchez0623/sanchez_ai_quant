
import sys
from pathlib import Path
# ── 项目路径 ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from quant_core.ai import MultiModelClient, DeepSeekClient,KimiClient

# 自动加载DeepSeek + Kimi(从.env读Key)
mc = MultiModelClient({
                "deepseek-v3": DeepSeekClient(model="r1"),
                "kimi-k2.5":     KimiClient(model="k2.5")})
mc.set_system_prompt("你是一位资深量化分析师。回答用严格JSON格式。")

# 同一问题对比两个模型
results = mc.compare_json("分析贵州茅台近期走势，给出JSON格式的趋势判断")
mc.print_comparison(results)

# 也可以单独用某个模型
answer = mc.chat("你好", model="kimi-k2.5")
print("\nKimi的回答：")
print(answer)