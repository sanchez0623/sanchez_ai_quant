"""
全局配置（API Key等敏感信息）
============================
⚠️ 此文件不要上传到GitHub！请加入.gitignore
"""

import os


def _load_dotenv():
	project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
	dotenv_path = os.path.join(project_root, ".env")
	if not os.path.exists(dotenv_path):
		return

	try:
		with open(dotenv_path, "r", encoding="utf-8") as handle:
			for raw_line in handle:
				line = raw_line.strip()
				if not line or line.startswith("#") or "=" not in line:
					continue
				key, value = line.split("=", 1)
				key = key.strip()
				value = value.strip().strip("\"").strip("'")
				if key and value and key not in os.environ:
					os.environ[key] = value
	except OSError:
		return

# 方式1：直接设置（仅本地测试用，不要提交到Git）
# DEEPSEEK_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxx"

# 方式2（推荐）：从 .env / 环境变量读取
_load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# 如何设置环境变量：
# Mac/Linux: 在 ~/.bashrc 或 ~/.zshrc 中添加：
#   export DEEPSEEK_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxx"
# Windows: 在命令行中执行：
#   setx DEEPSEEK_API_KEY "sk-xxxxxxxxxxxxxxxxxxxxxxxx"