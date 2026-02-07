"""
路径设置（每个notebook目录放一份）
==================================
确保能从任意子目录导入 quant_core
"""

import sys
from pathlib import Path

# 把项目根目录加入 Python 路径
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print(f"✅ 项目根目录: {project_root}")
print(f"   现在可以使用: from quant_core.data import fetch_stock")