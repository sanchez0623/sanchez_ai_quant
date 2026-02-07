"""
第2周：量化开发环境搭建
======================
建议用虚拟环境隔离，避免依赖冲突
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 方案一：使用 venv（推荐初学者）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 终端执行：
# python -m venv quant_env
# source quant_env/bin/activate      # Mac/Linux
# quant_env\Scripts\activate         # Windows
# pip install -r requirements.txt

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 方案二：使用 conda（推荐有经验的开发者）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# conda create -n quant python=3.11
# conda activate quant
# pip install -r requirements.txt

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 验证环境
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def check_environment():
    """检查所有依赖是否正确安装"""
    results = {}

    libs = {
        "pandas": "pd",
        "numpy": "np",
        "matplotlib": "matplotlib",
        "akshare": "ak",
        "sklearn": "sklearn",
    }

    for lib_name, import_name in libs.items():
        try:
            module = __import__(import_name if import_name != "pd"
                                else lib_name)
            version = getattr(module, "__version__", "已安装")
            results[lib_name] = f"✅ {version}"
        except ImportError:
            results[lib_name] = "❌ 未安装"

    print("📦 环境检查结果：")
    print("─" * 35)
    for lib, status in results.items():
        print(f"  {lib:15s} {status}")
    print("─" * 35)

    if all("✅" in v for v in results.values()):
        print("  🎉 所有依赖已就绪！")
    else:
        print("  ⚠️ 请安装缺失的依赖：pip install -r requirements.txt")

if __name__ == "__main__":
    check_environment()