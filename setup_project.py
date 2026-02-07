"""
量化投资学习项目 — 初始化脚本
==============================
运行此脚本来创建整个课程的项目结构
"""

import os

# 项目根目录结构
PROJECT_STRUCTURE = {
    "ai-quant-course/": {
        # ===== 按周组织的学习笔记和作业 =====
        "notebooks/": {
            "week01_intro/": ["01_hello_quant.py"],
            "week02_python_ds/": [],
            "week03_llm_api/": [],
            "week04_finance/": [],
            "week05_data/": [],
            "week06_stock_fund/": [],
            "week07_backtest_design/": [],
            "week08_technical/": [],
            "week09_fundamental/": [],
            "week10_strategy/": [],
            "week11_deltafq/": [],
            "week12_stock_picking/": [],
            "week13_fund_intro/": [],
            "week14_fund_strategy/": [],
            "week15_system_arch/": [],
            "week16_trading/": [],
            "week17_automation/": [],
            "week18_ai_agent/": [],
            "week19_deploy/": [],
            "week20_outlook/": [],
        },
        # ===== 核心量化库（逐步构建） =====
        "quant_core/": {
            "__init__.py": "",
            "data/": {          # 数据层（第2、5周开始）
                "__init__.py": "",
                "fetcher.py":   "# 数据获取模块",
                "cleaner.py":   "# 数据清洗模块",
            },
            "indicators/": {    # 指标层（第8、9周）
                "__init__.py": "",
                "technical.py": "# 技术指标：SMA/EMA/RSI/BOLL",
                "fundamental.py": "# 基本面指标：PE/PB/ROE",
            },
            "strategy/": {      # 策略层（第10-12周）
                "__init__.py": "",
                "base.py":      "# BaseStrategy 策略基类",
                "signals.py":   "# 信号生成模块",
            },
            "backtest/": {      # 回测层（第10-11周）
                "__init__.py": "",
                "engine.py":    "# BacktestEngine 回测引擎",
                "metrics.py":   "# 绩效评估指标",
            },
            "trading/": {       # 交易层（第15-17周）
                "__init__.py": "",
                "order.py":     "# 订单管理",
                "risk.py":      "# 风控模块",
            },
            "ai/": {            # AI层（第3、18周）
                "__init__.py": "",
                "llm_client.py": "# LLM API客户端",
                "agent.py":      "# AI Agent",
                "prompts.py":    "# Prompt模板库",
            },
        },
        # ===== 配置与工具 =====
        "config/": {
            "settings.py":  "# 全局配置（API Key等）",
        },
        "tests/": {
            "__init__.py": "",
        },
        "requirements.txt": "",
        "README.md": "",
    }
}


def create_structure(base_path, structure):
    """递归创建项目目录结构"""
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        if name.endswith("/"):
            # 这是一个目录
            dir_path = path.rstrip("/")
            os.makedirs(dir_path, exist_ok=True)
            if isinstance(content, dict):
                create_structure(dir_path, content)
            elif isinstance(content, list):
                for file in content:
                    file_path = os.path.join(dir_path, file)
                    if not os.path.exists(file_path):
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(f"# {file}\n# Created for AI Quant Course\n")
        else:
            # 这是一个文件
            if not os.path.exists(path):
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content if content else f"# {name}\n")


if __name__ == "__main__":
    create_structure(".", PROJECT_STRUCTURE)
    print("✅ 项目结构创建成功！")
    print("\n📁 目录结构：")
    print("""
    ai-quant-course/
    ├── notebooks/          ← 每周学习笔记和作业
    │   ├── week01_intro/
    │   ├── week02_python_ds/
    │   └── ...
    ├── quant_core/         ← 核心量化库（逐步构建）
    │   ├── data/           ← 数据管理
    │   ├── indicators/     ← 技术+基本面指标
    │   ├── strategy/       ← 策略引擎
    │   ├── backtest/       ← 回测引擎
    │   ├── trading/        ← 交易系统
    │   └── ai/             ← AI智能层
    ├── config/             ← 配置文件
    ├── tests/              ← 测试
    └── requirements.txt
    """)