# -*- coding: utf-8 -*-
"""
第4周-Part4: 个人资产健康度评估
================================
输入你的资产数据, 自动诊断健康度并给出配置建议
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from quant_core.ai import DeepSeekClient
import json


def assess_financial_health(
    age: int = 28,
    monthly_income: float = 15000,
    monthly_expense: float = 8000,
    savings: float = 100000,
    stock_value: float = 50000,
    fund_value: float = 30000,
    bond_value: float = 0,
    gold_value: float = 0,
    real_estate_value: float = 0,
    debt: float = 0,
    risk_tolerance: str = "medium",
):
    """
    个人资产健康度评估

    参数:
        age:              年龄
        monthly_income:   月收入(元)
        monthly_expense:  月支出(元)
        savings:          现金/存款(元)
        stock_value:      股票市值(元)
        fund_value:       基金市值(元)
        bond_value:       债券市值(元)
        gold_value:       黄金市值(元)
        real_estate_value:房产市值(元), 不含自住房
        debt:             负债总额(元)
        risk_tolerance:   风险偏好 "low"(保守) / "medium"(稳健) / "high"(激进)

    术语:
      储蓄率(Savings Rate): (收入-支出)/收入, 衡量你每月能存下多少钱
      应急储备(Emergency Fund): 建议保留3-6个月支出作为现金储备
      资产负债率: 负债/总资产, 越低越健康
    """
    total_assets = savings + stock_value + fund_value + bond_value + gold_value + real_estate_value
    net_worth = total_assets - debt
    monthly_savings = monthly_income - monthly_expense
    savings_rate = monthly_savings / monthly_income if monthly_income > 0 else 0
    emergency_months = savings / monthly_expense if monthly_expense > 0 else 0
    debt_ratio = debt / total_assets if total_assets > 0 else 0

    # 当前资产配比
    investable = stock_value + fund_value + bond_value + gold_value + savings
    if investable > 0:
        current_allocation = {
            "stock_pct": stock_value / investable * 100,
            "fund_pct": fund_value / investable * 100,
            "bond_pct": bond_value / investable * 100,
            "gold_pct": gold_value / investable * 100,
            "cash_pct": savings / investable * 100,
        }
    else:
        current_allocation = {k: 0 for k in ["stock_pct", "fund_pct", "bond_pct", "gold_pct", "cash_pct"]}

    print("=" * 60)
    print("  个人资产健康度评估报告")
    print("=" * 60)

    print(f"""
    基本信息:
      年龄: {age}岁
      月收入: {monthly_income:,.0f} 元
      月支出: {monthly_expense:,.0f} 元
      月结余: {monthly_savings:,.0f} 元
      储蓄率: {savings_rate:.1%} {"(优秀>30%)" if savings_rate > 0.3 else "(建议提高到30%+)"}

    资产负债:
      总资产: {total_assets:,.0f} 元
      总负债: {debt:,.0f} 元
      净资产: {net_worth:,.0f} 元
      资产负债率: {debt_ratio:.1%} {"(健康<50%)" if debt_ratio < 0.5 else "(偏高, 注意风险)"}

    应急储备:
      当前现金: {savings:,.0f} 元
      可支撑月数: {emergency_months:.1f} 个月
      健康标准: 3-6个月支出 = {monthly_expense * 3:,.0f} ~ {monthly_expense * 6:,.0f} 元
      评估: {"充足" if emergency_months >= 6 else "偏少" if emergency_months >= 3 else "严重不足!"}

    当前资产配比:
      股票: {current_allocation["stock_pct"]:.1f}%
      基金: {current_allocation["fund_pct"]:.1f}%
      债券: {current_allocation["bond_pct"]:.1f}%
      黄金: {current_allocation["gold_pct"]:.1f}%
      现金: {current_allocation["cash_pct"]:.1f}%
    """)

    # 建议配比(根据年龄和风险偏好)
    # 经典法则: 股票比例 = 100 - 年龄 (激进版用110)
    base_stock_pct = max(20, min(80, (100 if risk_tolerance == "medium" else 110 if risk_tolerance == "high" else 90) - age))

    suggested = {
        "stock_equity": base_stock_pct,
        "bond": max(10, 60 - base_stock_pct),
        "gold": 10,
        "cash": 100 - base_stock_pct - max(10, 60 - base_stock_pct) - 10,
    }
    # 确保不出现负数
    suggested["cash"] = max(5, suggested["cash"])

    # 重新归一化
    total_s = sum(suggested.values())
    suggested = {k: v / total_s * 100 for k, v in suggested.items()}

    print(f"""    建议配比(基于年龄{age}岁 + {risk_tolerance}风险偏好):
      股票+权益类基金: {suggested["stock_equity"]:.0f}%
      债券+债券基金:   {suggested["bond"]:.0f}%
      黄金:            {suggested["gold"]:.0f}%
      现金:            {suggested["cash"]:.0f}%

    经典法则: 股票比例 = 100 - 年龄
      年轻人(25岁): 股票75%, 有时间承受波动, 追求长期增长
      中年人(45岁): 股票55%, 平衡增长与稳定
      退休前(60岁): 股票40%, 以稳健为主, 减少波动
    """)

    return {
        "net_worth": net_worth,
        "savings_rate": savings_rate,
        "emergency_months": emergency_months,
        "current_allocation": current_allocation,
        "suggested_allocation": suggested,
    }


# 运行评估 -- 修改以下参数为你的真实数据!
result = assess_financial_health(
    age=35,
    monthly_income=20000,
    monthly_expense=10000,
    savings=100000,
    stock_value=1000000,
    fund_value=500000,
    bond_value=0,
    gold_value=0,
    debt=800000,
    risk_tolerance="medium",  # "low" / "medium" / "high"
)