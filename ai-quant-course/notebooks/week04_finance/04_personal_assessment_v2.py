# -*- coding: utf-8 -*-
"""
第4周-Part4 V2: 个人资产健康度评估（顾问版）
================================================
在V1基础上升级：
1) 健康评分(0-100)
2) 债务分层(好负债/坏负债)
3) 目标资金池(短中长期)
4) 压力测试(市场下跌情景)
5) 区间化资产配置建议
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from typing import Dict, List
import numpy as np


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _score_savings_rate(savings_rate: float) -> float:
    """储蓄率评分：30%以上优秀。"""
    return _clamp(savings_rate / 0.35 * 100, 0, 100)


def _score_emergency_months(months: float) -> float:
    """应急金评分：6个月满分，3个月及格。"""
    if months >= 6:
        return 100
    if months >= 3:
        return 60 + (months - 3) / 3 * 40
    return _clamp(months / 3 * 60, 0, 60)


def _score_debt_ratio(debt_ratio: float) -> float:
    """资产负债率评分：越低越好。"""
    if debt_ratio <= 0.2:
        return 100
    if debt_ratio <= 0.5:
        return 100 - (debt_ratio - 0.2) / 0.3 * 35
    if debt_ratio <= 0.7:
        return 65 - (debt_ratio - 0.5) / 0.2 * 35
    return _clamp(30 - (debt_ratio - 0.7) / 0.3 * 30, 0, 30)


def _target_equity_range(age: int, risk_tolerance: str) -> Dict[str, float]:
    """根据年龄+风险偏好给出权益资产建议区间。"""
    base = 100 - age
    if risk_tolerance == "high":
        base += 10
        width = 15
    elif risk_tolerance == "low":
        base -= 10
        width = 10
    else:
        width = 12

    center = _clamp(base, 20, 80)
    low = _clamp(center - width / 2, 15, 85)
    high = _clamp(center + width / 2, 15, 85)
    return {"low": round(low, 1), "high": round(high, 1), "center": round(center, 1)}


def _build_goal_buckets(goals: List[Dict]) -> Dict[str, float]:
    """
    根据目标期限分桶：
      短期 0-3年
      中期 3-10年
      长期 10年以上
    """
    short_amt = sum(g.get("amount", 0) for g in goals if g.get("years", 0) <= 3)
    mid_amt = sum(g.get("amount", 0) for g in goals if 3 < g.get("years", 0) <= 10)
    long_amt = sum(g.get("amount", 0) for g in goals if g.get("years", 0) > 10)
    total = short_amt + mid_amt + long_amt
    if total <= 0:
        return {"short": 0, "mid": 0, "long": 0, "total_goal_amount": 0}
    return {
        "short": short_amt / total,
        "mid": mid_amt / total,
        "long": long_amt / total,
        "total_goal_amount": total,
    }


def assess_financial_health_v2(
    age: int = 35,
    monthly_income: float = 20000,
    monthly_expense: float = 10000,
    cash_savings: float = 100000,
    stock_value: float = 500000,
    fund_value: float = 300000,
    bond_value: float = 100000,
    gold_value: float = 50000,
    real_estate_value: float = 0,
    debt_mortgage: float = 600000,
    debt_consumer: float = 50000,
    debt_other: float = 0,
    risk_tolerance: str = "medium",  # low / medium / high
    income_stability_score: float = 70,  # 0-100，工作/现金流稳定性主观评分
    goals: List[Dict] = None,  # [{"name":"买房首付", "years":3, "amount":500000}, ...]
):
    """
    顾问版个人资产评估。
    """
    if goals is None:
        goals = [
            {"name": "子女教育金", "years":8, "amount": 400000},
            {"name": "退休储备", "years":25, "amount": 5000000},
        ]

    total_debt = debt_mortgage + debt_consumer + debt_other
    total_assets = (
        cash_savings + stock_value + fund_value + bond_value + gold_value + real_estate_value
    )
    investable_assets = cash_savings + stock_value + fund_value + bond_value + gold_value
    net_worth = total_assets - total_debt

    monthly_surplus = monthly_income - monthly_expense
    savings_rate = monthly_surplus / monthly_income if monthly_income > 0 else 0
    emergency_months = cash_savings / monthly_expense if monthly_expense > 0 else 0
    debt_ratio = total_debt / total_assets if total_assets > 0 else 1

    equity_value = stock_value + fund_value
    equity_pct = equity_value / investable_assets * 100 if investable_assets > 0 else 0
    bond_pct = bond_value / investable_assets * 100 if investable_assets > 0 else 0
    gold_pct = gold_value / investable_assets * 100 if investable_assets > 0 else 0
    cash_pct = cash_savings / investable_assets * 100 if investable_assets > 0 else 0

    # 评分体系
    score_cashflow = _score_savings_rate(savings_rate)
    score_emergency = _score_emergency_months(emergency_months)
    score_debt = _score_debt_ratio(debt_ratio)
    score_stability = _clamp(income_stability_score, 0, 100)

    health_score = (
        score_cashflow * 0.30
        + score_emergency * 0.25
        + score_debt * 0.25
        + score_stability * 0.20
    )

    if health_score >= 85:
        health_level = "A(优秀)"
    elif health_score >= 70:
        health_level = "B(良好)"
    elif health_score >= 55:
        health_level = "C(中等)"
    else:
        health_level = "D(待改善)"

    # 债务分层建议
    bad_debt_ratio = debt_consumer / total_assets if total_assets > 0 else 0
    if debt_consumer > 0 and bad_debt_ratio > 0.1:
        debt_advice = "优先偿还消费贷/信用贷，再提高权益仓位。"
    elif debt_mortgage > 0 and debt_ratio < 0.5:
        debt_advice = "房贷在可控区间，可按计划偿还并继续长期投资。"
    elif total_debt == 0:
        debt_advice = "无债务结构优秀，可将更多现金流用于长期复利投资。"
    else:
        debt_advice = "控制总负债增速，避免新增高利率负债。"

    # 目标分桶与建议
    buckets = _build_goal_buckets(goals)
    target_equity = _target_equity_range(age, risk_tolerance)

    # 短中长期资金池建议（对可投资资产）
    # 给目标优先级：短期越高，现金/固收池越大
    short_pool_pct = _clamp(max(0.25, buckets["short"] * 1.2), 0.20, 0.60)
    mid_pool_pct = _clamp(max(0.20, buckets["mid"]), 0.15, 0.45)
    long_pool_pct = _clamp(1 - short_pool_pct - mid_pool_pct, 0.20, 0.65)

    # 归一化（防止clamp造成和不为1）
    pool_total = short_pool_pct + mid_pool_pct + long_pool_pct
    short_pool_pct, mid_pool_pct, long_pool_pct = [x / pool_total for x in (short_pool_pct, mid_pool_pct, long_pool_pct)]

    # 压力测试：权益-30%，债券+3%，黄金+12%，现金0%
    stress_return = (
        (equity_pct / 100) * (-0.30)
        + (bond_pct / 100) * (0.03)
        + (gold_pct / 100) * (0.12)
        + (cash_pct / 100) * (0.00)
    )
    stress_loss_amount = investable_assets * (-stress_return) if stress_return < 0 else 0

    print("=" * 68)
    print("  个人资产健康度评估 V2（顾问版）")
    print("=" * 68)

    print(f"""
【一、财务体检】
- 年龄: {age} 岁 | 风险偏好: {risk_tolerance}
- 月收入: {monthly_income:,.0f} 元 | 月支出: {monthly_expense:,.0f} 元 | 月结余: {monthly_surplus:,.0f} 元
- 总资产: {total_assets:,.0f} 元 | 总负债: {total_debt:,.0f} 元 | 净资产: {net_worth:,.0f} 元
- 储蓄率: {savings_rate:.1%} | 应急月数: {emergency_months:.1f} 个月 | 资产负债率: {debt_ratio:.1%}

【二、健康评分】
- 综合评分: {health_score:.1f}/100  -> {health_level}
- 现金流评分: {score_cashflow:.1f}
- 应急金评分: {score_emergency:.1f}
- 负债评分: {score_debt:.1f}
- 收入稳定评分: {score_stability:.1f}

【三、当前资产结构（可投资资产口径）】
- 权益(股票+基金): {equity_pct:.1f}%
- 债券: {bond_pct:.1f}%
- 黄金: {gold_pct:.1f}%
- 现金: {cash_pct:.1f}%

【四、顾问建议】
- 债务建议: {debt_advice}
- 权益建议区间: {target_equity['low']:.0f}% ~ {target_equity['high']:.0f}% (中枢 {target_equity['center']:.0f}%)
- 应急金目标: {monthly_expense*6:,.0f} 元（约6个月支出）

【五、目标资金池建议】
- 短期资金池(0-3年): {short_pool_pct:.0%}（以现金/货基/短债为主）
- 中期资金池(3-10年): {mid_pool_pct:.0%}（股债平衡）
- 长期资金池(10年以上): {long_pool_pct:.0%}（权益为主）
- 目标总资金需求: {buckets['total_goal_amount']:,.0f} 元

【六、压力测试（单期极端情景）】
- 假设: 权益-30%, 债券+3%, 黄金+12%, 现金0%
- 组合情景收益: {stress_return:.2%}
- 预计账面波动: {-stress_loss_amount:,.0f} 元
""")

    return {
        "health_score": round(health_score, 2),
        "health_level": health_level,
        "net_worth": net_worth,
        "metrics": {
            "savings_rate": savings_rate,
            "emergency_months": emergency_months,
            "debt_ratio": debt_ratio,
            "equity_pct": equity_pct,
            "bond_pct": bond_pct,
            "gold_pct": gold_pct,
            "cash_pct": cash_pct,
        },
        "advice": {
            "debt_advice": debt_advice,
            "target_equity_range": target_equity,
            "short_pool_pct": short_pool_pct,
            "mid_pool_pct": mid_pool_pct,
            "long_pool_pct": long_pool_pct,
        },
        "stress_test": {
            "scenario_return": stress_return,
            "scenario_loss_amount": stress_loss_amount,
        },
    }


# 运行示例（请替换为个人真实数据）
if __name__ == "__main__":
    result = assess_financial_health_v2(
        age=35,
        monthly_income=20000,
        monthly_expense=10000,
        cash_savings=100000,
        stock_value=1000000,
        fund_value=500000,
        bond_value=0,
        gold_value=0,
        real_estate_value=0,
        debt_mortgage=700000,
        debt_consumer=100000,
        debt_other=0,
        risk_tolerance="medium",
        income_stability_score=75,
        goals=[
            {"name": "3年后换车", "years": 3, "amount": 200000},
            {"name": "8年后子女教育", "years": 8, "amount": 500000},
            {"name": "25年后退休", "years": 25, "amount": 6000000},
        ],
    )
