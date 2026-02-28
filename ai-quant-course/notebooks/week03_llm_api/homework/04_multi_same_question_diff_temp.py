# -*- coding: utf-8 -*-
"""
第3周-作业4: Temperature 对比实验
=================================
同一个分析Prompt, 分别用 temperature=0, 0.3, 0.7, 1.0 各调用10次,
统计AI给出的趋势判断(uptrend/downtrend/sideways)分布,
用柱状图可视化, 验证"temperature越高, 输出越不稳定"。

实验设计思路
────────────
1. 控制变量: 同一份固定数据 + 同一个Prompt模板, 只改temperature
2. 重复实验: 每个temperature跑N轮(默认10轮), 保证统计学意义
3. 结果采集: 完整记录每轮的原始JSON, 供后续复查
4. 稳定性量化: 用信息熵(Shannon Entropy)客观衡量分布集中度
5. 可视化: 分组柱状图 + 稳定性曲线 + 详细统计表

术语说明
────────
Temperature(温度): 控制AI输出随机度的参数
    0   → 几乎确定性输出(贪婪解码), 每次回答基本相同
    1.0 → 高随机性, 同一问题可能得到不同方向的回答
信息熵(Entropy): 分布混乱度的度量, 越高=越不确定
    全部一样 → 熵=0; 均匀分布 → 熵=log2(类别数)
"""

import sys
import json
import time
import math
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── 项目路径 ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from quant_core.ai import DeepSeekClient, QuantPrompts

# ── 中文字体与Matplotlib全局设置 ──────────────────────────
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# ── 实验常量 ──────────────────────────────────────────────
TEMPERATURES = [0.0, 0.3, 0.7, 1.0]       # 待测试的Temperature列表
RUNS_PER_TEMP = 10                          # 每个Temperature重复调用次数
VALID_TRENDS = ["uptrend", "downtrend", "sideways"]  # 模型应返回的合法趋势值
MODEL = "v3"                                # 使用DeepSeek-V3(快速且便宜)

# ── 测试数据设计原则 ─────────────────────────────────────
#
# 关键: 必须使用**边界/模糊数据**, 让AI "犹豫"!
#
# 如果数据太一边倒(比如收盘价远高于均线、涨幅显著为正),
# 模型在所有temperature下都会给出相同的"uptrend"判断,
# 因为证据太强, 随机采样无法改变结论方向。
#
# 正确做法: 构造"均线粘合 + 涨跌幅接近零 + 高波动"的场景,
# 让模型在uptrend/downtrend/sideways之间真正纠结,
# 这样temperature的效果才能体现出来。
# ─────────────────────────────────────────────────────────

TEST_STOCK_DATA = {
    "name": "贵州茅台",
    "code": "600519",
    "close": 1780.00,        # 收盘价卡在MA5和MA20之间(模糊!)
    "sma5": 1782.50,         # MA5略高于收盘价(微弱空头信号)
    "sma20": 1778.30,        # MA20略低于收盘价(微弱多头信号)
    "return_5d": -0.0008,    # 近5日几乎持平, 方向不明
    "volatility_20d": 0.0260, # 波动率偏高, 增加不确定性
    "turnover_5d": 3.15,     # 换手率偏高, 多空博弈激烈
}


# ═══════════════════════════════════════════════════════════
# 数据结构: 单次调用记录 & 实验整体结果
# ═══════════════════════════════════════════════════════════

@dataclass
class SingleCallRecord:
    """一次API调用的完整记录"""
    temperature: float
    run_index: int           # 第几轮(从1开始)
    trend: str               # 提取到的趋势判断
    confidence: Optional[float] = None  # AI自评置信度
    strength: Optional[int] = None      # AI给的趋势强度
    raw_json: Optional[dict] = None     # 原始返回JSON(调试用)
    error: Optional[str] = None         # 若出错则记录错误信息
    latency_ms: float = 0.0             # 响应耗时(毫秒)


@dataclass
class ExperimentResult:
    """整个实验的汇总结果"""
    stock_data: dict                                  # 使用的测试数据
    temperatures: List[float] = field(default_factory=list)
    runs_per_temp: int = 0
    records: List[SingleCallRecord] = field(default_factory=list)

    # ── 分析方法 ──────────────────────────────────

    def get_records_by_temp(self, temp: float) -> List[SingleCallRecord]:
        """取某个temperature的所有记录"""
        return [r for r in self.records if r.temperature == temp]

    def trend_counter(self, temp: float) -> Counter:
        """统计某个temperature下各趋势的出现次数"""
        return Counter(r.trend for r in self.get_records_by_temp(temp))

    def trend_distribution(self, temp: float) -> Dict[str, float]:
        """趋势概率分布(归一化为百分比)"""
        counter = self.trend_counter(temp)
        total = sum(counter.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in counter.items()}

    def shannon_entropy(self, temp: float) -> float:
        """
        计算Shannon信息熵(bits)

        H = -Σ p(x) * log2(p(x))

        熵=0: 完全确定(每次输出一样)
        熵=log2(3)≈1.585: 三种趋势均匀分布(最大不确定性)
        """
        dist = self.trend_distribution(temp)
        if not dist:
            return 0.0
        return -sum(p * math.log2(p) for p in dist.values() if p > 0)

    def dominant_ratio(self, temp: float) -> float:
        """
        主导比率: 出现最多的趋势占比

        比率=1.0: 每次都一样(完全稳定)
        比率≈0.33: 三种趋势均匀(完全不稳定)
        """
        counter = self.trend_counter(temp)
        total = sum(counter.values())
        if total == 0:
            return 0.0
        return max(counter.values()) / total

    def avg_latency(self, temp: float) -> float:
        """某个temperature的平均响应时间(毫秒)"""
        records = self.get_records_by_temp(temp)
        if not records:
            return 0.0
        return sum(r.latency_ms for r in records) / len(records)

    def error_count(self, temp: float) -> int:
        """某个temperature的错误次数"""
        return sum(1 for r in self.get_records_by_temp(temp) if r.error)

    def to_dataframe(self) -> pd.DataFrame:
        """将所有记录转为DataFrame, 便于分析和导出"""
        rows = []
        for r in self.records:
            rows.append({
                "temperature": r.temperature,
                "run": r.run_index,
                "trend": r.trend,
                "confidence": r.confidence,
                "strength": r.strength,
                "latency_ms": round(r.latency_ms, 1),
                "error": r.error or "",
            })
        return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════
# 核心实验逻辑
# ═══════════════════════════════════════════════════════════

def run_experiment(
    stock_data: dict = None,
    temperatures: List[float] = None,
    runs: int = RUNS_PER_TEMP,
    model: str = MODEL,
) -> ExperimentResult:
    """
    执行Temperature对比实验

    参数：
        stock_data:    测试用的股票数据字典(默认用贵州茅台)
        temperatures:  待测试的temperature列表(默认[0, 0.3, 0.7, 1.0])
        runs:          每个temperature重复次数(默认10)
        model:         使用的DeepSeek模型(默认v3)

    返回：
        ExperimentResult 实验结果对象
    """
    stock_data = stock_data or TEST_STOCK_DATA
    temperatures = temperatures or TEMPERATURES

    # 初始化客户端 & 生成固定Prompt
    client = DeepSeekClient(model=model)
    prompt = QuantPrompts.stock_technical_analysis(stock_data)

    result = ExperimentResult(
        stock_data=stock_data,
        temperatures=temperatures,
        runs_per_temp=runs,
    )

    total_calls = len(temperatures) * runs
    call_idx = 0

    print("=" * 60)
    print("  Temperature 对比实验")
    print("=" * 60)
    print(f"  股票: {stock_data['name']}({stock_data['code']})")
    print(f"  模型: DeepSeek {model}")
    print(f"  Temperature列表: {temperatures}")
    print(f"  每组重复: {runs}次")
    print(f"  总调用次数: {total_calls}")
    print("=" * 60)

    for temp in temperatures:
        print(f"\n  ── Temperature = {temp} ──")
        for i in range(1, runs + 1):
            call_idx += 1
            record = SingleCallRecord(temperature=temp, run_index=i, trend="error")

            t0 = time.time()
            try:
                resp = client.chat_json(prompt, temperature=temp)
                record.latency_ms = (time.time() - t0) * 1000
                record.raw_json = resp

                # 提取趋势判断(做容错: 转小写 + 去空格)
                raw_trend = str(resp.get("trend", "")).strip().lower()
                record.trend = raw_trend if raw_trend in VALID_TRENDS else "unknown"

                # 提取置信度和趋势强度(可选字段)
                try:
                    record.confidence = float(resp.get("confidence", 0))
                except (ValueError, TypeError):
                    pass
                try:
                    record.strength = int(resp.get("strength", 0))
                except (ValueError, TypeError):
                    pass

            except Exception as e:
                record.latency_ms = (time.time() - t0) * 1000
                record.error = str(e)
                record.trend = "error"

            result.records.append(record)

            # 实时进度输出
            status = f"✓ {record.trend}" if not record.error else f"✗ {record.error[:40]}"
            print(f"    [{call_idx:>2}/{total_calls}] 第{i:>2}轮: {status}"
                  f"  ({record.latency_ms:.0f}ms)")

    print(f"\n{'=' * 60}")
    print("  实验完成！")
    print(f"{'=' * 60}")

    return result


# ═══════════════════════════════════════════════════════════
# 统计分析 & 结论输出
# ═══════════════════════════════════════════════════════════

def print_statistics(result: ExperimentResult) -> None:
    """打印详细的统计分析表"""

    print("\n" + "=" * 60)
    print("  📊 统计分析结果")
    print("=" * 60)

    # 1. 每个Temperature的趋势分布
    print("\n  ┌─ 趋势分布 ─────────────────────────────────────┐")
    for temp in result.temperatures:
        counter = result.trend_counter(temp)
        dist_str = "  |  ".join(f"{k}: {v}次" for k, v in sorted(counter.items()))
        print(f"  │ temp={temp:.1f}  →  {dist_str}")
    print("  └───────────────────────────────────────────────┘")

    # 2. 稳定性指标
    print("\n  ┌─ 稳定性指标 ───────────────────────────────────┐")
    print(f"  │ {'Temperature':>12} │ {'信息熵(bits)':>12} │ {'主导比率':>8} │ {'均延迟(ms)':>10} │ {'错误数':>6} │")
    print(f"  │ {'─' * 12} │ {'─' * 12} │ {'─' * 8} │ {'─' * 10} │ {'─' * 6} │")
    for temp in result.temperatures:
        entropy = result.shannon_entropy(temp)
        dominant = result.dominant_ratio(temp)
        avg_lat = result.avg_latency(temp)
        errors = result.error_count(temp)
        print(f"  │ {temp:>12.1f} │ {entropy:>12.4f} │ {dominant:>8.1%} │ {avg_lat:>10.0f} │ {errors:>6} │")
    print("  └───────────────────────────────────────────────┘")

    max_entropy = math.log2(len(VALID_TRENDS))  # ≈1.585 for 3 trends
    print(f"\n  📖 信息熵解读:")
    print(f"     熵 = 0.000  → 每次输出完全一致(最稳定)")
    print(f"     熵 = {max_entropy:.3f}  → 三种趋势均匀分布(最不稳定)")

    # 3. 置信度标准差(连续指标, 比趋势分类更敏感)
    print("\n  ┌─ 连续指标波动性(置信度 & 趋势强度) ────────────┐")
    print(f"  │ {'Temperature':>12} │ {'置信度均值':>10} │ {'置信度标准差':>12} │ {'强度均值':>8} │ {'强度标准差':>10} │")
    print(f"  │ {'─' * 12} │ {'─' * 10} │ {'─' * 12} │ {'─' * 8} │ {'─' * 10} │")
    for temp in result.temperatures:
        records = result.get_records_by_temp(temp)
        confs = [r.confidence for r in records if r.confidence is not None]
        strs = [r.strength for r in records if r.strength is not None]
        conf_mean = np.mean(confs) if confs else 0
        conf_std = np.std(confs) if confs else 0
        str_mean = np.mean(strs) if strs else 0
        str_std = np.std(strs) if strs else 0
        print(f"  │ {temp:>12.1f} │ {conf_mean:>10.4f} │ {conf_std:>12.4f} │ {str_mean:>8.2f} │ {str_std:>10.4f} │")
    print("  └───────────────────────────────────────────────┘")

    # 4. 综合结论
    entropies = [result.shannon_entropy(t) for t in result.temperatures]
    has_any_variation = any(e > 0 for e in entropies)
    entropy_increasing = (entropies[-1] > entropies[0] + 0.01) if has_any_variation else False

    # 连续指标的标准差是否随temperature递增
    conf_stds = []
    for temp in result.temperatures:
        confs = [r.confidence for r in result.get_records_by_temp(temp) if r.confidence is not None]
        conf_stds.append(np.std(confs) if confs else 0)
    conf_std_increasing = conf_stds[-1] > conf_stds[0] + 0.001

    print("\n  ── 实验结论 ──")
    if not has_any_variation:
        print("  ⚠️  所有Temperature下趋势判断完全一致!")
        print("     可能原因: 输入数据信号太强(如价格远高于均线),")
        print("     导致模型无论随机度多高都给出相同的分类判断。")
        print("     → 已使用边界模糊数据来规避此问题, 请检查TEST_STOCK_DATA。")
        if conf_std_increasing:
            print("  📊 但置信度/强度的波动确实随temperature增大,")
            print("     说明temperature对连续值输出仍有影响。")
    elif entropy_increasing:
        print("  ✅ 验证通过: Temperature越高, 信息熵越大, 输出确实越不稳定!")
        low_e = entropies[0]
        high_e = entropies[-1]
        print(f"     temp=0 熵={low_e:.3f} → temp=1.0 熵={high_e:.3f} (增幅{high_e - low_e:.3f})")
    else:
        print("  ⚠️  信息熵未呈现明显递增趋势(可能因为样本量较小)。")
        print("     建议: 增加runs_per_temp到20~30次, 或多跑几次实验取平均。")

    print(f"\n  💡 量化建议:")
    print(f"     技术分析类任务: temperature=0~0.3 (追求稳定可复现)")
    print(f"     创意生成类任务: temperature=0.7~1.0 (追求多样性)")


# ═══════════════════════════════════════════════════════════
# 可视化: 3合1大图
# ═══════════════════════════════════════════════════════════

def plot_results(result: ExperimentResult, save_path: str = None) -> None:
    """
    绘制实验结果可视化(3合1)

    上图: 分组柱状图 — 各Temperature下趋势判断的次数分布
    中图: 信息熵曲线 — 量化Temperature与输出稳定性的关系
    下图: 置信度箱线图 — 各Temperature下AI自评置信度的波动
    """

    fig = plt.figure(figsize=(14, 13))

    # 收集所有出现过的趋势类别(保持固定顺序)
    all_trends = []
    for t in VALID_TRENDS + ["unknown", "error"]:
        if any(t in result.trend_counter(temp) for temp in result.temperatures):
            all_trends.append(t)

    # 趋势中文映射
    trend_cn = {
        "uptrend": "上涨趋势", "downtrend": "下跌趋势",
        "sideways": "横盘震荡", "unknown": "未识别", "error": "调用错误",
    }
    trend_colors = {
        "uptrend": "#E74C3C", "downtrend": "#27AE60",
        "sideways": "#F39C12", "unknown": "#95A5A6", "error": "#7F8C8D",
    }

    # ────────────────────────────────────────────
    # 上图: 分组柱状图
    # ────────────────────────────────────────────
    ax1 = fig.add_subplot(3, 1, 1)

    x = np.arange(len(result.temperatures))
    n_trends = len(all_trends)
    bar_width = 0.6 / max(n_trends, 1)

    for i, trend in enumerate(all_trends):
        counts = [result.trend_counter(t).get(trend, 0) for t in result.temperatures]
        offset = (i - (n_trends - 1) / 2) * bar_width
        bars = ax1.bar(
            x + offset, counts, bar_width,
            label=f"{trend_cn.get(trend, trend)}({trend})",
            color=trend_colors.get(trend, "#BBBBBB"),
            edgecolor="white", linewidth=0.5,
        )
        # 在柱子上方标注数字
        for bar, count in zip(bars, counts):
            if count > 0:
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                         str(count), ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"temp={t}" for t in result.temperatures], fontsize=11)
    ax1.set_ylabel("出现次数", fontsize=11)
    ax1.set_title(
        f"Temperature对比实验 — 趋势判断分布\n"
        f"({result.stock_data['name']} | 模型: DeepSeek-V3 | 每组{result.runs_per_temp}次)",
        fontsize=13, fontweight="bold",
    )
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax1.set_ylim(0, result.runs_per_temp + 1.5)
    ax1.axhline(y=result.runs_per_temp, color="#CCCCCC", linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.grid(axis="y", alpha=0.3)

    # ────────────────────────────────────────────
    # 中图: 信息熵 + 主导比率 双Y轴
    # ────────────────────────────────────────────
    ax2 = fig.add_subplot(3, 1, 2)
    ax2_twin = ax2.twinx()

    temps = result.temperatures
    entropies = [result.shannon_entropy(t) for t in temps]
    dominants = [result.dominant_ratio(t) for t in temps]

    line1, = ax2.plot(temps, entropies, "o-", color="#E74C3C", linewidth=2.5,
                      markersize=10, label="信息熵(↑越不稳定)", zorder=3)
    line2, = ax2_twin.plot(temps, dominants, "s--", color="#2980B9", linewidth=2,
                           markersize=8, label="主导比率(↓越不稳定)", zorder=3)

    # 标注数值
    for t, e, d in zip(temps, entropies, dominants):
        ax2.annotate(f"{e:.3f}", (t, e), textcoords="offset points",
                     xytext=(0, 12), ha="center", fontsize=9, color="#E74C3C",
                     fontweight="bold")
        ax2_twin.annotate(f"{d:.0%}", (t, d), textcoords="offset points",
                          xytext=(0, -18), ha="center", fontsize=9, color="#2980B9",
                          fontweight="bold")

    # 理论最大熵参考线
    max_entropy = math.log2(len(VALID_TRENDS))
    ax2.axhline(y=max_entropy, color="#E74C3C", linestyle=":", linewidth=1, alpha=0.4)
    ax2.text(temps[-1] + 0.02, max_entropy, f"理论最大熵={max_entropy:.3f}",
             fontsize=8, color="#E74C3C", alpha=0.6, va="bottom")

    ax2.set_xlabel("Temperature", fontsize=11)
    ax2.set_ylabel("信息熵 / Shannon Entropy (bits)", fontsize=11, color="#E74C3C")
    ax2_twin.set_ylabel("主导比率 / Dominant Ratio", fontsize=11, color="#2980B9")
    ax2.set_title("稳定性量化指标 vs Temperature", fontsize=13, fontweight="bold")
    ax2.set_xticks(temps)

    # 合并图例
    ax2.legend(handles=[line1, line2], loc="center left", fontsize=9, framealpha=0.9)
    ax2.grid(alpha=0.3)

    # ────────────────────────────────────────────
    # 下图: 置信度(左) + 趋势强度(右) 并排箱线图
    # ────────────────────────────────────────────
    ax3 = fig.add_subplot(3, 1, 3)
    ax3_twin = ax3.twinx()

    n_temps = len(result.temperatures)
    box_colors = ["#D5F5E3", "#FCF3CF", "#FADBD8", "#F5B7B1"]

    # -- 左侧: 置信度箱线图 --
    conf_data = []
    for temp in result.temperatures:
        confs = [r.confidence for r in result.get_records_by_temp(temp)
                 if r.confidence is not None]
        conf_data.append(confs if confs else [0])

    positions_conf = np.arange(1, n_temps + 1) - 0.18
    bp1 = ax3.boxplot(
        conf_data, positions=positions_conf, widths=0.3, patch_artist=True,
        boxprops=dict(edgecolor="#2980B9"),
        medianprops=dict(color="#E74C3C", linewidth=2),
        whiskerprops=dict(color="#2980B9"),
        capprops=dict(color="#2980B9"),
        flierprops=dict(marker="o", markerfacecolor="#E74C3C", markersize=4),
    )
    for patch, color in zip(bp1["boxes"], box_colors[:n_temps]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    # -- 右侧: 趋势强度箱线图 --
    str_data = []
    for temp in result.temperatures:
        strs = [r.strength for r in result.get_records_by_temp(temp)
                if r.strength is not None]
        str_data.append(strs if strs else [0])

    positions_str = np.arange(1, n_temps + 1) + 0.18
    bp2 = ax3_twin.boxplot(
        str_data, positions=positions_str, widths=0.3, patch_artist=True,
        boxprops=dict(edgecolor="#8E44AD"),
        medianprops=dict(color="#F39C12", linewidth=2),
        whiskerprops=dict(color="#8E44AD"),
        capprops=dict(color="#8E44AD"),
        flierprops=dict(marker="s", markerfacecolor="#F39C12", markersize=4),
    )
    for patch, color in zip(bp2["boxes"], box_colors[:n_temps]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    ax3.set_xticks(range(1, n_temps + 1))
    ax3.set_xticklabels([f"temp={t}" for t in result.temperatures], fontsize=11)
    ax3.set_ylabel("置信度 confidence (●)", fontsize=11, color="#2980B9")
    ax3_twin.set_ylabel("趋势强度 strength (■)", fontsize=11, color="#8E44AD")
    ax3.set_title(
        "各Temperature下AI输出连续值的波动(置信度 vs 趋势强度)",
        fontsize=13, fontweight="bold",
    )
    ax3.set_ylim(-0.05, 1.15)
    ax3_twin.set_ylim(0, 11)
    ax3.grid(axis="y", alpha=0.3)

    # 手动图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#EAF2F8", edgecolor="#2980B9", label="置信度(左轴)"),
        Patch(facecolor="#EAF2F8", edgecolor="#8E44AD", label="趋势强度(右轴)"),
    ]
    ax3.legend(handles=legend_elements, loc="upper left", fontsize=9)

    # ── 底部总结文字 ──
    fig.subplots_adjust(bottom=0.08)

    # 根据实际实验数据动态生成结论
    entropies = [result.shannon_entropy(t) for t in result.temperatures]
    has_variation = any(e > 0 for e in entropies)
    if has_variation and entropies[-1] > entropies[0] + 0.01:
        conclusion = "✅ 验证通过: Temperature↑ → 熵↑ → 输出越不稳定"
    elif not has_variation:
        conclusion = "⚠️ 趋势分类完全一致, 但连续值(置信度/强度)波动随Temperature增大"
    else:
        conclusion = "⚠️ 趋势熵未严格递增, 建议增大样本量"
    summary = (
        f"{conclusion}  |  "
        f"量化建议: 分析类任务用temp=0~0.3, 创意类任务用temp=0.7~1.0"
    )
    fig.text(0.5, 0.01, summary, ha="center", va="center", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#EEF2FF",
                       edgecolor="#5B86E5", alpha=0.95))

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  📁 图表已保存: {save_path}")

    plt.show()


# ═══════════════════════════════════════════════════════════
# 导出原始数据(CSV)
# ═══════════════════════════════════════════════════════════

def export_csv(result: ExperimentResult, path: str = None) -> str:
    """
    导出实验原始记录为CSV, 便于后续二次分析

    返回实际保存路径
    """
    if path is None:
        path = str(Path(__file__).resolve().parent / "04_temp_experiment_results.csv")

    df = result.to_dataframe()
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  📁 原始数据已导出: {path}")
    return path


# ═══════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════

def main():
    """
    完整实验流程:
    1. 执行API调用实验(4×10=40次)
    2. 打印统计分析
    3. 可视化结果
    4. 导出CSV原始数据
    """
    # ── Step 1: 运行实验 ──
    result = run_experiment()

    # ── Step 2: 统计分析 ──
    print_statistics(result)

    # ── Step 3: 可视化 ──
    save_path = str(Path(__file__).resolve().parent / "04_temp_experiment_chart.png")
    plot_results(result, save_path=save_path)

    # ── Step 4: 导出CSV ──
    export_csv(result)

    # ── Step 5: 费用估算 ──
    client = DeepSeekClient(model=MODEL)
    prompt = QuantPrompts.stock_technical_analysis(TEST_STOCK_DATA)
    total_calls = len(TEMPERATURES) * RUNS_PER_TEMP
    cost = client.estimate_cost(prompt, output_tokens=500)
    total_cost = cost["estimated_cost_rmb（预估费用/元）"] * total_calls
    print(f"\n  💰 费用估算:")
    print(f"     单次调用: ≈ ¥{cost['estimated_cost_rmb（预估费用/元）']:.4f}")
    print(f"     总计({total_calls}次): ≈ ¥{total_cost:.4f}")


if __name__ == "__main__":
    main()