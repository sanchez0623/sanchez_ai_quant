# -*- coding: utf-8 -*-
"""
第4周-Part3: 全天候(All Weather)资产配置
=======================================
桥水基金达利奥的经典策略, 用Python模拟验证

经典全天候配比:
  股票    30%  -- 经济增长时赚钱
  长期国债 40%  -- 经济衰退/降息时赚钱
  中期国债 15%  -- 稳定收益的基石
  黄金    7.5% -- 通胀/地缘风险时赚钱
  大宗商品 7.5% -- 通胀上升时赚钱

核心逻辑:
  经济只有4种状态: 增长/衰退 x 通胀/通缩
  全天候在每种状态下都配了"能赚钱的资产"
  所以无论未来怎样, 组合都不会太差
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def _pick_gold_date_column(df: pd.DataFrame) -> str:
    """自动识别黄金日期列。"""
    candidates = ["日期", "交易时间", "交易日期", "时间", "date", "datetime"]
    for col in df.columns:
        if str(col).strip() in candidates:
            return col

    best_col = None
    best_valid = 0
    for col in df.columns:
        parsed = pd.to_datetime(df[col], errors="coerce")
        valid = parsed.notna().sum()
        if valid > best_valid:
            best_valid = valid
            best_col = col

    if best_col is None or best_valid == 0:
        raise ValueError(f"未识别到可用日期列，当前列: {list(df.columns)}")
    return best_col


def _pick_gold_price_column(df: pd.DataFrame, date_col: str = None) -> str:
    """优先按常见列名选择黄金价格列，失败则回退到第一个可转数值列。"""
    preferred = [
        "晚盘价", "收盘", "收盘价", "今收盘", "价格", "金价", "AU9999", "AU99.99", "基准价"
    ]
    for col in df.columns:
        col_name = str(col).strip()
        if any(key in col_name for key in preferred):
            return col

    for col in df.columns:
        if date_col is not None and col == date_col:
            continue
        if str(col).strip() in {"日期", "交易时间", "交易日期", "时间", "date", "datetime"}:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().sum() > 0:
            return col

    raise ValueError("未识别到可用黄金价格列")


def _fetch_gold_prices(start_date: str) -> pd.Series:
    """
    获取黄金价格，支持多重兜底：
    1) SGE现货基准(首选)
    2) 黄金ETF(518880)近似
    """
    try:
        try:
            df = ak.spot_golden_benchmark_sge(
                start_date=start_date.replace("-", ""),
                end_date="20261231",
            )
        except TypeError:
            df = ak.spot_golden_benchmark_sge()

        if df is None or df.empty:
            raise ValueError("SGE接口返回空数据")

        df.columns = [str(c).strip() for c in df.columns]
        date_col = _pick_gold_date_column(df)
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").set_index("date")

        price_col = _pick_gold_price_column(df, date_col=date_col)
        series = pd.to_numeric(df[price_col], errors="coerce").dropna()
        series = series[series.index >= pd.to_datetime(start_date)]
        if series.empty:
            raise ValueError("SGE过滤后无有效数据")
        print(
            f"         黄金数据源: SGE现货基准 "
            f"(日期列={date_col}, 价格列={price_col}, 样本={len(series)})"
        )
        return series
    except Exception as sge_error:
        try:
            # 兜底：黄金ETF（华安黄金ETF，518880）
            etf = ak.fund_etf_hist_sina(symbol="sh518880")
            if etf is None or etf.empty:
                raise ValueError("黄金ETF返回空数据")
            etf["date"] = pd.to_datetime(etf["date"], errors="coerce")
            etf = etf.dropna(subset=["date"]).sort_values("date").set_index("date")
            series = pd.to_numeric(etf["close"], errors="coerce").dropna()
            series = series[series.index >= pd.to_datetime(start_date)]
            if series.empty:
                raise ValueError("黄金ETF过滤后无有效数据")
            print(
                f"         黄金数据源: sh518880(黄金ETF兜底) "
                f"(价格列=close, 样本={len(series)}, 触发原因: {sge_error})"
            )
            return series
        except Exception as etf_error:
            raise ValueError(f"SGE失败: {sge_error}; 黄金ETF失败: {etf_error}")


# ================================================================
# 数据获取层: 支持多类资产, 失败自动跳过
# ================================================================

# 资产注册表: 新增资产只需在这里加一行
# key = 代码标识, name = 中文名, fetch_func = 获取函数名, category = 类别
ASSET_REGISTRY = {
    "hs300": {
        "name": "沪深300(股票)",
        "category": "股票",
        "description": "A股最大300只股票, 代表中国股市整体",
    },
    "csi500": {
        "name": "中证500(中小盘)",
        "category": "股票",
        "description": "中小盘股票指数, 与沪深300互补",
    },
    "bond_national": {
        "name": "国债指数(债券)",
        "category": "债券",
        "description": "中国国债, 经济衰退/降息时表现好",
    },
    "bond_enterprise": {
        "name": "企业债指数(债券)",
        "category": "债券",
        "description": "企业债, 收益高于国债但有违约风险",
    },
    "gold": {
        "name": "黄金",
        "category": "商品",
        "description": "避险资产, 通胀和地缘风险时表现好",
    },
    "commodity": {
        "name": "大宗商品",
        "category": "商品",
        "description": "石油/铜/农产品等, 通胀上升时表现好",
    },
    "money_fund": {
        "name": "货币基金(现金)",
        "category": "现金",
        "description": "类似余额宝, 几乎无风险, 代表持有现金的收益",
    },
}


def fetch_all_assets(start_date="20180101"):
    """
    批量获取多类资产数据, 获取失败的自动跳过并提示

    返回:
        prices_df: DataFrame, 每列为一类资产的每日价格
        available:  成功获取的资产列表
        failed:     获取失败的资产列表
    """
    prices = {}
    available = []
    failed = []

    # --- 1. 沪深300 ---
    print("  [1/7] 获取沪深300(股票)...")
    try:
        df = ak.stock_zh_index_daily(symbol="sh000300")
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        prices["hs300"] = df["close"]
        available.append("hs300")
        print(f"         成功, {len(df)} 条")
    except Exception as e:
        failed.append(("hs300", str(e)))
        print(f"         失败: {e}")

    # --- 2. 中证500 ---
    print("  [2/7] 获取中证500(中小盘)...")
    try:
        df = ak.stock_zh_index_daily(symbol="sh000905")
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        prices["csi500"] = df["close"]
        available.append("csi500")
        print(f"         成功, {len(df)} 条")
    except Exception as e:
        failed.append(("csi500", str(e)))
        print(f"         失败: {e}")

    # --- 3. 国债指数 ---
    print("  [3/7] 获取国债指数(债券)...")
    try:
        df = ak.bond_zh_us_rate(start_date="20180101")
        df = df[["日期", "中国国债收益率10年"]].dropna()
        df.columns = ["date", "rate"]
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
        # 国债收益率转换为"模拟债券价格"
        # 简化模型: 久期约8年, 收益率每变动1bp, 价格反向变动约0.08%
        # 以第一天为基准100
        df["rate_change"] = df["rate"].diff()
        df["bond_return"] = -df["rate_change"] * 0.08 / 100  # 久期效应
        df["bond_price"] = 100 * (1 + df["bond_return"].fillna(0)).cumprod()
        prices["bond_national"] = df["bond_price"]
        available.append("bond_national")
        print(f"         成功, {len(df)} 条 (由收益率模拟债券价格)")
    except Exception as e:
        failed.append(("bond_national", str(e)))
        print(f"         失败: {e}")

    # --- 4. 企业债指数 ---
    print("  [4/7] 获取企业债指数...")
    try:
        df = ak.bond_zh_us_rate(start_date="20180101")
        if "中国国债收益率2年" in df.columns:
            df = df[["日期", "中国国债收益率2年"]].dropna()
            df.columns = ["date", "rate"]
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
            # 企业债 = 国债 + 信用利差, 这里用短期国债近似
            df["rate_change"] = df["rate"].diff()
            df["bond_return"] = -df["rate_change"] * 0.03 / 100  # 短久期
            df["bond_price"] = 100 * (1 + df["bond_return"].fillna(0)).cumprod()
            prices["bond_enterprise"] = df["bond_price"]
            available.append("bond_enterprise")
            print(f"         成功, {len(df)} 条 (短期国债近似)")
        else:
            raise ValueError("缺少2年国债收益率列")
    except Exception as e:
        failed.append(("bond_enterprise", str(e)))
        print(f"         失败: {e}")

    # --- 5. 黄金 ---
    print("  [5/7] 获取黄金...")
    try:
        prices["gold"] = _fetch_gold_prices(start_date)
        available.append("gold")
        print(f"         成功, {len(prices['gold'])} 条")
    except Exception as e:
        failed.append(("gold", str(e)))
        print(f"         失败: {e}")

    # --- 6. 大宗商品(南华商品指数) ---
    print("  [6/7] 获取大宗商品指数...")
    try:
        df = ak.futures_index_symbol_table_nh()
        # 尝试南华商品指数
        df2 = ak.futures_nh_index_all()
        if "南华商品指数" in df2.columns:
            prices["commodity"] = pd.to_numeric(df2["南华商品指数"], errors="coerce")
            available.append("commodity")
            print(f"         成功")
        else:
            raise ValueError("未找到南华商品指数")
    except Exception as e:
        # 备选: 用螺纹钢主力合约近似
        try:
            df = ak.futures_main_sina(symbol="RB0", start_date=start_date, end_date="20261231")
            df["date"] = pd.to_datetime(df["日期"])
            df = df.set_index("date")
            prices["commodity"] = pd.to_numeric(df["收盘价"], errors="coerce")
            available.append("commodity")
            print(f"         成功 (螺纹钢主力合约近似)")
        except Exception as e2:
            failed.append(("commodity", f"{e} / {e2}"))
            print(f"         失败: {e}")

    # --- 7. 货币基金(模拟) ---
    print("  [7/7] 生成货币基金(现金)数据...")
    try:
        # 货币基金年化约2%, 用固定日收益率模拟
        if prices:
            first_key = list(prices.keys())[0]
            dates = prices[first_key].dropna().index
            daily_rate = (1 + 0.02) ** (1 / 252) - 1  # 年化2%转日收益
            money_prices = 100 * (1 + daily_rate) ** np.arange(len(dates))
            prices["money_fund"] = pd.Series(money_prices, index=dates)
            available.append("money_fund")
            print(f"         成功 (年化2%模拟)")
    except Exception as e:
        failed.append(("money_fund", str(e)))
        print(f"         失败: {e}")

    # 汇总
    print(f"\n  数据汇总: 成功 {len(available)}/{len(ASSET_REGISTRY)} 类资产")
    if failed:
        print(f"  获取失败的资产:")
        for name, err in failed:
            print(f"    - {ASSET_REGISTRY[name]['name']}: {err}")

    # 合并成DataFrame
    prices_df = pd.DataFrame(prices)
    prices_df = prices_df[prices_df.index >= start_date]
    prices_df = prices_df.ffill().dropna()

    return prices_df, available, failed


# ================================================================
# 组合模拟引擎
# ================================================================

def simulate_portfolio(
    prices_df: pd.DataFrame,
    weights: dict,
    portfolio_name: str = "My Portfolio",
    initial_capital: float = 100_000,
    rebalance_freq: str = "QE",
):
    """
    模拟资产配置组合的历史表现

    参数:
        prices_df:       各资产价格DataFrame(由fetch_all_assets返回)
        weights:         配置比例, 如 {"hs300": 0.3, "bond_national": 0.4, ...}
        portfolio_name:  组合名称
        initial_capital: 初始资金(元)
        rebalance_freq:  再平衡频率 "Q"=季度 / "M"=月度 / "Y"=年度

    术语:
      再平衡(Rebalance): 定期把组合比例调回目标比例.
        例: 目标是股票30%债券40%. 股票涨了变成40%:30%,
        就卖掉一些股票买入债券, 恢复30:40.
        作用: 强制"高抛低吸", 控制风险.

    返回:
        DataFrame, 包含每日净值、收益率、回撤
    """
    # 只保留有数据的资产
    valid_assets = [a for a in weights if a in prices_df.columns]
    if not valid_assets:
        print(f"  [{portfolio_name}] 没有可用的资产数据, 跳过")
        return None

    # 归一化权重
    valid_weights = {a: weights[a] for a in valid_assets}
    total_w = sum(valid_weights.values())
    valid_weights = {a: w / total_w for a, w in valid_weights.items()}

    if len(valid_assets) < len(weights):
        missing = [a for a in weights if a not in valid_assets]
        print(f"  [{portfolio_name}] 部分资产缺失{missing}, 已自动重新分配权重")

    # 计算日收益率
    returns_df = prices_df[valid_assets].pct_change().fillna(0)

    # 兼容新版本pandas频率别名
    freq_alias = {
        "Q": "QE", "QE": "QE",
        "M": "ME", "ME": "ME",
        "Y": "YE", "A": "YE", "YE": "YE",
    }
    normalized_freq = freq_alias.get(rebalance_freq.upper(), rebalance_freq)
    if normalized_freq != rebalance_freq:
        print(f"  [{portfolio_name}] 再平衡频率 {rebalance_freq} 已映射为 {normalized_freq}")

    # 标记再平衡日期
    rebalance_dates = set(returns_df.resample(normalized_freq).last().index)

    # 模拟
    nav = [initial_capital]
    current_w = valid_weights.copy()

    for i in range(1, len(returns_df)):
        date = returns_df.index[i]

        # 今日组合收益 = 各资产收益的加权和
        daily_ret = sum(
            current_w[a] * returns_df.iloc[i][a] for a in current_w
        )
        nav.append(nav[-1] * (1 + daily_ret))

        # 再平衡日: 恢复目标权重
        if date in rebalance_dates:
            current_w = valid_weights.copy()

    result = pd.DataFrame({
        "nav": nav,
    }, index=returns_df.index)

    result["daily_return"] = result["nav"].pct_change()
    result["cumulative_return"] = result["nav"] / initial_capital - 1
    result["drawdown"] = result["nav"] / result["nav"].cummax() - 1

    return result


# ================================================================
# 组合对比分析
# ================================================================

def compare_portfolios(
    portfolios: dict,
    start_date: str = "20180101",
    initial_capital: float = 1_000_000,
    risk_free_rate: float = 0.025,
    rebalance_freq: str = "QE",
):
    """
    对比多个资产配置方案

    参数:
        portfolios: {
            "方案名": {"asset_key": weight, ...},
            ...
        }
        start_date:      起始日期
        initial_capital:  初始资金
        risk_free_rate:   无风险利率(年化), 用于计算夏普比率
        rebalance_freq:   再平衡频率

    术语:
      夏普比率(Sharpe Ratio): (年化收益-无风险利率)/年化波动率
        衡量每承担1单位风险获得多少超额收益. >1优秀
      卡尔玛比率(Calmar Ratio): 年化收益/|最大回撤|
        衡量收益能否补偿最大亏损. >1较好
    """
    print("=" * 60)
    print("  全天候资产配置 -- 多方案对比")
    print("=" * 60)

    # 统一获取一次数据(避免重复请求)
    print("\n  [Step 1] 获取所有资产数据...")
    prices_df, available, failed = fetch_all_assets(start_date)

    if prices_df.empty:
        print("  数据不足, 无法模拟")
        return

    # 打印可用资产
    print(f"\n  可用资产({len(available)}类):")
    for a in available:
        info = ASSET_REGISTRY.get(a, {})
        print(f"    {info.get('name', a):20s} [{info.get('category', '?')}] {info.get('description', '')}")

    # 模拟各方案
    print(f"\n  [Step 2] 模拟各配置方案...")
    results = {}
    stats_list = []

    for name, weights in portfolios.items():
        print(f"\n  --- {name} ---")
        weight_str = " + ".join([
            f"{ASSET_REGISTRY.get(k, {}).get('name', k)}:{v:.0%}"
            for k, v in weights.items()
        ])
        print(f"  配比: {weight_str}")

        nav_df = simulate_portfolio(
            prices_df=prices_df,
            weights=weights,
            portfolio_name=name,
            initial_capital=initial_capital,
            rebalance_freq=rebalance_freq,
        )

        if nav_df is None:
            continue

        results[name] = nav_df

        # 绩效指标
        daily_ret = nav_df["daily_return"].dropna()
        trading_days = len(daily_ret)
        cum_ret = nav_df["cumulative_return"].iloc[-1]
        ann_ret = (1 + cum_ret) ** (252 / trading_days) - 1
        ann_vol = daily_ret.std() * np.sqrt(252)
        sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else 0
        max_dd = nav_df["drawdown"].min()
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

        stats_list.append({
            "方案": name,
            "累计收益率": f"{cum_ret:.2%}",
            "年化收益率": f"{ann_ret:.2%}",
            "年化波动率(风险)": f"{ann_vol:.2%}",
            "夏普比率": f"{sharpe:.2f}",
            "最大回撤": f"{max_dd:.2%}",
            "卡尔玛比率": f"{calmar:.2f}",
            "终值(元)": f"{nav_df['nav'].iloc[-1]:,.0f}",
        })

    if not stats_list:
        print("  所有方案模拟失败")
        return

    # 绩效表
    stats_df = pd.DataFrame(stats_list)
    print(f"\n  [Step 3] 绩效对比:")
    print(f"  分析区间: {prices_df.index[0].strftime('%Y-%m-%d')} ~ {prices_df.index[-1].strftime('%Y-%m-%d')}")
    print(f"  初始资金: {initial_capital:,.0f} 元")
    print(f"  再平衡频率: {rebalance_freq}")
    print(f"  无风险利率: {risk_free_rate:.2%}\n")
    print(stats_df.to_string(index=False))

    # ---- 可视化 ----
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFD93D", "#96CEB4", "#DDA0DD"]

    # 左上: 净值走势
    ax = axes[0, 0]
    for i, (name, nav_df) in enumerate(results.items()):
        ax.plot(nav_df.index, nav_df["nav"] / initial_capital,
                label=name, linewidth=1.5, color=colors[i % len(colors)])
    ax.set_title("净值走势对比(1.0=本金)", fontsize=13, fontweight="bold")
    ax.set_ylabel("净值倍数")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color="black", linewidth=0.5, linestyle="--")

    # 右上: 回撤走势
    ax = axes[0, 1]
    for i, (name, nav_df) in enumerate(results.items()):
        ax.fill_between(nav_df.index, nav_df["drawdown"] * 100,
                        alpha=0.15, color=colors[i % len(colors)])
        ax.plot(nav_df.index, nav_df["drawdown"] * 100,
                label=name, linewidth=1, color=colors[i % len(colors)])
    ax.set_title("回撤走势(越接近0越好)", fontsize=13, fontweight="bold")
    ax.set_ylabel("回撤(%)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 左下: 风险收益散点图
    ax = axes[1, 0]
    for i, row in pd.DataFrame(stats_list).iterrows():
        ann_ret_val = float(row["年化收益率"].strip("%")) / 100
        ann_vol_val = float(row["年化波动率(风险)"].strip("%")) / 100
        ax.scatter(ann_vol_val * 100, ann_ret_val * 100,
                   s=200, color=colors[i % len(colors)], zorder=5, edgecolors="black")
        ax.annotate(row["方案"], xy=(ann_vol_val * 100, ann_ret_val * 100),
                    xytext=(8, 8), textcoords="offset points", fontsize=9)
    ax.set_xlabel("年化波动率(风险) %")
    ax.set_ylabel("年化收益率 %")
    ax.set_title("风险-收益散点图(左上方最佳)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # 右下: 各方案配比饼图
    ax = axes[1, 1]
    # 取第一个全天候方案做饼图展示
    aw_key = None
    for k, v in portfolios.items():
        if len(v) >= 3:
            aw_key = k
            break
    if aw_key:
        w = portfolios[aw_key]
        labels = [ASSET_REGISTRY.get(k, {}).get("name", k) for k in w]
        sizes = list(w.values())
        pie_colors = colors[:len(sizes)]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct="%1.0f%%",
            colors=pie_colors, startangle=90, textprops={"fontsize": 10},
        )
        ax.set_title(f"[{aw_key}] 配置比例", fontsize=13, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "无多资产方案", ha="center", va="center", fontsize=14)
        ax.set_title("配置比例", fontsize=13, fontweight="bold")

    plt.suptitle("全天候资产配置 -- 多方案对比", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig("all_weather_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n  图表已保存: all_weather_comparison.png")

    return stats_df, results


# ================================================================
# 执行对比 -- 修改以下参数即可测试不同方案!
# ================================================================

if __name__ == "__main__":

    compare_portfolios(
        portfolios={
            # 方案1: 全仓股票(激进型, 对照组)
            "全仓沪深300": {
                "hs300": 1.0,
            },

            # 方案2: 经典60/40
            # 术语: 60/40组合是最经典的资产配置方案
            #   60%股票追求增长, 40%债券控制波动
            "经典60/40": {
                "hs300": 0.60,
                "bond_national": 0.40,
            },

            # 方案3: 经典全天候(达利奥版)
            # 股票30% + 长债40% + 中债15% + 黄金7.5% + 商品7.5%
            "经典全天候": {
                "hs300": 0.30,
                "bond_national": 0.40,
                "bond_enterprise": 0.15,
                "gold": 0.075,
                "commodity": 0.075,
            },

            # 方案4: 中国版全天候(适配A股市场)
            # A股波动大, 适当降低股票比例, 增加黄金和现金
            "中国版全天候": {
                "hs300": 0.20,
                "csi500": 0.10,
                "bond_national": 0.30,
                "bond_enterprise": 0.10,
                "gold": 0.15,
                "commodity": 0.05,
                "money_fund": 0.10,
            },

            # 方案5: 保守型(退休/低风险偏好)
            "保守稳健型": {
                "bond_national": 0.40,
                "bond_enterprise": 0.20,
                "gold": 0.10,
                "hs300": 0.15,
                "money_fund": 0.15,
            },
        },
        start_date="20180101",       # 起始日期, 可改
        initial_capital=1_000_000,     # 初始资金, 可改
        risk_free_rate=0.025,        # 无风险利率, 可改
        rebalance_freq="QE",         # 再平衡频率: QE=季度, ME=月度, YE=年度
    )