# -*- coding: utf-8 -*-
"""
基于AI分析结果的贵州茅台(600519)交易策略回测
策略规则: 收盘价>MA20 且 MA5>MA20时买入; MA5<MA20时卖出
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from dataclasses import dataclass
from typing import Optional
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from quant_core.data.fetcher import fetch_stock


@dataclass
class BacktestConfig:
    symbol: str = "600519"
    years: int = 2
    initial_capital: float = 100000
    ma_short: int = 5
    ma_long: int = 20
    cache_dir: Optional[str] = ".cache/stock"
    cache_ttl_seconds: int = 3600
    min_interval_seconds: float = 1.5
    sources: Optional[list[str]] = None
    trend: str = "上涨趋势"
    signal: str = "BUY"
    support: float = 1499.39
    resistance: float = 1550.0

# ==================== 1. 数据获取 ====================
def fetch_stock_data(config: BacktestConfig):
    """
    获取指定股票的日线数据
    """
    print(f"正在获取{config.symbol}数据...")

    try:
        # 使用项目统一的fetcher获取数据
        df = fetch_stock(
            config.symbol,
            days=365 * config.years,
            adjust="qfq",
            cache_dir=config.cache_dir,
            cache_ttl_seconds=config.cache_ttl_seconds,
            min_interval_seconds=config.min_interval_seconds,
            sources=config.sources,
        ).reset_index()

        # 按日期排序
        df = df.sort_values('date').reset_index(drop=True)
    except Exception as exc:
        raise Exception("数据获取失败，请检查akshare版本或网络连接") from exc

    print(f"数据获取成功! 时间范围: {df['date'].min()} 到 {df['date'].max()}")
    print(f"数据量: {len(df)} 条")

    return df

# ==================== 2. 技术指标计算 ====================
def calculate_technical_indicators(df, ma_short: int = 5, ma_long: int = 20):
    """
    计算技术指标
    """
    df = df.copy()

    # 计算移动平均线
    df[f"MA{ma_short}"] = df['close'].rolling(window=ma_short).mean()
    df[f"MA{ma_long}"] = df['close'].rolling(window=ma_long).mean()

    # 删除NaN值
    df = df.dropna().reset_index(drop=True)

    return df

# ==================== 3. 交易信号生成 ====================
def generate_trading_signals(df, ma_short: int = 5, ma_long: int = 20):
    """
    生成交易信号
    买入条件: 收盘价>MA_long 且 MA_short>MA_long
    卖出条件: MA_short<MA_long
    """
    df = df.copy()

    # 初始化信号列
    df['signal'] = 0  # 0: 无信号, 1: 买入, -1: 卖出

    # 生成买入信号
    ma_short_col = f"MA{ma_short}"
    ma_long_col = f"MA{ma_long}"
    buy_condition = (df['close'] > df[ma_long_col]) & (df[ma_short_col] > df[ma_long_col])
    df.loc[buy_condition, 'signal'] = 1

    # 生成卖出信号
    sell_condition = (df[ma_short_col] < df[ma_long_col])
    df.loc[sell_condition, 'signal'] = -1

    # 避免连续重复信号
    df['position'] = df['signal'].replace(0, np.nan).ffill().fillna(0)

    # 计算实际交易信号（只在仓位变化时产生交易）
    df['trade_signal'] = 0
    df.loc[df['position'] != df['position'].shift(1), 'trade_signal'] = df['position']

    return df

# ==================== 4. 策略回测 ====================
def backtest_strategy(df, initial_capital=100000):
    """
    执行策略回测
    """
    df = df.copy()

    # 初始化资金和持仓
    capital = initial_capital
    position = 0  # 持仓股数
    trades = []  # 记录交易

    # 添加回测结果列
    df['capital'] = float(capital)
    df['position'] = 0.0
    df['returns'] = 0.0
    df['strategy_returns'] = 0.0

    # 执行回测
    for i in range(len(df)):
        current_date = df.loc[i, 'date']
        current_price = df.loc[i, 'close']
        signal = df.loc[i, 'trade_signal']

        # 执行买入信号
        if signal == 1 and position == 0:
            # 全仓买入
            position = capital / current_price
            capital = 0
            trades.append({
                'date': current_date,
                'type': 'BUY',
                'price': current_price,
                'position': position
            })

        # 执行卖出信号
        elif signal == -1 and position > 0:
            # 全仓卖出
            capital = position * current_price
            trades.append({
                'date': current_date,
                'type': 'SELL',
                'price': current_price,
                'position': position
            })
            position = 0

        # 计算每日市值
        if position > 0:
            daily_value = position * current_price
        else:
            daily_value = capital

        # 记录结果
        df.loc[i, 'capital'] = daily_value
        df.loc[i, 'position'] = position

        # 计算收益率
        if i > 0:
            df.loc[i, 'returns'] = (current_price / df.loc[i-1, 'close']) - 1
            df.loc[i, 'strategy_returns'] = (daily_value / df.loc[i-1, 'capital']) - 1

    # 计算策略净值
    df['strategy_net_value'] = (1 + df['strategy_returns']).cumprod() * initial_capital

    # 计算基准净值（买入持有）
    df['benchmark_returns'] = df['returns']
    df['benchmark_net_value'] = (1 + df['benchmark_returns']).cumprod() * initial_capital

    return df, trades

# ==================== 5. 绩效指标计算 ====================
def calculate_performance_metrics(df, initial_capital=100000):
    """
    计算策略绩效指标
    """
    # 基本指标
    total_return = (df['strategy_net_value'].iloc[-1] / initial_capital) - 1

    # 年化收益率
    days = (df['date'].iloc[-1] - df['date'].iloc[0]).days
    years = days / 365.25
    annual_return = (1 + total_return) ** (1 / years) - 1

    # 最大回撤
    df['peak'] = df['strategy_net_value'].cummax()
    df['drawdown'] = (df['strategy_net_value'] - df['peak']) / df['peak']
    max_drawdown = df['drawdown'].min()

    # 夏普比率（假设无风险利率为3%）
    risk_free_rate = 0.03
    excess_returns = df['strategy_returns'] - risk_free_rate/252
    sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0

    # 胜率（如果有交易记录）
    positive_returns = df[df['strategy_returns'] > 0]['strategy_returns']
    win_rate = len(positive_returns) / len(df['strategy_returns'].dropna()) if len(df['strategy_returns'].dropna()) > 0 else 0

    metrics = {
        '累计收益率': f"{total_return:.2%}",
        '年化收益率': f"{annual_return:.2%}",
        '最大回撤': f"{max_drawdown:.2%}",
        '夏普比率': f"{sharpe_ratio:.2f}",
        '胜率': f"{win_rate:.2%}",
        '交易天数': days,
        '最终净值': f"{df['strategy_net_value'].iloc[-1]:.2f}元"
    }

    return metrics

# ==================== 6. 可视化 ====================
def plot_results(df, symbol: str, ma_short: int, ma_long: int):
    """
    绘制策略净值曲线对比图
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # 1. 净值曲线对比
    ax1 = axes[0]
    ax1.plot(df['date'], df['strategy_net_value'], label='策略净值', linewidth=2, color='red')
    ax1.plot(df['date'], df['benchmark_net_value'], label='买入持有净值', linewidth=2, color='blue', alpha=0.7)  
    ax1.set_title(f'{symbol}交易策略净值曲线 vs 买入持有', fontsize=14, fontweight='bold')
    ax1.set_ylabel('净值(元)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 股价和均线
    ax2 = axes[1]
    ax2.plot(df['date'], df['close'], label='收盘价', linewidth=1.5, color='black')
    ax2.plot(df['date'], df[f"MA{ma_short}"], label=f'MA{ma_short}', linewidth=1, color='orange')
    ax2.plot(df['date'], df[f"MA{ma_long}"], label=f'MA{ma_long}', linewidth=1, color='blue')

    # 标记买卖点
    buy_signals = df[df['trade_signal'] == 1]
    sell_signals = df[df['trade_signal'] == -1]

    ax2.scatter(buy_signals['date'], buy_signals['close'],
                color='green', marker='^', s=100, label='买入信号', zorder=5)
    ax2.scatter(sell_signals['date'], sell_signals['close'],
                color='red', marker='v', s=100, label='卖出信号', zorder=5)

    ax2.set_title('股价走势与交易信号', fontsize=14, fontweight='bold')
    ax2.set_ylabel('价格(元)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 回撤曲线
    ax3 = axes[2]
    ax3.fill_between(df['date'], 0, df['drawdown']*100,
                     color='red', alpha=0.3, label='回撤')
    ax3.plot(df['date'], df['drawdown']*100, color='red', linewidth=1)
    ax3.set_title('策略回撤曲线', fontsize=14, fontweight='bold')
    ax3.set_ylabel('回撤(%)')
    ax3.set_xlabel('日期')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# ==================== 7. 打印汇总表格 ====================
def print_summary_table(metrics, df, trades, config: BacktestConfig):
    """
    打印策略汇总表格
    """
    print("\n" + "="*60)
    print("策略回测汇总报告")
    print("="*60)

    # 打印AI分析结果
    print("\n📊 AI分析结果:")
    print(f"  股票: {config.symbol}")
    print(f"  趋势: {config.trend}")
    print(f"  信号: {config.signal}")
    print(f"  支撑位: {config.support}")
    print(f"  压力位: {config.resistance}")

    print("\n📈 策略规则:")
    print(f"  买入条件: 收盘价 > MA{config.ma_long} 且 MA{config.ma_short} > MA{config.ma_long}")
    print(f"  卖出条件: MA{config.ma_short} < MA{config.ma_long}")
    print(f"  初始资金: {config.initial_capital:,.2f}元")
    print("  仓位管理: 全仓操作")

    print("\n💰 绩效指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    print(f"\n📅 回测期间: {df['date'].iloc[0].strftime('%Y-%m-%d')} 到 {df['date'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"  总交易日数: {len(df)} 天")

    print(f"\n🔄 交易统计:")
    print(f"  总交易次数: {len(trades)} 次")
    if len(trades) > 0:
        buy_trades = [t for t in trades if t['type'] == 'BUY']
        sell_trades = [t for t in trades if t['type'] == 'SELL']
        print(f"  买入次数: {len(buy_trades)} 次")
        print(f"  卖出次数: {len(sell_trades)} 次")

    print("\n" + "="*60)

# ==================== 8. 主函数 ====================
def parse_args() -> BacktestConfig:
    import argparse

    parser = argparse.ArgumentParser(description="股票策略回测参数")
    parser.add_argument("--symbol", default="600519", help="股票代码")
    parser.add_argument("--years", type=int, default=2, help="回测年数")
    parser.add_argument("--initial-capital", type=float, default=100000, help="初始资金")
    parser.add_argument("--ma-short", type=int, default=5, help="短期均线窗口")
    parser.add_argument("--ma-long", type=int, default=20, help="长期均线窗口")
    parser.add_argument("--cache-dir", default=".cache/stock", help="缓存目录")
    parser.add_argument("--cache-ttl", type=int, default=3600, help="缓存有效期(秒)")
    parser.add_argument("--min-interval", type=float, default=1.5, help="最小请求间隔(秒)")
    parser.add_argument("--sources", default="eastmoney,tencent", help="数据源优先级(逗号分隔)")
    parser.add_argument("--trend", default="上涨趋势", help="AI趋势描述")
    parser.add_argument("--signal", default="BUY", help="AI信号")
    parser.add_argument("--support", type=float, default=1499.39, help="支撑位")
    parser.add_argument("--resistance", type=float, default=1550.0, help="压力位")

    args = parser.parse_args()
    sources = [item.strip() for item in args.sources.split(",") if item.strip()]

    return BacktestConfig(
        symbol=args.symbol,
        years=args.years,
        initial_capital=args.initial_capital,
        ma_short=args.ma_short,
        ma_long=args.ma_long,
        cache_dir=args.cache_dir,
        cache_ttl_seconds=args.cache_ttl,
        min_interval_seconds=args.min_interval,
        sources=sources,
        trend=args.trend,
        signal=args.signal,
        support=args.support,
        resistance=args.resistance,
    )


def main():
    """
    主函数：执行完整的回测流程
    """
    config = parse_args()

    print(f"开始执行{config.symbol}交易策略回测...")
    print("-" * 60)

    # 1. 获取数据
    df = fetch_stock_data(config)

    # 2. 计算技术指标
    df = calculate_technical_indicators(df, config.ma_short, config.ma_long)

    # 3. 生成交易信号
    df = generate_trading_signals(df, config.ma_short, config.ma_long)

    # 4. 执行回测
    df, trades = backtest_strategy(df, config.initial_capital)

    # 5. 计算绩效指标
    metrics = calculate_performance_metrics(df, config.initial_capital)

    # 6. 可视化结果
    plot_results(df, config.symbol, config.ma_short, config.ma_long)

    # 7. 打印汇总表格
    print_summary_table(metrics, df, trades, config)

    # 8. 显示AI分析的关键价位
    print("\n🎯 AI分析关键价位:")
    print(f"  当前收盘价: {df['close'].iloc[-1]:.2f}")
    print(f"  支撑位: {config.support:.2f} ({'高于' if df['close'].iloc[-1] > config.support else '低于'}当前价)")
    print(f"  压力位: {config.resistance:.2f} ({'高于' if df['close'].iloc[-1] > config.resistance else '低于'}当前价)")

    # 9. 显示最新信号
    latest_signal = df['trade_signal'].iloc[-1]
    signal_text = "买入" if latest_signal == 1 else "卖出" if latest_signal == -1 else "持有"
    print(f"\n📢 最新交易信号: {signal_text}")

    print("\n回测完成!")


# ==================== 执行程序 ====================
if __name__ == "__main__":
    main()