import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from quant_core.data.fetcher import fetch_stock

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def get_stock_data(years=5):
    """
    获取贵州茅台最近5年的日线数据

    Returns:
        pd.DataFrame: 包含日期、开盘价、收盘价等数据的DataFrame
    """
    print("正在获取贵州茅台股票数据...")

    try:
        # 使用项目统一的fetcher获取数据
        df = fetch_stock("600519", days=365 * years, adjust="qfq").reset_index()

        # 按日期排序
        df = df.sort_values('date').reset_index(drop=True)

        print(f"数据获取成功！共获取{len(df)}个交易日数据")
        print(f"数据时间范围：{df['date'].min()} 到 {df['date'].max()}")

        return df

    except Exception as e:
        print(f"数据获取失败: {e}")
        return None

def calculate_ma_signals(df):
    """
    计算双均线策略的买卖信号

    Args:
        df (pd.DataFrame): 股票数据

    Returns:
        pd.DataFrame: 添加了均线和信号的DataFrame
    """
    # 计算5日和20日移动平均线
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()

    # 计算均线交叉信号
    # 当MA5上穿MA20时，产生买入信号（金叉）
    # 当MA5下穿MA20时，产生卖出信号（死叉）

    # 初始化信号列
    df['信号'] = 0  # 0表示无信号，1表示买入，-1表示卖出

    # 计算金叉（买入信号）：MA5从下方上穿MA20
    df['金叉'] = (df['MA5'] > df['MA20']) & (df['MA5'].shift(1) <= df['MA20'].shift(1))

    # 计算死叉（卖出信号）：MA5从上方下穿MA20
    df['死叉'] = (df['MA5'] < df['MA20']) & (df['MA5'].shift(1) >= df['MA20'].shift(1))

    # 将信号转换为数值
    df.loc[df['金叉'], '信号'] = 1
    df.loc[df['死叉'], '信号'] = -1

    # 删除前20行（因为MA20需要20个数据点）
    df = df.dropna().reset_index(drop=True)

    return df

def backtest_strategy(df, initial_capital=1000000):
    """
    执行双均线策略回测

    Args:
        df (pd.DataFrame): 包含信号的股票数据
        initial_capital (float): 初始资金

    Returns:
        tuple: (回测结果DataFrame, 最终净值, 收益率)
    """
    # 初始化回测参数
    capital = initial_capital  # 当前资金
    position = 0  # 持仓数量
    hold_value = 0  # 持仓市值

    # 创建回测结果DataFrame
    backtest_df = df.copy()
    backtest_df['持仓数量'] = 0
    backtest_df['持仓市值'] = 0.0
    backtest_df['可用资金'] = 0.0
    backtest_df['总资产'] = 0.0
    backtest_df['净值'] = 1.0  # 初始净值为1

    # 交易记录
    trades = []

    # 持仓状态：0表示空仓，1表示持仓
    holding = 0

    # 遍历每个交易日进行回测
    for i in range(len(backtest_df)):
        current_price = backtest_df.loc[i, 'close']
        signal = backtest_df.loc[i, '信号']

        # 买入信号且当前空仓
        if signal == 1 and holding == 0:
            # 计算可买入数量（全仓买入）
            position = int(capital / current_price)
            if position > 0:
                # 更新资金和持仓
                capital -= position * current_price
                hold_value = position * current_price
                holding = 1

                # 记录交易
                trades.append({
                    '日期': backtest_df.loc[i, 'date'],
                    '类型': '买入',
                    '价格': current_price,
                    '数量': position,
                    '金额': position * current_price
                })

        # 卖出信号且当前持仓
        elif signal == -1 and holding == 1:
            # 卖出所有持仓
            capital += position * current_price
            hold_value = 0
            position = 0
            holding = 0

            # 记录交易
            trades.append({
                '日期': backtest_df.loc[i, 'date'],
                '类型': '卖出',
                '价格': current_price,
                '数量': position,
                '金额': position * current_price
            })

        # 更新每日数据
        backtest_df.loc[i, '持仓数量'] = position
        backtest_df.loc[i, '持仓市值'] = position * current_price
        backtest_df.loc[i, '可用资金'] = capital
        backtest_df.loc[i, '总资产'] = capital + position * current_price

        # 计算净值（相对于初始资金）
        backtest_df.loc[i, '净值'] = backtest_df.loc[i, '总资产'] / initial_capital

    # 计算最终结果
    final_value = backtest_df.loc[len(backtest_df)-1, '总资产']
    total_return = (final_value - initial_capital) / initial_capital * 100

    # 如果有持仓，最后一天卖出
    if holding == 1:
        final_price = backtest_df.loc[len(backtest_df)-1, 'close']
        final_value = capital + position * final_price
        total_return = (final_value - initial_capital) / initial_capital * 100

    return backtest_df, final_value, total_return, trades

def buy_and_hold_strategy(df, initial_capital=100000):
    """
    执行买入持有策略回测

    Args:
        df (pd.DataFrame): 股票数据
        initial_capital (float): 初始资金

    Returns:
        tuple: (回测结果DataFrame, 最终净值, 收益率)
    """
    # 创建回测结果DataFrame
    bh_df = df.copy()

    # 第一天全仓买入
    first_price = bh_df.loc[0, 'close']
    position = int(initial_capital / first_price)
    capital = initial_capital - position * first_price

    # 计算每日总资产和净值
    bh_df['持仓数量'] = position
    bh_df['持仓市值'] = bh_df['close'] * position
    bh_df['可用资金'] = capital
    bh_df['总资产'] = bh_df['持仓市值'] + capital
    bh_df['净值'] = bh_df['总资产'] / initial_capital

    # 计算最终结果
    final_value = bh_df.loc[len(bh_df)-1, '总资产']
    total_return = (final_value - initial_capital) / initial_capital * 100

    return bh_df, final_value, total_return

def plot_results(ma_df, bh_df, trades):
    """
    绘制回测结果图表

    Args:
        ma_df (pd.DataFrame): 双均线策略回测结果
        bh_df (pd.DataFrame): 买入持有策略回测结果
        trades (list): 交易记录
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # 1. 价格和均线图
    ax1 = axes[0]
    ax1.plot(ma_df['date'], ma_df['close'], label='收盘价', linewidth=1, alpha=0.7)
    ax1.plot(ma_df['date'], ma_df['MA5'], label='5日均线', linewidth=1.5, alpha=0.8)
    ax1.plot(ma_df['date'], ma_df['MA20'], label='20日均线', linewidth=1.5, alpha=0.8)

    # 标记买卖点
    buy_signals = ma_df[ma_df['信号'] == 1]
    sell_signals = ma_df[ma_df['信号'] == -1]

    ax1.scatter(buy_signals['date'], buy_signals['close'],
                color='red', marker='^', s=100, label='买入信号', zorder=5)
    ax1.scatter(sell_signals['date'], sell_signals['close'],
                color='green', marker='v', s=100, label='卖出信号', zorder=5)

    ax1.set_title('贵州茅台价格与双均线策略信号', fontsize=14)
    ax1.set_ylabel('价格(元)', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # 2. 净值曲线对比
    ax2 = axes[1]
    ax2.plot(ma_df['date'], ma_df['净值'], label='双均线策略', linewidth=2, color='blue')
    ax2.plot(bh_df['date'], bh_df['净值'], label='买入持有策略', linewidth=2, color='orange', alpha=0.8)
    ax2.set_title('策略净值曲线对比', fontsize=14)
    ax2.set_ylabel('净值', fontsize=12)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    # 3. 资产构成图
    ax3 = axes[2]
    width = 0.35
    x = np.arange(len(ma_df))

    # 选择部分日期显示，避免过于密集
    step = max(1, len(ma_df) // 20)
    display_indices = range(0, len(ma_df), step)

    ax3.bar([ma_df.loc[i, 'date'] for i in display_indices],
            [ma_df.loc[i, '持仓市值'] for i in display_indices],
            width, label='持仓市值', alpha=0.7)
    ax3.bar([ma_df.loc[i, 'date'] for i in display_indices],
            [ma_df.loc[i, '可用资金'] for i in display_indices],
            width, bottom=[ma_df.loc[i, '持仓市值'] for i in display_indices],
            label='可用资金', alpha=0.7)

    ax3.set_title('双均线策略资产构成', fontsize=14)
    ax3.set_ylabel('资产(元)', fontsize=12)
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 打印交易统计
    if trades:
        print("\n交易记录统计:")
        print(f"总交易次数: {len(trades)}")

        buy_trades = [t for t in trades if t['类型'] == '买入']
        sell_trades = [t for t in trades if t['类型'] == '卖出']

        print(f"买入次数: {len(buy_trades)}")
        print(f"卖出次数: {len(sell_trades)}")

        if buy_trades:
            print("\n买入交易记录:")
            for trade in buy_trades[-5:]:  # 显示最后5次买入
                print(f"日期: {trade['日期'].date()}, 价格: {trade['价格']:.2f}, 数量: {trade['数量']}")

        if sell_trades:
            print("\n卖出交易记录:")
            for trade in sell_trades[-5:]:  # 显示最后5次卖出
                print(f"日期: {trade['日期'].date()}, 价格: {trade['价格']:.2f}, 数量: {trade['数量']}")

def main():
    """
    主函数：执行完整的双均线策略回测
    """
    print("=" * 60)
    print("双均线策略回测系统")
    print("策略规则：5日均线上穿20日均线买入，下穿卖出")
    print("=" * 60)

    years = 10
    # 1. 获取数据
    df = get_stock_data(years)
    if df is None:
        print("无法获取数据，程序退出")
        return

    # 2. 计算均线和信号
    df_with_signals = calculate_ma_signals(df)

    # 3. 执行双均线策略回测
    print("\n执行双均线策略回测...")
    initial_capital = 10000000
    ma_backtest_df, ma_final_value, ma_return, trades = backtest_strategy(df_with_signals, initial_capital)       

    # 4. 执行买入持有策略回测
    print("执行买入持有策略回测...")
    bh_backtest_df, bh_final_value, bh_return = buy_and_hold_strategy(df_with_signals, initial_capital)

    # 5. 打印回测结果
    print("\n" + "=" * 60)
    print("回测结果对比")
    print("=" * 60)
    print(f"回测期间: {df_with_signals['date'].min().date()} 到 {df_with_signals['date'].max().date()}")
    print(f"初始资金: {initial_capital:,.2f}元")
    print("\n双均线策略:")
    print(f"  最终资产: {ma_final_value:,.2f}元")
    print(f"  总收益率: {ma_return:.2f}%")
    print(f"  年化收益率: {ma_return/years:.2f}% (按{years}年计算)")

    print("\n买入持有策略:")
    print(f"  最终资产: {bh_final_value:,.2f}元")
    print(f"  总收益率: {bh_return:.2f}%")
    print(f"  年化收益率: {bh_return/years:.2f}% (按{years}年计算)")

    print("\n策略对比:")
    if ma_return > bh_return:
        print(f"  双均线策略跑赢买入持有策略: {ma_return - bh_return:.2f}%")
    else:
        print(f"  双均线策略跑输买入持有策略: {bh_return - ma_return:.2f}%")

    # 6. 绘制图表
    print("\n生成回测图表...")
    plot_results(ma_backtest_df, bh_backtest_df, trades)

    # 7. 显示详细数据
    print("\n最近10个交易日的策略表现:")
    recent_data = ma_backtest_df[['date', 'close', 'MA5', 'MA20', '信号', '净值']].tail(10)
    print(recent_data.to_string(index=False))

    print("\n回测完成！")

if __name__ == "__main__":
    # 执行主函数
    main()