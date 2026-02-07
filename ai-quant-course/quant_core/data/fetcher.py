"""
数据获取模块
============
所有数据获取的函数集中在这里，全课程统一调用
"""

import akshare as ak
import os
import pandas as pd
from contextlib import contextmanager
from datetime import datetime, timedelta


@contextmanager
def _temporary_disable_proxies():
    # Temporarily disable common proxy environment variables for requests.
    proxy_keys = [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "NO_PROXY",
        "no_proxy",
    ]
    backup = {key: os.environ.get(key) for key in proxy_keys}
    try:
        for key in proxy_keys:
            if key in ("NO_PROXY", "no_proxy"):
                os.environ[key] = "*"
            else:
                os.environ.pop(key, None)
        yield
    finally:
        for key, value in backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def fetch_stock(
    code: str,
    days: int = 365,
    adjust: str = "qfq",
    use_proxy: bool = False,
) -> pd.DataFrame:
    """
    获取A股日线数据（标准化版本）

    参数：
        code:   股票代码，如 "600519"
        days:   获取最近多少天的数据
        adjust: 复权方式，"qfq"=前复权, "hfq"=后复权, ""=不复权

    返回：
        DataFrame，列名统一英文，日期为索引

    使用示例：
        from quant_core.data import fetch_stock
        df = fetch_stock("600519", days=365)
    """
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")

    if use_proxy:
        df = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )
    else:
        with _temporary_disable_proxies():
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
            )

    df = df.rename(columns={
        "日期": "date", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low", "成交量": "volume",
        "成交额": "amount", "振幅": "amplitude",
        "涨跌幅": "change_pct", "涨跌额": "change_amt",
        "换手率": "turnover",
    })

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    return df