"""
数据获取模块
============
所有数据获取的函数集中在这里，全课程统一调用
"""

import akshare as ak
import os
import time
import random
import pandas as pd
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Optional


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
    sources: Optional[list[str]] = None,
    cache_dir: Optional[str] = None,
    cache_ttl_seconds: int = 3600,
    min_interval_seconds: float = 1.0,
    max_retries: int = 3,
    backoff_base_seconds: float = 0.8,
) -> pd.DataFrame:
    """
    获取A股日线数据（标准化版本）

    参数：
        code:   股票代码，如 "600519"
        days:   获取最近多少天的数据
        adjust: 复权方式，"qfq"=前复权, "hfq"=后复权, ""=不复权
        sources: 数据源优先级列表，如 ["eastmoney", "tencent"]

    返回：
        DataFrame，列名统一英文，日期为索引

    使用示例：
        from quant_core.data import fetch_stock
        df = fetch_stock("600519", days=365)
    """
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")

    source_list = _normalize_sources(sources)
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        source_key = "-".join(source_list)
        cache_key = f"{code}_{start_date}_{end_date}_{adjust}_{source_key}".replace("/", "_")
        cache_path = os.path.join(cache_dir, f"{cache_key}.csv")
        if os.path.exists(cache_path):
            age = time.time() - os.path.getmtime(cache_path)
            if age <= cache_ttl_seconds:
                cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                if "date" not in cached.columns:
                    cached.index.name = "date"
                return cached

    _rate_limit_sleep(min_interval_seconds)

    df = _safe_fetch_stock(
        code=code,
        start_date=start_date,
        end_date=end_date,
        adjust=adjust,
        use_proxy=use_proxy,
        sources=source_list,
        max_retries=max_retries,
        backoff_base_seconds=backoff_base_seconds,
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

    if cache_path:
        df.to_csv(cache_path)

    return df


def _safe_fetch_stock(
    code: str,
    start_date: str,
    end_date: str,
    adjust: str,
    use_proxy: bool,
    sources: list[str],
    max_retries: int,
    backoff_base_seconds: float,
) -> pd.DataFrame:
    last_error = None
    resolved_sources = _resolve_sources(
        sources=sources,
        code=code,
        start_date=start_date,
        end_date=end_date,
        adjust=adjust,
    )
    if not resolved_sources:
        raise ValueError("未找到可用的数据源，请检查AkShare版本或sources参数。")
    for attempt in range(1, max_retries + 1):
        try:
            for name, fetcher in resolved_sources:
                try:
                    if use_proxy:
                        return fetcher()
                    with _temporary_disable_proxies():
                        return fetcher()
                except Exception as exc:
                    last_error = exc
                    continue
        except Exception as exc:
            last_error = exc
            sleep_seconds = backoff_base_seconds * (2 ** (attempt - 1))
            sleep_seconds += random.uniform(0.0, 0.4)
            time.sleep(sleep_seconds)

    raise last_error


def _normalize_sources(sources: Optional[list[str]]) -> list[str]:
    if not sources:
        return ["eastmoney", "tencent"]
    normalized = []
    for item in sources:
        if not item or not item.strip():
            continue
        name = item.strip().lower()
        name = _SOURCE_ALIASES.get(name, name)
        normalized.append(name)
    return normalized


def _resolve_sources(
    sources: list[str],
    code: str,
    start_date: str,
    end_date: str,
    adjust: str,
) -> list[tuple[str, callable]]:
    resolved = []
    for name in sources:
        if name == "eastmoney" and hasattr(ak, "stock_zh_a_hist"):
            resolved.append((
                "eastmoney",
                lambda c=code, sd=start_date, ed=end_date, adj=adjust: ak.stock_zh_a_hist(
                    symbol=c,
                    period="daily",
                    start_date=sd,
                    end_date=ed,
                    adjust=adj,
                ),
            ))
        elif name == "tencent" and hasattr(ak, "stock_zh_a_hist_tx"):
            resolved.append((
                "tencent",
                lambda c=code, sd=start_date, ed=end_date, adj=adjust: ak.stock_zh_a_hist_tx(
                    symbol=c,
                    start_date=sd,
                    end_date=ed,
                    adjust=adj,
                ),
            ))
    return resolved


_SOURCE_ALIASES = {
    "hist": "eastmoney",
    "em": "eastmoney",
    "eastmoney": "eastmoney",
    "tx": "tencent",
    "tencent": "tencent",
}


_last_call_timestamp = 0.0


def _rate_limit_sleep(min_interval_seconds: float) -> None:
    global _last_call_timestamp
    if min_interval_seconds <= 0:
        return
    now = time.time()
    elapsed = now - _last_call_timestamp
    if elapsed < min_interval_seconds:
        time.sleep(min_interval_seconds - elapsed + random.uniform(0.0, 0.2))
    _last_call_timestamp = time.time()