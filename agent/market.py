# --- Market Data Functions ---
# Two data sources behind the same functions:
#   live     - yfinance, used for the interactive demo
#   snapshot - a recorded JSON file, used for evals and benchmarks so answers are
#              checkable and concurrent runs don't hit yfinance rate limits
import json
import os
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

VALID_HISTORICAL_PERIODS = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "5y"}

SNAPSHOT = None


def use_snapshot(path):
    """Serve market data from a recorded snapshot file. Pass None to go back to live."""
    global SNAPSHOT
    if path is None:
        SNAPSHOT = None
        return
    with open(path, encoding="utf-8") as f:
        SNAPSHOT = json.load(f)


if os.environ.get("MARKET_SNAPSHOT"):
    use_snapshot(os.environ["MARKET_SNAPSHOT"])


def snapshot_active() -> bool:
    return SNAPSHOT is not None


def _from_snapshot(section: str, key: str, what: str) -> dict:
    entry = SNAPSHOT.get(section, {}).get(key)
    if entry is None:
        return {"error": f"No {what} available for {key.split('|')[0]}"}
    return entry


def normalize_ticker(ticker: str) -> str:
    return ticker.strip().upper()


def safe_float(value):
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def get_stock_price(ticker: str) -> dict:
    ticker = normalize_ticker(ticker)

    if snapshot_active():
        return _from_snapshot("get_stock_price", ticker, "price data")

    try:
        tk = yf.Ticker(ticker)
        fi = tk.fast_info

        last_price = safe_float(fi.get("lastPrice"))
        previous_close = safe_float(fi.get("previousClose"))

        if last_price is None:
            return {"error": f"Could not retrieve live price for {ticker}"}

        change = None
        change_pct = None
        if previous_close not in (None, 0):
            change = last_price - previous_close
            change_pct = (change / previous_close) * 100

        return {
            "ticker": ticker,
            "current_price": round(last_price, 2),
            "previous_close": None if previous_close is None else round(previous_close, 2),
            "change": None if change is None else round(change, 2),
            "change_percent": None if change_pct is None else round(change_pct, 2),
            "day_high": None if fi.get("dayHigh") is None else round(float(fi.get("dayHigh")), 2),
            "day_low": None if fi.get("dayLow") is None else round(float(fi.get("dayLow")), 2),
            "year_high": None if fi.get("yearHigh") is None else round(float(fi.get("yearHigh")), 2),
            "year_low": None if fi.get("yearLow") is None else round(float(fi.get("yearLow")), 2),
            "market_cap": safe_float(fi.get("marketCap")),
        }

    except Exception as e:
        return {"error": f"Failed to get stock price for {ticker}: {str(e)}"}


def get_option_chain(ticker: str, expiration: str = None, top_n: int = 5) -> dict:
    ticker = normalize_ticker(ticker)

    if snapshot_active():
        chain = _from_snapshot("get_option_chain", ticker, "option data")
        if "error" not in chain and expiration not in (None, chain["expiration"]):
            return {
                "error": f"Invalid expiration for {ticker}",
                "available_expirations": [chain["expiration"]],
            }
        return chain

    try:
        tk = yf.Ticker(ticker)
        expirations = list(tk.options or [])

        if not expirations:
            return {"error": f"No option expirations available for {ticker}"}

        if expiration is None:
            expiration = expirations[0]
        elif expiration not in expirations:
            return {
                "error": f"Invalid expiration for {ticker}",
                "available_expirations": expirations[:10]
            }

        current_price_data = get_stock_price(ticker)
        if "error" in current_price_data:
            return current_price_data

        current_price = current_price_data["current_price"]
        chain = tk.option_chain(expiration)

        calls = chain.calls.copy()
        puts = chain.puts.copy()

        if calls.empty and puts.empty:
            return {"error": f"No option data found for {ticker} {expiration}"}

        if not calls.empty:
            calls["distance"] = (calls["strike"] - current_price).abs()
            calls = calls.sort_values("distance").head(top_n)

        if not puts.empty:
            puts["distance"] = (puts["strike"] - current_price).abs()
            puts = puts.sort_values("distance").head(top_n)

        def format_contracts(df):
            if df.empty:
                return []
            cols = ["contractSymbol", "strike", "lastPrice", "bid", "ask", "volume", "openInterest", "impliedVolatility"]
            out = []
            for _, row in df[cols].iterrows():
                out.append({
                    "contract": row["contractSymbol"],
                    "strike": safe_float(row["strike"]),
                    "last_price": safe_float(row["lastPrice"]),
                    "bid": safe_float(row["bid"]),
                    "ask": safe_float(row["ask"]),
                    "volume": 0 if pd.isna(row["volume"]) else int(row["volume"]),
                    "open_interest": 0 if pd.isna(row["openInterest"]) else int(row["openInterest"]),
                    "implied_volatility": None if pd.isna(row["impliedVolatility"]) else round(float(row["impliedVolatility"]), 4),
                })
            return out

        return {
            "ticker": ticker,
            "expiration": expiration,
            "current_price": current_price,
            "calls_near_atm": format_contracts(calls),
            "puts_near_atm": format_contracts(puts),
        }

    except Exception as e:
        return {"error": f"Failed to get option chain for {ticker}: {str(e)}"}


def get_stock_news(ticker: str, max_items: int = 5) -> dict:
    """
    Handles both old and new yfinance news schemas:
      - old (< 0.2.40): flat dict with 'title', 'publisher', 'link', 'providerPublishTime'
      - new (>= 0.2.40): nested under 'content' with 'title', 'provider.displayName',
        'clickThroughUrl.url', 'pubDate'/'displayTime'
    """
    ticker = normalize_ticker(ticker)

    if snapshot_active():
        return _from_snapshot("get_stock_news", ticker, "news")

    try:
        tk = yf.Ticker(ticker)
        news_items = tk.news or []

        if not news_items:
            return {"ticker": ticker, "articles": []}

        articles = []
        for item in news_items[:max_items]:
            content = item.get("content", item)

            title = content.get("title")

            # publisher: 'publisher' (old) or provider.displayName (new)
            provider = content.get("provider")
            if isinstance(provider, dict):
                publisher = provider.get("displayName")
            else:
                publisher = content.get("publisher")

            # link: 'link' (old) or clickThroughUrl.url / canonicalUrl.url (new)
            link = content.get("link")
            if not link:
                ct = content.get("clickThroughUrl") or content.get("canonicalUrl")
                if isinstance(ct, dict):
                    link = ct.get("url")

            # time: 'providerPublishTime' (old, unix int) or 'pubDate'/'displayTime' (new, ISO)
            published = content.get("pubDate") or content.get("displayTime")
            if not published:
                pub_ts = content.get("providerPublishTime") or item.get("providerPublishTime")
                if pub_ts:
                    try:
                        published = datetime.fromtimestamp(pub_ts, tz=timezone.utc).isoformat(sep=" ", timespec="seconds")
                    except Exception:
                        pass

            articles.append({
                "title": title,
                "publisher": publisher,
                "link": link,
                "published": published,
            })

        return {
            "ticker": ticker,
            "articles": articles,
        }

    except Exception as e:
        return {"error": f"Failed to get news for {ticker}: {str(e)}"}


def get_historical_data(ticker: str, period: str = "1mo") -> dict:
    ticker = normalize_ticker(ticker)
    period = period.strip()

    if period not in VALID_HISTORICAL_PERIODS:
        return {
            "error": f"Invalid period '{period}'",
            "valid_periods": sorted(VALID_HISTORICAL_PERIODS),
        }

    if snapshot_active():
        return _from_snapshot("get_historical_data", f"{ticker}|{period}", "historical data")

    try:
        hist = yf.Ticker(ticker).history(period=period, auto_adjust=False)

        if hist.empty:
            return {"error": f"No historical data found for {ticker} over period {period}"}

        hist = hist.reset_index()

        # Compute summary stats over the FULL period
        first_close   = safe_float(hist.iloc[0]["Close"])
        last_close    = safe_float(hist.iloc[-1]["Close"])
        period_high   = safe_float(hist["High"].max())
        period_low    = safe_float(hist["Low"].min())
        pct_change    = (
            round((last_close - first_close) / first_close * 100, 2)
            if first_close and last_close else None
        )

        # Cap the per-row records returned to keep tool-result tokens bounded.
        # Returning every daily OHLCV row over multi-month periods can push the
        # input context well past 8k tokens and OOM smaller models on a T4.
        MAX_RECORDS = 10
        recent = hist.tail(MAX_RECORDS)

        records = []
        for _, row in recent.iterrows():
            dt_col = row["Date"]
            records.append({
                "date": pd.to_datetime(dt_col).strftime("%Y-%m-%d"),
                "open": safe_float(row.get("Open")),
                "high": safe_float(row.get("High")),
                "low": safe_float(row.get("Low")),
                "close": safe_float(row.get("Close")),
                "volume": 0 if pd.isna(row.get("Volume")) else int(row.get("Volume")),
            })

        return {
            "ticker": ticker,
            "period": period,
            "num_points_total": len(hist),
            "summary": {
                "first_close":  first_close,
                "last_close":   last_close,
                "period_high":  period_high,
                "period_low":   period_low,
                "pct_change":   pct_change,
            },
            "recent_records": records,
            "note": f"Summary covers the full {period} period; recent_records shows the last {len(records)} of {len(hist)} trading days.",
        }

    except Exception as e:
        return {"error": f"Failed to get historical data for {ticker}: {str(e)}"}


def get_company_info(ticker: str) -> dict:
    """Company name and trailing P/E (used by compare_stocks)."""
    ticker = normalize_ticker(ticker)

    if snapshot_active():
        return SNAPSHOT.get("info", {}).get(ticker, {"company_name": ticker, "trailing_pe": None})

    company_name = ticker
    pe_ratio = None
    try:
        info = yf.Ticker(ticker).info or {}
        company_name = info.get("shortName", ticker)
        pe_ratio = safe_float(info.get("trailingPE"))
    except Exception:
        pass

    return {
        "company_name": company_name,
        "trailing_pe": None if pe_ratio is None else round(pe_ratio, 2),
    }


def get_latest_prices(tickers: list[str]) -> dict:
    """
    Batch-fetch latest close prices for multiple tickers.
    Returns: {ticker: latest_price_or_none}
    """
    tickers = [normalize_ticker(t) for t in tickers if t]
    if not tickers:
        return {}

    if snapshot_active():
        return {t: get_stock_price(t).get("current_price") for t in tickers}

    try:
        batch = yf.download(
            tickers=tickers,
            period="1d",
            progress=False,
            threads=False,
            auto_adjust=False,
        )

        if batch.empty:
            return {t: None for t in tickers}

        close = batch["Close"]

        # single ticker case
        if isinstance(close, pd.Series):
            return {tickers[0]: safe_float(close.iloc[-1])}

        latest = close.iloc[-1].to_dict()
        return {t: safe_float(latest.get(t)) for t in tickers}

    except Exception:
        return {t: None for t in tickers}
