# --- Portfolio Analysis Functions ---
from agent.db import get_user_info, get_holdings
from agent.market import (
    normalize_ticker, get_stock_price, get_company_info, get_latest_prices,
)

SECTOR_MAP = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "GOOGL": "Communication Services",
    "AMZN": "Consumer Discretionary",
    "NVDA": "Technology",
    "JPM": "Financials",
    "JNJ": "Healthcare",
    "VTI": "ETF",
    "BND": "ETF",
    "TSLA": "Consumer Discretionary",
}


def get_portfolio_summary(user_id: int = 1) -> dict:
    user_info = get_user_info(user_id)
    if "error" in user_info:
        return user_info

    holdings = get_holdings(user_id)
    if not holdings:
        return {
            "user_name": user_info["name"],
            "cash_balance": user_info["cash_balance"],
            "holdings_count": 0,
            "portfolio_value": round(user_info["cash_balance"], 2),
            "invested_value": 0.0,
            "total_cost_basis": 0.0,
            "unrealized_gain_loss": 0.0,
            "unrealized_gain_loss_percent": None,
            "positions": [],
        }

    tickers = [h["ticker"] for h in holdings]
    price_map = get_latest_prices(tickers)

    positions = []
    invested_value = 0.0
    total_cost_basis = 0.0

    for h in holdings:
        ticker = h["ticker"]
        shares = h["shares"]
        avg_cost = h["avg_cost_basis"]
        cost_basis = shares * avg_cost
        current_price = price_map.get(ticker)

        current_value = None if current_price is None else shares * current_price
        gain_loss = None if current_value is None else current_value - cost_basis
        gain_loss_pct = None if cost_basis == 0 or gain_loss is None else (gain_loss / cost_basis) * 100

        if current_value is not None:
            invested_value += current_value
        total_cost_basis += cost_basis

        positions.append({
            "ticker": ticker,
            "shares": round(shares, 4),
            "avg_cost_basis": round(avg_cost, 2),
            "current_price": None if current_price is None else round(current_price, 2),
            "cost_basis": round(cost_basis, 2),
            "current_value": None if current_value is None else round(current_value, 2),
            "gain_loss": None if gain_loss is None else round(gain_loss, 2),
            "gain_loss_percent": None if gain_loss_pct is None else round(gain_loss_pct, 2),
        })

    portfolio_value = user_info["cash_balance"] + invested_value
    unrealized_gain_loss = invested_value - total_cost_basis
    unrealized_gain_loss_pct = None if total_cost_basis == 0 else (unrealized_gain_loss / total_cost_basis) * 100

    return {
        "user_name": user_info["name"],
        "cash_balance": round(user_info["cash_balance"], 2),
        "holdings_count": len(holdings),
        "portfolio_value": round(portfolio_value, 2),
        "invested_value": round(invested_value, 2),
        "total_cost_basis": round(total_cost_basis, 2),
        "unrealized_gain_loss": round(unrealized_gain_loss, 2),
        "unrealized_gain_loss_percent": None if unrealized_gain_loss_pct is None else round(unrealized_gain_loss_pct, 2),
        "positions": positions,
    }


def get_sector_allocation(user_id: int = 1) -> dict:
    holdings = get_holdings(user_id)
    if not holdings:
        return {"sector_allocation": [], "total_invested_value": 0.0}

    tickers = [h["ticker"] for h in holdings]
    price_map = get_latest_prices(tickers)

    sector_values = {}
    total_value = 0.0

    for h in holdings:
        ticker = h["ticker"]
        shares = h["shares"]
        current_price = price_map.get(ticker)

        if current_price is None:
            continue

        current_value = shares * current_price
        sector = SECTOR_MAP.get(ticker, "Other")

        sector_values[sector] = sector_values.get(sector, 0.0) + current_value
        total_value += current_value

    allocation = []
    for sector, value in sorted(sector_values.items(), key=lambda x: x[1], reverse=True):
        pct = (value / total_value * 100) if total_value > 0 else 0.0
        allocation.append({
            "sector": sector,
            "value": round(value, 2),
            "percentage": round(pct, 2),
        })

    return {
        "total_invested_value": round(total_value, 2),
        "sector_allocation": allocation,
    }


def compare_stocks(ticker_1: str, ticker_2: str) -> dict:
    ticker_1 = normalize_ticker(ticker_1)
    ticker_2 = normalize_ticker(ticker_2)

    def get_stock_snapshot(ticker: str) -> dict:
        price_data = get_stock_price(ticker)
        if "error" in price_data:
            return price_data

        info = get_company_info(ticker)

        return {
            "ticker": ticker,
            "company_name": info.get("company_name", ticker),
            "current_price": price_data.get("current_price"),
            "change_percent": price_data.get("change_percent"),
            "year_high": price_data.get("year_high"),
            "year_low": price_data.get("year_low"),
            "market_cap": price_data.get("market_cap"),
            "trailing_pe": info.get("trailing_pe"),
        }

    return {
        "stock_1": get_stock_snapshot(ticker_1),
        "stock_2": get_stock_snapshot(ticker_2),
    }
