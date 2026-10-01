# --- Trade Execution with Guardrails ---
from agent.db import (
    session_scope, User, Holding, Transaction, utcnow_naive,
    get_user_info, get_holding_by_ticker,
)
from agent.market import normalize_ticker, get_stock_price

MAX_SHARES_PER_ORDER = 500
RESTRICTED_TICKERS = {"GME", "AMC"}


def check_order(action, ticker, shares, order_type="MARKET", limit_price=None) -> dict:
    """
    Run the guardrails that don't need the database and price the order.
    Returns {"error": ...} or the cleaned-up order with its price and total.
    """
    ticker = normalize_ticker(ticker)
    action = action.strip().upper()
    order_type = order_type.strip().upper()

    if action not in {"BUY", "SELL"}:
        return {"error": f"Invalid action '{action}'. Must be 'BUY' or 'SELL'."}

    if order_type not in {"MARKET", "LIMIT"}:
        return {"error": f"Invalid order_type '{order_type}'. Must be 'MARKET' or 'LIMIT'."}

    if shares <= 0:
        return {"error": "Number of shares must be positive."}

    if shares > MAX_SHARES_PER_ORDER:
        return {"error": f"Order too large. Maximum {MAX_SHARES_PER_ORDER} shares per order."}

    if ticker in RESTRICTED_TICKERS:
        return {"error": f"{ticker} is restricted and cannot be traded."}

    if order_type == "LIMIT" and limit_price is None:
        return {"error": "limit_price is required for LIMIT orders."}

    price_data = get_stock_price(ticker)
    if "error" in price_data:
        return {"error": f"Cannot trade {ticker}: {price_data['error']}"}

    exec_price = float(limit_price) if (order_type == "LIMIT" and limit_price is not None) else price_data["current_price"]

    return {
        "action": action,
        "ticker": ticker,
        "shares": shares,
        "order_type": order_type,
        "price_per_share": exec_price,
        "total_amount": round(shares * exec_price, 2),
    }


def trade_key(arguments: dict) -> tuple:
    """A trade's identity, so 'the same trade' can be recognised across turns."""
    limit_price = arguments.get("limit_price")
    return (
        str(arguments.get("action", "")).strip().upper(),
        str(arguments.get("ticker", "")).strip().upper(),
        float(arguments.get("shares", 0) or 0),
        str(arguments.get("order_type") or "MARKET").strip().upper(),
        None if limit_price is None else float(limit_price),
    )


def preview_trade(
    action: str,
    ticker: str,
    shares: float,
    order_type: str = "MARKET",
    limit_price: float = None
) -> dict:
    """
    Check and price a trade WITHOUT executing it. Returned in place of
    execute_trade the first time a trade is proposed, so the user can confirm.
    """
    order = check_order(action, ticker, shares, order_type, limit_price)
    if "error" in order:
        return order

    cash = get_user_info()["cash_balance"]
    holding = get_holding_by_ticker(order["ticker"])
    owned = 0 if "error" in holding else holding["shares"]

    if order["action"] == "BUY" and order["total_amount"] > cash:
        return {"error": f"Insufficient funds. Cost: ${order['total_amount']:,.2f}, Available: ${cash:,.2f}"}

    if order["action"] == "SELL" and owned < shares:
        return {"error": f"Cannot sell {shares} shares of {order['ticker']}. You own {owned}."}

    return {
        "status": "CONFIRMATION_REQUIRED",
        "message": "This trade has NOT been executed. Show the user these details and ask them to confirm. "
                   "After they confirm, call execute_trade again with the same arguments.",
        **order,
        "price_per_share": round(order["price_per_share"], 2),
        "cash_balance": cash,
        "shares_currently_owned": owned,
    }


def execute_trade(
    action: str,
    ticker: str,
    shares: float,
    order_type: str = "MARKET",
    limit_price: float = None
) -> dict:
    """Execute a stock trade with guardrails and DB updates."""
    order = check_order(action, ticker, shares, order_type, limit_price)
    if "error" in order:
        return order

    ticker, action, order_type = order["ticker"], order["action"], order["order_type"]
    exec_price, total_cost = order["price_per_share"], order["total_amount"]

    with session_scope() as session:
        try:
            user = session.query(User).filter_by(id=1).first()
            if not user:
                return {"error": "Demo user not found."}

            holding = session.query(Holding).filter_by(user_id=1, ticker=ticker).first()

            if action == "BUY":
                if total_cost > user.cash_balance:
                    return {
                        "error": f"Insufficient funds. Cost: ${total_cost:,.2f}, Available: ${user.cash_balance:,.2f}"
                    }

                user.cash_balance -= total_cost

                if holding:
                    new_total_shares = holding.shares + shares
                    new_total_cost = (holding.shares * holding.avg_cost_basis) + total_cost
                    holding.shares = new_total_shares
                    holding.avg_cost_basis = new_total_cost / new_total_shares
                else:
                    session.add(
                        Holding(
                            user_id=1,
                            ticker=ticker,
                            shares=shares,
                            avg_cost_basis=exec_price
                        )
                    )

            else:  # SELL
                owned = holding.shares if holding else 0
                if owned < shares:
                    return {"error": f"Cannot sell {shares} shares of {ticker}. You own {owned}."}

                user.cash_balance += total_cost
                holding.shares -= shares

                # Use a small epsilon to avoid float equality bugs after partial sells.
                if holding.shares < 1e-6:
                    session.delete(holding)

            session.add(
                Transaction(
                    user_id=1,
                    ticker=ticker,
                    action=action,
                    shares=shares,
                    price_per_share=exec_price,
                    total_amount=total_cost,
                    order_type=order_type,
                    timestamp=utcnow_naive(),
                )
            )

            session.commit()

            return {
                "status": "SUCCESS",
                "action": action,
                "ticker": ticker,
                "shares": shares,
                "price_per_share": round(exec_price, 2),
                "total_amount": total_cost,
                "order_type": order_type,
                "new_cash_balance": round(user.cash_balance, 2),
            }

        except Exception as e:
            session.rollback()
            return {"error": f"Trade failed unexpectedly: {str(e)}"}
