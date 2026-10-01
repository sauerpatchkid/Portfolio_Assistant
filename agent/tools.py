# --- Tool Registry + Schemas ---
import json

from agent.db import (
    get_user_info, get_holdings, get_holding_by_ticker, get_recent_transactions,
)
from agent.market import (
    VALID_HISTORICAL_PERIODS, get_stock_price, get_option_chain, get_stock_news,
    get_historical_data,
)
from agent.analysis import get_portfolio_summary, get_sector_allocation, compare_stocks
from agent.trade import execute_trade, preview_trade, trade_key
from agent.web import web_search


def make_tool_schema(name, description, properties=None, required=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
            }
        }
    }

TOOL_REGISTRY = [
    {
        "name": "get_user_info",
        "fn": get_user_info,
        "description": "Get the user's account information including name and available cash balance. Use when the user asks about their account, balance, or available funds.",
        "properties": {},
        "required": [],
    },
    {
        "name": "get_holdings",
        "fn": get_holdings,
        "description": "Get all current stock positions in the portfolio. Returns ticker, shares, cost basis, and total cost for each. Use when the user asks what stocks they own or their positions.",
        "properties": {},
        "required": [],
    },
    {
        "name": "get_holding_by_ticker",
        "fn": get_holding_by_ticker,
        "description": "Get details about a specific stock position (shares, cost basis). Use when the user asks about a specific stock they own.",
        "properties": {
            "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. 'AAPL'"}
        },
        "required": ["ticker"],
    },
    {
        "name": "get_recent_transactions",
        "fn": get_recent_transactions,
        "description": "Get the user's recent trade history (buys and sells). Use when the user asks about recent trades, transaction history, or past orders.",
        "properties": {
            "days": {"type": "integer", "description": "Days of history to retrieve.", "default": 60}
        },
        "required": [],
    },
    {
        "name": "get_stock_price",
        "fn": get_stock_price,
        "description": "Get current market price and statistics for any stock (price, change, 52-week range, market cap). Use when the user asks about a stock's current price or market data.",
        "properties": {
            "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. 'NVDA'"}
        },
        "required": ["ticker"],
    },
    {
        "name": "get_option_chain",
        "fn": get_option_chain,
        "description": "Get the options chain (calls and puts near the money) for a stock. Returns strikes, premiums, volume, open interest, and implied volatility.",
        "properties": {
            "ticker": {"type": "string", "description": "Stock ticker symbol"},
            "expiration": {"type": "string", "description": "Expiration date in YYYY-MM-DD format. Defaults to nearest expiration."}
        },
        "required": ["ticker"],
    },
    {
        "name": "get_stock_news",
        "fn": get_stock_news,
        "description": "Get recent news articles about a stock. Use when the user asks about news or why a stock moved.",
        "properties": {
            "ticker": {"type": "string", "description": "Stock ticker symbol"}
        },
        "required": ["ticker"],
    },
    {
        "name": "get_historical_data",
        "fn": get_historical_data,
        "description": "Get historical price data (OHLCV) for a stock over a time period. Use for past performance or price trends.",
        "properties": {
            "ticker": {"type": "string", "description": "Stock ticker symbol"},
            "period": {
                "type": "string",
                "description": "Time period for historical data.",
                "enum": sorted(list(VALID_HISTORICAL_PERIODS)),
                "default": "1mo"
            }
        },
        "required": ["ticker"],
    },
    {
        "name": "get_portfolio_summary",
        "fn": get_portfolio_summary,
        "description": "Get a complete portfolio summary with current market values, gain/loss per position, and total portfolio value. Use when the user asks to analyze their portfolio, see P&L, or get an investment overview.",
        "properties": {},
        "required": [],
    },
    {
        "name": "get_sector_allocation",
        "fn": get_sector_allocation,
        "description": "Get the portfolio's sector allocation breakdown with percentages. Use when the user asks about diversification or sector exposure.",
        "properties": {},
        "required": [],
    },
    {
        "name": "compare_stocks",
        "fn": compare_stocks,
        "description": "Compare two stocks side by side: current price, daily change, P/E ratio, and other key metrics. Use when the user asks to compare two tickers.",
        "properties": {
            "ticker_1": {"type": "string", "description": "First stock ticker symbol"},
            "ticker_2": {"type": "string", "description": "Second stock ticker symbol"},
        },
        "required": ["ticker_1", "ticker_2"],
    },
    {
        "name": "web_search",
        "fn": web_search,
        "description": "Search the open web (via DuckDuckGo) for general information NOT covered by the database or yfinance — e.g., central-bank decisions, macro/economic news, what is happening in markets broadly today, recent geopolitical events, or definitions of obscure terms. Returns top web results with titles, snippets, and URLs. Do NOT use for stock prices, holdings, or news on a specific ticker — those have dedicated tools.",
        "properties": {
            "query": {"type": "string", "description": "What to search the web for. Be specific."},
            "max_results": {"type": "integer", "description": "Maximum number of results.", "default": 5},
        },
        "required": ["query"],
    },
    {
        "name": "execute_trade",
        "fn": execute_trade,
        "description": "Execute a stock trade (buy or sell). ALWAYS confirm trade details with the user before calling this. Returns confirmation or error.",
        "properties": {
            "action": {"type": "string", "description": "'BUY' or 'SELL'", "enum": ["BUY", "SELL"]},
            "ticker": {"type": "string", "description": "Stock ticker symbol"},
            "shares": {"type": "number", "description": "Number of shares"},
            "order_type": {"type": "string", "description": "'MARKET' or 'LIMIT'", "enum": ["MARKET", "LIMIT"], "default": "MARKET"},
            "limit_price": {"type": "number", "description": "Limit price. Required for LIMIT orders."}
        },
        "required": ["action", "ticker", "shares"],
    },
]

TOOLS = [
    make_tool_schema(
        tool["name"],
        tool["description"],
        tool["properties"],
        tool["required"]
    )
    for tool in TOOL_REGISTRY
]

TOOL_FUNCTIONS = {
    tool["name"]: tool["fn"]
    for tool in TOOL_REGISTRY
}


# Tools that change something for real. When the agent's confirmation gate is on,
# the first call returns this preview instead, and the tool only runs once the
# same call is made again in a later turn (see agent/agent.py).
PREVIEW_FUNCTIONS = {"execute_trade": preview_trade}
CALL_KEYS = {"execute_trade": trade_key}


def preview_tool_call(name: str, arguments: dict) -> str:
    """Run a tool's preview (no side effects) and return a JSON string."""
    return execute_tool_call(name, arguments, functions=PREVIEW_FUNCTIONS)


def execute_tool_call(name: str, arguments: dict, functions: dict = None) -> str:
    """Execute one tool call and return a JSON string."""
    arguments = arguments or {}
    functions = TOOL_FUNCTIONS if functions is None else functions

    if name not in functions:
        return json.dumps({"error": f"Unknown tool: {name}"})

    try:
        result = functions[name](**arguments)
        return json.dumps(result, default=str)
    except TypeError as e:
        return json.dumps({
            "error": f"Tool argument mismatch for {name}: {str(e)}",
            "arguments_received": arguments
        })
    except Exception as e:
        return json.dumps({"error": f"Tool execution failed: {str(e)}"})
