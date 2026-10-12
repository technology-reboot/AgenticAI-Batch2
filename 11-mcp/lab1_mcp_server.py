"""
Lab 1 - Build and connect an MCP server
========================================

Objective
---------
Create an MCP server that exposes financial data tools, test it in Claude
Desktop, then connect it to a LangChain ReAct agent.

Tools exposed by this server
-----------------------------
  - get_stock_price(ticker)        -> latest closing price (yfinance)
  - get_news_headlines(query)      -> recent headlines (NewsAPI)
  - get_fundamentals(ticker)       -> P/E ratio, EPS, market cap, ... (yfinance)

Setup
-----
  pip install -r requirements.txt

  Environment variables (set in your shell or a .env file):
    NEWSAPI_KEY        - required for get_news_headlines (https://newsapi.org)
    OPENAI_API_KEY      - required for --test-client / --compare (the ReAct agent's LLM)
    LLM_MODEL           - optional, defaults to "gpt-4o-mini"

Usage
-----
  1) Run the MCP server directly (stdio transport):
        python lab1_mcp_server.py

  2) Print the Claude Desktop config snippet, then paste it into
     claude_desktop_config.json and restart Claude Desktop. Check
     Settings -> Developer -> MCP Servers to see the 3 tools listed:
        python lab1_mcp_server.py --print-config

  3) Connect with LangChain (MultiServerMCPClient + load_mcp_tools) and run
     a ReAct agent that answers "What is the current P/E ratio of TCS?":
        python lab1_mcp_server.py --test-client

  4) Compare the same agent/question using raw Python tool functions vs.
     the same logic exposed over MCP:
        python lab1_mcp_server.py --compare

Notes
-----
  - Indian (NSE) tickers such as 'TCS' are auto-suffixed with '.NS' for
    yfinance (e.g. 'TCS' -> 'TCS.NS'), since this lab's sample question is
    about Tata Consultancy Services.
  - "--test-client" spawns this same file as a subprocess (python
    lab1_mcp_server.py, no flags) over stdio, exactly like Claude Desktop
    would.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

from mcp.server.fastmcp import FastMCP

SERVER_NAME = "financial-data-server"
mcp_server = FastMCP(SERVER_NAME)

NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", "")
NEWSAPI_URL = "https://newsapi.org/v2/everything"

QUESTION = "What is the current P/E ratio of TCS?"


# ---------------------------------------------------------------------------
# Shared implementation (used by both the MCP tool wrappers below and the
# raw LangChain @tool wrappers in run_raw_agent, so the two execution paths
# share identical logic and only differ in how the tools are exposed).
# ---------------------------------------------------------------------------


def _normalize_ticker(ticker: str) -> str:
    ticker = ticker.strip().upper()
    if "." not in ticker and not ticker.endswith("-USD"):
        # NSE-listed Indian tickers (e.g. TCS) need a ".NS" suffix for yfinance.
        return f"{ticker}.NS"
    return ticker


def _get_stock_price_impl(ticker: str) -> dict[str, Any]:
    symbol = _normalize_ticker(ticker)
    stock = yf.Ticker(symbol)
    history = stock.history(period="1d")
    if history.empty:
        return {"ticker": symbol, "error": f"No price data found for '{symbol}'."}
    last_row = history.iloc[-1]
    try:
        currency = stock.fast_info["currency"]
    except Exception:
        currency = "N/A"
    return {
        "ticker": symbol,
        "price": round(float(last_row["Close"]), 2),
        "currency": currency,
        "as_of": str(last_row.name.date()),
    }


def _get_news_headlines_impl(query: str, page_size: int = 5) -> dict[str, Any]:
    if not NEWSAPI_KEY:
        return {
            "query": query,
            "error": "NEWSAPI_KEY is not set. Export it or add it to a .env file.",
            "headlines": [],
        }
    params = {
        "q": query,
        "pageSize": max(1, min(page_size, 20)),
        "sortBy": "publishedAt",
        "language": "en",
        "apiKey": NEWSAPI_KEY,
    }
    response = requests.get(NEWSAPI_URL, params=params, timeout=10)
    response.raise_for_status()
    articles = response.json().get("articles", [])
    headlines = [
        {
            "title": a.get("title"),
            "source": (a.get("source") or {}).get("name"),
            "publishedAt": a.get("publishedAt"),
            "url": a.get("url"),
        }
        for a in articles
    ]
    return {"query": query, "headlines": headlines}


def _get_fundamentals_impl(ticker: str) -> dict[str, Any]:
    symbol = _normalize_ticker(ticker)
    info = yf.Ticker(symbol).info
    if not info or (info.get("trailingPE") is None and info.get("regularMarketPrice") is None):
        return {"ticker": symbol, "error": f"No fundamentals found for '{symbol}'."}
    return {
        "ticker": symbol,
        "pe_ratio": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "eps": info.get("trailingEps"),
        "market_cap": info.get("marketCap"),
        "dividend_yield": info.get("dividendYield"),
        "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
        "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
        "sector": info.get("sector"),
    }


# ---------------------------------------------------------------------------
# Step 2: MCP server with 3 registered tools
# ---------------------------------------------------------------------------


@mcp_server.tool()
def get_stock_price(ticker: str) -> dict:
    """Get the latest closing price for a stock ticker.

    Args:
        ticker: Stock ticker symbol, e.g. 'TCS', 'INFY', 'AAPL'. Indian NSE
            tickers are auto-suffixed with '.NS' if no suffix is given.
    """
    return _get_stock_price_impl(ticker)


@mcp_server.tool()
def get_news_headlines(query: str, page_size: int = 5) -> dict:
    """Get recent news headlines about a company or topic.

    Args:
        query: Company name or search topic, e.g. 'Tata Consultancy Services'.
        page_size: Number of headlines to return (1-20, default 5).
    """
    return _get_news_headlines_impl(query, page_size)


@mcp_server.tool()
def get_fundamentals(ticker: str) -> dict:
    """Get key fundamentals (P/E ratio, EPS, market cap, etc.) for a stock.

    Args:
        ticker: Stock ticker symbol, e.g. 'TCS', 'INFY', 'AAPL'. Indian NSE
            tickers are auto-suffixed with '.NS' if no suffix is given.
    """
    return _get_fundamentals_impl(ticker)


# ---------------------------------------------------------------------------
# Step 4-5: LangChain client wiring (MultiServerMCPClient + load_mcp_tools)
# and a ReAct agent built from the MCP tools.
# ---------------------------------------------------------------------------


def _build_llm():
    from langchain_openai import ChatOpenAI

    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model, temperature=0)


async def run_mcp_agent(question: str = QUESTION) -> str:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools
    from langgraph.prebuilt import create_react_agent

    server_script = str(Path(__file__).resolve())
    client = MultiServerMCPClient(
        {
            SERVER_NAME: {
                "command": sys.executable,
                "args": [server_script],
                "transport": "stdio",
            }
        }
    )

    async with client.session(SERVER_NAME) as session:
        tools = await load_mcp_tools(session)
        print(f"[MCP] Loaded {len(tools)} tools: {[t.name for t in tools]}")
        agent = create_react_agent(_build_llm(), tools)
        result = await agent.ainvoke({"messages": [("user", question)]})
        return result["messages"][-1].content


# ---------------------------------------------------------------------------
# Step 6: Same agent/question, but with raw Python tools (no MCP) for
# comparison against the MCP-tools version above.
# ---------------------------------------------------------------------------


def run_raw_agent(question: str = QUESTION) -> str:
    from langchain_core.tools import tool
    from langgraph.prebuilt import create_react_agent

    @tool
    def get_stock_price(ticker: str) -> dict:
        """Get the latest closing price for a stock ticker."""
        return _get_stock_price_impl(ticker)

    @tool
    def get_news_headlines(query: str, page_size: int = 5) -> dict:
        """Get recent news headlines about a company or topic."""
        return _get_news_headlines_impl(query, page_size)

    @tool
    def get_fundamentals(ticker: str) -> dict:
        """Get key fundamentals (P/E ratio, EPS, market cap, etc.) for a stock."""
        return _get_fundamentals_impl(ticker)

    raw_tools = [get_stock_price, get_news_headlines, get_fundamentals]
    print(f"[Raw] Loaded {len(raw_tools)} tools: {[t.name for t in raw_tools]}")
    agent = create_react_agent(_build_llm(), raw_tools)
    result = agent.invoke({"messages": [("user", question)]})
    return result["messages"][-1].content


# ---------------------------------------------------------------------------
# Step 3: Claude Desktop config helper
# ---------------------------------------------------------------------------


def print_claude_desktop_config() -> None:
    config = {
        "mcpServers": {
            SERVER_NAME: {
                "command": sys.executable,
                "args": [str(Path(__file__).resolve())],
            }
        }
    }
    print(json.dumps(config, indent=2))
    print("\nAdd the block above to claude_desktop_config.json, then restart Claude Desktop.")
    print("Settings -> Developer -> MCP Servers should then list:", SERVER_NAME)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab 1: MCP server with financial data tools")
    parser.add_argument("--print-config", action="store_true", help="Print Claude Desktop config snippet and exit")
    parser.add_argument("--test-client", action="store_true", help="Run a LangChain ReAct agent against this server via MCP")
    parser.add_argument("--compare", action="store_true", help="Compare raw Python tools vs MCP tools on the same question")
    parser.add_argument("--question", default=QUESTION, help="Question to ask the agent in --test-client / --compare modes")
    args = parser.parse_args()

    if args.print_config:
        print_claude_desktop_config()
        return

    if args.compare:
        print("=" * 70)
        print("RAW PYTHON TOOLS (Session 3 style)")
        print("=" * 70)
        print(run_raw_agent(args.question))
        print("\n" + "=" * 70)
        print("MCP TOOLS (this lab)")
        print("=" * 70)
        print(asyncio.run(run_mcp_agent(args.question)))
        return

    if args.test_client:
        print(asyncio.run(run_mcp_agent(args.question)))
        return

    # Default: run the MCP server itself (stdio transport). This is what
    # Claude Desktop launches, and what MultiServerMCPClient spawns as a
    # subprocess in run_mcp_agent() above.
    mcp_server.run(transport="stdio")


if __name__ == "__main__":
    main()
