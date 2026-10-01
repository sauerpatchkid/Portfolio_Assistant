# --- Database Schema + Mock Data + Query Functions ---
import os
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from sqlalchemy import (
    create_engine, Column, Integer, Float, String, DateTime,
    ForeignKey, UniqueConstraint, Index
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

# Set PORTFOLIO_DB to point runs at their own file (the eval harness does this).
DB_PATH = os.environ.get("PORTFOLIO_DB", "portfolio.db")


def utcnow_naive():
    """
    Return UTC time as a naive datetime.
    Using one consistent convention avoids SQLite timezone weirdness.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


engine = create_engine(f"sqlite:///{DB_PATH}", echo=False, future=True)
Base = declarative_base()
Session = sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def session_scope():
    session = Session()
    try:
        yield session
    finally:
        session.close()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    cash_balance = Column(Float, nullable=False, default=0.0)

    holdings = relationship("Holding", back_populates="user", cascade="all, delete-orphan")
    transactions = relationship("Transaction", back_populates="user", cascade="all, delete-orphan")


class Holding(Base):
    __tablename__ = "holdings"
    __table_args__ = (
        UniqueConstraint("user_id", "ticker", name="uq_holdings_user_ticker"),
        Index("ix_holdings_user_ticker", "user_id", "ticker"),
    )

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    ticker = Column(String, nullable=False)
    shares = Column(Float, nullable=False)
    avg_cost_basis = Column(Float, nullable=False)

    user = relationship("User", back_populates="holdings")


class Transaction(Base):
    __tablename__ = "transactions"
    __table_args__ = (
        Index("ix_transactions_user_timestamp", "user_id", "timestamp"),
    )

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    ticker = Column(String, nullable=False)
    action = Column(String, nullable=False)   # BUY or SELL
    shares = Column(Float, nullable=False)
    price_per_share = Column(Float, nullable=False)
    total_amount = Column(Float, nullable=False)
    order_type = Column(String, nullable=False, default="MARKET")
    timestamp = Column(DateTime, nullable=False, default=utcnow_naive)

    user = relationship("User", back_populates="transactions")


Base.metadata.create_all(engine)

MOCK_CASH_BALANCE = 15420.50

MOCK_HOLDINGS = [
    ("AAPL", 25, 185.20),
    ("MSFT", 18, 412.75),
    ("GOOGL", 12, 168.40),
    ("AMZN", 10, 178.30),
    ("NVDA", 15, 905.10),
    ("JPM", 20, 192.80),
    ("JNJ", 22, 151.90),
    ("VTI", 30, 255.45),
    ("BND", 40, 72.10),
    ("TSLA", 8, 210.60),
]

# (ticker, action, shares, price, order_type, days_ago)
MOCK_TRANSACTIONS = [
    ("AAPL", "BUY", 10, 178.50, "MARKET", 55),
    ("MSFT", "BUY", 8, 398.20, "MARKET", 50),
    ("GOOGL", "BUY", 6, 162.10, "MARKET", 47),
    ("AMZN", "BUY", 5, 171.40, "MARKET", 44),
    ("NVDA", "BUY", 5, 880.00, "MARKET", 40),
    ("JPM", "BUY", 10, 188.20, "MARKET", 38),
    ("JNJ", "BUY", 10, 148.75, "MARKET", 36),
    ("VTI", "BUY", 15, 248.10, "MARKET", 34),
    ("BND", "BUY", 20, 71.40, "MARKET", 32),
    ("TSLA", "BUY", 4, 198.25, "MARKET", 29),
    ("AAPL", "BUY", 15, 189.67, "MARKET", 22),
    ("MSFT", "BUY", 10, 424.39, "MARKET", 20),
    ("GOOGL", "BUY", 6, 174.70, "MARKET", 18),
    ("NVDA", "BUY", 10, 917.65, "MARKET", 15),
    ("BND", "BUY", 20, 72.80, "MARKET", 12),
    ("TSLA", "BUY", 4, 222.95, "MARKET", 8),
]


def seed_mock_data():
    with session_scope() as session:
        existing_user = session.query(User).filter_by(id=1).first()
        if existing_user:
            return

        user = User(id=1, name="Alex Johnson", cash_balance=MOCK_CASH_BALANCE)
        session.add(user)
        session.flush()

        for ticker, shares, avg_cost_basis in MOCK_HOLDINGS:
            session.add(
                Holding(
                    user_id=user.id,
                    ticker=ticker,
                    shares=shares,
                    avg_cost_basis=avg_cost_basis,
                )
            )

        now = utcnow_naive()
        for ticker, action, shares, price, order_type, days_ago in MOCK_TRANSACTIONS:
            session.add(
                Transaction(
                    user_id=user.id,
                    ticker=ticker,
                    action=action,
                    shares=shares,
                    price_per_share=price,
                    total_amount=round(shares * price, 2),
                    order_type=order_type,
                    timestamp=now - timedelta(days=days_ago),
                )
            )

        session.commit()


def reset_database():
    """Wipe all data and re-seed with mock data."""
    with session_scope() as session:
        session.query(Transaction).delete()
        session.query(Holding).delete()
        session.query(User).delete()
        session.commit()
    seed_mock_data()


seed_mock_data()


def get_user_info(user_id: int = 1) -> dict:
    with session_scope() as session:
        user = session.query(User).filter_by(id=user_id).first()
        if not user:
            return {"error": "User not found"}

        return {
            "user_id": user.id,
            "name": user.name,
            "cash_balance": round(user.cash_balance, 2),
        }


def get_holdings(user_id: int = 1) -> list[dict]:
    with session_scope() as session:
        holdings = (
            session.query(Holding)
            .filter_by(user_id=user_id)
            .order_by(Holding.ticker.asc())
            .all()
        )

        return [
            {
                "ticker": h.ticker,
                "shares": round(h.shares, 4),
                "avg_cost_basis": round(h.avg_cost_basis, 2),
                "total_cost_basis": round(h.shares * h.avg_cost_basis, 2),
            }
            for h in holdings
        ]


def get_holding_by_ticker(ticker: str, user_id: int = 1) -> dict:
    ticker = ticker.strip().upper()

    with session_scope() as session:
        holding = (
            session.query(Holding)
            .filter_by(user_id=user_id, ticker=ticker)
            .first()
        )

        if not holding:
            return {"error": f"No holding found for ticker '{ticker}'"}

        return {
            "ticker": holding.ticker,
            "shares": round(holding.shares, 4),
            "avg_cost_basis": round(holding.avg_cost_basis, 2),
            "total_cost_basis": round(holding.shares * holding.avg_cost_basis, 2),
        }


def get_recent_transactions(user_id: int = 1, days: int = 60) -> list[dict]:
    cutoff = utcnow_naive() - timedelta(days=days)

    with session_scope() as session:
        txns = (
            session.query(Transaction)
            .filter(Transaction.user_id == user_id, Transaction.timestamp >= cutoff)
            .order_by(Transaction.timestamp.desc())
            .all()
        )

        return [
            {
                "ticker": t.ticker,
                "action": t.action,
                "shares": round(t.shares, 4),
                "price_per_share": round(t.price_per_share, 2),
                "total_amount": round(t.total_amount, 2),
                "order_type": t.order_type,
                "timestamp": t.timestamp.isoformat(sep=" ", timespec="seconds"),
            }
            for t in txns
        ]
