# Tests run against a throwaway database and the recorded market snapshot.
# The environment has to be set before anything imports `agent`.
import os
import tempfile

os.environ["PORTFOLIO_DB"] = os.path.join(tempfile.mkdtemp(), "test_portfolio.db")

import eval.offline  # noqa: E402,F401  (sets MARKET_SNAPSHOT)

import pytest  # noqa: E402

from agent.db import reset_database  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_database():
    reset_database()
    yield
    reset_database()
