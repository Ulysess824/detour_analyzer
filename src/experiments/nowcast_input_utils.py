"""
Month in progress completed with the mean of its first days, used as the input of ml_mean for the month after it.

The logic moved to the engine (`month_in_progress.completed_month`, `strategies.nowcast_input_forecast`) so the production forecast
and this experiment run the same code; the names are kept here for the scripts and tests.
"""

from src.engine.month_in_progress import completed_month  # noqa: F401
from src.engine.strategies import nowcast_input_forecast  # noqa: F401
