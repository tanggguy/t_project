
import sys
from unittest.mock import MagicMock

# Mock scripts modules
sys.modules["scripts"] = MagicMock()
sys.modules["scripts.run_backtest"] = MagicMock()
sys.modules["scripts.run_optimization"] = MagicMock()

try:
    from optimization import dashboard_runner
    print("Import successful")
except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()
