import sys
import os

# Make customer_churn_prediction/ the root for imports
# so that `from api.index import app` works in tests
sys.path.insert(0, os.path.dirname(__file__))
