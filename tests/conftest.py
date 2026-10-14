import os
import sys

# The audits import their shared helpers (_common, fixtures) as top-level modules.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
