"""Pytest configuration."""
import sys
from pathlib import Path

# Ensure the project root is on the Python path when running tests
sys.path.insert(0, str(Path(__file__).parent.parent))
