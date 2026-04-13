"""
Shared pytest fixtures and path setup for FIN-bench-eval tests.
"""
import sys
from pathlib import Path

# Make scripts/ importable without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
