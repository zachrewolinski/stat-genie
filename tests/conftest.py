"""
Pytest configuration for stat-genie tests.

Adds necessary directories to Python path so imports work:
- blade/ directory for blade_bench
- src/ directory for stat_genie
"""
import sys
from pathlib import Path

# Get the repo root directory
repo_root = Path(__file__).parent.parent

# Add blade/ directory to Python path so blade_bench can be imported
blade_dir = repo_root / "blade"
if str(blade_dir) not in sys.path:
    sys.path.insert(0, str(blade_dir))

# Add src/ directory to Python path so stat_genie can be imported
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

