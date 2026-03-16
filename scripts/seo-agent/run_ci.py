#!/usr/bin/env python3
"""
CI wrapper – runs the SEO agent and prints full traceback on failure.
Run from repo root: python scripts/seo-agent/run_ci.py
"""
import sys
import traceback
from pathlib import Path

# Ensure we're in project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
import os
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(SCRIPT_DIR))

if __name__ == "__main__":
    try:
        from run_agent import main
        sys.exit(main())
    except Exception as e:
        print("=" * 60)
        print("SEO Agent failed with error:")
        print("=" * 60)
        traceback.print_exc()
        sys.exit(1)
