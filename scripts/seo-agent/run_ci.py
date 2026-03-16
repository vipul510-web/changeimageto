#!/usr/bin/env python3
"""
CI wrapper – runs the SEO agent and prints full traceback on failure.
"""
import sys
import traceback

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
