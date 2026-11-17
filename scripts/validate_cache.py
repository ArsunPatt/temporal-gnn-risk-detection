"""Audit or recover saved Stage 4 caches without inference or fitting."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cache_provenance import audit_stage4_cache

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adopt-legacy", action="store_true", help="Date and seal verified existing caches; not an original code-freeze proof")
    parser.add_argument("--recover-tables", action="store_true", help="Rebuild aggregate tables from validated complete scenario records only")
    args = parser.parse_args()
    result = audit_stage4_cache(ROOT, args.adopt_legacy, args.recover_tables)
    print(json.dumps({"status": "passed", "new_inference": False, **result}, indent=2))
