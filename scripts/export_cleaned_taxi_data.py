#!/usr/bin/env python
"""一次性导出清洗后的轨迹 CSV 到 data/processed/（需从仓库根目录运行）。"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.data_loader import export_cleaned_corpus  # noqa: E402

if __name__ == "__main__":
    export_cleaned_corpus()
