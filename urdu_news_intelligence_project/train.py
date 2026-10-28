"""Compatibility CLI entry point for training and model comparison."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from urdu_news_intelligence.compare_models import main

if __name__ == "__main__":
    main()
