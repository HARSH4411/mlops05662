"""Step 1: download the raw UCI file and write the cleaned CSV.

Usage:  python scripts/download_data.py [--force]
"""
import argparse
import logging

from heart import config
from heart.data import clean, download, load_raw

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="re-download even if file exists")
    args = p.parse_args()

    download(force=args.force)
    raw = load_raw()
    df = clean(raw)
    config.CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.CLEAN_PATH, index=False)
    logging.info("raw rows=%d  clean rows=%d  missing cells=%d  positive rate=%.3f",
                 len(raw), len(df), int(df.isna().sum().sum()), df[config.TARGET].mean())
    logging.info("Wrote %s", config.CLEAN_PATH)
