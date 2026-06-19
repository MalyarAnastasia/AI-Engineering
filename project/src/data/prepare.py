import argparse
import logging
import os

from src.data.preprocessing import preprocess_from_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Prepare cleaned estate dataset")
    parser.add_argument("--input", default=os.getenv("RAW_DATA_PATH", "data/all_v2.csv"))
    parser.add_argument("--output", default=os.getenv("DATA_PATH", "data/cleaned_data.csv"))
    args = parser.parse_args()

    logger.info("Starting data preparation: %s -> %s", args.input, args.output)
    df = preprocess_from_file(args.input, output_path=args.output, filter_moscow=True)
    logger.info("Preparation completed: %s rows", len(df))


if __name__ == "__main__":
    main()
