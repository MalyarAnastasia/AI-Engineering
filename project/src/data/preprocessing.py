import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MOSCOW_REGION_CODES = [
    "77", "97", "99", "177", "197", "199",
    "777", "797", "799", "50", "81",
]

REGION_MAPPING = {
    11416: "01", 6817: "04", 7929: "28", 6543: "29", 10582: "30",
    2722: "02", 5952: "31", 821: "32", 9579: "03", 2885: "20",
    5282: "74", 10160: "75", 3019: "21", 2661: "78", 6937: "82",
    4007: "05", 6309: "04", 13913: "06", 5368: "38", 8894: "37",
    9648: "07", 7896: "39", 8509: "08", 2328: "40", 13098: "41",
    11991: "09", 8090: "10", 2860: "42", 4086: "27", 2359: "19",
    2484: "86", 2594: "43", 4417: "11", 4189: "44", 2843: "23",
    3870: "24", 5703: "45", 7121: "46", 3446: "47", 4240: "48",
    16705: "49", 4982: "12", 5241: "13", 81: "50", 3: "77",
    14368: "51", 61888: "83", 2871: "52", 13919: "15", 5736: "53",
    9654: "54", 8640: "55", 2814: "57", 3153: "56", 5993: "58",
    5520: "59", 4963: "25", 7793: "60", 3230: "61", 1491: "62",
    11171: "14", 1901: "65", 3106: "63", 2528: "64", 7873: "92",
    5794: "67", 2900: "26", 6171: "66", 2806: "68", 2922: "16",
    10201: "70", 5143: "71", 5178: "17", 2880: "69", 3991: "72",
    1010: "18", 4249: "73", 5789: "33", 4695: "34", 4374: "35",
    2072: "36", 14880: "89", 2604: "76", 69: "79",
}


class EstatePreprocessing:
    """Пайплайн очистки данных, перенесённый из real_estate_analytics."""

    def __init__(self, file_path: str):
        logger.info("Loading raw data from %s", file_path)
        self.df = pd.read_csv(file_path)

    def cleaning(self) -> pd.DataFrame:
        if "rooms" in self.df.columns:
            studios_count = (self.df["rooms"] == -1).sum()
            self.df["rooms"] = self.df["rooms"].replace(-1, 0)
            if studios_count:
                logger.info("Studios fixed: %s values -1 -> 0", studios_count)

        num_cols_for_neg = self.df.select_dtypes(include=[np.number]).columns.tolist()
        for skip_col in ["date", "region"]:
            if skip_col in num_cols_for_neg:
                num_cols_for_neg.remove(skip_col)

        neg_mask = (self.df[num_cols_for_neg] < 0).any(axis=1)
        neg_count = int(neg_mask.sum())
        if neg_count > 0:
            self.df = self.df[~neg_mask]
            logger.info("Removed rows with negative values: %s", neg_count)

        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        logger.info("Removed duplicate rows: %s", initial_rows - len(self.df))

        if "region" in self.df.columns:
            self.df["region"] = self.df["region"].replace(REGION_MAPPING)
            self.df["region"] = self.df["region"].astype(str)
            logger.info("Region codes normalized to string auto codes")

        num_cols = self.df.select_dtypes(include=["float64"]).columns
        if len(num_cols) > 0:
            self.df = self.df.dropna(subset=num_cols)

        for col in ["price", "geo_lat", "geo_lon", "area"]:
            if col in self.df.columns:
                zeros = int((self.df[col] == 0).sum())
                if zeros > 0:
                    self.df[col] = self.df[col].replace(0, np.nan)
                    logger.info("Column %s: replaced %s zeros with NaN", col, zeros)

        self.df = self.df.dropna(
            subset=[c for c in ["price", "geo_lat", "geo_lon", "area", "kitchen_area", "rooms"] if c in self.df.columns]
        )
        return self.df

    def apply_iqr(self) -> pd.DataFrame:
        initial_len = len(self.df)
        for col in ["price", "area", "kitchen_area"]:
            if col not in self.df.columns:
                continue
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            before_count = len(self.df)
            self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
            logger.info(
                "IQR on %s: removed %s rows, bounds [%.2f, %.2f]",
                col, before_count - len(self.df), lower_bound, upper_bound,
            )

        logger.info("IQR total removed rows: %s", initial_len - len(self.df))
        return self.df

    def filter_moscow_region(self) -> pd.DataFrame:
        if "region" not in self.df.columns:
            logger.warning("Column region not found, Moscow filter skipped")
            return self.df

        initial_len = len(self.df)
        self.df["region"] = self.df["region"].astype(str)
        self.df = self.df[self.df["region"].isin(MOSCOW_REGION_CODES)]
        logger.info("Moscow/MO filter: kept %s rows (was %s)", len(self.df), initial_len)
        return self.df

    def save_data(self, output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        self.df.to_csv(output_path, index=False)
        logger.info("Saved cleaned dataset: %s (%s rows)", output_path, len(self.df))

    def run_full_pipeline(self) -> pd.DataFrame:
        self.cleaning()
        self.apply_iqr()
        self.filter_moscow_region()
        return self.df


def preprocess_from_file(
    input_path: str,
    output_path: str | None = None,
    filter_moscow: bool = True,
) -> pd.DataFrame:
    processor = EstatePreprocessing(input_path)
    processor.cleaning()
    processor.apply_iqr()
    if filter_moscow:
        processor.filter_moscow_region()
    if output_path:
        processor.save_data(output_path)
    return processor.df
