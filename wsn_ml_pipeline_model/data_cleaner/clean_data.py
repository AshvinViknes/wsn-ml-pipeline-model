# File: wsn_ml_pipeline_model/data_cleaner/clean_data.py
# This script cleans raw sensor data from text files, extracting relevant fields and saving them in a structured CSV format.
# It handles parsing errors gracefully and logs the number of valid lines processed.

import os
import csv
import logging
from typing import Optional, Tuple, List
from wsn_ml_pipeline_model.config.constants import DATA_COLUMNS


class DataCleaner:
    """
    A class to clean raw sensor data from text files, extracting relevant fields
    and saving them in a structured CSV format.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def parse_line(self, line: str) -> Optional[Tuple[int, int]]:
        """
        Parse one line of sensor data from a legacy TXT file.

        Args:
            line: raw line string e.g. "2025-07-03 20:07:24,891 # -62,124"

        Returns:
            (rssi, lqi) if successful, else None.
        """
        try:
            line = line.strip()
            if not line:
                return None
            if "#" in line:
                _, data_str = line.split("#", 1)
                data_str = data_str.strip()
            else:
                data_str = line  
            rssi_str, lqi_str = [p.strip() for p in data_str.split(",", 1)]
            rssi = int(rssi_str)
            lqi = int(lqi_str)
            return rssi, lqi
        except Exception as e:
            self.logger.warning(f"Failed to parse line: {line!r} - {e}")
            return None

    def clean_file(self, input_path: str, output_path: str) -> int:
        """
        Clean raw txt and save as 2-col CSV (rssi,lqi).

        Args:
            input_path: .txt (legacy) 
            output_path: Path to output CSV file.

        Returns:
            Number of valid rows written.
        """
        valid_rows: List[Tuple[int, int]] = []

        with open(input_path, "r", encoding="utf-8") as fin:
            for line_num, line in enumerate(fin, start=1):
                parsed = self.parse_line(line)
                if parsed:
                    valid_rows.append(parsed)
                else:
                    self.logger.debug(f"Skipping invalid line #{line_num}")

        if not valid_rows:
            self.logger.error(f"No valid data found in: {input_path}")
            return 0

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as fout:
            writer = csv.writer(fout)
            writer.writerow(DATA_COLUMNS) 
            writer.writerows(valid_rows)

        self.logger.info(
            f"Cleaned '{input_path}' → '{output_path}', rows written: {len(valid_rows)}"
        )
        return len(valid_rows)
