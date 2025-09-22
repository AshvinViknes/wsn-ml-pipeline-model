# File: wsn_ml_pipeline_model/data_cleaner/clean_data.py
# This script cleans raw sensor data from text files, extracting relevant fields and saving them in a structured CSV format.
# It handles parsing errors gracefully and logs the number of valid lines processed.
import os
import csv
import logging
from datetime import datetime
from typing import Optional, Tuple, List
from wsn_ml_pipeline_model.config.constants import TIMESTAMP_FORMAT, DATA_COLUMNS

class DataCleaner:
    """
    A class to clean raw sensor data from text files, extracting relevant fields
    and saving them in a structured CSV format.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def parse_line(self, line: str) -> Optional[Tuple[float, int, int]]:
        """
        Parse one line of raw sensor data.

        Args:
            line: raw line string e.g. "2025-07-03 20:07:24,891 # -62,124"

        Returns:
            Tuple of (unix_timestamp, rssi, lqi) if successful, else None.
        """
        try:
            # Be tolerant to extra spaces and multiple separators
            if '#' not in line:
                raise ValueError("Missing '#' separator")
            timestamp_str, data_str = line.split('#', 1)
            timestamp_str = timestamp_str.strip()
            data_str = data_str.strip()

            # Allow both comma- and dot-microseconds (e.g., 12:34:56,789 vs 12:34:56.789)
            try:
                dt = datetime.strptime(timestamp_str, TIMESTAMP_FORMAT)
            except ValueError:
                dt = datetime.strptime(timestamp_str.replace('.', ','), TIMESTAMP_FORMAT)
            unix_time = int(dt.timestamp()) 

            parts = [p for p in data_str.replace('\t', ',').split(',') if p != '']
            if len(parts) < 2:
                raise ValueError("Missing RSSI/LQI values")
            rssi_str, lqi_str = parts[0].strip(), parts[1].strip()
            rssi = int(rssi_str)
            lqi = int(lqi_str)

            return unix_time, rssi, lqi

        except Exception as e:
            self.logger.warning(f"Failed to parse line: {line.strip()} - {e}")
            return None

    def clean_file(self, input_path: str, output_path: str) -> int:
        valid_rows: List[Tuple[int, int]] = []
        with open(input_path, 'r', encoding='utf-8') as fin:
            for line_num, line in enumerate(fin, start=1):
                parsed = self.parse_line(line)
                if parsed:
                    # parsed = (unix_time, rssi, lqi) → keep only (rssi, lqi)
                    _, rssi, lqi = parsed
                    valid_rows.append((rssi, lqi))
                else:
                    self.logger.debug(f"Skipping invalid line #{line_num}")

        if not valid_rows:
            self.logger.error(f"No valid data found in file: {input_path}")
            return 0

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', newline='', encoding='utf-8') as fout:
            writer = csv.writer(fout)
            writer.writerow(DATA_COLUMNS)
            writer.writerows(valid_rows)

        self.logger.info(f"Cleaned file '{input_path}' → '{output_path}', rows written: {len(valid_rows)}")
        return len(valid_rows)

