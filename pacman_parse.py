import re
import csv
from datetime import datetime
from pathlib import Path

PACMAN_LOG_PATH = Path("/var/log/pacman.log")
OUTPUT_CSV_PATH = "data/pacman_history.csv"

LOG_PATTERN_OLD = re.compile(
    r'^\[(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2})\] '
    r'\[ALPM\] (?P<action>installed|upgraded|removed|reinstalled) '
    r'(?P<package>[^\s]+) \((?P<version_info>.+)\)$'
)

LOG_PATTERN_NEW = re.compile(
    r'^\[(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{4})\] '
    r'\[ALPM\] (?P<action>installed|upgraded|removed|reinstalled) '
    r'(?P<package>[^\s]+) \((?P<version_info>.+)\)$'
)


def parse_timestamp_old(ts_str):
    try:
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M")
    except ValueError as e:
        print(f"Failed to parse timestamp: {ts_str} | Error: {e}")
        return None

def parse_timestamp_new(ts_str):
    try:
        return datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S%z")
    except ValueError as e:
        print(f"Failed to parse timestamp: {ts_str} | Error: {e}")
        return None

csv_rows = []

if not PACMAN_LOG_PATH.exists():
    print(f"File not found: {PACMAN_LOG_PATH}")
    exit(1)

with PACMAN_LOG_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        match = LOG_PATTERN_OLD.match(line)

        if match:
            ts = parse_timestamp_old(match.group("timestamp"))
        else:
            match = LOG_PATTERN_NEW.match(line)

            if not match:
                # print(f"Skipping unmatched line: {line}")
                continue
            
            ts = parse_timestamp_new(match.group("timestamp"))

        if not ts:
            continue

        action = match.group("action")
        package = match.group("package")
        version_info = match.group("version_info")

        version_before, version_after = "", ""

        if action == "upgraded":
            if " -> " in version_info:
                version_before, version_after = version_info.split(" -> ")
            else:
                print(f"Unexpected upgrade format: {version_info}")
                continue
        elif action == "installed":
            version_after = version_info
        elif action == "removed":
            version_before = version_info
        elif action == "reinstalled":
            version_before = version_info
            version_after = version_info

        csv_rows.append({
            "package": package,
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "action": action,
            "version_before": version_before,
            "version_after": version_after
        })

# Write CSV output
if csv_rows:
    with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["package", "timestamp", "action", "version_before", "version_after"])
        writer.writeheader()
        writer.writerows(csv_rows)
else:
    print("No matching entries found in the log.")
    exit(1)
