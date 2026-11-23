"""Download 實價登錄 season batch files (all cities, CSV) into data/raw/.

    python scripts/download_data.py                 # 111S1 .. 115S3 (default)
    python scripts/download_data.py --start 112S1 --end 115S2

Source: 內政部不動產交易實價查詢服務網 https://plvr.land.moi.gov.tw/DownloadOpenData
(非本期下載), Open Government Data License v1.0.

Notes
* Each season is one zip of ~15 MB holding CSVs for every city.
* A season that is not published yet comes back as a small HTML page with
  HTTP 200, so every download is checked for the zip signature ("PK").
* Already-downloaded seasons are skipped.  Requests are spaced out politely.
* If the URL pattern ever changes, download the files by hand from the page
  above and drop them into data/raw/ (any file names are fine).
"""
from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
URL = "https://plvr.land.moi.gov.tw/DownloadSeason?season={season}&type=zip&fileName=lvr_landcsv.zip"


def seasons(start: str, end: str) -> list[str]:
    def parse(s):
        y, q = s.upper().split("S")
        return int(y), int(q)

    (y, q), (y_end, q_end) = parse(start), parse(end)
    out = []
    while (y, q) <= (y_end, q_end):
        out.append(f"{y}S{q}")
        y, q = (y + 1, 1) if q == 4 else (y, q + 1)
    return out


def download(season: str, dest: Path, retries: int = 3) -> str:
    req = urllib.request.Request(URL.format(season=season), headers={"User-Agent": "Mozilla/5.0 (research script)"})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            break
        except Exception as exc:  # network hiccup -> retry
            if attempt == retries:
                return f"failed ({exc})"
            time.sleep(5 * attempt)
    if data[:2] != b"PK":
        return f"not published yet / not a zip ({len(data)} bytes) – skipped"
    dest.write_bytes(data)
    return f"ok ({len(data) / 1e6:.1f} MB)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="111S1")
    ap.add_argument("--end", default="115S3")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    for s in seasons(args.start, args.end):
        dest = RAW / f"{s}.zip"
        if dest.exists() and dest.read_bytes()[:2] == b"PK":
            print(f"{s}: already downloaded")
            continue
        print(f"{s}: {download(s, dest)}", flush=True)
        time.sleep(2)


if __name__ == "__main__":
    main()
