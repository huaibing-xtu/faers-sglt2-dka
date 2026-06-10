from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from .utils import ensure_dir, normalize_quarter, quarter_range

FDA_FAERS_QDE_URL = "https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html"


def discover_ascii_links(index_url: str = FDA_FAERS_QDE_URL) -> dict[str, str]:
    """
    Scrape the FDA quarterly extract page and return {YYYYQn: zip_url} for ASCII files.
    FDA link names have varied in case over time, so discovery is safer than hard-coding.
    """
    resp = requests.get(index_url, timeout=60)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    links = {}
    for a in soup.find_all("a"):
        text = " ".join(a.get_text(" ", strip=True).split())
        href = a.get("href", "")
        if "ASCII" not in text.upper() and "faers_ascii" not in href.lower():
            continue
        url = urljoin(index_url, href)
        m = re.search(r"faers_ascii_(20\d{2})[Qq]([1-4])\.zip", url)
        if not m:
            continue
        quarter = f"{m.group(1)}Q{m.group(2)}"
        links[quarter] = url
    if not links:
        raise RuntimeError("No FAERS ASCII links discovered. The FDA page format may have changed.")
    return links


def download_file(url: str, out_path: Path, overwrite: bool = False, chunk_size: int = 1024 * 1024, max_retries: int = 3) -> Path:
    ensure_dir(out_path.parent)
    if out_path.exists() and out_path.stat().st_size > 0 and not overwrite:
        print(f"[skip] {out_path.name} exists")
        return out_path

    tmp = out_path.with_suffix(out_path.suffix + ".part")
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[download] {url} (attempt {attempt}/{max_retries})")
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
            tmp.rename(out_path)
            return out_path
        except Exception as exc:
            print(f"[warn] Attempt {attempt} failed: {exc}")
            if attempt == max_retries:
                raise
            import time
            time.sleep(5)
    return out_path


def download_quarters(start: str, end: str, raw_dir: str | Path, overwrite: bool = False) -> list[Path]:
    start = normalize_quarter(start)
    end = normalize_quarter(end)
    raw_dir = ensure_dir(raw_dir)

    links = discover_ascii_links()
    paths = []
    missing = []
    for q in quarter_range(start, end):
        if q not in links:
            missing.append(q)
            continue
        out_path = raw_dir / f"faers_ascii_{q}.zip"
        paths.append(download_file(links[q], out_path, overwrite=overwrite))

    if missing:
        raise RuntimeError(f"Missing download links for quarters: {missing}")
    return paths
