"""Bulk downloader for the OpenAI VPT Minecraft dataset.

Reads a VPT JSON index file (``{"basedir": ..., "relpaths": [...]}``) and
downloads every ``.mp4`` recording plus its matching ``.jsonl`` action log
into a target directory. Uses ``aria2c`` when available for multi-segment
parallel downloads; falls back to a warning otherwise.

Typical usage:
    python download_vpt.py --input_path 8xx_Jun_29.json \\
        --output_path ./data/vpt-recordings --num_downloads 100 --workers 8
"""
import os
import json
import shutil
import argparse
import subprocess
from pathlib import Path


def build_urls(filename, num_downloads):
    """Build the list of URLs to fetch from a VPT index JSON.

    Args:
        filename: Path to a VPT index file. Must contain ``basedir`` (a
            URL prefix) and ``relpaths`` (a list of relative paths).
        num_downloads: Number of ``(mp4, jsonl)`` pairs to take from the
            front of ``relpaths``. Pass ``0`` / falsy for "all".

    Returns:
        list[str]: URLs in ``[mp4, jsonl, mp4, jsonl, ...]`` order —
        twice ``num_downloads`` entries when truncated.
    """
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    paths = data["relpaths"][:num_downloads] if num_downloads else data["relpaths"]
    urls = []
    for path in paths:
        mp4 = data["basedir"] + path
        urls.extend([mp4, mp4[:-3] + "jsonl"])
    return urls


def download_aria2c(urls, output, workers):
    """Fetch ``urls`` into ``output`` using aria2c.

    Writes a temporary ``_urls.txt`` manifest (one URL per entry with a
    per-entry ``dir=`` override) and invokes ``aria2c`` with ``--async-dns=false``
    so that the system resolver is used — this fixes intermittent DNS
    resolution failures against the VPT CDN seen on university networks.

    Args:
        urls: List of URLs to download.
        output: Target directory (created by the caller).
        workers: Max concurrent downloads. aria2c also splits each file
            into 8 connections internally.

    Returns:
        None. Prints a warning when aria2c exits non-zero so the caller
        can re-run to resume.
    """
    url_file = Path(output) / "_urls.txt"
    with open(url_file, "w", encoding="utf-8") as f:
        for url in urls:
            f.write(url + f"\n  dir={output}\n")

    cmd = [
        "aria2c",
        f"--input-file={url_file}",
        f"--max-concurrent-downloads={workers}",
        "--split=8",
        "--max-connection-per-server=8",
        "--min-split-size=5M",
        "--continue=true",
        "--max-tries=5",
        "--retry-wait=3",
        "--async-dns=false",  # use system DNS — fixes resolution failures
        "--console-log-level=warn",
        "--summary-interval=0",
    ]

    print(f"[aria2c] Downloading {len(urls)} files "
          f"({workers} concurrent, 8 segments/file)...")
    result = subprocess.run(cmd, check=False)
    url_file.unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"[aria2c] Finished with exit code {result.returncode}. "
              "Some downloads may have failed — re-run to resume.")


def main():
    """CLI entry point: parse arguments and dispatch to aria2c.

    Parses ``--input_path``, ``--output_path``, ``--num_downloads`` and
    ``--workers``, creates the output directory, builds the URL list
    via :func:`build_urls`, and hands it to :func:`download_aria2c`
    when ``aria2c`` is on ``PATH``. Prints an install hint otherwise.
    """
    p = argparse.ArgumentParser(description="Download VPT Minecraft data")
    p.add_argument("--input_path",    type=str, required=True,
                   help="Path to the JSON index file")
    p.add_argument("--output_path",   type=str, required=True,
                   help="Directory to save downloaded files")
    p.add_argument("--num_downloads", type=int, default=0,
                   help="Number of mp4/jsonl pairs (0 = all)")
    p.add_argument("--workers",       type=int, default=8,
                   help="Parallel download slots (default: 8)")
    args = p.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    urls = build_urls(args.input_path, args.num_downloads)
    print(f"Preparing to download {len(urls)} files into '{args.output_path}'...")

    if shutil.which("aria2c"):
        download_aria2c(urls, args.output_path, args.workers)
        print("Done.")
    else:
        print("[info] aria2c not found. Install aria2c for faster multi-segment downloads.")


if __name__ == "__main__":
    main()
