import os
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
 
 
def build_urls(filename, num_downloads):
    with open(filename, "r") as f:
        data = json.load(f)
    paths = data["relpaths"][:num_downloads] if num_downloads else data["relpaths"]
    urls = []
    for path in paths:
        mp4 = data["basedir"] + path
        urls.extend([mp4, mp4[:-3] + "jsonl"])
    return urls
 
 
def download_aria2c(urls, output, workers):
    """Use aria2c with system DNS resolver (--async-dns=false)."""
    url_file = Path(output) / "_urls.txt"
    with open(url_file, "w") as f:
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
        "--async-dns=false",        # use system DNS — fixes resolution failures
        "--console-log-level=warn",
        "--summary-interval=0",
    ]
 
    print(f"[aria2c] Downloading {len(urls)} files "
          f"({workers} concurrent, 8 segments/file)...")
    result = subprocess.run(cmd)
    url_file.unlink(missing_ok=True)
 
    if result.returncode != 0:
        print(f"[aria2c] Finished with exit code {result.returncode}. "
              "Some downloads may have failed — re-run to resume.")
 
 
def main():
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
