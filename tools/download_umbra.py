#!/usr/bin/env python3
"""Download an explicit manifest of public Umbra assets over HTTPS."""

import argparse
import csv
import hashlib
import time
import urllib.parse
import urllib.request
from pathlib import Path


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def download(url: str, destination: Path, retries: int) -> None:
    temporary = destination.with_suffix(destination.suffix + '.part')
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url, headers={'User-Agent': 'sar-gfm-bench/1.0'})
            with urllib.request.urlopen(request, timeout=60) as response, \
                    temporary.open('wb') as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
            temporary.replace(destination)
            return
        except Exception:
            if attempt == retries:
                raise
            time.sleep(2 ** (attempt - 1))


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Download public Umbra assets listed in a CSV manifest.')
    parser.add_argument('manifest', type=Path,
                        help='CSV with url,filename and optional sha256 columns')
    parser.add_argument('output', type=Path)
    parser.add_argument('--retries', type=int, default=3)
    args = parser.parse_args()

    with args.manifest.open(newline='') as handle:
        reader = csv.DictReader(
            line for line in handle if not line.lstrip().startswith('#'))
        if not reader.fieldnames or not {'url', 'filename'} <= set(reader.fieldnames):
            parser.error('manifest must have url,filename columns')
        rows = list(reader)

    for index, row in enumerate(rows, 1):
        url = row['url'].strip()
        filename = row['filename'].strip()
        expected = row.get('sha256', '').strip().lower()
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != 'https':
            parser.error(f'row {index}: only HTTPS URLs are accepted')
        destination = (args.output / filename).resolve()
        try:
            destination.relative_to(args.output.resolve())
        except ValueError:
            parser.error(f'row {index}: filename escapes output directory')

        if destination.exists() and (not expected or digest(destination) == expected):
            print(f'[{index}/{len(rows)}] already verified: {filename}')
            continue
        print(f'[{index}/{len(rows)}] downloading: {filename}')
        download(url, destination, args.retries)
        actual = digest(destination)
        if expected and actual != expected:
            destination.unlink()
            raise RuntimeError(
                f'{filename}: SHA-256 mismatch; expected {expected}, got {actual}')
        print(f'  sha256={actual}')

    print(f'downloaded/verified {len(rows)} asset(s) under {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
