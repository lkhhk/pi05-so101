"""Download and CRC32C-verify the official checkpoint; never delete existing data."""
import base64
import concurrent.futures
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request

import google_crc32c

ROOT = Path('/root/pi05-base-deploy/checkpoint')
SPILL = Path('/root/autodl-tmp/pi05-base-deploy-shard')
PREFIX = 'checkpoints/pi05_base/'
MIRROR = 'https://hf-mirror.com/robotgeneralist/openpi_checkpoint_mirrors2/resolve/1fda6b04e83d625eb1a55d68db32050e248cac2b/pi05_base/'


def main():
    url = 'https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix=checkpoints%2Fpi05_base%2F'
    items = json.load(urllib.request.urlopen(url, timeout=30))['items']
    ROOT.mkdir(parents=True, exist_ok=True)
    SPILL.mkdir(parents=True, exist_ok=True)
    # Place one large shard on the data disk to leave space on the system disk.
    spill_item = max(items, key=lambda x: int(x['size']))
    def remaining(item, path):
        return max(0, int(item['size']) - (path.stat().st_size if path.exists() else 0))
    required_root = sum(remaining(x, ROOT / x['name'].removeprefix(PREFIX))
                        for x in items if x is not spill_item)
    if shutil.disk_usage(ROOT).free < required_root + 500_000_000:
        raise RuntimeError('Insufficient system disk space; no files removed')
    if shutil.disk_usage(SPILL).free < remaining(spill_item, SPILL / Path(spill_item['name']).name) + 100_000_000:
        raise RuntimeError('Insufficient data disk space; no files removed')
    (ROOT.parent / 'official-manifest.json').write_text(json.dumps(items, indent=2))

    def get(item):
        rel = item['name'].removeprefix(PREFIX)
        dest = ROOT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if item is spill_item:
            physical = SPILL / Path(rel).name
            if not dest.exists() and not dest.is_symlink():
                dest.symlink_to(physical)
        else:
            physical = dest
        size = int(item['size'])
        control_file = physical.with_name(physical.name + '.aria2')
        if not physical.exists() or physical.stat().st_size != size or control_file.exists():
            # Mirror is a transport only: every byte is checked against Google's manifest.
            sources = ([MIRROR + rel] if size > 1_000_000 else [])
            sources.append('https://storage.googleapis.com/openpi-assets/' + item['name'])
            for source in sources:
                print('Downloading', rel, 'bytes=', size, flush=True)
                command = (['aria2c', '--continue=true', '--auto-file-renaming=false',
                            '--allow-overwrite=true', '--file-allocation=none', '-x', '8',
                            '-s', '8', '-k', '16M', '--max-tries=3', '--retry-wait=3',
                            '--connect-timeout=20', '--timeout=60', '--summary-interval=0',
                            '--console-log-level=warn', '--download-result=hide',
                            '--dir', str(physical.parent), '--out', physical.name, source]
                           if shutil.which('aria2c') else
                           ['curl', '-fL', '--connect-timeout', '20', '--max-time', '1800',
                            '--retry', '2', '-C', '-', '-o', str(physical), source])
                result = subprocess.run(command,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if result.returncode == 0 and physical.stat().st_size == size:
                    break
            else:
                raise RuntimeError('Download incomplete: ' + rel)
        checksum = google_crc32c.Checksum()
        with physical.open('rb') as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
                checksum.update(block)
        if base64.b64encode(checksum.digest()).decode() != item['crc32c']:
            raise RuntimeError('CRC32C mismatch: ' + rel)
        print('Verified', rel, flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(get, items))
    (ROOT.parent / 'VERIFIED').write_text('All files match official Google CRC32C manifest.\n')
    print('ALL VERIFIED', flush=True)


if __name__ == '__main__':
    main()
