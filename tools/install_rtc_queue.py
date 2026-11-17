"""Explicitly install the recorded RTC queue into a project environment."""
import argparse
import hashlib
from importlib import metadata
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    dist = metadata.distribution('lerobot')
    if dist.version != '0.7.0':
        raise SystemExit('This snapshot was recorded with lerobot==0.7.0')
    target = Path(dist.locate_file('lerobot/policies/rtc/action_queue.py')).resolve()
    source = Path(__file__).resolve().parents[1] / 'third_party/lerobot_rtc/action_queue.py'
    if not target.is_file():
        raise SystemExit('Cannot locate installed queue module')
    print('Source:', source)
    print('Target:', target)
    print('SHA256:', hashlib.sha256(source.read_bytes()).hexdigest())
    if not args.apply:
        print('Inspection only. Pass --apply inside your project virtual environment to install.')
        return
    if source.read_bytes() == target.read_bytes():
        print('Snapshot already installed')
        return
    backup = target.with_suffix('.py.before-pi05-so101')
    if backup.exists():
        raise SystemExit('Backup already exists; inspect it before another replacement')
    shutil.copy2(target, backup)
    shutil.copy2(source, target)
    print('Installed; original module saved at:', backup)


if __name__ == '__main__':
    main()
