"""Check syntax, JSON and reported outcome counts using only the standard library."""
import ast
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    count = 0
    for folder in ('scripts', 'tools', 'third_party'):
        for path in (root / folder).rglob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
            count += 1
    for folder in ('configs', 'data', 'results', 'media'):
        for path in (root / folder).rglob('*.json'):
            json.loads(path.read_text(encoding='utf-8'))
    result_root = root / 'results/blockbox-ab-formal-20261117'
    audit = json.loads((result_root / 'audit.json').read_text(encoding='utf-8'))
    report = json.loads((result_root / 'report.json').read_text(encoding='utf-8'))
    rows = audit['scored_trials']
    if audit['errors'] or len(rows) != 40:
        raise SystemExit('Unexpected audit errors or number of scored trials')
    for duration in (30, 60):
        for variant in ('base', 'lora'):
            group = [r for r in rows if r['duration_s'] == duration and r['variant'] == variant]
            observed = {'successes': sum(r['outcome'] == 'success' for r in group), 'scored_trials': len(group)}
            if observed != report['by_duration_s'][str(duration)][variant]:
                raise SystemExit(f'Outcome count mismatch: {duration}, {variant}')
    print(f'OK: {count} Python files, JSON artifacts, and 40 scored trial counts')


if __name__ == '__main__':
    main()
