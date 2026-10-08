"""Command-line interface: python3 analyze_profile.py RUN_DIR [...]"""
import argparse
import json
import os
import sys

from . import amdahl, report
from .reader import load_run
from .stats import compute_run_stats


def parse_iters(text):
    if not text:
        return None
    lo, _, hi = text.partition(':')
    return (int(lo) if lo else 0, int(hi) if hi else 10 ** 9)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description='Analyse SDPB per-rank profiles (ck.profiling/profiling.<rank>).')
    ap.add_argument('run_dir', help='directory containing ck.profiling/ (and out/), '
                                    'or the profiling directory itself')
    ap.add_argument('--skip-first', type=int, default=1,
                    help='ignore the first N iterations (default 1)')
    ap.add_argument('--iters', help='iteration range a:b (inclusive)')
    ap.add_argument('--report', choices=['phase1', 'blocks', 'tree', 'ranks'],
                    default='phase1')
    ap.add_argument('--rank', type=int, default=0, help='rank for --report tree')
    ap.add_argument('--min-share', type=float, default=0.005,
                    help='--report tree: hide entries below this share')
    ap.add_argument('--max-depth', type=int, help='--report tree: max depth')
    ap.add_argument('--md', help='write the markdown report to this file')
    ap.add_argument('--csv', help='write CSV to this file')
    ap.add_argument('--json', help='write machine-readable JSON to this file')
    ap.add_argument('--speedup', nargs='*', metavar='CAT=S',
                    help='what-if: speed up category CAT (or prefix) by S')
    ap.add_argument('--compare', metavar='RUN_DIR',
                    help='side-by-side with another run (A = run_dir, B = this)')
    args = ap.parse_args(argv)

    iters = parse_iters(args.iters)
    profiles = load_run(args.run_dir)
    st = compute_run_stats(profiles, args.run_dir, args.skip_first, iters)
    if st.num_iterations == 0:
        print('no complete iterations found (try --skip-first 0)', file=sys.stderr)
        return 1

    out = []
    out.extend(report.run_summary_lines(st))
    out.extend(report.crosscheck_lines(st, args.run_dir))
    out.append('')
    csv_text = None
    json_obj = report.phase1_json(st)
    if args.report == 'phase1':
        out.append(report.phase1_table(st))
        csv_text = report.phase1_csv(st)
    elif args.report == 'ranks':
        out.append(report.ranks_table(st))
        csv_text = report.phase1_csv(st)
    elif args.report == 'blocks':
        if st.legacy:
            print('blocks report needs the JSON profile format', file=sys.stderr)
            return 1
        md, csv_text, fits = report.blocks_report(profiles, args.skip_first, iters)
        out.append(md)
        json_obj['block_fits'] = fits
    elif args.report == 'tree':
        p = next((p for p in profiles if p.rank == args.rank), profiles[0])
        out.append(report.tree_report(p, args.skip_first, iters, args.min_share,
                                      args.max_depth))
    if args.speedup:
        out.append('')
        out.append(amdahl.amdahl_table(st, amdahl.parse_speedups(args.speedup)))
    if args.compare:
        other = compute_run_stats(load_run(args.compare), args.compare,
                                  args.skip_first, iters)
        out.append('')
        out.append(f'### Compare: A = {args.run_dir}, B = {args.compare}')
        out.append(report.compare_table(st, other))

    text = '\n'.join(out)
    print(text)
    if args.md:
        with open(args.md, 'w') as f:
            f.write(text + '\n')
    if args.csv and csv_text is not None:
        with open(args.csv, 'w') as f:
            f.write(csv_text)
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(json_obj, f, indent=1)
    return 0
