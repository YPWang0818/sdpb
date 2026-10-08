#!/usr/bin/env python3
"""Run sdpb on a matrix of problems x rank counts x precisions with profiling on,
collect the per-rank profiles and summarise them.

Example:
  python3 scripts/profile/run_matrix.py --ranks 2,6 --precision 768 \\
      --max-iterations 20 --problem singlet --problem dfibo \\
      --problem mine=/path/to/pmp.json --out ~/sdpb-profiles/2026-10-08 --analyze

A problem is a built-in name (singlet, dfibo: the end-to-end test problems),
or NAME=PATH where PATH is an sdp directory/zip (output of pmp2sdp) or a PMP
input file (pmp.json, pmp.m, pmp.xml, pmp.nsv), which is converted once with
pmp2sdp into <out>/sdp_cache/.
"""
import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

E2E = os.path.join(REPO, 'test', 'data', 'end-to-end_tests')
BUILTIN = {
    'singlet': {
        'sdp': os.path.join(E2E, 'SingletScalar_cT_test_nmax6', 'primal_dual_optimal',
                            'output', 'sdp'),
        'args': '--dualityGapThreshold 1.0e-30 --primalErrorThreshold 1.0e-30 '
                '--dualErrorThreshold 1.0e-30 --initialMatrixScalePrimal 1.0e20 '
                '--initialMatrixScaleDual 1.0e20 --feasibleCenteringParameter 0.1 '
                '--infeasibleCenteringParameter 0.3 --stepLengthReduction 0.7 '
                '--maxComplementarity 1.0e100 --procGranularity 1',
    },
    'dfibo': {
        'sdp': os.path.join(E2E, 'dfibo-0-0-j=3-c=3.0000-d=3-s=6', 'output', 'sdp'),
        'args': '--findDualFeasible --findPrimalFeasible --initialMatrixScalePrimal 1e10 '
                '--initialMatrixScaleDual 1e10 --maxComplementarity 1e30 '
                '--dualErrorThreshold 1e-10 --primalErrorThreshold 1e-153 '
                '--feasibleCenteringParameter=0.1 --infeasibleCenteringParameter=0.3 '
                '--stepLengthReduction=0.7 --maxSharedMemory=100K',
    },
}
PMP_EXT = ('.json', '.m', '.xml', '.nsv')


def is_sdp(path):
    if os.path.isdir(path):
        return os.path.exists(os.path.join(path, 'control.json'))
    return path.endswith('.zip')


def run_logged(cmd, cwd, stdout_path, time_path=None):
    full = cmd
    if time_path and shutil.which('/usr/bin/time'):
        full = ['/usr/bin/time', '-v', '-o', time_path] + cmd
    with open(stdout_path, 'w') as f:
        t0 = time.time()
        rc = subprocess.call(full, cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
    return rc, time.time() - t0


def parse_time_v(path):
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path):
        if 'Elapsed (wall clock)' in line:
            out['elapsed'] = line.split(':', 1)[1].strip().split(': ')[-1]
        elif 'Maximum resident set size' in line:
            out['max_rss_kb'] = int(line.rsplit(':', 1)[1])
    return out


def ensure_sdp(name, path, args, out_dir):
    """Return an sdp path for `path`, converting PMP inputs with pmp2sdp."""
    if is_sdp(path):
        return path
    cache = os.path.join(out_dir, 'sdp_cache', f'{name}_p{args.precision}')
    if os.path.exists(os.path.join(cache, 'control.json')):
        return cache
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    cmd = shlex.split(args.mpirun) + ['-n', str(args.pmp2sdp_ranks), args.pmp2sdp,
                                      '--input', path, '--output', cache,
                                      '--precision', str(args.precision)]
    print('converting', name, 'with pmp2sdp ...', flush=True)
    rc, _ = run_logged(cmd, REPO, os.path.join(out_dir, 'sdp_cache', f'{name}.pmp2sdp.log'))
    if rc != 0:
        raise SystemExit(f'pmp2sdp failed for {name}, see sdp_cache/{name}.pmp2sdp.log')
    return cache


def read_profile_meta(run_dir):
    p = os.path.join(run_dir, 'ck.profiling', 'profiling.0')
    if not os.path.exists(p):
        return {}
    try:
        with open(p) as f:
            doc = json.load(f)
    except ValueError:
        return {'legacy': True}
    meta = doc.get('meta', {})
    out = {'precision_bits': doc.get('precision_bits'), 'num_ranks': doc.get('num_ranks'),
           'N': meta.get('N'), 'P': meta.get('P'),
           'num_blocks': len(meta.get('blocks', [])),
           'num_primes': meta.get('bigint_syrk', {}).get('num_primes')}
    blocks = meta.get('blocks', [])
    if blocks:
        out['dims'] = sorted({b['dim'] for b in blocks})
        out['num_points'] = [min(b['num_points'] for b in blocks),
                             max(b['num_points'] for b in blocks)]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sdpb', default=os.path.join(REPO, 'build', 'sdpb'))
    ap.add_argument('--pmp2sdp', default=os.path.join(REPO, 'build', 'pmp2sdp'))
    ap.add_argument('--mpirun', default='mpirun --oversubscribe')
    ap.add_argument('--pmp2sdp-ranks', type=int, default=2)
    ap.add_argument('--ranks', default='2', help='comma-separated rank counts')
    ap.add_argument('--precision', type=int, default=768)
    ap.add_argument('--max-iterations', type=int, default=20)
    ap.add_argument('--profile-detail', type=int, default=1)
    ap.add_argument('--problem', action='append', default=[],
                    help='built-in name or NAME=PATH (repeatable)')
    ap.add_argument('--problem-args', action='append', default=[],
                    help='NAME="extra sdpb args" for a problem (repeatable)')
    ap.add_argument('--extra-args', default='', help='extra sdpb args for every run')
    ap.add_argument('--out', required=True)
    ap.add_argument('--analyze', action='store_true',
                    help='run analyze_profile on every run and write summary.md')
    ap.add_argument('--skip-first', type=int, default=1)
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    extra_per_problem = {}
    for item in args.problem_args:
        name, _, val = item.partition('=')
        extra_per_problem[name] = val
    problems = []
    for item in args.problem or ['singlet']:
        name, _, path = item.partition('=')
        if not path:
            if name not in BUILTIN:
                raise SystemExit(f'unknown built-in problem {name!r}; use NAME=PATH')
            path, pargs = BUILTIN[name]['sdp'], BUILTIN[name]['args']
        else:
            pargs = ''
        pargs = (pargs + ' ' + extra_per_problem.get(name, '')).strip()
        problems.append((name, os.path.abspath(os.path.expanduser(path)), pargs))

    manifest_path = os.path.join(args.out, 'manifest.json')
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else []
    for name, path, pargs in problems:
        sdp = ensure_sdp(name, path, args, args.out)
        for ranks in [int(r) for r in args.ranks.split(',')]:
            run_name = f'{name}_r{ranks}_p{args.precision}'
            run_dir = os.path.join(args.out, run_name)
            if os.path.exists(run_dir):
                shutil.rmtree(run_dir)
            os.makedirs(run_dir)
            cmd = shlex.split(args.mpirun) + ['-n', str(ranks), args.sdpb,
                                              '--sdpDir', sdp,
                                              '--outDir', os.path.join(run_dir, 'out'),
                                              '--checkpointDir', os.path.join(run_dir, 'ck'),
                                              '--precision', str(args.precision),
                                              '--maxIterations', str(args.max_iterations),
                                              '--verbosity', '2',
                                              '--profileDetail', str(args.profile_detail),
                                              '--checkpointInterval', '1000000',
                                              '--noFinalCheckpoint']
            cmd += shlex.split(pargs) + shlex.split(args.extra_args)
            with open(os.path.join(run_dir, 'cmd.txt'), 'w') as f:
                f.write(' '.join(shlex.quote(c) for c in cmd) + '\n')
            print(f'running {run_name} ...', flush=True)
            rc, wall = run_logged(cmd, REPO, os.path.join(run_dir, 'stdout.log'),
                                  os.path.join(run_dir, 'time.txt'))
            entry = {'run': run_name, 'problem': name, 'sdp': sdp, 'ranks': ranks,
                     'precision': args.precision, 'exit_code': rc, 'wall_s': round(wall, 2)}
            entry.update(parse_time_v(os.path.join(run_dir, 'time.txt')))
            entry.update(read_profile_meta(run_dir))
            manifest = [m for m in manifest if m.get('run') != run_name] + [entry]
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=1)
            print(f'  exit={rc} wall={wall:.1f}s', flush=True)

    if args.analyze:
        from sdpb_profile.reader import load_run
        from sdpb_profile.stats import compute_run_stats
        from sdpb_profile import report
        lines = ['# Profiling summary', '']
        shape_rows = []
        for m in manifest:
            run_dir = os.path.join(args.out, m['run'])
            try:
                profiles = load_run(run_dir)
                st = compute_run_stats(profiles, run_dir, skip_first=args.skip_first)
            except Exception as e:  # noqa: BLE001
                lines += [f'## {m["run"]}', '', f'no profile: {e}', '']
                continue
            lines += [f'## {m["run"]}', ''] + report.run_summary_lines(st) + \
                     report.crosscheck_lines(st, run_dir) + ['', report.phase1_table(st), '']
            top = sorted(st.categories, key=lambda c: -st.mean_over_ranks(c))[:6]
            shape_rows.append([m['run'], m.get('num_blocks', '?'), m.get('N', '?'),
                               f'{st.mean_iter / 1e9:.3f}'] +
                              [f'{c} {100 * st.mean_over_ranks(c) / st.mean_iter:.0f}%'
                               for c in top])
        if shape_rows:
            width = max(len(r) for r in shape_rows)
            headers = ['run', 'blocks', 'N', 's/iter'] + [f'top {i + 1}' for i in range(width - 4)]
            lines = lines[:2] + ['## Shape vs share', '',
                                 report.md_table(headers, [r + [''] * (width - len(r))
                                                           for r in shape_rows],
                                                 ['---'] * width), ''] + lines[2:]
        with open(os.path.join(args.out, 'summary.md'), 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print('wrote', os.path.join(args.out, 'summary.md'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
