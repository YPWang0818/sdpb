"""Text/CSV/JSON reports."""
import csv
import io
import json
import math
from collections import OrderedDict, defaultdict

from .categories import sort_categories
from .stats import RunStats, load_iterations_json
from .tree import build_tree, complete_iterations

NS = 1e9


def fmt_s(ns):
    return f'{ns / NS:.4f}'


def fmt_pct(x):
    return f'{100 * x:5.1f}%'


def md_table(headers, rows, align=None):
    out = ['| ' + ' | '.join(headers) + ' |']
    if align is None:
        align = ['---'] + ['---:'] * (len(headers) - 1)
    out.append('| ' + ' | '.join(align) + ' |')
    for r in rows:
        out.append('| ' + ' | '.join(str(x) for x in r) + ' |')
    return '\n'.join(out)


def run_summary_lines(st: RunStats):
    h, m = st.header, st.meta
    lines = []
    lines.append(f'run: {st.run_dir}')
    if st.legacy:
        lines.append('format: legacy (coarse timers, no attributes)')
    else:
        lines.append(f'sdpb {h.get("sdpb_version", "?")}, ranks={h.get("num_ranks")}, '
                     f'nodes={h.get("num_nodes")}, precision={h.get("precision_bits")} bits, '
                     f'run_kind={m.get("run_kind", "?")}')
        if 'blocks' in m:
            blocks = m['blocks']
            dims = sorted({b['dim'] for b in blocks})
            pts = [b['num_points'] for b in blocks]
            lines.append(f'blocks={len(blocks)}, dim in {dims}, num_points {min(pts)}..{max(pts)}, '
                         f'N={m.get("N")}, P={m.get("P")}, '
                         f'num_primes={m.get("bigint_syrk", {}).get("num_primes")}')
    lines.append(f'iterations analysed: {st.num_iterations} '
                 f'({st.ranks[0].iterations[0] if st.num_iterations else "-"}..'
                 f'{st.ranks[0].iterations[-1] if st.num_iterations else "-"}), '
                 f'mean iteration time {fmt_s(st.mean_iter)} s '
                 f'(critical rank {st.critical_rank.rank}: {fmt_s(st.critical_rank.mean_iter)} s)')
    return lines


def phase1_table(st: RunStats):
    """The Phase 1 deliverable: share of iteration time per category."""
    crit = st.critical_rank
    rows = []
    for c in st.categories:
        mean = st.mean_over_ranks(c)
        rows.append([c, fmt_s(mean), fmt_pct(mean / st.mean_iter if st.mean_iter else 0),
                     fmt_s(st.min_over_ranks(c)), fmt_s(st.max_over_ranks(c)),
                     fmt_pct(crit.mean.get(c, 0.0) / crit.mean_iter if crit.mean_iter else 0)])
    rows.append(['**iteration**', fmt_s(st.mean_iter), fmt_pct(1.0),
                 fmt_s(min(r.mean_iter for r in st.ranks)),
                 fmt_s(max(r.mean_iter for r in st.ranks)), fmt_pct(1.0)])
    return md_table(['category', 'mean s/iter', 'share', 'min rank', 'max rank',
                     'critical-rank share'], rows)


def phase1_csv(st: RunStats):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(['category', 'rank', 'mean_s', 'min_s', 'max_s', 'share'])
    for r in st.ranks:
        for c in st.categories:
            vals = [d.get(c, 0) for d in r.per_iter] or [0]
            w.writerow([c, r.rank, sum(vals) / len(vals) / NS, min(vals) / NS,
                        max(vals) / NS,
                        (sum(vals) / len(vals)) / r.mean_iter if r.mean_iter else 0])
    return buf.getvalue()


def phase1_json(st: RunStats):
    return {
        'run_dir': st.run_dir,
        'header': st.header,
        'legacy': st.legacy,
        'num_iterations': st.num_iterations,
        'mean_iter_s': st.mean_iter / NS,
        'critical_rank': st.critical_rank.rank,
        'categories': {c: {'mean_s': st.mean_over_ranks(c) / NS,
                           'min_s': st.min_over_ranks(c) / NS,
                           'max_s': st.max_over_ranks(c) / NS,
                           'share': st.mean_over_ranks(c) / st.mean_iter if st.mean_iter else 0}
                       for c in st.categories},
        'ranks': {r.rank: {'mean_iter_s': r.mean_iter / NS,
                           'categories': {c: v / NS for c, v in r.mean.items()}}
                  for r in st.ranks},
    }


def ranks_table(st: RunStats):
    rows = []
    for r in st.ranks:
        rows.append([r.rank, fmt_s(r.mean_iter), fmt_s(r.mean.get('mpi', 0.0)),
                     fmt_pct(r.mean.get('mpi', 0.0) / r.mean_iter if r.mean_iter else 0)])
    return md_table(['rank', 'mean s/iter', 'mpi s/iter', 'mpi share'], rows)


def crosscheck_lines(st: RunStats, run_dir):
    """Compare the profile's per-iteration time with out/iterations.json."""
    ref = load_iterations_json(run_dir)
    if not ref or not st.ranks:
        return []
    r0 = st.ranks[0]
    diffs = []
    for k, t in zip(r0.iterations, r0.iter_times):
        if k in ref:
            diffs.append(abs(t / NS - ref[k]))
    if not diffs:
        return []
    return [f'cross-check vs iterations.json (rank 0, {len(diffs)} iterations): '
            f'max |profile - iter_time| = {max(diffs):.4f} s']


def tree_report(profile, skip_first=1, iters=None, min_share=0.005, max_depth=None):
    """Average tree of the complete iterations of one rank, as indented text."""
    roots, _ = build_tree(profile)
    its = complete_iterations(roots, skip_first=skip_first, iters=iters)
    if not its:
        return 'no complete iterations'
    n = len(its)
    # aggregate by path (names joined), summing elapsed and counts
    agg = OrderedDict()
    for _, it in its:
        for node in it.walk():
            key = node.path(stop_at=it.parent)
            a = agg.setdefault(key, {'elapsed': 0, 'count': 0, 'depth': node.entry.depth,
                                     'kind': node.attrs.get('kind', '')})
            a['elapsed'] += node.elapsed
            a['count'] += node.entry.count
    total = sum(it.elapsed for _, it in its)
    lines = [f'rank {profile.rank}: mean over {n} iterations; '
             f'entries below {100 * min_share:.1f}% of the iteration are hidden']
    base_depth = its[0][1].entry.depth
    for key, a in agg.items():
        share = a['elapsed'] / total if total else 0
        depth = a['depth'] - base_depth
        if share < min_share or (max_depth is not None and depth > max_depth):
            continue
        name = key.split('/')[-1]
        kind = f' [{a["kind"]}]' if a['kind'] else ''
        lines.append(f'{"  " * depth}{name}{kind}: {a["elapsed"] / n / NS:.4f} s '
                     f'({100 * share:.1f}%, x{a["count"] / n:.0f})')
    return '\n'.join(lines)


# Kernels reported per block: (column, path regex relative to iter)
BLOCK_KERNELS = [
    ('cholesky_S', r'^step/initializeSchurComplementSolver/Q/cholesky$'),
    ('solve_P', r'^step/initializeSchurComplementSolver/Q/solve$'),
    ('schur', r'^step/initializeSchurComplementSolver/schur_complement/block$'),
    ('A_X_inv', r'^bilinear_pairings/A_X_inv/block$'),
    ('A_Y', r'^bilinear_pairings/A_Y/block$'),
    ('cholesky_X', r'^cholesky_XY/X$'),
    ('cholesky_Y', r'^cholesky_XY/Y$'),
    ('eig', r'^step/step_length/eig/hermitian_eig$'),
    ('congruence', r'^step/step_length/congruence/block$'),
    ('cholesky_solve', r'^step/search_direction/cholesky_solve_\w+/cholesky_solve$'),
    ('XY_gemm', r'^step/(XY/product|search_direction/\w+_product)/gemm$'),
]


def blocks_report(profiles, skip_first=1, iters=None):
    """Per-block mean time per iteration of the main kernels (summed over
    parities and over the ranks of the block's group), joined with the block
    shapes from meta. Returns (markdown, csv_text, fits)."""
    import re
    kernels = [(col, re.compile(rx)) for col, rx in BLOCK_KERNELS]
    meta = profiles[0].meta
    blocks = {b['index']: b for b in meta.get('blocks', [])}
    times = defaultdict(lambda: defaultdict(float))   # block -> col -> ns/iter
    group_size = {}
    for p in profiles:
        local = p.meta.get('local_blocks', [])
        group_size.update({b: p.meta.get('group_size', 1) for b in local})
        roots, _ = build_tree(p)
        its = complete_iterations(roots, skip_first=skip_first, iters=iters)
        if not its:
            continue
        n = len(its)
        for _, it in its:
            for node in it.walk():
                path = node.path(stop_at=it)
                for col, rx in kernels:
                    if rx.match(path):
                        b = node.attrs.get('block')
                        if b is None and 'local_block' in node.attrs:
                            lb = int(node.attrs['local_block'])
                            b = local[lb] if lb < len(local) else None
                        if b is None:
                            continue
                        times[int(b)][col] += node.elapsed / n
                        break
    cols = [c for c, _ in kernels]
    headers = ['block', 'dim', 'K', 'S', 'psd0', 'psd1', 'ranks'] + cols
    rows = []
    for b in sorted(times):
        shape = blocks.get(b, {})
        psd = shape.get('psd_sizes', [None, None])
        rows.append([b, shape.get('dim'), shape.get('num_points'), shape.get('schur_size'),
                     psd[0], psd[1], group_size.get(b)] +
                    [times[b].get(c, 0.0) / NS for c in cols])
    md_rows = [[r[0], r[1], r[2], r[3], r[4], r[5], r[6]] + [f'{x:.4f}' for x in r[7:]]
               for r in rows]
    md = md_table(headers, md_rows)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(headers)
    w.writerows(rows)
    # power-law fits t ~ S^a (least squares on logs), per kernel
    fits = {}
    for j, c in enumerate(cols):
        pts = [(r[3], r[7 + j]) for r in rows if r[3] and r[7 + j] > 0]
        if len({s for s, _ in pts}) >= 3:
            xs = [math.log(s) for s, _ in pts]
            ys = [math.log(t) for _, t in pts]
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            sxx = sum((x - mx) ** 2 for x in xs)
            if sxx > 0:
                a = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
                fits[c] = a
    if fits:
        md += '\n\nfit t ∝ S^a (over blocks with S = Schur block size): ' + \
              ', '.join(f'{c}: a={a:.2f}' for c, a in fits.items())
    return md, buf.getvalue(), fits


def compare_table(a: RunStats, b: RunStats):
    cats = sort_categories(set(a.categories) | set(b.categories))
    rows = []
    for c in cats:
        ta, tb = a.mean_over_ranks(c), b.mean_over_ranks(c)
        rows.append([c, fmt_s(ta), fmt_s(tb), f'{ta / tb:.2f}x' if tb else '-'])
    rows.append(['**iteration**', fmt_s(a.mean_iter), fmt_s(b.mean_iter),
                 f'{a.mean_iter / b.mean_iter:.2f}x' if b.mean_iter else '-'])
    return md_table(['category', 'A s/iter', 'B s/iter', 'A/B'], rows)
