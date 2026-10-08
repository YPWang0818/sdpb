"""What-if model: apply per-category speedups to a measured iteration."""
from .report import fmt_pct, fmt_s, md_table


def parse_speedups(items):
    """['Q.syrk.blas=20', 'bilinear_pairings=5'] -> {prefix: factor}.
    A key matches every category equal to it or starting with it + '.'."""
    out = {}
    for item in items or []:
        key, _, val = item.partition('=')
        if not val:
            raise ValueError(f'--speedup expects category=factor, got {item!r}')
        out[key.strip()] = float(val)
    return out


def factor_for(category, speedups):
    best = None
    for key, f in speedups.items():
        if category == key or category.startswith(key + '.'):
            if best is None or len(key) > len(best[0]):
                best = (key, f)
    return best[1] if best else 1.0


def predict(means, speedups):
    """means: category -> ns. Returns (new_total, upper_bound_total, rows)."""
    total = sum(means.values())
    new_total = 0.0
    bound_total = 0.0
    rows = []
    for c, t in means.items():
        f = factor_for(c, speedups)
        new_t = t / f
        new_total += new_t
        bound_total += 0.0 if f != 1.0 else t
        rows.append((c, t, f, new_t))
    return total, new_total, bound_total, rows


def amdahl_table(run_stats, speedups):
    out = []
    for label, means, iter_time in (
            ('mean rank', {c: run_stats.mean_over_ranks(c) for c in run_stats.categories},
             run_stats.mean_iter),
            (f'critical rank {run_stats.critical_rank.rank}',
             dict(run_stats.critical_rank.mean), run_stats.critical_rank.mean_iter)):
        total, new_total, bound, rows = predict(means, speedups)
        table_rows = [[c, fmt_s(t), f'{f:g}' if f != 1.0 else '', fmt_s(nt)]
                      for c, t, f, nt in rows if f != 1.0]
        table_rows.append(['**iteration**', fmt_s(total), '', fmt_s(new_total)])
        out.append(f'### What-if, {label}: predicted iteration time '
                   f'{fmt_s(new_total)} s vs {fmt_s(total)} s '
                   f'(speedup {total / new_total:.2f}x; upper bound with infinite '
                   f'speedup of the same categories: {total / bound if bound else float("inf"):.2f}x)')
        out.append(md_table(['category', 'now s/iter', 'speedup', 'predicted s/iter'],
                            table_rows))
    return '\n\n'.join(out)
