"""Per-rank and cross-rank statistics of category times."""
import json
import os
import statistics
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List

from .categories import categorize_iteration, sort_categories
from .tree import build_tree, complete_iterations


@dataclass
class RankStats:
    rank: int
    iterations: List[int]
    iter_times: List[int]                       # ns per iteration
    per_iter: List[Dict[str, int]]              # category -> ns, per iteration
    mean: Dict[str, float] = field(default_factory=dict)   # category -> mean ns
    mean_iter: float = 0.0

    def finalize(self):
        cats = set()
        for d in self.per_iter:
            cats.update(d)
        n = max(len(self.per_iter), 1)
        self.mean = {c: sum(d.get(c, 0) for d in self.per_iter) / n for c in cats}
        self.mean_iter = sum(self.iter_times) / n if self.iter_times else 0.0


@dataclass
class RunStats:
    run_dir: str
    ranks: List[RankStats]
    meta: Dict
    header: Dict
    legacy: bool
    categories: List[str]

    @property
    def num_iterations(self):
        return len(self.ranks[0].iterations) if self.ranks else 0

    def mean_over_ranks(self, cat):
        return statistics.fmean(r.mean.get(cat, 0.0) for r in self.ranks)

    def min_over_ranks(self, cat):
        return min(r.mean.get(cat, 0.0) for r in self.ranks)

    def max_over_ranks(self, cat):
        return max(r.mean.get(cat, 0.0) for r in self.ranks)

    @property
    def mean_iter(self):
        return statistics.fmean(r.mean_iter for r in self.ranks)

    @property
    def critical_rank(self):
        """Rank with the largest mean iteration time."""
        return max(self.ranks, key=lambda r: r.mean_iter)


def compute_run_stats(profiles, run_dir, skip_first=1, iters=None):
    ranks = []
    cats = set()
    for p in profiles:
        roots, _ = build_tree(p)
        its = complete_iterations(roots, skip_first=skip_first, iters=iters)
        per_iter = [categorize_iteration(n) for _, n in its]
        for d in per_iter:
            cats.update(d)
        rs = RankStats(rank=p.rank, iterations=[k for k, _ in its],
                       iter_times=[n.elapsed for _, n in its],
                       per_iter=per_iter)
        rs.finalize()
        ranks.append(rs)
    p0 = profiles[0]
    return RunStats(run_dir=run_dir, ranks=ranks, meta=p0.meta,
                    header=p0.header, legacy=p0.legacy,
                    categories=sort_categories(cats))


def load_iterations_json(run_dir):
    """Return {iteration: iter_time_seconds} from out/iterations.json if present."""
    for cand in (os.path.join(run_dir, 'out', 'iterations.json'),
                 os.path.join(os.path.dirname(run_dir.rstrip('/')), 'out',
                              'iterations.json')):
        if os.path.exists(cand):
            try:
                with open(cand) as f:
                    data = json.load(f)
            except (OSError, ValueError):
                return {}
            return {int(d['iteration']): float(d['iter_time']) for d in data
                    if 'iteration' in d and 'iter_time' in d}
    return {}
