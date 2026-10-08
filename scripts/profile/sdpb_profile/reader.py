"""Read per-rank profile files: the JSON format (schema_version 1) written by
Timers::write_profile, and the legacy pseudo-JSON format of older SDPB versions.
"""
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Entry:
    id: int
    parent: int
    depth: int
    name: str
    elapsed: int          # ns
    cpu: int = -1         # ns, -1 if unknown
    start: Optional[int] = None  # ns since Timers creation, None for legacy
    count: int = 1
    max: Optional[int] = None
    attrs: Dict[str, str] = field(default_factory=dict)


@dataclass
class Profile:
    path: str
    rank: int
    num_ranks: int
    entries: List[Entry]
    meta: Dict = field(default_factory=dict)
    header: Dict = field(default_factory=dict)
    legacy: bool = False


# Legacy names carried instance indices in the name; normalise them to
# the (name, attrs) convention of the JSON format.
_LEGACY_RULES = [
    (re.compile(r'^iter_(\d+)$'), 'iter', 'iter'),
    (re.compile(r'^cholesky_(\d+)$'), 'cholesky', 'block'),
    (re.compile(r'^solve_(\d+)$'), 'solve', 'block'),
    (re.compile(r'^split_P_(\d+)$'), 'split_P', 'k'),
    (re.compile(r'^offset=(\d+)$'), 'reduce_step', 'offset'),
]
_LEGACY_FIXED = {
    'choleskyDecomposition': ('cholesky_XY', {}),
    'computeSearchDirection(betaPredictor)': ('search_direction', {'phase': 'predictor'}),
    'computeSearchDirection(betaCorrector)': ('search_direction', {'phase': 'corrector'}),
    'stepLength(XCholesky)': ('step_length', {'which': 'X'}),
    'stepLength(YCholesky)': ('step_length', {'which': 'Y'}),
}
_LEGACY_Q = re.compile(r'^Q_(\d+)_(\d+)$')


def normalize_legacy_name(name):
    if name in _LEGACY_FIXED:
        return _LEGACY_FIXED[name]
    m = _LEGACY_Q.match(name)
    if m:
        return 'Q_tile', {'i': m.group(1), 'j': m.group(2)}
    for rx, new_name, attr in _LEGACY_RULES:
        m = rx.match(name)
        if m:
            return new_name, {attr: m.group(1)}
    return name, {}


_LEGACY_LINE = re.compile(r'^\s*\{\s*"(.*)"\s*,\s*([0-9.eE+-]+)\s*\}\s*,?\s*$')


def parse_legacy(text, path):
    """Parse the legacy `{ {"a.b.c", 1.234}, ... }` format.

    The tree is rebuilt from the prefix relation between full names
    (child full name == parent full name + "." + child name), which is
    robust to names that themselves contain dots.
    """
    entries = []
    stack = []  # (full_name, id)
    for line in text.splitlines():
        m = _LEGACY_LINE.match(line)
        if not m:
            continue
        full, seconds = m.group(1), float(m.group(2))
        while stack and not full.startswith(stack[-1][0] + '.'):
            stack.pop()
        if stack:
            parent_full, parent_id = stack[-1]
            leaf = full[len(parent_full) + 1:]
        else:
            parent_id, leaf = -1, full
        name, attrs = normalize_legacy_name(leaf)
        e = Entry(id=len(entries), parent=parent_id, depth=len(stack),
                  name=name, elapsed=int(round(seconds * 1e9)), attrs=attrs)
        entries.append(e)
        stack.append((full, e.id))
    m = re.search(r'profiling\.(\d+)$', os.path.basename(path))
    rank = int(m.group(1)) if m else 0
    return Profile(path=path, rank=rank, num_ranks=-1, entries=entries,
                   legacy=True)


def parse_json(doc, path):
    entries = []
    for t in doc['timers']:
        entries.append(Entry(
            id=t['id'], parent=t['parent'], depth=t['depth'], name=t['name'],
            elapsed=t['elapsed'], cpu=t.get('cpu', -1), start=t.get('start'),
            count=t.get('count', 1), max=t.get('max'),
            attrs=dict(t.get('attrs', {}))))
    header = {k: v for k, v in doc.items() if k not in ('timers', 'meta')}
    return Profile(path=path, rank=doc['rank'], num_ranks=doc['num_ranks'],
                   entries=entries, meta=doc.get('meta', {}), header=header)


def load_profile(path):
    with open(path) as f:
        text = f.read()
    stripped = text.lstrip()
    if stripped.startswith('{') and '"schema_version"' in stripped[:200]:
        return parse_json(json.loads(text), path)
    return parse_legacy(text, path)


def find_profiling_dir(run_dir):
    """RUN_DIR may be the profiling dir itself, or a directory containing
    <something>.profiling/ (e.g. the directory holding ck/, out/)."""
    if os.path.isdir(run_dir) and any(
            f.startswith('profiling.') for f in os.listdir(run_dir)):
        return run_dir
    candidates = sorted(
        d for d in os.listdir(run_dir)
        if d.endswith('.profiling') and os.path.isdir(os.path.join(run_dir, d)))
    if not candidates:
        raise FileNotFoundError(
            f'no profiling.<rank> files or *.profiling/ directory in {run_dir}')
    return os.path.join(run_dir, candidates[0])


def load_run(run_dir):
    """Load all profiling.<rank> files of a run, sorted by rank."""
    pdir = find_profiling_dir(run_dir)
    files = [f for f in os.listdir(pdir)
             if re.match(r'^profiling\.\d+$', f)]
    profiles = [load_profile(os.path.join(pdir, f)) for f in files]
    profiles.sort(key=lambda p: p.rank)
    if not profiles:
        raise FileNotFoundError(f'no profiling.<rank> files in {pdir}')
    return profiles
