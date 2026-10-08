"""Timer tree built from Entry.parent ids, with iteration selection."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .reader import Entry, Profile


@dataclass
class Node:
    entry: Entry
    children: List['Node'] = field(default_factory=list)
    parent: Optional['Node'] = None

    @property
    def name(self):
        return self.entry.name

    @property
    def attrs(self):
        return self.entry.attrs

    @property
    def elapsed(self):
        return self.entry.elapsed

    @property
    def self_time(self):
        return self.elapsed - sum(c.elapsed for c in self.children)

    def path(self, stop_at=None):
        """Names from `stop_at` (exclusive) down to this node, joined by '/'."""
        parts = []
        n = self
        while n is not None and n is not stop_at:
            parts.append(n.name)
            n = n.parent
        return '/'.join(reversed(parts))

    def child(self, name):
        for c in self.children:
            if c.name == name:
                return c
        return None

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()


def build_tree(profile: Profile):
    nodes = [Node(e) for e in profile.entries]
    roots = []
    for n in nodes:
        p = n.entry.parent
        if p < 0:
            roots.append(n)
        else:
            parent = nodes[p]
            n.parent = parent
            parent.children.append(n)
    return roots, nodes


def find_all(roots, name):
    out = []
    for r in roots:
        for n in r.walk():
            if n.name == name:
                out.append(n)
    return out


def iteration_nodes(roots):
    """All `iter` nodes, in order, keyed by iteration number."""
    result = []
    for n in find_all(roots, 'iter'):
        try:
            k = int(n.attrs.get('iter', '0'))
        except ValueError:
            k = 0
        result.append((k, n))
    result.sort(key=lambda t: t[0])
    return result


def complete_iterations(roots, skip_first=1, iters=None):
    """Iterations that ran a full step: have both `step` and `print_iteration`
    children (the terminating iteration stops before `step`; the
    max-complementarity exit stops before `print_iteration`).
    `iters` is an optional (lo, hi) inclusive range of iteration numbers."""
    out = []
    for k, n in iteration_nodes(roots):
        if n.child('step') is None or n.child('print_iteration') is None:
            continue
        if k <= skip_first:
            continue
        if iters is not None and not (iters[0] <= k <= iters[1]):
            continue
        out.append((k, n))
    return out
