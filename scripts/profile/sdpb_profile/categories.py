"""Assign every timer node of an iteration to a category.

A node's category is the first rule whose regex matches the node's path
relative to the `iter` node (e.g. "step/initializeSchurComplementSolver/Q/syrk");
a node matching no rule inherits its parent's category. Nodes with attribute
kind=mpi are always "mpi". Self times (elapsed minus children) are summed per
category, so the categories of one iteration add up to its elapsed time.
"""
import re
from collections import OrderedDict

# Order matters: first match wins. Paths are relative to the iter node.
RULES = [
    (r'^step/initializeSchurComplementSolver/Q/syrk/.*/compute_residues', 'Q.syrk.residues'),
    (r'^step/initializeSchurComplementSolver/Q/syrk/.*/split_P/syrk', 'Q.syrk.blas'),
    (r'^step/initializeSchurComplementSolver/Q/syrk/.*/restore_and_reduce', 'Q.syrk.restore_reduce'),
    (r'^step/initializeSchurComplementSolver/Q/syrk', 'Q.syrk.other'),
    (r'^step/initializeSchurComplementSolver/Q/cholesky', 'Q.cholesky_S'),
    (r'^step/initializeSchurComplementSolver/Q/solve', 'Q.solve_P'),
    (r'^step/initializeSchurComplementSolver/schur_complement', 'schur_complement'),
    (r'^step/initializeSchurComplementSolver/Cholesky_Q', 'Cholesky_Q'),
    (r'^cholesky_XY', 'cholesky_XY'),
    (r'^bilinear_pairings/A_X_inv', 'bilinear_pairings.A_X_inv'),
    (r'^bilinear_pairings/A_Y', 'bilinear_pairings.A_Y'),
    (r'^bilinear_pairings', 'bilinear_pairings.other'),
    (r'^step/search_direction/cholesky_solve', 'search_direction.cholesky_solve'),
    (r'^step/search_direction/schur_RHS', 'search_direction.schur_RHS'),
    (r'^step/search_direction/solve_schur/solve_Q', 'search_direction.solve_Q'),
    (r'^step/search_direction/solve_schur/trsm_S', 'search_direction.solve_S'),
    (r'^step/search_direction/solve_schur', 'search_direction.solve_other'),
    (r'^step/search_direction/weighted_sum', 'search_direction.weighted_sum'),
    (r'^step/search_direction/\w+_product', 'search_direction.products'),
    (r'^step/XY', 'search_direction.products'),
    (r'^step/search_direction', 'search_direction.other'),
    (r'^step/step_length/congruence', 'step_length.congruence'),
    (r'^step/step_length/eig', 'step_length.eig'),
    (r'^step/step_length', 'step_length.other'),
    (r'^step/condition_numbers', 'condition_numbers'),
    (r'^step/update_xXyY', 'update'),
    (r'^step/(mu|R_error)', 'residues_objectives'),
    (r'^(objectives|computeDualResidues|computePrimalResidues)', 'residues_objectives'),
    (r'^(save_checkpoint|print_iteration)', 'io'),
]
_COMPILED = [(re.compile(rx), cat) for rx, cat in RULES]

# Display order of categories in reports (others appended alphabetically)
ORDER = [c for _, c in RULES]
for extra in ('mpi', 'other'):
    if extra not in ORDER:
        ORDER.append(extra)
# de-duplicate, keep first occurrence
_seen = set()
ORDER = [c for c in ORDER if not (c in _seen or _seen.add(c))]


def category_of_path(path):
    for rx, cat in _COMPILED:
        if rx.match(path):
            return cat
    return None


def categorize_iteration(iter_node):
    """Return OrderedDict category -> ns (self times summed), for one iteration."""
    totals = OrderedDict()

    def visit(node, inherited):
        if node.attrs.get('kind') == 'mpi':
            cat = 'mpi'
        elif node is iter_node:
            cat = 'other'
        else:
            cat = category_of_path(node.path(stop_at=iter_node)) or inherited
        totals[cat] = totals.get(cat, 0) + node.self_time
        for c in node.children:
            visit(c, cat)

    visit(iter_node, 'other')
    return totals


def sort_categories(cats):
    known = [c for c in ORDER if c in cats]
    rest = sorted(c for c in cats if c not in ORDER)
    return known + rest
