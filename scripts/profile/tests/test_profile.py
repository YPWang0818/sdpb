import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sdpb_profile import amdahl, reader, report  # noqa: E402
from sdpb_profile.categories import categorize_iteration, category_of_path  # noqa: E402
from sdpb_profile.stats import compute_run_stats  # noqa: E402
from sdpb_profile.tree import build_tree, complete_iterations  # noqa: E402

LEGACY = '''{
    {"sdpb.solve", 10.5},
    {"sdpb.solve.run", 10.0},
    {"sdpb.solve.run.iter_1", 2.0},
    {"sdpb.solve.run.iter_1.step", 1.5},
    {"sdpb.solve.run.iter_1.step.stepLength(XCholesky)", 0.5},
    {"sdpb.solve.run.iter_1.print_iteration", 0.001},
    {"sdpb.solve.run.iter_2", 3.0},
    {"sdpb.solve.run.iter_2.choleskyDecomposition", 0.25},
    {"sdpb.solve.run.iter_2.step", 2.5},
    {"sdpb.solve.run.iter_2.step.initializeSchurComplementSolver", 1.0},
    {"sdpb.solve.run.iter_2.step.initializeSchurComplementSolver.Q", 0.8},
    {"sdpb.solve.run.iter_2.step.initializeSchurComplementSolver.Q.cholesky_3", 0.2},
    {"sdpb.solve.run.iter_2.step.initializeSchurComplementSolver.Q.syrk", 0.5},
    {"sdpb.solve.run.iter_2.step.initializeSchurComplementSolver.Q.syrk.bigint_syrk_blas", 0.4},
    {"sdpb.solve.run.iter_2.step.initializeSchurComplementSolver.Q.syrk.bigint_syrk_blas.Q_0_0", 0.4},
    {"sdpb.solve.run.iter_2.step.initializeSchurComplementSolver.Q.syrk.bigint_syrk_blas.Q_0_0.shmem", 0.3},
    {"sdpb.solve.run.iter_2.step.initializeSchurComplementSolver.Q.syrk.bigint_syrk_blas.Q_0_0.shmem.split_P_0", 0.3},
    {"sdpb.solve.run.iter_2.step.initializeSchurComplementSolver.Q.syrk.bigint_syrk_blas.Q_0_0.shmem.split_P_0.syrk", 0.2},
    {"sdpb.solve.run.iter_2.step.initializeSchurComplementSolver.Q.syrk.bigint_syrk_blas.Q_0_0.shmem.split_P_0.syrk.blas_jobs", 0.15},
    {"sdpb.solve.run.iter_2.step.computeSearchDirection(betaPredictor)", 0.6},
    {"sdpb.solve.run.iter_2.step.stepLength(YCholesky)", 0.3},
    {"sdpb.solve.run.iter_2.print_iteration", 0.001},
    {"sdpb.solve.run.iter_3", 0.5}
}
'''


def make_json_doc(rank=0, num_ranks=2):
    """Two complete iterations with a few timers and attributes."""
    timers = []

    def add(parent, depth, name, elapsed, attrs=None, **kw):
        t = {'id': len(timers), 'parent': parent, 'depth': depth, 'name': name,
             'start': 0, 'elapsed': elapsed, 'cpu': elapsed, 'count': 1,
             'attrs': attrs or {}}
        t.update(kw)
        timers.append(t)
        return t['id']

    root = add(-1, 0, 'sdpb.solve', 100)
    run = add(root, 1, 'run', 90)
    for k in (1, 2, 3):
        it = add(run, 2, 'iter', 30, {'iter': str(k)})
        add(it, 3, 'cholesky_XY', 2)
        bp = add(it, 3, 'bilinear_pairings', 4)
        axi = add(bp, 4, 'A_X_inv', 3)
        blk = add(axi, 5, 'block', 3, {'block': '7', 'parity': '0'})
        add(blk, 6, 'trsm', 2, {'kind': 'trsm', 'm': '5', 'n': '5', 'ranks': '1'})
        step = add(it, 3, 'step', 20)
        sl = add(step, 4, 'step_length', 6, {'which': 'X'})
        eig = add(sl, 5, 'eig', 5)
        add(eig, 6, 'hermitian_eig', 4, {'kind': 'eig', 'n': '5', 'ranks': '1',
                                         'local_block': '0', 'parity': '0'})
        add(eig, 6, 'allreduce', 1, {'kind': 'mpi', 'op': 'allreduce'})
        add(step, 4, 'block_timings_AllReduce', 1, {'kind': 'mpi', 'op': 'allreduce'})
        sd = add(step, 4, 'search_direction', 3, {'phase': 'predictor'})
        acc = add(sd, 5, 'weighted_sum', 3)
        add(acc, 6, 'gemm_tile', 2, {'kind': 'gemm'}, count=10, max=1)
        add(it, 3, 'print_iteration', 1)
    return {'schema_version': 1, 'sdpb_version': 'test', 'rank': rank,
            'num_ranks': num_ranks, 'node': 0, 'node_rank': rank, 'num_nodes': 1,
            'hostname': 'h', 'precision_bits': 768, 't0_unix_ns': 0,
            't0_monotonic_ns': 0, 'cpu_time': True,
            'meta': {'blocks': [{'index': 7, 'dim': 1, 'num_points': 5,
                                 'schur_size': 5, 'psd_sizes': [3, 2],
                                 'bilinear_pairing_size': 5}],
                     'local_blocks': [7], 'group_size': 1, 'N': 20, 'P': 5},
            'timers': timers}


class LegacyTest(unittest.TestCase):
    def test_parse_and_tree(self):
        p = reader.parse_legacy(LEGACY, 'x/profiling.3')
        self.assertTrue(p.legacy)
        self.assertEqual(p.rank, 3)
        names = [e.name for e in p.entries]
        self.assertEqual(names[0], 'sdpb.solve')
        self.assertIn('iter', names)
        self.assertIn('Q_tile', names)
        self.assertIn('step_length', names)
        roots, nodes = build_tree(p)
        self.assertEqual(len(roots), 1)
        its = complete_iterations(roots, skip_first=0)
        self.assertEqual([k for k, _ in its], [1, 2])   # iter_3 is partial
        it2 = its[1][1]
        cats = categorize_iteration(it2)
        self.assertAlmostEqual(sum(cats.values()), it2.elapsed)
        self.assertAlmostEqual(cats['Q.syrk.blas'] / 1e9, 0.2)
        self.assertAlmostEqual(cats['Q.cholesky_S'] / 1e9, 0.2)
        self.assertAlmostEqual(cats['cholesky_XY'] / 1e9, 0.25)
        self.assertAlmostEqual(cats['step_length.other'] / 1e9, 0.3)

    def test_normalize(self):
        self.assertEqual(reader.normalize_legacy_name('iter_12'), ('iter', {'iter': '12'}))
        self.assertEqual(reader.normalize_legacy_name('offset=2'),
                         ('reduce_step', {'offset': '2'}))
        self.assertEqual(reader.normalize_legacy_name('plain'), ('plain', {}))


class JsonTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.pdir = os.path.join(self.tmp.name, 'ck.profiling')
        os.makedirs(self.pdir)
        for r in (0, 1):
            with open(os.path.join(self.pdir, f'profiling.{r}'), 'w') as f:
                json.dump(make_json_doc(r), f)

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_and_stats(self):
        profiles = reader.load_run(self.tmp.name)
        self.assertEqual([p.rank for p in profiles], [0, 1])
        self.assertFalse(profiles[0].legacy)
        st = compute_run_stats(profiles, self.tmp.name, skip_first=1)
        self.assertEqual(st.num_iterations, 2)
        self.assertAlmostEqual(st.mean_iter, 30)
        total = sum(st.mean_over_ranks(c) for c in st.categories)
        self.assertAlmostEqual(total, 30)
        self.assertAlmostEqual(st.mean_over_ranks('mpi'), 2)
        self.assertAlmostEqual(st.mean_over_ranks('step_length.eig'), 4)
        self.assertAlmostEqual(st.mean_over_ranks('bilinear_pairings.A_X_inv'), 3)
        self.assertAlmostEqual(st.mean_over_ranks('search_direction.weighted_sum'), 3)
        self.assertIn('| **iteration** |', report.phase1_table(st))
        js = report.phase1_json(st)
        self.assertAlmostEqual(js['categories']['mpi']['share'], 2 / 30)

    def test_blocks_report(self):
        profiles = reader.load_run(self.tmp.name)
        md, csv_text, fits = report.blocks_report(profiles, skip_first=1)
        self.assertIn('| 7 | 1 | 5 | 5 |', md)
        self.assertIn('A_X_inv', csv_text.splitlines()[0])
        # eig attributed via local_block -> global block 7, summed over 2 ranks
        row = [r for r in csv_text.splitlines() if r.startswith('7,')][0]
        self.assertIn('8e-09', row)   # 2 ranks x 4 ns

    def test_tree_report(self):
        profiles = reader.load_run(self.tmp.name)
        text = report.tree_report(profiles[0], skip_first=1, min_share=0.0)
        self.assertIn('hermitian_eig [eig]', text)


class AmdahlTest(unittest.TestCase):
    def test_factor(self):
        sp = amdahl.parse_speedups(['Q.syrk=10', 'Q.syrk.blas=100'])
        self.assertEqual(amdahl.factor_for('Q.syrk.blas', sp), 100)
        self.assertEqual(amdahl.factor_for('Q.syrk.residues', sp), 10)
        self.assertEqual(amdahl.factor_for('schur_complement', sp), 1.0)
        total, new, bound, rows = amdahl.predict({'a': 80, 'b': 20}, {'a': 4})
        self.assertEqual((total, new, bound), (100, 40, 20))


class CategoryTest(unittest.TestCase):
    def test_paths(self):
        self.assertEqual(category_of_path('step/initializeSchurComplementSolver/Q/syrk/bigint_syrk_blas/Q_tile/shmem/split_P/compute_residues/compute_and_write'),
                         'Q.syrk.residues')
        self.assertEqual(category_of_path('step/search_direction/solve_schur/solve_Q'),
                         'search_direction.solve_Q')
        self.assertIsNone(category_of_path('step/does_not_exist'))


if __name__ == '__main__':
    unittest.main()
