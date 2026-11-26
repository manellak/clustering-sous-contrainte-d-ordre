"""
Tests unitaires pour utils.py :
  - diam et clustfisher ;
  - méthodes comparatives (K-means, CAH de Ward) ;
  - critères de comparaison (inertie, ARI, silhouette).

Lancement depuis la racine du projet :
    python -m unittest test.testutils -v
ou :
    python -m pytest test/testutils.py -v
"""
import itertools
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import (  # noqa: E402
    diam, clustfisher, clustkmeans, clustward,
    compare_partitions, inertie, silhouette,
)


def critere_force_brute(X, K):
    """Critère optimal par énumération de toutes les segmentations (petits n)."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = X.shape[0]
    best = np.inf
    # ruptures = débuts des segments 2..K (1-based), choisis parmi 2..n
    for ruptures in itertools.combinations(range(2, n + 1), K - 1):
        bornes = [1] + list(ruptures) + [n + 1]
        c = sum(diam(X, bornes[k], bornes[k + 1] - 1) for k in range(K))
        best = min(best, c)
    return best


class TestDiam(unittest.TestCase):

    def test_un_seul_point_vaut_zero(self):
        X = np.array([1.0, 5.0, 3.0])
        for a in range(1, 4):
            self.assertAlmostEqual(diam(X, a, a), 0.0)

    def test_segment_constant_vaut_zero(self):
        X = np.array([2.0, 2.0, 2.0, 2.0])
        self.assertAlmostEqual(diam(X, 1, 4), 0.0)

    def test_valeur_connue_1d(self):
        # moyenne de [1, 2, 3] = 2 -> (1 + 0 + 1) = 2
        X = np.array([1.0, 2.0, 3.0, 10.0])
        self.assertAlmostEqual(diam(X, 1, 3), 2.0)

    def test_indices_1_based(self):
        # le segment (x_2, x_3) = [2, 3] -> moyenne 2.5 -> 0.25 + 0.25
        X = np.array([1.0, 2.0, 3.0, 10.0])
        self.assertAlmostEqual(diam(X, 2, 3), 0.5)

    def test_multivarie(self):
        X = np.array([[0.0, 0.0],
                      [2.0, 4.0]])
        # moyenne (1, 2) -> (1 + 4) * 2 = 10
        self.assertAlmostEqual(diam(X, 1, 2), 10.0)

    def test_1d_equivaut_colonne(self):
        x = np.array([1.0, 4.0, 2.0, 8.0])
        self.assertAlmostEqual(diam(x, 1, 4), diam(x.reshape(-1, 1), 1, 4))

    def test_accepte_liste(self):
        self.assertAlmostEqual(diam([1, 2, 3], 1, 3), 2.0)

    def test_egal_variance_fois_effectif(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 3))
        seg = X[4:15]  # a=5, b=15
        attendu = seg.var(axis=0).sum() * len(seg)
        self.assertAlmostEqual(diam(X, 5, 15), attendu)

    def test_positif(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(15, 2))
        for a in range(1, 16):
            for b in range(a, 16):
                self.assertGreaterEqual(diam(X, a, b), 0.0)


class TestClustfisher(unittest.TestCase):

    def test_k1_un_seul_segment(self):
        X = np.array([1.0, 3.0, 5.0, 7.0])
        t, cluster, C = clustfisher(X, 1)
        self.assertEqual(len(t), 0)
        np.testing.assert_array_equal(cluster, [1, 1, 1, 1])
        self.assertAlmostEqual(C, diam(X, 1, 4))

    def test_k_egal_n_critere_nul(self):
        X = np.array([4.0, 1.0, 7.0, 2.0])
        t, cluster, C = clustfisher(X, 4)
        np.testing.assert_array_equal(t, [2, 3, 4])
        np.testing.assert_array_equal(cluster, [1, 2, 3, 4])
        self.assertAlmostEqual(C, 0.0)

    def test_deux_paliers_evidents(self):
        X = np.array([0.0, 0.1, -0.1, 0.0, 10.0, 10.1, 9.9, 10.0])
        t, cluster, C = clustfisher(X, 2)
        np.testing.assert_array_equal(t, [5])
        np.testing.assert_array_equal(cluster, [1, 1, 1, 1, 2, 2, 2, 2])
        self.assertAlmostEqual(C, diam(X, 1, 4) + diam(X, 5, 8))

    def test_trois_paliers_evidents(self):
        X = np.array([0, 0, 0, 5, 5, 5, 5, -3, -3], dtype=float)
        t, cluster, C = clustfisher(X, 3)
        np.testing.assert_array_equal(t, [4, 8])
        np.testing.assert_array_equal(cluster, [1, 1, 1, 2, 2, 2, 2, 3, 3])
        self.assertAlmostEqual(C, 0.0)

    def test_multivarie(self):
        X = np.vstack([np.zeros((5, 2)), np.ones((4, 2)) * [3, -2]])
        t, cluster, C = clustfisher(X, 2)
        np.testing.assert_array_equal(t, [6])
        self.assertAlmostEqual(C, 0.0)

    def test_formes_des_sorties(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(12, 2))
        K = 4
        t, cluster, C = clustfisher(X, K)
        self.assertEqual(len(t), K - 1)
        self.assertEqual(len(cluster), 12)
        self.assertTrue(np.isscalar(C) or np.ndim(C) == 0)

    def test_ruptures_strictement_croissantes_et_dans_bornes(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=30)
        t, _, _ = clustfisher(X, 5)
        self.assertTrue(np.all(np.diff(t) > 0))
        self.assertTrue(np.all((t >= 2) & (t <= 30)))

    def test_labels_coherents_avec_ruptures(self):
        rng = np.random.default_rng(4)
        X = rng.normal(size=25)
        K = 4
        t, cluster, _ = clustfisher(X, K)
        # labels croissants, valant 1..K, changent exactement aux ruptures
        self.assertEqual(cluster[0], 1)
        self.assertEqual(cluster[-1], K)
        self.assertTrue(np.all(np.diff(cluster) >= 0))
        changements = np.where(np.diff(cluster) != 0)[0] + 2  # 1-based
        np.testing.assert_array_equal(changements, t)
        self.assertEqual(set(cluster), set(range(1, K + 1)))

    def test_critere_egal_somme_des_diametres(self):
        rng = np.random.default_rng(5)
        X = rng.normal(size=(20, 3))
        K = 3
        t, _, C = clustfisher(X, K)
        bornes = [1] + list(t) + [21]
        somme = sum(diam(X, bornes[k], bornes[k + 1] - 1) for k in range(K))
        self.assertAlmostEqual(C, somme)

    def test_optimalite_vs_force_brute(self):
        rng = np.random.default_rng(6)
        for n, K in [(6, 2), (7, 3), (8, 3), (9, 4), (10, 5)]:
            X = rng.normal(size=(n, 2))
            _, _, C = clustfisher(X, K)
            with self.subTest(n=n, K=K):
                self.assertAlmostEqual(C, critere_force_brute(X, K))

    def test_critere_decroissant_en_k(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=15)
        criteres = [clustfisher(X, K)[2] for K in range(1, 16)]
        self.assertTrue(np.all(np.diff(criteres) <= 1e-12))
        self.assertAlmostEqual(criteres[-1], 0.0)

    def test_k_invalide(self):
        X = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            clustfisher(X, 0)
        with self.assertRaises(ValueError):
            clustfisher(X, 4)

    def test_accepte_liste(self):
        t, cluster, C = clustfisher([0, 0, 9, 9], 2)
        np.testing.assert_array_equal(t, [3])
        np.testing.assert_array_equal(cluster, [1, 1, 2, 2])


class TestMethodesComparatives(unittest.TestCase):

    def test_deux_groupes_separes(self):
        X = np.array([[0, 0], [0, 2], [10, 10], [10, 12]])
        for methode in (clustkmeans, clustward):
            with self.subTest(methode=methode.__name__):
                cluster, C = methode(X, 2)
                self.assertEqual(compare_partitions(cluster, [1, 1, 2, 2]), 1.0)
                self.assertAlmostEqual(C, 4.0)
                self.assertEqual(set(cluster), {1, 2})

    def test_groupes_non_contigus(self):
        for methode in (clustkmeans, clustward):
            with self.subTest(methode=methode.__name__):
                cluster, C = methode([0, 10, 0, 10], 2)
                self.assertEqual(compare_partitions(cluster, [1, 2, 1, 2]), 1.0)
                self.assertAlmostEqual(C, 0.0)

    def test_k1(self):
        for methode in (clustkmeans, clustward):
            cluster, C = methode([0, 2, 4], 1)
            np.testing.assert_array_equal(cluster, [1, 1, 1])
            self.assertAlmostEqual(C, 8.0)

    def test_k_egal_n(self):
        for methode in (clustkmeans, clustward):
            cluster, C = methode([0, 2, 4], 3)
            self.assertEqual(len(np.unique(cluster)), 3)
            self.assertAlmostEqual(C, 0.0)

    def test_un_seul_point(self):
        for methode in (clustkmeans, clustward):
            cluster, C = methode([[3, 7]], 1)
            np.testing.assert_array_equal(cluster, [1])
            self.assertAlmostEqual(C, 0.0)

    def test_k_invalide(self):
        for methode in (clustkmeans, clustward):
            for K in (0, 4, 1.5):
                with self.subTest(methode=methode.__name__, K=K):
                    with self.assertRaises(ValueError):
                        methode([0, 2, 4], K)

    def test_kmeans_une_initialisation(self):
        X = np.random.default_rng(0).normal(size=(20, 2))
        cluster1, C1 = clustkmeans(X, 3, n_init=1, random_state=5)
        cluster2, C2 = clustkmeans(X, 3, n_init=1, random_state=5)
        np.testing.assert_array_equal(cluster1, cluster2)
        self.assertAlmostEqual(C1, C2)
        self.assertEqual(set(cluster1), {1, 2, 3})

    def test_kmeans_reproductible(self):
        X = np.random.default_rng(0).normal(size=(20, 2))
        cluster1, C1 = clustkmeans(X, 3)
        cluster2, C2 = clustkmeans(X, 3)
        np.testing.assert_array_equal(cluster1, cluster2)
        self.assertAlmostEqual(C1, C2)


class TestComparaison(unittest.TestCase):

    def test_labels_permutés(self):
        self.assertEqual(compare_partitions([1, 1, 2, 2], [8, 8, 3, 3]), 1.0)

    def test_partitions_differentes(self):
        self.assertAlmostEqual(compare_partitions([1, 1, 2, 2], [1, 2, 1, 2]), -0.5)

    def test_longueurs_differentes(self):
        with self.assertRaises(ValueError):
            compare_partitions([1, 2], [1, 2, 3])

    def test_inertie_fisher(self):
        X = np.array([[0, 1], [1, 2], [9, 8], [10, 9]])
        _, cluster, C = clustfisher(X, 2)
        self.assertAlmostEqual(inertie(X, cluster), C)

    def test_inertie_labels_non_consecutifs(self):
        self.assertAlmostEqual(inertie([0, 2, 10, 12], [5, 5, 9, 9]), 4.0)

    def test_inertie_longueur_invalide(self):
        with self.assertRaises(ValueError):
            inertie([0, 2, 10], [1, 2])


class TestSilhouette(unittest.TestCase):

    def test_valeur_connue_1d(self):
        # Distances dans chaque groupe : 2 ; distances moyennes à l'autre : 11 ou 9
        attendu = ((1 - 2 / 11) + (1 - 2 / 9)) / 2
        self.assertAlmostEqual(silhouette([0, 2, 10, 12], [1, 1, 2, 2]), attendu)

    def test_multivarie_et_labels_permutés(self):
        X = [[0, 0], [0, 0], [10, 10], [10, 10]]
        self.assertAlmostEqual(silhouette(X, [8, 8, 3, 3]), 1.0)

    def test_nombre_de_groupes_invalide(self):
        for cluster in ([1, 1, 1], [1, 2, 3]):
            with self.assertRaises(ValueError):
                silhouette([0, 2, 10], cluster)

    def test_longueur_invalide(self):
        with self.assertRaises(ValueError):
            silhouette([0, 2, 10], [1, 2])


if __name__ == "__main__":
    unittest.main()
