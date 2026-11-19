"""
Tests des méthodes comparatives.

Lancement depuis la racine du dépôt :
    python3 -B -m unittest discover -s test -p 'test*.py' -v
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import clustfisher, clustkmeans, clustward, compare_partitions, inertie, silhouette


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
