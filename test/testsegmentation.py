"""
Tests de la segmentation de Fisher : propriétés, séquences simulées et jeux de données.

Ils complètent testutils.py (diam, clustfisher, K-means, Ward, inertie,
ARI, silhouette) sans les répéter. Ils portent sur :
  - les propriétés du critère (translation, échelle, inversion de la séquence) ;
  - le comportement sur une séquence simulée comme sequencesimu
    (régime récurrent, fusion des régimes, effet de l'ordre) ;
  - la comparaison de Fisher avec K-means et Ward ;
  - la non-régression sur les jeux de données aiguillage et sequencesimu
    (dont le choix de K par la méthode du coude).

Lancement depuis la racine du dépôt :
    python3 -B -m unittest discover -s test -p 'test*.py' -v
"""
import os
import sys
import unittest

import numpy as np

RACINE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, RACINE)

from utils import (  # noqa: E402
    clustfisher, clustkmeans, clustward, compare_partitions, inertie,
)

CHEMIN_AIGUILLAGE = os.path.join(RACINE, "data", "aiguillage.txt")
CHEMIN_SEQUENCESIMU = os.path.join(RACINE, "data", "sequencesimu.txt")


def sequence_simulee(seed=0, bruit=0.3):
    """
    Séquence 2D construite comme sequencesimu : quatre segments de moyennes
    (0, 0), (0, 3), (0, 0), (4, 0), d'effectifs 50, 30, 50, 30, plus un bruit gaussien.
    Les segments 1 et 3 ont la même moyenne.
    """
    rng = np.random.default_rng(seed)
    moyennes = [(0, 0), (0, 3), (0, 0), (4, 0)]
    effectifs = [50, 30, 50, 30]
    X = np.vstack([np.asarray(m) + bruit * rng.normal(size=(e, 2))
                   for m, e in zip(moyennes, effectifs)])
    return X, np.array([51, 81, 131])


def criteres_fisher(X, K_max):
    """Critère optimal C(K) de Fisher pour K = 1, ..., K_max."""
    return np.array([clustfisher(X, K)[2] for K in range(1, K_max + 1)])


# ---------------------------------------------------------------------------
# 1. Propriétés du critère de Fisher
# ---------------------------------------------------------------------------


class TestFisherProprietes(unittest.TestCase):

    def setUp(self):
        self.X, _ = sequence_simulee(bruit=1.0)

    def test_invariance_par_translation(self):
        t1, c1, C1 = clustfisher(self.X, 4)
        t2, c2, C2 = clustfisher(self.X + [10.0, -5.0], 4)
        np.testing.assert_array_equal(t1, t2)
        np.testing.assert_array_equal(c1, c2)
        self.assertAlmostEqual(C1, C2, places=8)

    def test_mise_a_echelle(self):
        # multiplier X par a multiplie le critère par a², sans changer les ruptures
        t1, _, C1 = clustfisher(self.X, 4)
        t2, _, C2 = clustfisher(3.0 * self.X, 4)
        np.testing.assert_array_equal(t1, t2)
        self.assertAlmostEqual(C2, 9.0 * C1, places=6)

    def test_sequence_inversee(self):
        # inverser l'ordre donne les ruptures symétriques et le même critère
        n = len(self.X)
        t, _, C = clustfisher(self.X, 4)
        t_inv, _, C_inv = clustfisher(self.X[::-1], 4)
        np.testing.assert_array_equal(t_inv, np.sort(n + 2 - t))
        self.assertAlmostEqual(C, C_inv, places=8)


# ---------------------------------------------------------------------------
# 2. Séquence simulée comme sequencesimu
# ---------------------------------------------------------------------------


class TestFisherSequenceSimulee(unittest.TestCase):

    def test_regime_recurrent(self):
        # ruptures retrouvées ; segments 1 et 3 de même moyenne mais labels distincts
        X, ruptures = sequence_simulee()
        t, cluster, _ = clustfisher(X, 4)
        np.testing.assert_array_equal(t, ruptures)
        moyenne_1 = X[cluster == 1].mean(axis=0)
        moyenne_3 = X[cluster == 3].mean(axis=0)
        self.assertLess(np.linalg.norm(moyenne_1 - moyenne_3), 0.3)

    def test_fusion_regimes_cout_faible(self):
        # réunir les segments 1 et 3 ne coûte presque rien en inertie
        X, _ = sequence_simulee()
        _, cluster, C = clustfisher(X, 4)
        regimes = np.where(cluster == 3, 1, cluster)
        C_regimes = inertie(X, regimes)
        self.assertGreaterEqual(C_regimes, C - 1e-9)
        self.assertLess(C_regimes - C, 0.01 * C)

    def test_melange_degrade_le_critere(self):
        # sans l'ordre, quatre segments ne décrivent plus les régimes
        X, _ = sequence_simulee()
        _, _, C = clustfisher(X, 4)
        permutation = np.random.default_rng(0).permutation(len(X))
        _, _, C_melange = clustfisher(X[permutation], 4)
        self.assertGreater(C_melange, 2 * C)



# ---------------------------------------------------------------------------
# 3. Fisher face aux méthodes non contraintes
# ---------------------------------------------------------------------------


class TestFisherContreNonContraintes(unittest.TestCase):

    def test_ward_reproductible(self):
        X = np.random.default_rng(0).normal(size=(20, 2))
        cluster1, C1 = clustward(X, 3)
        cluster2, C2 = clustward(X, 3)
        np.testing.assert_array_equal(cluster1, cluster2)
        self.assertAlmostEqual(C1, C2)

    def test_ari_symetrique(self):
        a = [1, 1, 2, 2, 3, 3]
        b = [1, 2, 2, 2, 3, 1]
        self.assertAlmostEqual(compare_partitions(a, b), compare_partitions(b, a))

    def test_inertie_non_contrainte_plus_faible(self):
        # sans contrainte d'ordre, K-means et Ward peuvent réunir les segments 1 et 3
        X, _ = sequence_simulee(bruit=1.0)
        _, _, C_fisher = clustfisher(X, 4)
        for methode in (clustkmeans, clustward):
            with self.subTest(methode=methode.__name__):
                _, C = methode(X, 4)
                self.assertLess(C, C_fisher)

    def test_non_contraintes_ne_separent_pas_le_regime_recurrent(self):
        X, _ = sequence_simulee()
        _, cluster_fisher, _ = clustfisher(X, 4)
        for methode in (clustkmeans, clustward):
            with self.subTest(methode=methode.__name__):
                cluster, _ = methode(X, 4)
                self.assertLess(compare_partitions(cluster_fisher, cluster), 0.9)

    def test_kmeans_k3_retrouve_les_regimes(self):
        X, _ = sequence_simulee()
        _, cluster_fisher, _ = clustfisher(X, 4)
        regimes = np.where(cluster_fisher == 3, 1, cluster_fisher)
        cluster, _ = clustkmeans(X, 3)
        self.assertGreater(compare_partitions(regimes, cluster), 0.9)


# ---------------------------------------------------------------------------
# 4. Non-régression sur les jeux de données
# ---------------------------------------------------------------------------


@unittest.skipUnless(os.path.exists(CHEMIN_AIGUILLAGE), "data/aiguillage.txt absent")
class TestDonneesAiguillage(unittest.TestCase):
    """Résultats présentés dans aiguillage.ipynb."""

    @classmethod
    def setUpClass(cls):
        donnees = np.loadtxt(CHEMIN_AIGUILLAGE, delimiter=",")
        cls.X, cls.y = donnees[:, :-1], donnees[:, -1].astype(int)

    def test_fisher_retrouve_la_verite(self):
        t, cluster, _ = clustfisher(self.X, 4)
        np.testing.assert_array_equal(t, [36, 76, 121])
        self.assertEqual(compare_partitions(self.y, cluster), 1.0)


@unittest.skipUnless(os.path.exists(CHEMIN_SEQUENCESIMU), "data/sequencesimu.txt absent")
class TestDonneesSequencesimu(unittest.TestCase):
    """Résultats présentés dans sequencesimu.ipynb."""

    @classmethod
    def setUpClass(cls):
        cls.X = np.loadtxt(CHEMIN_SEQUENCESIMU, skiprows=1)

    def test_ruptures_k4(self):
        t, _, C = clustfisher(self.X, 4)
        np.testing.assert_array_equal(t, [51, 81, 131])
        self.assertAlmostEqual(C, 300.5963, places=3)

    def test_coude_en_k4(self):
        C = criteres_fisher(self.X, 8)
        gains = -np.diff(C)
        self.assertAlmostEqual(C[0], 884.1367, places=3)
        self.assertGreater(gains[2], 100)          # C(3) - C(4) ≈ 161
        self.assertTrue(np.all(gains[3:] < 10))    # gains suivants ≈ 7 à 8


if __name__ == "__main__":
    unittest.main()
