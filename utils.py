"""
Fonctions de classification et d'évaluation des partitions.

Les lignes de X représentent les observations, les colonnes les variables.
Fisher respecte l'ordre des observations ; K-means et Ward sont non contraints.
Les méthodes de classification renvoient des labels à partir de 1.
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score


# ---------------------------------------------------------------------------
# 1. Classification sous contrainte d’ordre : Fisher
# ---------------------------------------------------------------------------


def diam(X, a, b):
    """
    Diamètre D(a, b) du segment (x_a, ..., x_b)

    X    : array (n, p) (ou (n,) pour des données 1D)
    a, b : indices de début et de fin
         avec a <= b
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    seg = X[a - 1:b]                 # x_a, ..., x_b (passage en indices Python)
    mu = seg.mean(axis=0)            # moyenne de chaque variable j
    return ((seg - mu) ** 2).sum()   # somme sur i puis sur j


def clustfisher(X, K):
    """
    Algorithme de programmation dynamique de Fisher.

    X : array (n, p) (ou (n,) pour des données 1D)
    K : nombre de segments souhaité (K >= 1)

    Retourne :
      t       : instants de rupture (t_1, ..., t_{K-1}), 1-based
      cluster : labels des classes (1..K), de longueur n
      C       : critère d'inertie optimal M[n, K]
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = X.shape[0]
    if not isinstance(K, (int, np.integer)) or not 1 <= K <= n:
        raise ValueError("K doit être un entier vérifiant 1 <= K <= n")


    # Initialisation
    D = np.zeros((n + 1, n + 1))
    M = np.zeros((n + 1, K + 1))
    A = np.zeros((n + 1, K + 1), dtype=int)
    t = np.zeros(K, dtype=int)            # t[1..K-1]
    cluster = np.zeros(n + 1, dtype=int)  # cluster[1..n]

    # Étape 1 : matrice triangulaire supérieure D
    for a in range(1, n + 1):
        for b in range(a, n + 1):
            D[a, b] = diam(X, a, b)

    # Étape 2 : critères optimaux
    for i in range(1, n + 1):
        M[i, 1] = D[1, i]

    for k in range(2, K + 1):
        for i in range(k, n + 1):
            ts = np.arange(k, i + 1)                    # t = k, ..., i
            valeurs = M[ts - 1, k - 1] + D[ts, i]
            M[i, k] = valeurs.min()
            A[i, k] = ts[np.argmin(valeurs)]

    # Étape 3 : points de rupture (backtracking)
    k = K - 1
    m = n
    while k >= 1:
        t[k] = A[m, k + 1]
        m = t[k] - 1
        k -= 1

    # Étape 4 : critère et labels
    C = M[n, K]
    bornes = [1] + list(t[1:K]) + [n + 1]   # débuts des segments, puis n+1
    for k in range(1, K + 1):
        cluster[bornes[k - 1]:bornes[k]] = k

    return t[1:], cluster[1:], C


# ---------------------------------------------------------------------------
# 2. Méthodes comparatives : K-means et CAH Ward
# ---------------------------------------------------------------------------


def clustkmeans(X, K):
    """
    Classification en K groupes par K-means (scikit-learn).

    X : array (n, p) (ou (n,) pour des données 1D)
    K : nombre de groupes souhaité

    Retourne :
      cluster : labels des classes (1..K), de longueur n
      C       : inertie intra-classe

    Les groupes ne sont pas nécessairement contigus dans la séquence.
    Si des points sont identiques, moins de K groupes peuvent être obtenus.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = X.shape[0]
    if not isinstance(K, (int, np.integer)) or not 1 <= K <= n:
        raise ValueError("K doit être un entier vérifiant 1 <= K <= n")

    # Plusieurs initialisations, avec une graine fixe pour reproduire les résultats
    modele = KMeans(n_clusters=K, n_init=10, random_state=0)
    cluster = modele.fit_predict(X) + 1
    C = inertie(X, cluster)

    return cluster, C


def clustward(X, K):
    """
    Classification ascendante hiérarchique de Ward (scikit-learn).

    X : array (n, p) (ou (n,) pour des données 1D)
    K : nombre de groupes souhaité

    Retourne :
      cluster : labels des classes (1..K), de longueur n
      C       : inertie intra-classe

    Les groupes ne sont pas nécessairement contigus dans la séquence.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = X.shape[0]
    if not isinstance(K, (int, np.integer)) or not 1 <= K <= n:
        raise ValueError("K doit être un entier vérifiant 1 <= K <= n")

    # Un seul point forme directement un groupe
    if n == 1:
        return np.ones(1, dtype=int), 0.0

    modele = AgglomerativeClustering(n_clusters=K, linkage="ward")
    cluster = modele.fit_predict(X) + 1
    C = inertie(X, cluster)

    return cluster, C


# ---------------------------------------------------------------------------
# 3. Critères d’évaluation : inertie, ARI et silhouette
# ---------------------------------------------------------------------------


def inertie(X, cluster):
    """
    Inertie intra-classe d'une partition (même critère que Fisher).

    X       : array (n, p) (ou (n,) pour des données 1D)
    cluster : labels des classes, de longueur n
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    cluster = np.asarray(cluster)
    if cluster.ndim != 1 or len(cluster) != len(X):
        raise ValueError("cluster doit contenir un label par observation")

    C = 0.0
    for k in np.unique(cluster):
        groupe = X[cluster == k]
        mu = groupe.mean(axis=0)
        C += ((groupe - mu) ** 2).sum()

    return C


def compare_partitions(cluster1, cluster2):
    """
    Compare deux partitions des mêmes observations avec l'indice de Rand ajusté.

    cluster1, cluster2 : labels des classes, de même longueur

    Retourne :
      ARI : 1 pour deux partitions identiques, même si les labels diffèrent ;
            proche de 0 pour un accord au hasard, peut être négatif.
    """
    return adjusted_rand_score(cluster1, cluster2)


def silhouette(X, cluster):
    """
    Silhouette moyenne d'une partition, avec la distance euclidienne.

    X       : array (n, p) (ou (n,) pour des données 1D)
    cluster : labels des classes, de longueur n

    Retourne :
      S : score entre -1 et 1 ; proche de 1 pour des groupes bien séparés.

    Il faut entre 2 et n-1 groupes. Les vraies classes ne sont pas nécessaires.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    cluster = np.asarray(cluster)
    if cluster.ndim != 1 or len(cluster) != len(X):
        raise ValueError("cluster doit contenir un label par observation")

    K = len(np.unique(cluster))
    if not 2 <= K <= len(X) - 1:
        raise ValueError("La silhouette nécessite entre 2 et n-1 groupes")

    return silhouette_score(X, cluster, metric="euclidean")
