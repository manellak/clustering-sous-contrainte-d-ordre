import numpy as np


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
    if not 1 <= K <= n:
        raise ValueError("K doit vérifier 1 <= K <= n")

    
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

    # Étape 5 : critère et labels
    C = M[n, K]
    bornes = [1] + list(t[1:K]) + [n + 1]   # débuts des segments, puis n+1
    for k in range(1, K + 1):
        cluster[bornes[k - 1]:bornes[k]] = k

    return t[1:], cluster[1:], C