def masna_togostna_matrika(m1, m2, k1, k2):

    M = np.zeros((2, 2))
    K = np.zeros((2, 2))
    M[0, 0] = m1
    M[1, 1] = m2
    K[0, 0] = k1
    K[0, 1] = -k1
    K[1, 0] = -k1
    K[1, 1] = k1+k2

    M = np.array([[m1, 0], [0, m2]])
    K = np.array([[k1+k2, -k2], [-k2, k2]])
    return [M, K]
