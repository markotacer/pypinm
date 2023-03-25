import numpy as np;
import matplotlib.pyplot as plt;

def neodvisna_spremenljivka_t(t):
    ele = t * 1000;
    t = np.linspace(0, ele/1000, ele + 1)
    return t;

def mass_matrix(m1, m2):
    M = np.zeros((2,2))
    M[0,0] = m1
    M[1,1] = m2
    return M;

def stiffness_matrix(k1, k2):
    K = np.zeros((2,2))
    K[0,0] = k1+k2
    K[0,1] = -k2
    K[1,0] = -k2
    K[1,1] = k2
    return K;

def eig(eigenval, eigenvec):
    eigenvalues = np.zeros((len(eigenval)))
    eigenvalues[0] = eigenval[1]
    eigenvalues[1] = eigenval[0]
    eigenvalues = np.sqrt(eigenvalues) # korenimo, da dobimo lastne frekvence
    print("eigenvalues")
    #print(np.sqrt(eigenval))
    print(eigenvalues)
    print(" ")
    eigenvectors = np.zeros((len(eigenvec[0]),len(eigenvec)))
    eigenvectors[:, 0] = eigenvec[:, 1]
    eigenvectors[:, 1] = eigenvec[:, 0]
    #eigenvectors[:, 0] = eigenvectors[:, 0]/eigenvectors[0,0]
    #eigenvectors[:, 1] = eigenvectors[:, 1]/eigenvectors[0,1]
    print("eigenvectors")
    #print(eigenvec)
    print(eigenvectors)
    return eigenvalues, eigenvectors

def modal_mass_stiffness(eigenvectors, M, K):
    M1 = eigenvectors[:,0].T @ M @ eigenvectors[:,0]
    M2 = eigenvectors[:,1].T @ M @ eigenvectors[:,1]
    modal_mass = np.zeros((2,2))
    modal_mass[0,0] = M1
    modal_mass[1,1] = M2
    
    K1 = eigenvectors[:,0].T @ K @ eigenvectors[:,0]
    K2 = eigenvectors[:,1].T @ K @ eigenvectors[:,1]
    modal_stiffness = np.zeros((2,2))
    modal_stiffness[0,0] = K1
    modal_stiffness[1,1] = K2
    return modal_mass, modal_stiffness
