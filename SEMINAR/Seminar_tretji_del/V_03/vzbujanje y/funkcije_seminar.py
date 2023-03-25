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
    eigenvalues = np.sqrt(eigenvalues)
    eigenvectors = np.zeros((len(eigenvec[0]),len(eigenvec)))
    eigenvectors[:, 0] = eigenvec[:, 1]
    eigenvectors[:, 1] = eigenvec[:, 0]
    #eigenvectors[:, 0] = eigenvectors[:, 0]/eigenvectors[0,0]
    #eigenvectors[:, 1] = eigenvectors[:, 1]/eigenvectors[0,1]
    return eigenvalues, eigenvectors

def print_eig(eigenvalues, eigenvectors):
    print("eigenvalues")
    #print(np.sqrt(eigenval))
    print(eigenvalues)
    print(" ")
    print("eigenvectors")
    print(eigenvectors)

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

## dodatne funkcije

def analiza1(m1=10, m2=350, k1=260*1000, k2=40*1000, delta1=0.1, delta2= 0.3):
    ''' vrača :
    M, K, delta1, delta2, eigenvalues, eigenvectors, omega1, 
    omega2, omega1D, omega2D, modal_mass_m, modal_stiffness_m'''
    
    M = mass_matrix(m1, m2)
    K = stiffness_matrix(k1, k2)

    A = np.linalg.inv(M) @ K

    eigenval, eigenvec = np.linalg.eig(A)
    eigenvalues, eigenvectors = eig(eigenval, eigenvec)
    omega1 = eigenvalues[0]
    omega2 = eigenvalues[1] 
    omega1D = omega1 * np.sqrt(1 - delta1**2)
    omega2D = omega2 * np.sqrt(1 - delta2**2)
    modal_mass_m, modal_stiffness_m = modal_mass_stiffness(eigenvectors, M, K)
    return M, K, delta1, delta2, eigenvalues, eigenvectors, omega1, omega2, omega1D, omega2D, modal_mass_m, modal_stiffness_m

def izpiseig(eigenvalues, eigenvectors):
    print("eigenvalues")
    #print(np.sqrt(eigenval))
    print(eigenvalues)
    print(" ")
    print("eigenvectors")
    print(eigenvectors)
def izpisMK(M, K):
    print("M")
    print(M)
    print(" ")
    print("K")
    print(K)
def izpisomega(omega1, omega2, omega1D, omega2D):
    print(f'omega1: \t {omega1:.2f} rad/s')
    print(f'omega2: \t {omega2:.2f} rad/s')
    print(f'omega1D: \t {omega1D:.2f} rad/s')
    print(f'omega2D: \t {omega2D:.2f} rad/s')

def vzbujanje(M, t, delta_t, eigenvectors, hitrost=10, L=0.5, o=0.1):
    '''enote podanih podatkov:
    hitrost: km/h
    L: m
    o: m
    delta_t: s
    
    return vrača:
    ff, Y, Y_v, Y_a, Ft, Modal_Ft 
    '''
    hitrost = hitrost/3.6; # m/s
    ff = 1/(2*L/hitrost); # Hz
    y = np.sin(2 * np.pi * ff * t + 3*np.pi/2) + 1
    t1 = 2/(2*ff);
    t11 = t1 % (t[1] - t[0])
    T1 = t1 - t11
    Y = np.zeros_like(t)
    t_index = np.where(t == T1)
    t2 = 1000 + t_index[0][0] + 1
    Y[1000:t2] = y[0:t2-1000] * o;
    Y_v = np.gradient(Y, delta_t)
    Y_a = np.gradient(Y_v, delta_t)
    B = np.ones((2,1))
    Ft = np.zeros((2, len(t)))
    Ft = - M @ B * Y_a
    Modal_Ft = np.zeros((2, len(t)))
    Modal_Ft = eigenvectors.T @ Ft
    return ff, Y, Y_v, Y_a, Ft, Modal_Ft

def konvolucija(t, m1, m2, Modal_Ft, omega1, omega2, omega1D, omega2D, delta1, delta2, delta_t=0.0001):
    '''return: 
    eta
    etaD'''
    g_1 = 1/omega1 * np.sin(omega1 * t)
    g_2 = 1/omega2 * np.sin(omega2 * t)

    g_1d = 1/omega1 * np.exp(-delta1 * omega1 * t) * np.sin(omega1D * t)
    g_2d = 1/omega2 * np.exp(-delta2 * omega2 * t) * np.sin(omega2D * t)
    #1
    eta1 = 1/m1 * np.convolve(Modal_Ft[0,:], g_1) * (t[1] - t[0])
    eta1d = 1/m1 * np.convolve(Modal_Ft[0,:], g_1d) * (t[1] - t[0])

    #2
    eta2 = 1/m2 * np.convolve(Modal_Ft[1,:], g_2) * (t[1] - t[0])
    eta2d = 1/m2 * np.convolve(Modal_Ft[1,:], g_2d) * (t[1] - t[0])

    eta = np.zeros((2, 2*len(t) - 1))
    eta[0,:] = eta1
    eta[1,:] = eta2

    etaD = np.zeros((2, 2*len(t) - 1))
    etaD[0,:] = eta1d
    etaD[1,:] = eta2d
    return eta, etaD

#
def rezultat(eta, etaD, eigenvectors, t, delta_t, Y):
    '''return:
    u, uD'''
    z1_1 = eigenvectors[:,0] @ eta
    z1_2 = eigenvectors[:,1] @ eta

    z1_1D = eigenvectors[:,0] @ etaD
    z1_2D = eigenvectors[:,1] @ etaD
    
    x1_1 = z1_1[:len(t)] + Y[:len(t)]
    x1_2 = z1_2[:len(t)] + Y[:len(t)]

    x1_1D = z1_1D[:len(t)] + Y[:len(t)]
    x1_2D = z1_2D[:len(t)] + Y[:len(t)]
    
    v11 = np.gradient(x1_1)/delta_t
    a11 = np.gradient(v11)/delta_t
    v12 = np.gradient(x1_2, delta_t)
    a12 = np.gradient(v12, delta_t)

    v11D = np.gradient(x1_1D, delta_t)
    a11D = np.gradient(v11D, delta_t)
    v12D = np.gradient(x1_2D, delta_t)
    a12D = np.gradient(v12D, delta_t)
    x = np.array([x1_1, x1_2])
    xD = np.array([x1_1D, x1_2D])
    a = np.array([a11, a12])
    aD = np.array([a11D, a12D])
    return x, xD, a, aD

def abc(a=1):
    return a + 1