import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# NOTE: There are 5 frequency, 1000 time data in this homework.
# n = 1000, k = 5

n = 1000
threshold = 0.1
max_sample_range = 70

def plot_wave(x, path = './wave.png'):
    plt.gcf().clear()
    plt.plot(x)
    plt.xlabel('n')
    plt.ylabel('xn')
    plt.savefig(path)

def plot_ak(a, path = './freq.png'):
    plt.gcf().clear()
    # Only plot the mag of a
    a = np.abs(a)
    plt.plot(a)
    plt.xlabel('k')
    plt.ylabel('ak')
    plt.savefig(path)

def CosineTrans(x, B):
    # implement cosine transform
    return np.matmul(np.linalg.inv(B),x)

def InvCosineTrans(a, B):
    # implement inverse cosine transform
    return np.matmul(B,a)

"""
def gen_basis(N):
    basis = np.zeros(n)
    return basis"""

def gen_basis(a, rng):
    basis = np.zeros(n)
    for i in range(rng[0], rng[1]):
        basis[i] = a[i]
    return basis

def main():
    signal_path = sys.argv[1]
    if len(signal_path)< 1:
        print("Invalid file path!")
        return

    x = np.loadtxt(signal_path)
    plot_wave(x, 'x.png')

    # Generate transform matrix
    B = np.ndarray(shape=(n,n), dtype=float)
    for i in range(n):
        B[0,i] = 1/(n ** 0.5)
    for i in range(1,n):
        for j in range(n):
            B[i,j] = ((2.0/n)**0.5)*np.cos(((i+0.5)*(j)*np.pi)/n)

    #np.savetxt('B.txt', B)
    #np.savetxt('inv_b.txt', np.linalg.inv(B))
    
    a = CosineTrans(x, B)
    plot_ak(a, 'a.png')

    #np.savetxt('a.txt', a)
    peak_rng = np.ndarray(shape=(0,2), dtype=int)
    peaks = np.array([], dtype=int)
    peak_filted = np.where(a > threshold)
    print(peak_filted)
    # Get appropriate interval of frequency
    first = 0
    max_peak = 0
    
    for i in range(len(peak_filted[0])):
        if peak_filted[0][i] > max_peak:
            
            max_peak = peak_filted[0][i]
        if peak_filted[0][i] < max_peak:
            peaks = np.append(peaks, peak_filted[0][i])
            max_peak = 0

    for k in range(len(peak_filted[0])):
        if (peak_filted[0][k]-first) > max_sample_range or k == len(peak_filted[0])-1:
            peak_rng = np.row_stack((peak_rng, np.array([first, (peak_filted[0][k-1])])))
            first = peak_filted[0][k]

    print(peak_rng)     

    f1 = InvCosineTrans(gen_basis(a, peak_rng[0]), B)
    #f1 = InvCosineTrans(gen_basis(a, [0,50]), B)
    #f3 = InvCosineTrans(gen_basis(a, peak_rng[1]), B)
    #f3 = InvCosineTrans(gen_basis(a, [190,210]), B)
    f3 = InvCosineTrans(gen_basis(a, [363,375]), B)

    np.savetxt('f1.txt', f1)
    np.savetxt('f3.txt', f3)

    plot_wave(f1, 'f1.png')
    plot_wave(f3, 'f3.png')

if __name__ == '__main__':
    main()