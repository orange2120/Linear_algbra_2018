import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# NOTE: There are 5 frequency, 1000 time data in this homework.
# n = 1000, k = 5

n = 1000

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

    # Generate transform matrix
    B = np.ndarray(shape=(n,n), dtype=float)
    for i in range(n):
        B[0,i] = 1/(n ** 0.5)
    for i in range(1,n):
        for j in range(n):
            B[i,j] = ((2.0/n)**0.5)*np.cos(((i+0.5)*(j)*np.pi)/n)
    
    a = CosineTrans(x, B)
    plot_ak(a, 'b06602037_freq.png')

    f1 = InvCosineTrans(gen_basis(a, [30,40]), B)
    f3 = InvCosineTrans(gen_basis(a, [363,375]), B)

    np.savetxt('b06602037_f1.txt', f1)
    np.savetxt('b06602037_f3.txt', f3)

if __name__ == '__main__':
    main()