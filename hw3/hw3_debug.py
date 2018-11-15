import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# NOTE: There are 5 frequency, 1000 time data in this homework.
# n = 1000, k = 5

n = 1000
threshold = 5
max_sample_range = 6

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
    #return np.matmul(B,a)
    return np.matmul(np.transpose(a),B)

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
    for i in range(len(peak_filted[0])):
        peaks = np.append(peaks, peak_filted[0][i])
    k = 0
    while k < len(peaks):
        if peaks[k]-peaks[k-1] <= 1 and k > 1:
            peaks = np.delete(peaks, k-1)
        k = k + 1
    print(peaks)

    for i in range(len(peaks)):
        start = peaks[i]-max_sample_range
        end = peaks[i]+max_sample_range
        if start < 0:
            start = 0
        if end > peaks[len(peaks)-1]:
            end = peaks[len(peaks)-1]
        peak_rng = np.row_stack((peak_rng, np.array([start, end])))

    print(peak_rng)

    f1 = InvCosineTrans(gen_basis(a, peak_rng[0]), B)
    #f1 = InvCosineTrans(gen_basis(a, [30,50]), B)
    f3 = InvCosineTrans(gen_basis(a, peak_rng[2]), B)
    #f3 = InvCosineTrans(gen_basis(a, [220,225]), B)
    #f3 = InvCosineTrans(gen_basis(a, [363,375]), B)

    np.savetxt('f1.txt', f1)
    np.savetxt('f3.txt', f3)

    plot_wave(f1, 'f1.png')
    plot_wave(f3, 'f3.png')


if __name__ == '__main__':
    main()