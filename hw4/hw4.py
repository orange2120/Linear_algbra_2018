import sys
import numpy as np
import pandas as pd

def load(fname):
    f = open(fname, 'r').readlines()
    n = len(f)
    ret = {}
    for l in f:
        l = l.split('\n')[0].split(',')
        i = int(l[0])
        ret[i] = {}
        for j in range(n):
            if str(j) in l[1:]:
                ret[i][j] = 1
            else:
                ret[i][j] = 0
    ret = pd.DataFrame(ret).values
    return ret

def get_tran(g):
    cnt = 0
    n = g.shape[0]
    g = np.transpose(g)
    tm = np.zeros(shape=(n,n), dtype=float)

    for i in range(n):
        cnt = np.count_nonzero(g[i])
        for j in range(n):
            if g[i][j] == 1.0:
                tm[j][i] = 1/cnt
        cnt = 0 # reset counter
    return tm

def cal_rank(t, d = 0.85, max_iterations = 1000, alpha = 0.001):
    cnt = 0
    n = t.shape[0]
    r_zero = np.zeros(shape=(n,1), dtype=float)
    r = np.ndarray(shape=(n,1), dtype=float)
    rn = np.zeros(shape=(n,1), dtype=float)

    for i in range(n):
            r_zero[i] = 1/n # fill the zero column
    
    #print(r_zero)
    #print((1-d)*r_zero)

    while cnt < max_iterations:
        rn = (1-d)*r_zero + d*(np.matmul(t, r))
        if dist(rn, t) <= alpha:
            print(cnt)
            break
        r = rn
        cnt = cnt + 1
    #print(cnt)
    return rn

def save(t, r):
    np.savetxt('1.txt', t)
    np.savetxt('2.txt', r)
    return

def dist(a, b):
    return np.sum(np.abs(a-b))

def main():
    graph = load(sys.argv[1])
    transition_matrix = get_tran(graph)
    rank = cal_rank(transition_matrix)
    save(transition_matrix, rank)

if __name__ == '__main__':
    main()

