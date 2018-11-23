import sys
import numpy as np
import pandas as pd

def load(fname):
    f = open(fname, 'r').readlines()
    n = len(f)
    ret = {}
    for l in f:
        l = l.split('\n')[0].split(',')
        i = l[0]
        ret[i] = {}
        for j in range(n):
            if str(j) in l[1:]:
                ret[i][str(j)] = 1
            else:
                ret[i][str(j)] = 0
    ret = pd.DataFrame(ret).values
    return ret

def get_tran(g):
    # TODO

def cal_rank(t, d = 0.85, max_iterations = 1000, alpha = 0.001):
    # TODO

def save(t, r):
    # TODO

def dist(a, b):
    return np.sum(np.abs(a-b))

def main():
    graph = load(sys.argv[1])
    transition_matrix = get_tran(graph)
    rank = cal_rank(transition_matrix)
    save(transition_matrix, rank)

if __name__ == '__main__':
    main()

