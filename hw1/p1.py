import sys
import numpy as np
from graph_gen import *

def has_cycle(sets):
    # TODO
    # return True if the graph has cycle; return False if not

    return False

def main():
    p1_list = list()
    if len(sys.argv) <= 1:
        p1_list = get_p1('r07')
    else:
        p1_list = get_p1(sys.argv[1])
    for sets in p1_list:
        '''
          HINT: You can `print(sets)` to show what the matrix looks like
            If we have a directed graph with 2->3 4->1 3->5 5->2 0->1
                   0  1  2  3  4  5
                0  0  0 -1  1  0  0
                1  0  1  0  0 -1  0
                2  0  0  0 -1  0  1
                3  0  0  1  0  0 -1
                4 -1  1  0  0  0  0
            The size of the matrix is (5,6)
        '''
        if has_cycle(sets):
            print('Yes')
        else:
            print('No')

if __name__ == '__main__':
    main()
