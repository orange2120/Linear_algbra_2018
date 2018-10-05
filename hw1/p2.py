import sys
import numpy as np
from graph_gen import *

def has_cycle(sets):
    # TODO
    # return True if the graph has cycle; return False if not

    return False

def main():
    p2_list = list()
    if len(sys.argv) <= 1:
        p2_list = get_p2('r07')
    else:
        p2_list = get_p2(sys.argv[1])
    for sets in p2_list:
        '''
          HINT: You can `print(sets)` to show what the matrix looks like
            If we have a directed graph with 2->3 4->1 3->5 5->2 0->1
                   0  1  2  3  4  5
                0  0  1  0  0  0  0
                1  0  0  0  0  0  0
                2  0  0  0  1  0  0
                3  0  0  0  0  0  1
                4  0  1  0  0  0  0
                5  0  0  1  0  0  0
            The size of the matrix is (6,6)
        '''
        if has_cycle(sets):
            print('Yes')
        else:
            print('No')

if __name__ == '__main__':
    main()
