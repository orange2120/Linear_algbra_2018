import sys
import numpy as np
from graph_gen import *

# Find elements >1 in diagonal of matrix
def has_big_than_one(arr):
    for i in range(0, arr.shape[0]-1):
            if arr[i][i] > 1:
                return True
    return False

def has_cycle(sets):
    # return True if the graph has cycle; return False if not
    a =  np.array(sets) # Create a 2d array object

    node_arr = a[(a == 1)]
    nodes = node_arr.size # Get amount of nodes
    n = nodes

    '''print('---------------')
    print('n=', end="")
    print(n)
    print(a)
    print('---------------')'''

    b = a

    while n > 0:
        b = np.dot(b,a)

        '''print(n)
        print(b)
        print("")'''

        if has_big_than_one(b):
            return True
        n = n-1

    diag_arr = np.diag(b)
    diag_arr = diag_arr[(diag_arr != 0)]
    if diag_arr.size > 0:
        return True
    
    return False

def main():


    test_list = [[[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,1,0,0]],[[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,1,0,0]]]

    p2_list = list()
    if len(sys.argv) <= 1:
        p2_list = get_p2('r07')
    else:
        p2_list = get_p2(sys.argv[1])

    #for sets in test_list:
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
