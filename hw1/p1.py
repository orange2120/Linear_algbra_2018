import sys
import numpy as np
from graph_gen import *

def has_cycle(sets):
    # return True if the graph has cycle; return False if not
    arr = np.array(sets)

    for j in range(arr.shape[1]): # Search in column
        for i in range(arr.shape[0]): # Search in row
            if arr[i,j] == 1:
                col_mark = j
                row_mark = i
                one_flag = True
                has_min_one = False

                for k in range(arr.shape[0]):
                    if arr[k,col_mark] == -1:
                        has_min_one = True
                        arr = np.vstack([arr,arr[k]])
                        b = arr[row_mark]
                        arr[-1] = arr[-1]+b

                if has_min_one:
                    arr = np.delete(arr,(row_mark),axis=0)

                for m in range(arr.shape[0]): # Search in column
                    All_zero = True
                    for n in range(arr.shape[1]): # Search in row
                        if arr[m,n] != 0:
                            All_zero = False
                            break
                    if All_zero:
                        return True
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
