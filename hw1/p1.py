import sys
import numpy as np
from graph_gen import *

# Check if the row is all zero or not
def isZeroRow(row):
    for k in row:
        if row[k] != 0:
            return False
    return True

# Add a row to another row
# Using : add_row(row_1, row_2)
# row_1 : source row, row_2 : target row to be added  
def add_row(row_in, row_out):
    for i in range(len(row_in)):
        row_out[i] += row_in[i]

# Add two rows to another row
# Using : add_row(row_1, row_2)
# row_1, row_2 : source row  
def combine_row(row_in1, row_in2):
    row_out = [0 for col in range(len(row_in1))]
    for i in range(len(row_in1)):
        row_out[i] = row_in1[i]+row_in2[i]
    return row_out

# 1     2     3 < row_size = 2
# 4     5     6 < 
# ^     ^     ^
#    col_size = 3
#    -> 2 x 3 matrix
# Define : sets[row][column]
#               ( y ,  x )
def has_cycle(sets):
    #np1 = np.array(sets)

    # ***Column size will not change, so let it be fixed***
    col_size = len(sets[0])

    minor_one_list = list() # List to store -1 component position

    print('MATRIX=' ,end="")
    print(len(sets),end="")
    print('x', end="")
    print(len(sets[0]))
    print(sets)

    while True:
        # Find 1 as leading entry
        
        print(range(len(sets)-1),end="")
        print('x', end="")
        print(range(len(sets[0])-1))

        found_flag = bool(False)
        for i in sets: # Get column size of matrix
            for j in i: # Get row size of matrix
        #for i in range(col_size-2): # Get column size of matrix
        #    for j in range(len(sets)-1): # Get row size of matrix
                if sets[i][j] == 1: # Record current leading entry position
                    found_flag = True
                    break
            if found_flag:
                print(j, end="")
                print(',', end="")
                print(i)
                cursor_col = j 
                cursor_row = i
                break

        curr_row_size = len(sets) # Record current row size
        '''
        print('Found 1 in (', end="")
        print(cursor_x, end="")
        print(',', end="")
        print(cursor_y, end="")
        print(')')'''

        for m in range(curr_row_size): # Search rows for within value is -1
            #        row   colum
            if sets[m][cursor_col] == -1: # Once value is -1, add () to the row
                minor_one_list.append(m)
                '''print('Found -1 in <', end="")
                print(cursor_x, end="")
                print(',', end="")
                print(m, end="")
                print('>')'''
        
        print('-1 list:', end="")
        print(minor_one_list)

        for m in minor_one_list:
            sets.append(combine_row(sets[cursor_row], sets[m]))

            if isZeroRow(sets[-1]):
                return True
        
        sets.pop(cursor_row)

        print(sets)

        if len(minor_one_list) == 0:
            print('---------------')
            return False
        else:
            minor_one_list.clear() # Clear the list for further use

            print('---------------')

    # return True if the graph has cycle; return False if not

    return False

def main():
    p1_list = list()

    test_list = [[[0,1,-1,0],[1,-1,0,0],[-1,0,1,0],[0,-1,0,1]],[[-1,1,0,0],[-1,0,0,1],[0,-1,1,0],[0,-1,0,1]]]
    test_list2 = [[[0, 0, -1, 1, 0, 0],[0, 1, 0, 0, -1, 0],[0 ,0 ,0 ,-1, 0 ,1],[0 ,0, 1,0, 0,-1],[-1, 1, 0, 0, 0, 0]],[[0, 0, -1, 1, 0, 0],[0, 1, 0, 0, -1, 0],[0 ,0 ,0 ,-1, 0 ,1],[0 ,0, 1,0, 0,-1],[-1, 1, 0, 0, 0, 0]]]

    if len(sys.argv) <= 1:
        p1_list = get_p1('r07')
    else:
        p1_list = get_p1(sys.argv[1])
    for sets in p1_list:
    #for sets in test_list2:
    #for sets in test_list:
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

        #print(sets)

        if has_cycle(sets):
            print('Yes')
        else:
            print('No')

if __name__ == '__main__':
    main()
