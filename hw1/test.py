
a = [[1,2,3,4],[5,6,7,8]]
b = [[-1,0,-3,4],[1,1,1,1]]

def add_row(row_in, row_out):
    for i in range(len(row_in)):
        row_out[i] += row_in[i]

def combine_row(row_in1, row_in2):
    row_out = [0 for col in range(len(row_in1))]
    for i in range(len(row_in1)):
        row_out[i] = row_in1[i]+row_in2[i]
    return row_out

def main():

    print(len(a))
    print(len(a[0]))

    '''
    for i in range(len(a[0])):
        print('[ ', end="")
        for j in range(len(a)):
            print(a[i][j], end="")
            print(' ', end="")
        print(']')
     print(a)
     '''

    print(combine_row(a[0],a[1]))

    add_row(b[0], a[0])
    print(a)

if __name__ == '__main__':
    main()