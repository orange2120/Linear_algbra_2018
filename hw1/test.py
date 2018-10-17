import numpy as np

a = [[1,2,3,4],[5,6,7,8]]
b = [[0,0,1,0],[0,1,0,1],[1,1,1,0],[0,1,0,0]]
c = [[1,0,2,0],[1,1,1,1],[0,0,0,0],[0,1,0,0]]

def check_zero(ar):
    for m in range(ar.shape[0]): # Search in column
        All_zero = True
        for n in range(ar.shape[1]): # Search in row
            if ar[m,n] != 0:
                All_zero = False
                break
        if All_zero:
            return True
    return False

def main():
    arr = np.array([[0,1,2,3,4],[5,6,7,8,9]])
    a2 = np.array(c)

    

    print(arr[0,0])
    print(arr[1,3])
          #row col 
    print(arr[1][2])
    print(arr[1,2])
    print(arr[0])
    print(arr.shape)
    print(arr.shape[1])

    print('a=')
    print(a[1][2])
    print(a[1])

    print(check_zero(a2))

    


if __name__ == '__main__':
    main()
