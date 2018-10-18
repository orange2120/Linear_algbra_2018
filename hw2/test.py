import util as u
import numpy as np

def c2i_table(arg):
    sw = {
        'A' : 0,
        'B' : 1,
        'C' : 2,
        'D' : 3,
        'E' : 4,
        'F' : 5,
        'G' : 6,
        'H' : 7,
        'I' : 8,
        'J' : 9,
        'K' : 10,
        'L' : 11,
        'M' : 12,
        'N' : 13,
        'O' : 14,
        'P' : 15,
        'Q' : 16,
        'R' : 17,
        'S' : 18,
        'T' : 19,
        'U' : 20,
        'V' : 21,
        'W' : 22,
        'X' : 23,
        'Y' : 24,
        'Z' : 25,
        '_' : 26,
        '.' : 27,
        ',' : 28,
        '?' : 29,
        '!' : 30
    }
    return sw.get(arg, "None")

def i2c_table(arg):
    sw = {
        0  : 'A',
        1  : 'B',
        2  : 'C',
        3  : 'D',
        4  : 'E',
        5  : 'F',
        6  : 'G',
        7  : 'H',
        8  : 'I',
        9  : 'J',
        10 : 'K',
        11 : 'L',
        12 : 'M',
        13 : 'N',
        14 : 'O',
        15 : 'P',
        16 : 'Q',
        17 : 'R',
        18 : 'S',
        19 : 'T',
        20 : 'U',
        21 : 'V',
        22 : 'W',
        23 : 'X',
        24 : 'Y',
        25 : 'Z',
        26 : '_',
        27 : '.',
        28 : ',',
        29 : '?',
        30 : '!'
    }
    return sw.get(arg, "None")

#          string int
def decode(cipher, key):
    col_length = int(len(cipher)/3)
    c = np.array([], dtype=int)

    k = np.fromstring(key, dtype=int, sep=' ')
    k = np.reshape(k,(3,3))

    for i in range(len(cipher)):
        c = np.append(c, c2i_table(cipher[i]))
    c = np.reshape(c,(3,col_length))

    print(c)
    print(k)

    plain = np.matmul(u.inv_key(k),c)
    print(plain)
    plain = np.mod(plain, 31)
    print(plain)

    p_text = str()
    for i in plain.flat:
        p_text = p_text+i2c_table(i)

    return p_text

def main():

    cipher_text = 'VJWUV,EDI'
    key = '25 8 25 9 9 16 28 21 18'
    #print(len(cipher_text))
    #print(cipher_text)
    print('Plain:')
    print(decode(cipher_text, key))

if __name__ == '__main__':
    main()