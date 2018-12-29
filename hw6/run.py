import hw6_b06602037 as u
from hw6_b06602037 import svd_compress
import matplotlib.pyplot as plt

img_path = 'img/vegetable_english.jpg'
imArr = u.load_image(img_path)

def main():
    ks = [1, 5, 50, 150, 400, 1050, 1289]
    err = []
    for k in ks: 
        print("Perform SVD for k=%d ..." % k, end='\r\n')
        imArr_compressed = svd_compress(imArr, K=k)
        err += [u.approx_error(imArr, imArr_compressed)]
        u.save_image(imArr_compressed, 'result_{}.jpg'.format(k))

    plt.imshow(imArr_compressed)
    plt.show()
    u.plot_curve(ks, err, show=False)

if __name__ == '__main__':
    main()