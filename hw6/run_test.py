import svd_img as u
from svd_img import svd_compress
import matplotlib.pyplot as plt

img_path = 'img/vegetable_english.jpg'
imArr = u.load_image(img_path)

def main():
    #ks = [1, 5, 50, 150, 400, 1050, 1289]
    ks = [1, 2, 3, 4, 5]
    err = []
    channel = 1
    for k in ks: 
        print("Perform SVD for k=%d ..." % k, end='\r\n')
        imArr_compressed = svd_compress(imArr, K=k, ch=channel)
        err += [u.approx_error(imArr, imArr_compressed)]
        u.save_image(imArr_compressed, 'test/result_{}_{}.jpg'.format(channel, k))

    #plt.imshow(imArr_compressed)
    plt.show()
    fp = 'test/curve_{}.png'.format(channel)
    u.plot_curve(ks, err, fpath=fp, show=False)

if __name__ == '__main__':
    main()