import numpy as np
import svd_img as u
from svd_img import svd_compress
from svd_img import svd_compress_single
import matplotlib.pyplot as plt

img_path = 'img/vegetable_english.jpg'
imArr = u.load_image(img_path)

def main():
    #ks = [1, 5, 50, 150, 400, 1050, 1289]
    ks = [1, 2, 3, 4, 5]
    err = []
    channel = 1

    rank_img_sum = np.zeros((imArr.shape[0], imArr.shape[1]))

    for k in ks: 
        print("Perform SVD for k=%d ..." % k, end='\r\n')
        imArr_compressed = svd_compress_single(imArr, K=k, ch=channel)
        
        rank_img_sum += imArr_compressed

        err += [u.approx_error(imArr[:, :, channel], imArr_compressed)]
        u.save_image(imArr_compressed, 'test/result_{}_{}.jpg'.format(channel, k))
    
    rank_img_sum /= len(ks)
    rank_img_sum.astype(np.uint8)
    rank_img_sum[:, :] -= rank_img_sum[:, :].min()
    rank_img_sum[:, :] /= rank_img_sum[:, :].max()
    rank_img_sum[:, :] *= 255
    u.save_image(rank_img_sum, 'test/result_sum_{}.jpg'.format(channel))
    

    #plt.imshow(imArr_compressed)
    plt.show()
    fp = 'test/curve_{}.png'.format(channel)
    u.plot_curve(ks, err, fpath=fp, show=False)

if __name__ == '__main__':
    main()