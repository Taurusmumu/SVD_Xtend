import argparse

from defocus_estimate import *


def get_args():
    parser = argparse.ArgumentParser(description='Defocus map estimation from a single image, '
                                                 'S. Zhuo, T. Sim - Pattern Recognition, 2011 - Elsevier \n')

    parser.add_argument('-i', metavar='--image', default='/ssd2/AMC_zstack_2_patches/pngs_mid/24S 056115;F;9;;FA0824;1_241226_045830/z00/patch_39_5268_10000.png',
                        type=str, help='Defocused image \n')

    args = parser.parse_args()
    image = args.i

    return {'image': image}


if __name__ == '__main__':

    args = get_args()

    img = cv2.imread(args['image'])
    fblurmap = estimate_bmap_laplacian(img, sigma_c = 1, std1 = 1, std2 = 1.5)

    cv2.imwrite(args['image'] + '_bmap.png', np.uint8((fblurmap / fblurmap.max()) * 255))
    np.save(args['image'] + '_bmap.npy', fblurmap)


