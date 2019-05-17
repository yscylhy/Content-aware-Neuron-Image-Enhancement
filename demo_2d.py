from cane_2d import CaNE
from time import time

if __name__ == "__main__":
    image_path = "./images/2d_example.png"
    write_path = "./results"
    smooth_degree = 0.006

    tic = time()
    cane = CaNE()
    cane.read_img(image_path)
    cane.enhance(smooth_degree)
    cane.write_img(write_path)
    toc = time()
    print('Enhanced in {:.2}s. Enhanced image is written to ./results'.format(toc-tic))




