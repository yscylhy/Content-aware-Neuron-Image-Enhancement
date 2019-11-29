from cane import cane_2d
import imageio
import time
import numpy as np


if __name__ == "__main__":
    image_path = "./images/2d_example.png"
    write_path = "./results/2d_results.png"
    smooth_degree = 0.006

    # -- read image
    image = imageio.imread(image_path)
    image = image / np.max(image)

    # -- do enhancement
    tic = time.time()
    enhanced_image, enhance_hist = cane_2d(image, smooth_degree)
    toc = time.time()
    print("Enhancement done: {:.2f}s. The image is of size {}*{}".format(toc-tic, image.shape[0], image.shape[1]))

    # -- write the enhanced results
    imageio.imwrite(write_path, (enhanced_image*255).astype(np.uint8))



