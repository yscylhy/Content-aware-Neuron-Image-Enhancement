from cane import cane_3d
import imageio
import numpy as np
import time

if __name__ == "__main__":
    image_path = "./images/3d_example.tif"
    volume_write_path = "./results/3d_results.tif"
    projection_before_enhance_write_path = "./results/3d_projection_before.png"
    projection_aftter_enhance_write_path = "./results/3d_projection_after.png"

    smooth_degree = 0.006

    # -- read image
    image = imageio.volread(image_path)
    image = image / np.max(image)

    # -- do enhancement
    tic = time.time()
    enhanced_image, enhance_hist = cane_3d(image, smooth_degree)
    toc = time.time()
    print("Enhancement done: {:.2f}s. The image is of size {}*{}*{}".format(toc - tic, image.shape[0], image.shape[1], image.shape[2]))

    # -- write the enhanced results
    before_project = (np.max(image, axis=0)*255).astype(np.uint8)
    after_project = (np.max(enhanced_image, axis=0)*255).astype(np.uint8)

    imageio.volsave(volume_write_path, (enhanced_image * 255).astype(np.uint8))
    imageio.imwrite(projection_before_enhance_write_path, before_project)
    imageio.imwrite(projection_aftter_enhance_write_path, after_project)

