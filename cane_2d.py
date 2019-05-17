# pip3 install scikit-image

import os
import numpy as np
from skimage.io import imread, imsave
from scipy.fftpack import fft2, ifft2, fftshift


class CaNE:
    def __init__(self, kappa1=2, kappa2=2, kappa3=2000):
        # kappa1, kappa2 and kappa3 are empirically set parameters for numerical stability.
        # It is the safe to use the default settings in most cases.
        self.kappa1 = kappa1
        self.kappa2 = kappa2
        self.kappa3 = kappa3

        self.img_id = None
        self.org_img = None
        self.enhanced_img = None
        self.Denormin2 = None
        self.fourier_window = None

    def read_img(self, read_path):
        img_name = os.path.split(read_path)[-1]
        img = np.array(imread(read_path))
        assert img.ndim == 2, "Check the dimensionality of input image!"
        self.org_img = img / img.max()
        self.img_id = img_name.split('.')[0]

    def write_img(self, write_path):
        if not os.path.exists(write_path):
            os.mkdir(write_path)
        enhanced_img = (self.enhanced_img*255).astype(np.uint8)
        imsave(os.path.join(write_path, self.img_id + '.png'), enhanced_img)

    def _get_neuron_prominence(self, img, window_size):
        H, W = img.shape
        # Centered gradients
        dx, dy = np.zeros((H, W)), np.zeros((H, W))
        dx[:, 1:W - 1] = (img[:, 2:] - img[:, :W - 2]) / 2
        dy[1:H - 1, :] = (img[2:, :] - img[:H - 2, :]) / 2

        E_dx = np.real(fftshift(ifft2(fft2(dx**2) * self.fourier_window) * window_size ** 2))
        E_dy = np.real(fftshift(ifft2(fft2(dy**2) * self.fourier_window) * window_size ** 2))
        E_dxy = np.real(fftshift(ifft2(fft2(dx*dy) * self.fourier_window) * window_size ** 2))
        delta = np.abs((E_dx + E_dy) ** 2 - 4 * (E_dx * E_dy - E_dxy ** 2)) ** 0.5
        b = E_dx + E_dy
        lamada1 = np.abs((b - delta) / 2) ** 0.5
        lamada2 = np.abs((b + delta) / 2) ** 0.5
        Q_map = lamada2 * (abs(lamada2 - lamada1) / (lamada2 + lamada1))
        return Q_map

    def enhance(self, smooth_degree, iter_N=20, window_size=9):
        # smooth_degree
        # window_size:

        H, W = self.org_img.shape
        # get the optical transfer function of gradient on x and y directions. (similar to psf2otf in Matlab)
        dx, dy, spatial_window = np.zeros((H, W)), np.zeros((H, W)), np.zeros((H, W))
        dx[H // 2, W // 2 - 1:W // 2 + 1] = [-1, 1]
        dy[H // 2 - 1:H // 2 + 1, W // 2] = [-1, 1]
        h_start, h_end = H//2 - window_size//2, H//2 + window_size//2 + 1
        w_start, w_end = W//2 - window_size//2, W//2 + window_size//2 + 1
        spatial_window[h_start:h_end, w_start:w_end] = 1/(window_size**2)
        otfDx, otfDy = fft2(dx), fft2(dy)
        self.fourier_window = fft2(spatial_window)

        self.Denormin2 = abs(otfDx) ** 2 + abs(otfDy) ** 2

        beta = self.kappa2 * smooth_degree
        gamma = self.kappa3 * smooth_degree

        # record the convergence by mean squared difference to the original image
        diff_hist = []
        img = self.org_img.copy()

        for ite_num in range(iter_N):
            if ite_num % 10 == 0:
                neuron_prominence = self._get_neuron_prominence(img, window_size)
                weight = 1. / neuron_prominence
            denorminator2 = 1 + beta * self.Denormin2
            # Step (3) in Alg. 1 in [1].
            h = np.hstack([img[:, 1:] - img[:, :W-1], img[:, 0:1] - img[:, W-1:]])
            v = np.vstack([img[1:, :] - img[:H-1, :], img[0:1, :] - img[H-1:, :]])

            # Step (4) in Alg. 1 in [1].
            diff_norm2 = (h ** 2 + v ** 2)
            t = diff_norm2 < (smooth_degree * weight) / beta
            h[t] = 0
            v[t] = 0

            # Step (5) in Alg. 1 in [1].
            d = img - self.org_img

            # Step (6) in Alg. 1 in [1].
            t = (d**2) < (1/gamma)
            d[t] = 0

            # Step(2) in Alg.1 in [1].
            Normin2 = -np.hstack([h[:, 0:1] - h[:, W-1:], h[:, 1:] - h[:, :W-1]]) \
                        - np.vstack([v[0:1, :] - v[H-1:, :], v[1:, :] - v[:H-1, :]])

            Normin1 = fft2(self.org_img + d)
            FS = (Normin1 + beta * fft2(Normin2)) / denorminator2
            img = ifft2(FS).real

            # Step(7) in Alg.1 in [1].
            beta = beta * self.kappa1
            gamma = gamma * self.kappa1
            diff_hist.append(np.mean((img - self.org_img) ** 2))

        self.enhanced_img = img
        return img, diff_hist

