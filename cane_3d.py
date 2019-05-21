# pip3 install scikit-image

import os
import numpy as np
from skimage.io import imread, imsave
from skimage.external import tifffile
from numpy.fft import fftn, ifftn, fftshift

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
        assert img.ndim == 3, "Check the dimensionality of input image!"
        self.org_img = img / img.max()
        self.img_id = img_name.split('.')[0]

    def write_img(self, write_path):
        if not os.path.exists(write_path):
            os.mkdir(write_path)
        enhanced_img = (self.enhanced_img*255).astype(np.uint8)

        project0 = np.max(enhanced_img, axis=0)
        project1 = np.max(enhanced_img, axis=1)
        project2 = np.max(enhanced_img, axis=2)

        imsave(os.path.join(write_path, self.img_id + '0.png'), project0)
        imsave(os.path.join(write_path, self.img_id + '1.png'), project1)
        imsave(os.path.join(write_path, self.img_id + '2.png'), project2)
        tifffile.imsave(os.path.join(write_path, self.img_id + '.tif'), enhanced_img)

    def _get_neuron_prominence(self, img, window_size):
        D, H, W = img.shape
        # Centered gradients
        dz, dx, dy = np.zeros((D, H, W)), np.zeros((D, H, W)), np.zeros((D, H, W))
        dz[1:D - 1, :, :] = (img[2:, :, :] - img[:D - 2, :, :]) / 2
        dx[:, :, 1:W - 1] = (img[:, :, 2:] - img[:, :, :W - 2]) / 2
        dy[:, 1:H - 1, :] = (img[:, 2:, :] - img[:, :H - 2, :]) / 2

        E_dzz = np.real(fftshift(ifftn(fftn(dz**2)* self.fourier_window) * window_size**3))
        E_dxx = np.real(fftshift(ifftn(fftn(dx**2)* self.fourier_window) * window_size**3))
        E_dyy = np.real(fftshift(ifftn(fftn(dy**2)* self.fourier_window) * window_size**3))
        E_dxz = np.real(fftshift(ifftn(fftn(dx*dz)* self.fourier_window) * window_size**3))
        E_dxy = np.real(fftshift(ifftn(fftn(dx*dy)* self.fourier_window) * window_size**3))
        E_dyz = np.real(fftshift(ifftn(fftn(dy*dz)* self.fourier_window) * window_size**3))

        a = -1
        b = E_dxx + E_dyy + E_dzz
        c = E_dxy**2 + E_dxz**2 + E_dyz**2 - E_dxx*E_dyy - E_dxx*E_dzz - E_dyy*E_dzz
        d = E_dxx*E_dyy*E_dzz + 2 * E_dxy*E_dyz*E_dxz - E_dxx*E_dyz**2 - E_dzz*E_dxy**2 - E_dyy*E_dxz**2
        delta0 = b**2 - 3 * a* c
        delta1 = 2 * b**3 - 9 * a*b*c + 27 * a**2 * d
        beta = (delta1**2 - 4 * delta0**3+0j)**0.5

        C = ((delta1 + beta) / 2)**(1 / 3)

        _lambda = np.zeros((D, H, W, 3))
        _lambda[:,:,:, 0] = np.real(-1/(3 * a) * (b + C + delta0 / C))
        k1 = (-1/2 + 1/2 * np.sqrt(3) * 1)*1j
        _lambda[:,:,:, 1] = np.real(-1 / (3 * a) * (b + k1 * C + delta0 / C / k1))
        k2 = (-1 / 2 - 1 / 2 * np.sqrt(3) * 1)*1j
        _lambda[:,:,:, 2] = np.real(-1 / (3 * a) * (b + k2 * C + delta0/C/k2))

        lambda_sort = np.sort(_lambda, axis=3)

        lambda_min = lambda_sort[:,:,:, 0]
        lambda_mid = lambda_sort[:,:,:, 1]
        lambda_max = lambda_sort[:,:,:, 2]

        Q_map = lambda_max * lambda_mid * (abs(lambda_max - lambda_min) / (lambda_max + lambda_min))

        return Q_map

    def enhance(self, smooth_degree, iter_N=20, window_size=9):
        # smooth_degree
        # window_size:

        D, H, W = self.org_img.shape
        # get the optical transfer function of gradient on x and y directions. (similar to psf2otf in Matlab)
        dz, dx, dy = np.zeros((D, H, W)), np.zeros((D, H, W)), np.zeros((D, H, W))
        spatial_window = np.zeros((D, H, W))
        dz[D//2 - 1: D//2+1, H//2, W//2] = [-1, 1]
        dx[D//2, H//2, W//2 - 1:W//2 + 1] = [-1, 1]
        dy[D//2, H//2 - 1:H//2 + 1, W//2] = [-1, 1]
        z_start, z_end = D//2 - window_size//2, D//2 + window_size//2 + 1
        h_start, h_end = H//2 - window_size//2, H//2 + window_size//2 + 1
        w_start, w_end = W//2 - window_size//2, W//2 + window_size//2 + 1
        spatial_window[z_start:z_end, h_start:h_end, w_start:w_end] = 1/(window_size**3)
        otfDz, otfDx, otfDy = fftn(dz), fftn(dx), fftn(dy)
        self.fourier_window = fftn(spatial_window)

        self.Denormin2 = abs(otfDz) ** 2 + abs(otfDx) ** 2 + abs(otfDy) ** 2

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
            p = np.concatenate((img[1:, :, :] - img[:D-1, :, :], img[0:1, :, :] - img[D-1:, :, :]), axis=0)
            h = np.concatenate((img[:, :, 1:] - img[:, :, :W-1], img[:, :, 0:1] - img[:, :, W-1:]), axis=2)
            v = np.concatenate((img[:, 1:, :] - img[:, :H-1, :], img[:, 0:1, :] - img[:, H-1:, :]), axis=1)

            # Step (4) in Alg. 1 in [1].
            diff_norm2 = (p**2 + h ** 2 + v ** 2)
            t = diff_norm2 < (smooth_degree * weight) / beta
            p[t] = 0
            h[t] = 0
            v[t] = 0

            # Step (5) in Alg. 1 in [1].
            d = img - self.org_img

            # Step (6) in Alg. 1 in [1].
            t = (d**2) < (1/gamma)
            d[t] = 0

            # Step(2) in Alg.1 in [1].
            Normin2 = -np.concatenate((h[:, :, 0:1] - h[:, :, W-1:], h[:, :, 1:] - h[:, :, :W-1]), axis=2) \
                        - np.concatenate((v[:, 0:1, :] - v[:, H-1:, :], v[:, 1:, :] - v[:, :H-1, :]), axis=1) \
                        - np.concatenate((p[0:1, :, :] - p[D-1:, :, :], p[1:, :, :] - p[:D-1, :, :]), axis=0)

            Normin1 = fftn(self.org_img + d)
            FS = (Normin1 + beta * fftn(Normin2)) / denorminator2
            img = ifftn(FS).real

            # Step(7) in Alg.1 in [1].
            beta = beta * self.kappa1
            gamma = gamma * self.kappa1
            diff_hist.append(np.mean((img - self.org_img) ** 2))

        self.enhanced_img = img
        return img, diff_hist

