import numpy as np
from numpy.fft import fft2, ifft2, fftshift, fftn, ifftn


def cane_2d(org_img, smooth_degree, window_size=9, iter_N=20, kappa1=2, kappa2=2, kappa3=2000):
    def get_neuron_prominence_2d(img, window_size):
        H, W = img.shape
        # -- Centered gradients
        dx, dy = np.zeros((H, W)), np.zeros((H, W))
        dx[:, 1:W - 1] = (img[:, 2:] - img[:, :W - 2]) / 2
        dy[1:H - 1, :] = (img[2:, :] - img[:H - 2, :]) / 2

        E_dx = np.real(fftshift(ifft2(fft2(dx ** 2) * fourier_window) * window_size ** 2))
        E_dy = np.real(fftshift(ifft2(fft2(dy ** 2) * fourier_window) * window_size ** 2))
        E_dxy = np.real(fftshift(ifft2(fft2(dx * dy) * fourier_window) * window_size ** 2))
        delta = np.abs((E_dx + E_dy) ** 2 - 4 * (E_dx * E_dy - E_dxy ** 2)) ** 0.5
        b = E_dx + E_dy
        lambda1 = np.abs((b - delta) / 2) ** 0.5
        lambda2 = np.abs((b + delta) / 2) ** 0.5
        Q_map = lambda2 * (abs(lambda2 - lambda1) / (lambda1 + lambda2))
        return Q_map


    H, W = org_img.shape
    # -- get the optical transfer function of gradient on x and y directions. (similar to psf2otf in Matlab)
    dx, dy, spatial_window = np.zeros((H, W)), np.zeros((H, W)), np.zeros((H, W))
    dx[H // 2, W // 2 - 1:W // 2 + 1] = [-1, 1]
    dy[H // 2 - 1:H // 2 + 1, W // 2] = [-1, 1]
    h_start, h_end = H // 2 - window_size // 2, H // 2 + window_size // 2 + 1
    w_start, w_end = W // 2 - window_size // 2, W // 2 + window_size // 2 + 1
    spatial_window[h_start:h_end, w_start:w_end] = 1 / (window_size ** 2)
    otfDx, otfDy = fft2(dx), fft2(dy)
    fourier_window = fft2(spatial_window)

    Denormin2 = abs(otfDx) ** 2 + abs(otfDy) ** 2

    beta = kappa2 * smooth_degree
    gamma = kappa3 * smooth_degree

    # -- record the convergence by mean squared difference to the original image
    diff_hist = []
    img = org_img.copy()
    weight = None

    for ite_num in range(iter_N):
        if ite_num % 10 == 0:
            neuron_prominence = get_neuron_prominence_2d(img, window_size)
            weight = 1. / neuron_prominence
        denorminator2 = 1 + beta * Denormin2
        # -- Step (3) in Alg. 1 in [1].
        h = np.hstack([img[:, 1:] - img[:, :W - 1], img[:, 0:1] - img[:, W - 1:]])
        v = np.vstack([img[1:, :] - img[:H - 1, :], img[0:1, :] - img[H - 1:, :]])

        # -- Step (4) in Alg. 1 in [1].
        diff_norm2 = (h ** 2 + v ** 2)
        t = diff_norm2 < (smooth_degree * weight) / beta
        h[t] = 0
        v[t] = 0

        # -- Step (5) in Alg. 1 in [1].
        d = img - org_img

        # -- Step (6) in Alg. 1 in [1].
        t = (d ** 2) < (1 / gamma)
        d[t] = 0

        # -- Step(2) in Alg.1 in [1].
        Normin2 = -np.hstack([h[:, 0:1] - h[:, W - 1:], h[:, 1:] - h[:, :W - 1]]) \
                  - np.vstack([v[0:1, :] - v[H - 1:, :], v[1:, :] - v[:H - 1, :]])

        Normin1 = fft2(org_img + d)
        FS = (Normin1 + beta * fft2(Normin2)) / denorminator2
        img = ifft2(FS).real

        # -- Step(7) in Alg.1 in [1].
        beta = beta * kappa1
        gamma = gamma * kappa1
        diff_hist.append(np.mean((img - org_img) ** 2))

    return img, diff_hist


def cane_3d(org_img, smooth_degree, window_size=9, iter_N=20, kappa1=2, kappa2=2, kappa3=2000):

    def get_neuron_prominence_3d(img, window_size):
        D, H, W = img.shape
        # -- Centered gradients
        dz, dx, dy = np.zeros((D, H, W)), np.zeros((D, H, W)), np.zeros((D, H, W))
        dz[1:D - 1, :, :] = (img[2:, :, :] - img[:D - 2, :, :]) / 2
        dx[:, :, 1:W - 1] = (img[:, :, 2:] - img[:, :, :W - 2]) / 2
        dy[:, 1:H - 1, :] = (img[:, 2:, :] - img[:, :H - 2, :]) / 2

        E_dzz = np.real(fftshift(ifftn(fftn(dz**2)* fourier_window) * window_size**3))
        E_dxx = np.real(fftshift(ifftn(fftn(dx**2)* fourier_window) * window_size**3))
        E_dyy = np.real(fftshift(ifftn(fftn(dy**2)* fourier_window) * window_size**3))
        E_dxz = np.real(fftshift(ifftn(fftn(dx*dz)* fourier_window) * window_size**3))
        E_dxy = np.real(fftshift(ifftn(fftn(dx*dy)* fourier_window) * window_size**3))
        E_dyz = np.real(fftshift(ifftn(fftn(dy*dz)* fourier_window) * window_size**3))

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

    D, H, W = org_img.shape
    # -- get the optical transfer function of gradient on x and y directions. (similar to psf2otf in Matlab)
    dz, dx, dy = np.zeros((D, H, W)), np.zeros((D, H, W)), np.zeros((D, H, W))
    spatial_window = np.zeros((D, H, W))
    dz[D // 2 - 1: D // 2 + 1, H // 2, W // 2] = [-1, 1]
    dx[D // 2, H // 2, W // 2 - 1:W // 2 + 1] = [-1, 1]
    dy[D // 2, H // 2 - 1:H // 2 + 1, W // 2] = [-1, 1]
    z_start, z_end = D // 2 - window_size // 2, D // 2 + window_size // 2 + 1
    h_start, h_end = H // 2 - window_size // 2, H // 2 + window_size // 2 + 1
    w_start, w_end = W // 2 - window_size // 2, W // 2 + window_size // 2 + 1
    spatial_window[z_start:z_end, h_start:h_end, w_start:w_end] = 1 / (window_size ** 3)
    otfDz, otfDx, otfDy = fftn(dz), fftn(dx), fftn(dy)
    fourier_window = fftn(spatial_window)

    Denormin2 = abs(otfDz) ** 2 + abs(otfDx) ** 2 + abs(otfDy) ** 2

    beta = kappa2 * smooth_degree
    gamma = kappa3 * smooth_degree

    # -- record the convergence by mean squared difference to the original image
    diff_hist = []
    img = org_img.copy()
    weight = None

    for ite_num in range(iter_N):
        if ite_num % 10 == 0:
            neuron_prominence = get_neuron_prominence_3d(img, window_size)
            weight = 1. / neuron_prominence
        denorminator2 = 1 + beta * Denormin2
        # -- Step (3) in Alg. 1 in [1].
        p = np.concatenate((img[1:, :, :] - img[:D - 1, :, :], img[0:1, :, :] - img[D - 1:, :, :]), axis=0)
        h = np.concatenate((img[:, :, 1:] - img[:, :, :W - 1], img[:, :, 0:1] - img[:, :, W - 1:]), axis=2)
        v = np.concatenate((img[:, 1:, :] - img[:, :H - 1, :], img[:, 0:1, :] - img[:, H - 1:, :]), axis=1)

        # -- Step (4) in Alg. 1 in [1].
        diff_norm2 = (p ** 2 + h ** 2 + v ** 2)
        t = diff_norm2 < (smooth_degree * weight) / beta
        p[t] = 0
        h[t] = 0
        v[t] = 0

        # -- Step (5) in Alg. 1 in [1].
        d = img - org_img

        # -- Step (6) in Alg. 1 in [1].
        t = (d ** 2) < (1 / gamma)
        d[t] = 0

        # -- Step(2) in Alg.1 in [1].
        Normin2 = -np.concatenate((h[:, :, 0:1] - h[:, :, W - 1:], h[:, :, 1:] - h[:, :, :W - 1]), axis=2) \
                  - np.concatenate((v[:, 0:1, :] - v[:, H - 1:, :], v[:, 1:, :] - v[:, :H - 1, :]), axis=1) \
                  - np.concatenate((p[0:1, :, :] - p[D - 1:, :, :], p[1:, :, :] - p[:D - 1, :, :]), axis=0)

        Normin1 = fftn(org_img + d)
        FS = (Normin1 + beta * fftn(Normin2)) / denorminator2
        img = ifftn(FS).real

        # -- Step(7) in Alg.1 in [1].
        beta = beta * kappa1
        gamma = gamma * kappa1
        diff_hist.append(np.mean((img - org_img) ** 2))

    return img, diff_hist
