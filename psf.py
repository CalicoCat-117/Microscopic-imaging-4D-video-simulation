from scipy import integrate
from scipy.special import jv
import numpy as np
import multiprocessing
import time
from skimage.io import imread, imsave
from scipy.interpolate import interp2d
import pdb

# 
# wide field fluorescence microscope
# 

def h_wffm(_x, _y, _z, _lambda, _NA_n, _x0, _y0, _z0, x_step, y_step, z_step):
    k = 2 * np.pi / _lambda
    rho = np.sqrt(((_x - _x0) * x_step) ** 2 + ((_y - _y0) * y_step) ** 2)
    alpha = np.arcsin(_NA_n)
    def _func(_theta):
        return np.sqrt(np.cos(_theta)) * jv(0, k * rho * np.sin(_theta)) \
             * np.exp(- 1j * k * (_z - _z0) * z_step * np.cos(_theta)) * np.sin(_theta)
    def real_func(x):
        return np.real(_func(x))
    def imag_func(x):
        return np.imag(_func(x))
    real_res, real_err = integrate.quad(real_func, 0, alpha)
    imag_res, imag_err = integrate.quad(imag_func, 0, alpha)
    wffm_res = np.abs(real_res + 1j * imag_res) ** 2
    wffm_err = 2 * np.abs(real_res + 1j * imag_res) * np.abs(real_err + 1j * imag_err) 
    return wffm_res, wffm_err

def _h_wffm(i, _x, _y, _z, _lambda, _NA_n, _x0, _y0, _z0, x_step, y_step, z_step):
    return i, h_wffm(_x, _y, _z, _lambda, _NA_n, _x0, _y0, _z0, x_step, y_step, z_step)


def wide_field_psf(size, _lambda, _NA_n, x_step, y_step, z_step, _x0, _y0, _z0, n_jobs=1):
    xs = np.linspace(0, size[0] - 1, size[0]).astype(np.float32)
    ys = np.linspace(0, size[1] - 1, size[1]).astype(np.float32)
    zs = np.linspace(0, size[2] - 1, size[2]).astype(np.float32)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing='ij')
    # flatten
    xx_list = xx.flatten()
    yy_list = yy.flatten()
    zz_list = zz.flatten()
    # integrate
    pool = multiprocessing.Pool(processes=n_jobs)
    parameter_list = []
    N = size[0]*size[1]*size[2]
    psf = np.ones([N,])
    for i in range(N):
        parameter_list.append([i, xx_list[i], yy_list[i], zz_list[i], \
            _lambda, _NA_n, _x0, _y0, _z0, x_step, y_step, z_step])
    for i, h_result in pool.starmap(_h_wffm, parameter_list):
        psf[i] = h_result[0]
        # print("%d/%d"%(i+1, N), end='\r')
    pool.close()
    pool.join()
    psf = psf.reshape(size)
    return psf

# 
# laser scanning confocal microscope
# 

def h_lscm(_x, _y, _z, _r, _lambda_ex, _lambda_em, _NA_n, _x0, _y0, _z0, x_step, y_step, z_step):
    alpha = np.arcsin(_NA_n)
    x_real = (_x - _x0) * x_step
    y_real = (_y - _y0) * y_step
    z_real = (_z - _z0) * y_step
    def _h(x_real, y_real, z_real, _lambda, _NA_n):
        k = 2 * np.pi / _lambda
        rho = np.sqrt(x_real ** 2 + y_real ** 2)
        def _func(_theta):
            return np.sqrt(np.cos(_theta)) * jv(0, k * rho * np.sin(_theta)) \
             * np.exp(- 1j * k * z_real * np.cos(_theta)) * np.sin(_theta)
        def real_func(x):
            return np.real(_func(x))
        def imag_func(x):
            return np.imag(_func(x))
        real_res, real_err = integrate.quad(real_func, 0, alpha)
        imag_res, imag_err = integrate.quad(imag_func, 0, alpha)
        return np.abs(real_res + 1j * imag_res)
    # 
    # interpolate _h
    # 
    px1 = np.linspace(_r, -_r, 100)
    py1 = np.linspace(_r, -_r, 100)
    px1_px1, py1_py1 = np.meshgrid(px1, py1, indexing='ij')
    px1_px1_list = px1_px1.flatten()
    py1_py1_list = py1_py1.flatten()
    N = len(px1) * len(py1)
    debye = np.zeros([N,])
    for i in range(N):
        debye[i] = _h(x_real-px1_px1_list[i], y_real-py1_py1_list[i], z_real, _lambda_em, _NA_n)
    debye = debye.reshape([len(px1), len(py1)])
    _debye_inter = interp2d(x_real-px1, y_real-py1, debye, kind='cubic')

    # 
    # integrate
    # 
    def _func(x1, y1):
        return np.abs(_debye_inter(x_real+x1, y_real+y1)) ** 2
    def bounds_y():
        return [-_r, _r]
    def bounds_x(y):
        xd = np.sqrt(_r**2-y**2)
        return [-xd, xd]
    t_res = integrate.nquad(_func, [bounds_x, bounds_y])
    wffm_res = h_wffm(_x, _y, _z, _lambda_ex, _NA_n, _x0, _y0, _z0, x_step, y_step, z_step)
    return t_res[0] * wffm_res[0], wffm_res[0] * t_res[1] + t_res[0] * wffm_res[1]

def _h_lscm(i, _x, _y, _z, _r, _lambda_ex, _lambda_em, _NA_n, _x0, _y0, _z0, x_step, y_step, z_step):
    return i, h_lscm(_x, _y, _z, _r, _lambda_ex, _lambda_em, _NA_n, _x0, _y0, _z0, x_step, y_step, z_step)


def confocal_psf(size, _r, _lambda_ex, _lambda_em, _NA_n, x_step, y_step, z_step, n_jobs=1):
    xs = np.linspace(0, size[0] - 1, size[0]).astype(np.float32)
    ys = np.linspace(0, size[1] - 1, size[1]).astype(np.float32)
    zs = np.linspace(0, size[2] - 1, size[2]).astype(np.float32)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing='ij')
    _x0 = (0.5 + size[0] - 1 + 0.5) / 2
    _y0 = (0.5 + size[1] - 1 + 0.5) / 2
    _z0 = (0.5 + size[2] - 1 + 0.5) / 2
    # flatten
    xx_list = xx.flatten()
    yy_list = yy.flatten()
    zz_list = zz.flatten()
    # integrate
    pool = multiprocessing.Pool(processes=n_jobs)
    parameter_list = []
    N = size[0]*size[1]*size[2]
    psf = np.ones([N,])
    for i in range(N):
        parameter_list.append([i, xx_list[i], yy_list[i], zz_list[i], _r, \
            _lambda_ex, _lambda_em, _NA_n, _x0, _y0, _z0, x_step, y_step, z_step])
    for i, h_result in pool.starmap(_h_lscm, parameter_list):
        psf[i] = h_result[0]
    pool.close()
    pool.join()
    psf = psf.reshape(size)
    return psf

# 
# laser scanning confocal microscope (approximate)
# 
def h_lscm_app(_x, _y, _z, _r, pinhole_psf, _lambda_ex, _lambda_em, _NA_n, _x0, _y0, _z0, x_step, y_step, z_step):
    alpha = np.arcsin(_NA_n)
    pinhole_psf_z = pinhole_psf[:, :, int(_z)]
    size = pinhole_psf_z.shape
    xs = np.linspace(0, size[0] - 1, size[0]).astype(np.float32)
    ys = np.linspace(0, size[1] - 1, size[1]).astype(np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    mask = (((xx - _x) * x_step) ** 2 + ((yy - _y) * y_step) ** 2 <= _r ** 2)
    t_res = np.sum(pinhole_psf_z[mask])
    wffm_res = h_wffm(_x, _y, _z, _lambda_ex, _NA_n, _x0, _y0, _z0, x_step, y_step, z_step)
    return t_res * wffm_res[0], t_res * wffm_res[1]

def _h_lscm_app(i, _x, _y, _z, _r, pinhole_psf, _lambda_ex, _lambda_em, _NA_n, _x0, _y0, _z0, x_step, y_step, z_step):
    return i, h_lscm_app(_x, _y, _z, _r, pinhole_psf, _lambda_ex, _lambda_em, _NA_n, _x0, _y0, _z0, x_step, y_step, z_step)


def confocal_psf_app(size, pinhole_psf, _r, _lambda_ex, _lambda_em, _NA_n, x_step, y_step, z_step, _x0, _y0, _z0, n_jobs=1):
    xs = np.linspace(0, size[0] - 1, size[0]).astype(np.float32)
    ys = np.linspace(0, size[1] - 1, size[1]).astype(np.float32)
    zs = np.linspace(0, size[2] - 1, size[2]).astype(np.float32)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing='ij')

    # flatten
    xx_list = xx.flatten()
    yy_list = yy.flatten()
    zz_list = zz.flatten()
    # integrate
    pool = multiprocessing.Pool(processes=n_jobs)
    parameter_list = []
    N = size[0]*size[1]*size[2]
    psf = np.ones([N,])
    for i in range(N):
        parameter_list.append([i, xx_list[i], yy_list[i], zz_list[i], _r, pinhole_psf, \
            _lambda_ex, _lambda_em, _NA_n, _x0, _y0, _z0, x_step, y_step, z_step])
    for i, h_result in pool.starmap(_h_lscm_app, parameter_list):
        psf[i] = h_result[0]
        # print("%d/%d"%(i+1, N), end='\r')
    pool.close()
    pool.join()
    psf = psf.reshape(size)
    return psf


if __name__ == '__main__':

    ''' config '''
    size = [128, 128, 32]
    _lambda_ex = 561
    _lambda_em = 610
    _NA_n = 0.8
    x_step = 50
    y_step = 50
    z_step = 200
    _r = 20000 / 63 # objective magnification times a fixed internal magnification

    _x0 = (0.5 + size[0] - 1 + 0.5) / 2
    _y0 = (0.5 + size[1] - 1 + 0.5) / 2
    _z0 = (0.5 + size[2] - 1 + 0.5) / 2

    
    ''' wide_field '''
    # t = time.time()
    # psf = wide_field_psf(size, _lambda_em, _NA_n, x_step, y_step, z_step, _x0, _y0, _z0, n_jobs=8)
    # psf = psf.transpose([2, 1, 0])
    # print(psf.shape)
    # imsave("wffm_psf_s.tif", psf.astype(np.float32))
    # print("t=%.2fs"%(time.time()-t))
    
    ''' confocal '''
    # t = time.time()
    # psf = confocal_psf(size, _r, _lambda_ex, _lambda_em, _NA_n, x_step, y_step, z_step, n_jobs=20)
    # psf = psf.transpose([2, 1, 0])
    # print(psf.shape)
    # imsave("lscm_psf.tif", psf.astype(np.float32))
    # print("t=%.2fs"%(time.time()-t))

    ''' confocal (approximate)'''
    t = time.time()
    pinhole_psf = wide_field_psf(size, _lambda_em, _NA_n, x_step, y_step, z_step, _x0, _y0, _z0, n_jobs=20)
    psf = confocal_psf_app(size, pinhole_psf, _r, _lambda_ex, _lambda_em, _NA_n, x_step, y_step, z_step, _x0, _y0, _z0, n_jobs=20)
    psf = psf.transpose([2, 1, 0])
    print(psf.shape)
    imsave("lscm_psf.tif", psf.astype(np.float32), check_contrast=False)
    print("t=%.2fs"%(time.time()-t))
    
    