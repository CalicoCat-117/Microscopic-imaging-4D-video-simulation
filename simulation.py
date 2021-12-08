# -*- coding: utf-8 -*-
''' @Author: Zehao Wang (wangze@mail.ustc.edu.cn) '''
from skimage.io import imread, imsave
from psf import wide_field_psf, confocal_psf_app
import numpy as np
import time
import os
import pdb
from utils_imageJ import save_tiff_imagej_compatible
from skimage.transform import rescale
from tqdm import tqdm

DEBUG = False
class wffm:
    def __init__(self, intensity):
        self.name = "wide field"
        self.intensity = intensity
        self.size = [32, 32, 9]
        self.n = 1.5
        self._lambda_em = 610
        self._NA = [1.2, 1.3]
        self.x_step = 40
        self.y_step = 40
        self.z_step = 300
        self.scale = 1
    
    def build_psfs(self, psf_num, n_jobs):
        size = np.int32(np.asarray(self.size) * self.scale)
        x_step = self.x_step / self.scale
        y_step = self.y_step / self.scale
        z_step = self.z_step / self.scale
        _x0 = (0.5 + size[0] - 1 + 0.5) / 2
        _y0 = (0.5 + size[1] - 1 + 0.5) / 2
        _z0 = (0.5 + size[2] - 1 + 0.5) / 2
        _lambda_em = self._lambda_em / self.n
        _NA_n = [self._NA[0] / self.n, self._NA[1] / self.n]
        PSF_list = []
        for i in tqdm(range(psf_num)):
            _NA_n_i = np.random.uniform(*_NA_n)
            psf = wide_field_psf(size, _lambda_em, _NA_n_i, x_step, y_step, z_step, _x0, _y0, _z0, n_jobs).astype(np.float32)
            if DEBUG:
                print(psf.shape)
                save_path = "data/sim_3dmov/wffm_psf_scaled"
                os.makedirs(save_path, exist_ok=True)
                imsave(os.path.join(save_path, "wffm_psf_scaled_%03d.tif"%i), psf.transpose([2, 1, 0]), check_contrast=False)
            PSF_list.append(psf)
        return PSF_list


class lscm:
    def __init__(self, intensity):
        self.name = "confocal"
        self.intensity = intensity
        self.size = [32, 32, 32]
        self.n = 1.5
        self._lambda_ex = 561
        self._lambda_em = 610
        self._NA = [1.2, 1.3]
        self.pinhole = 20000
        self.magnification = 100
        self.x_step = 40
        self.y_step = 40
        self.z_step = 300
        self.scale = 1
    
    def build_psfs(self, psf_num, n_jobs):
        size = np.int32(np.asarray(self.size) * self.scale)
        x_step = self.x_step / self.scale
        y_step = self.y_step / self.scale
        z_step = self.z_step / self.scale
        _x0 = (0.5 + size[0] - 1 + 0.5) / 2
        _y0 = (0.5 + size[1] - 1 + 0.5) / 2
        _z0 = (0.5 + size[2] - 1 + 0.5) / 2
        _lambda_ex = self._lambda_ex / self.n
        _lambda_em = self._lambda_em / self.n
        _NA_n = [self._NA[0] / self.n, self._NA[1] / self.n]
        _r = self.pinhole / self.magnification
        PSF_list = []
        for i in tqdm(range(psf_num)):
            _NA_n_i = np.random.uniform(*_NA_n)
            pinhole_psf = wide_field_psf(size, _lambda_em, _NA_n_i, x_step, y_step, z_step, _x0, _y0, _z0, n_jobs=n_jobs)
            psf = confocal_psf_app(size, pinhole_psf, _r, _lambda_ex, _lambda_em, _NA_n_i, x_step, y_step, z_step, _x0, _y0, _z0, n_jobs=n_jobs)
            if DEBUG:
                print(psf.shape)
                save_path = "data/4d/wffm_psf_scaled"
                os.makedirs(save_path, exist_ok=True)
                imsave(os.path.join(save_path, "wffm_psf_scaled_%03d.tif"%i), psf.transpose([2, 1, 0]), check_contrast=False)
            PSF_list.append(psf)
        return PSF_list

class Minkowski_Point_Cloud():
    def __init__(self, obj_num, prototype_num, img_method, n_jobs):
        self._pos = np.zeros([obj_num, 3])
        self.obj_num = obj_num
        self.n_jobs = n_jobs
        self.prototype_num = prototype_num
        self.img_method = img_method
        # init prototype
        if img_method.name == "wide field" or img_method.name == "confocal":
            print("init prototype", img_method.name, "psf")
            self.prototype = img_method.build_psfs(self.prototype_num, n_jobs=self.n_jobs)
        else:
            assert False
        for i in range(len(self.prototype)):
            self.prototype[i] = self.prototype[i] / np.max(self.prototype[i]) * np.random.uniform(*img_method.intensity)
        # build points
        self.batch_proto_num = np.random.randint(len(self.prototype), size=[obj_num,1])

    def init_gauss_pos(self, mu, std):
        self._pos = np.random.normal(mu, std, [self.obj_num, 3])

    def init_uniform_pos(self, low, high):
        self._pos = np.random.uniform(low, high, [self.obj_num, 3])
    
    # Object movement mode I
    def update_random_walk(self, step_mu, step_std):
        step = np.random.normal(step_mu, step_std, [self.obj_num, 3])
        self._pos += step
    
    # Object movement mode II
    def update_diffusion(self, step_mu, step_std, center, m_dir_std):
        step = np.random.normal(step_mu, step_std, [self.obj_num, 3])
        m_dir_mu = self._pos - np.array(center)
        m_dir_mu /= np.linalg.norm(m_dir_mu, 2)
        m_dir = np.random.normal(m_dir_mu, m_dir_std, [self.obj_num, 3])
        m_dir /= np.linalg.norm(m_dir, 2)
        self._pos += step * m_dir

    def draw_now(self, mesh_in):
        mesh_out = mesh_in.copy()
        for i in range(self.obj_num):
            proto = self.batch_proto_num[i]
            pos = self._pos[i] * self.img_method.scale
            psf_size = np.asarray(self.img_method.size) * self.img_method.scale
            pos_min = np.int32(np.maximum(np.rint(pos-psf_size//2), 0))
            pos_max = np.int32(np.minimum(np.rint(pos-psf_size//2)+psf_size-1, np.asarray(mesh_in.shape)-1))
            label = np.zeros_like(mesh_out)
            if np.all(pos_min < (np.array(mesh_out.shape)-1)) and np.all(pos_max > 0):
                psf_min = np.int32(0 + (pos_min - np.rint(pos-psf_size//2)))
                psf_max = np.int32((psf_size-1) - (np.rint(pos-psf_size//2)+psf_size-1 - pos_max))
                psf_ind = tuple([slice(psf_min[i], psf_max[i]+1, 1) for i in range(len(psf_min))])
                pos_ind = tuple([slice(pos_min[i], pos_max[i]+1, 1) for i in range(len(pos_min))])
                mesh_out[pos_ind] += self.prototype[int(proto)][psf_ind]
                label[pos_ind] = 1
                mesh_out[label == 0] += self.prototype[int(proto)].min()
        return mesh_out

if __name__ == '__main__':
    t0 = time.time()
    ''' config '''
    img_size = [128, 128, 9] # XYZ
    ntimes = 20
    obj_num = 200
    intensity = [100, 200] # min, max (Number of particle radiated photons)

    # img_method = wffm(intensity)
    img_method = lscm(intensity)

    ''' init point cloud ''' 
    pt1 = Minkowski_Point_Cloud(obj_num=obj_num, prototype_num=3, img_method=img_method, n_jobs=32)
    c1 = np.asarray([30, 64, 4]) # mux, muy, muz
    pt1.init_gauss_pos(c1, [20, 20, 10]) # sigmax, sigmay, sigmaz
    
    pt2 = Minkowski_Point_Cloud(obj_num=obj_num, prototype_num=3, img_method=img_method, n_jobs=32)
    c2 = np.asarray([90, 64, 4])
    pt2.init_gauss_pos(c2, [20, 20, 10])

    canvas_nsy_list = []
    canvas_cln_list = []
    scale = img_method.scale

    ''' Simulating ''' 
    print("Simulating ...")
    for t in tqdm(range(ntimes)):
        canvas_cln_t = np.zeros([img_size[0]*scale, img_size[1]*scale, img_size[2]*scale])
        ''' random_walk '''
        pt1.update_diffusion(30 * 10/(t+10), 10 * 3/(t+3), c1, 0.5)
        pt2.update_diffusion(100 * 10/(t+10), 10 * 3/(t+3), c2, 0.5)
        ''' draw '''
        canvas_cln_t = pt1.draw_now(canvas_cln_t)
        canvas_cln_t = pt2.draw_now(canvas_cln_t)
        canvas_cln_t = canvas_cln_t[:, :, :, np.newaxis, np.newaxis]
        ''' add shot noise '''
        canvas_nsy_t = np.random.poisson(canvas_cln_t)
        canvas_nsy_list.append(canvas_nsy_t)
        canvas_cln_list.append(canvas_cln_t)
    canvas_nsy = np.concatenate(canvas_nsy_list, axis=-1)
    canvas_cln = np.concatenate(canvas_cln_list, axis=-1)
    
    ''' save ''' 
    save_path = "data/"
    os.makedirs(save_path, exist_ok=True)
    canvas_save = np.concatenate([canvas_cln, canvas_nsy], axis=3).astype(np.uint16)
    save_tiff_imagej_compatible(os.path.join(save_path, img_method.name + "_simulation.tif"), canvas_save, "XYZCT")
    print("simulation saved")
    print("build_psfs t=%.2fs"%(time.time()-t0))
    
        
