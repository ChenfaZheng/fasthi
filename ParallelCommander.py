from os import listdir, mkdir, remove
from os.path import exists as path_exists
from time import strftime
import numpy as np
from pandas import read_csv
from yaml import load as yaml_load
from yaml import FullLoader as YamlFullLoader

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from multiprocessing import shared_memory

from lib.FileIO import FitsInfo, SDSSCatalog

from astropy.io import fits
from astropy.wcs import WCS

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from scipy.optimize import curve_fit

import os
import wget


class Dir():
    def __init__(self, workdir='.', fitsdir='./fits', libdir='./lib'):
        self.work = workdir
        self.fits = fitsdir
        self.lib = libdir
        self.temp = '/'.join((self.work, 'temp'))
        self.SDSS = '/'.join((self.work, 'SDSS'))
        self.optimg = '/'.join((self.work, 'optimgs'))
        self.results = '/'.join((self.work, 'results'))
        self.update()
    
    def update(self):
        for path in (self.temp, self.SDSS, self.optimg, self.results):
            if not path_exists(path):
                mkdir(path)


class Manager():
    def __init__(self, config='./config.yml'):
        # read config and create dirs
        with open(config, 'r') as f:
            self.conf = yaml_load(f, Loader=YamlFullLoader)
        self.dir = Dir(self.conf['DIR']['work_dir'], self.conf['DIR']['fits_dir'], self.conf['DIR']['lib_dir'])
        self.dir.update()
        # read fits info
        fits_files = sorted(filter(lambda f: f[-11:].lower() == self.conf['FITS']['back_str'], listdir(self.dir.fits)))
        self.fits_info = list(map(FitsInfo, \
            map(lambda f: '/'.join((self.conf['DIR']['fits_dir'], f)), fits_files))) # should NOT delete 'list'
        # query and download SDSS catalog
        self.cat_info = list(map(SDSSCatalog, self.fits_info))
        pass

    def get_fits_list(self):
        return list(map(lambda f: f.fname, self.fits_info))

    def run(self, nfits=1):
        if nfits == -1:
            nfits = len(self.fits_info)
        nfits = min(nfits, int(self.conf['RUN']['n_fits']))
        for idx in range(nfits):
            self.run_one_fits(idx)

    def run_one_fits(self, fits_idx):
        # read fits info @idx
        fits_info = self.fits_info[fits_idx]
        # read catalog info @idx
        cat_info = self.cat_info[fits_idx]
        # multiprocessing
        cores_num = int(self.conf['RUN']['n_cores']) % cpu_count()
        # if cores_num != 1:
        pool = ProcessPoolExecutor(max_workers=cores_num)
        cat_status = cat_info.download(self.dir.SDSS, skip_saved=self.conf['SDSS']['skip_save'])
        print('cat status:', cat_status)
        if cat_status == -1:
            print('sdss catalog failed to download and save!')
            return None
        if cat_status == -2:
            print('sdss catalog null data queried!')
            return None
        # read fitsfile data
        hdu = fits.open(fits_info.path)
        data = hdu[0].data
        self.shape = data.shape
        self.dtype = data.dtype
        self.header = hdu[0].header
        # set shared memory
        self.shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
        # copy data into shared memory
        shared_data = np.ndarray(shape=data.shape, dtype=data.dtype, buffer=self.shm.buf)
        shared_data[:] = data[:]
        # distribute job configuration
        cat_df = read_csv(cat_info.path, header=0, skip_blank_lines=True, skipinitialspace=True, skiprows=[2])
        self.timestr = strftime(r'%Y%m%d-%H%M%S')
        dir_type_dict = {
            'fits'  :   fits_info.fname, 
            'time'  :   self.timestr,
            'both'  :   '-'.join((self.timestr, fits_info.fname)), 
            'both_r':   '-'.join((fits_info.fname, self.timestr))
        }
        self.dirwork = '/'.join((self.dir.results, dir_type_dict[self.conf['SAVE']['dir_type']]))
        if not path_exists(self.dirwork):
            mkdir(self.dirwork)
        self.bsl_dir = '/'.join((self.dirwork, 'baseline'))
        self.opt_dir = '/'.join((self.dirwork, 'optimize'))
        self.sig_dir = '/'.join((self.dirwork, 'signal'))
        self.mea_dir = '/'.join((self.dirwork, 'measure'))
        for the_dir in [self.bsl_dir, self.opt_dir, self.sig_dir, self.mea_dir]:
            if not path_exists(the_dir):
                mkdir(the_dir)
        self.cat_path = '/'.join((self.dirwork, self.conf['SAVE']['name_catalog'] + fits_info.fname + '.csv'))
        # self.data, self.wcs = Reader.read_fits(self.fits_info.path)
        self.works = [] # will be added when self.distribute_work called
        n_tasks = cat_df.shape[0]
        job_works = np.array_split(np.arange(n_tasks), cores_num)
        cat_paths = list(map(lambda idx: self.cat_path + '.part%d'%idx, range(cores_num)))
        # parallel part
        def gen_job_conf(job_id_and_work):
            job_id, job_work = job_id_and_work
            return [
                job_id, 
                job_work,
                cat_paths[job_id],
                self.shm.name,
                self.shape,
                self.dtype,
                self.header,
                cat_df.loc[job_work, ['#bestObjID', 'ra', 'dec', 'redshift']].to_numpy(), 
                self.conf
            ]
        res = list(pool.map(self.run_one_job, map(gen_job_conf, zip(range(cores_num), job_works ))))
        # collect results
        cats = list(map(lambda f: open(f, 'r'), filter(path_exists, cat_paths)))
        with open(self.cat_path, 'w') as f:
            for idx, cat in enumerate(cats):
                if idx == 0:
                    f.writelines(cat.readlines())
                else:
                    f.writelines(cat.readlines()[1:])
                cat.close()
                remove(cat_paths[idx])
        print('Catalog merged at %s'%self.cat_path)
        # clean up
        self.shm.close()
        self.shm.unlink()
        pass

    def run_one_job(self, job_conf):
        job_id, work_ids, cat_path, shm_name, fits_shape, fits_dtype, fits_header, cat_lines, conf = job_conf
        # get shared memory
        data_shm = shared_memory.SharedMemory(name=shm_name)
        data = np.ndarray(shape=fits_shape, dtype=fits_dtype, buffer=data_shm.buf)
        cat_header = 'id,ra,dec,vel,peakvel,w50,w20,flux,rms,snr,intf,intw,inta,intsnr,interr'
        cat_strs = []
        # create log file
        log_path = '%s/log_%s.txt'%(self.dirwork, str(job_id))
        # set fig path
        fig_dir = self.dirwork
        mea_dir = self.mea_dir
        # do work in this job
        for cat_idx, cat_line in enumerate(cat_lines):
            work_id = work_ids[cat_idx]
            cat_str = Main.analyse(data, work_id, cat_line, fits_header, conf['ANALYSE'], log_path, fig_dir, mea_dir)
            if cat_str is None:
                Log.log_and_print(log_path, 
                    '[%s] catalog line id%08d %d/%d failed'%(strftime(r'%Y%m%d-%H:%M:%S'), work_id, cat_idx+1, len(cat_lines)))
                continue
            cat_strs.append(cat_str)
            Log.log_and_print(log_path, 
                '[%s] catalog line id%08d %d/%d done'%(strftime(r'%Y%m%d-%H:%M:%S'), work_id, cat_idx+1, len(cat_lines)))
        # write to catalog
        with open(cat_path, 'w') as f:
            f.write(cat_header + '\n')
            for cat_str in cat_strs:
                f.write(cat_str + '\n')
        # clean up
        del data
        pass



class Const():
    c = 3e8 # m/s


class Coord3d():
    '''
    A simple struct to store pos/idx in fitscube for convience
    '''
    def __init__(self, lst, vecarr=False):
        if vecarr:
            self.ra, self.dec, self.z = np.array(lst).T
        else:
            self.ra, self.dec, self.vel = lst


class Log():
    @staticmethod
    def log_and_print(log_path, msg):
        with open(log_path, 'a') as f:
            f.write(msg + '\n')
        print(msg)


# main process to analyse
class Main():
    @staticmethod
    def analyse(data, catline_id, catline, header, anaconf, log_path, fig_dir, mea_dir):
        # --------------------------------------------------
        # STEP 1: get data
        # --------------------------------------------------
        n_vel, n_dec, n_ra = data.shape
        # read signal
        w50 = anaconf['w50'] * 1e3 # km/s -> m/s
        vel_view_scale = anaconf['view_scale']
        sid, ra, dec, z = catline
        # try:
        #     sid = int(sid)
        # except:
        #     print(f'catalog line {sid} failed to get sid')
        #     return
        sid = int(catline_id)
        vel = Const.c * z
        world = Coord3d([ra, dec, vel])
        wcs = WCS(header)
        pix = Coord3d(list(map(int, np.array(wcs.world_to_pixel_values([[ra, dec, vel]]))[0, :])))
        datashape = data.shape
        if pix.ra < 0 or pix.ra >= datashape[2]:
            Log.log_and_print(log_path, 'ra pix %d out of range [0, %d]'%(pix.ra, datashape[2]))
            return
        if pix.dec < 0 or pix.dec >= datashape[1]:
            Log.log_and_print(log_path, 'dec pix %d out of range [0, %d]'%(pix.dec, datashape[1]))
            return
        if pix.vel < 0 or pix.vel >= datashape[0]:
            Log.log_and_print(log_path, 'vel pix %d out of range [0, %d]'%(pix.vel, datashape[0]))
            return
        scale = Coord3d(wcs.pixel_scale_matrix.diagonal())   # deg/pix
        # cut velocity
        pix_w50_width = int(abs(w50 / scale.vel))
        # print('pix_w50_width', pix_w50_width, 'w50', w50, 'scale.vel', scale.vel)
        pix_vel_width = int(abs(w50 / scale.vel * vel_view_scale))
        pix_vel_range = set(range(max(0, pix.vel - pix_vel_width), min(n_vel, pix.vel + pix_vel_width)))
        if len(pix_vel_range) == 0:
            Log.log_and_print(log_path, 'empty vel range!')
            return
        if len(pix_vel_range) < 2 * pix_vel_width:
            Log.log_and_print(log_path, 'vel range too small to analyse!')
            return
        # get signal
        x = np.array(wcs.pixel_to_world_values(list(map(
            lambda pix_vel: [pix.ra, pix.dec, pix_vel], pix_vel_range
        ))))[:, 2]
        y = data[list(pix_vel_range), pix.dec, pix.ra]
        if np.all(np.isnan(y)):
            Log.log_and_print(log_path, 'vel value all nan!')
            return
        if np.any(np.isnan(y)):
            y = np.nan_to_num(y)
            Log.log_and_print(log_path, 'vel value has nan, turn into 0!')
        # TODO any tags to be added?

        # --------------------------------------------------
        # STEP 2: optimize
        # --------------------------------------------------
        # no need to optimize signal 

        # --------------------------------------------------
        # STEP 3: reduce baseline
        # --------------------------------------------------
        lim_inner, lim_outer = float(anaconf['bsl_inner']) / 60, float(anaconf['bsl_outer']) / 60 # arcmin -> deg
        # calculate baseline position
        bsl_ra_l = np.arange(world.ra - lim_outer, world.ra - lim_inner, abs(scale.ra))
        bsl_ra_r = np.arange(world.ra + lim_inner, world.ra + lim_outer, abs(scale.ra))
        bsl_ra = np.append(bsl_ra_l, bsl_ra_r)
        bsl_dec = np.full(len(bsl_ra), world.dec)
        bsl_vel = np.full(len(bsl_ra), world.vel)
        bsl_world = Coord3d((bsl_ra, bsl_dec, bsl_vel))
        bsl_pix0 = wcs.world_to_pixel_values(np.hstack([
            bsl_ra.reshape(-1, 1), bsl_dec.reshape(-1, 1), bsl_vel.reshape(-1, 1)
        ]))
        bsl_pixarr = np.array(bsl_pix0, dtype=np.int64)
        # select points with unique RA only
        _, bsl_pixarr_uargs = np.unique(bsl_pixarr[:, :1], axis=0, return_index=True)
        if bsl_pixarr_uargs.shape[0] < bsl_pixarr.shape[0]:
            bsl_pixarr = bsl_pixarr[bsl_pixarr_uargs, :]
        # check if bsl out of boundary
        bsl_pixarr_oora = (bsl_pixarr[:, 0] < 0) | (bsl_pixarr[:, 0] >= n_ra)
        bsl_pixarr_oodec = (bsl_pixarr[:, 1] < 0) | (bsl_pixarr[:, 1] >= n_dec)
        OOB = -9223372036854775808  # constant for 'out of boundary'
        bsl_pixarr_oob = np.any(bsl_pixarr == OOB, axis=1) | bsl_pixarr_oora | bsl_pixarr_oodec
        max_num_of_oob = np.sum(bsl_pixarr_oob)
        if max_num_of_oob > 0:
            bsl_pixarr = bsl_pixarr[bsl_pixarr_oob == False]
        # median method to calculate baseline of the signal
        bsl_pixarr_num = bsl_pixarr.shape[0]
        bsls = data[min(pix_vel_range):max(pix_vel_range)+1, bsl_pixarr[:, 1], bsl_pixarr[:, 0]].T
        if not anaconf['bsl_rm_nan']:
            bsl0 = np.nanmedian(bsls, axis=0)
        else:
            bsl_nan = np.isnan(bsls).any(axis=1)
            bsls = bsls[~bsl_nan, :]
            bsl_pixarr = bsl_pixarr[~bsl_nan, :]
            bsl_pixarr_num = bsl_pixarr.shape[0]
            if bsl_pixarr_num < int(anaconf['bsl_vn']):
                print('only %d samples of baseline are valid, skip this source!' % bsl_pixarr_num)
                return 
            bsl0 = np.nanmedian(bsls, axis=0)
        # now we get `bsl0` as the real baseline of the signal
        # then we need to reduce baseline from signal
        de_fft = anaconf['fft_run']
        y_raw = np.copy(y)
        if anaconf['reduce_ifma']:
            rdc_ma = int(anaconf['reduce_ma'])
            y = np.convolve(y, np.ones(rdc_ma), 'same') / rdc_ma
            bsl0 = np.convolve(bsl0, np.ones(rdc_ma), 'same') / rdc_ma
        if de_fft:
            # TODO use fft to reduce baseline
            pass
        else:
            y = y - bsl0
        # now `y` is the reduced signal

        # --------------------------------------------------
        # STEP 4: measure
        # --------------------------------------------------
        # NOTICE: use np.convolve to replace user-made moving_average function
        #         there are border effects using 'same' method in np.convolve
        measure_ma = int(anaconf['measure_ma'])
        xm, ym = x.copy(), np.convolve(y, np.ones(measure_ma), 'same') / measure_ma
        center_arg = pix.vel - min(pix_vel_range)
        signal_rest = np.append(y[:center_arg-pix_w50_width], y[center_arg+pix_w50_width:])
        # measure rms
        m_rms = np.sqrt(np.mean(np.square(np.nan_to_num(signal_rest))))
        # measure peak
        signal_sel = y[center_arg-pix_w50_width:center_arg+pix_w50_width]
        m_peak = np.nanmax(signal_sel)
        m_peakvel = x[center_arg-pix_w50_width:center_arg+pix_w50_width][np.nanargmax(signal_sel)]
        m_snr = abs(m_peak / m_rms)
        # measure flux
        m_int_flux = np.sum(signal_sel) * abs(scale.vel)
        # TODO measure w50 and w20
        m_w50 = -1
        m_w20 = -1

        # int measurements
        intm_range = float(anaconf['intm_range']) * 1e3 # km/s -> m/s
        intm_valid = float(anaconf['intm_valid']) * 1e3 # km/s -> m/s
        intm_ap = float(anaconf['intm_ap']) / 60.0 # arcmin -> deg
        intn_in = float(anaconf['intn_in']) / 60.0 # arcmin -> deg
        intn_out = float(anaconf['intn_out']) / 60.0 # arcmin -> deg
        int_ww, int_hh = intn_out, intn_out # width and height of the integration region
        
        int_world_arr = [
            [world.ra - int_ww, world.dec - int_hh, world.vel - intm_range], 
            [world.ra + int_hh, world.dec + int_hh, world.vel + intm_range]
        ]
        int_pix_arr = np.array(wcs.world_to_pixel_values(int_world_arr), dtype=int)
        int_pix_ra1, int_pix_dec1, int_pix_vel1 = int_pix_arr[0]
        int_pix_ra2, int_pix_dec2, int_pix_vel2 = int_pix_arr[1]
        int_pix_ra1, int_pix_ra2 = max(min(int_pix_ra1, int_pix_ra2), 0), min(max(int_pix_ra1, int_pix_ra2), n_ra)
        int_pix_dec1, int_pix_dec2 = max(min(int_pix_dec1, int_pix_dec2), 0), min(max(int_pix_dec1, int_pix_dec2), n_dec)
        int_pix_vel1, int_pix_vel2 = max(min(int_pix_vel1, int_pix_vel2), 0), min(max(int_pix_vel1, int_pix_vel2), n_vel)
        proper_int_pix_vel_range = min(abs(pix.vel - int_pix_vel1), abs(pix.vel - int_pix_vel2))
        if abs(proper_int_pix_vel_range * scale.vel) < intm_valid:
            Log.log_and_print(log_path, 'the integration region is too small, set measure value to default (-1)!')
            m_int_cal = -1
            m_int_fw = -1
            m_int_Ap = -1
            m_int_snr = -1
            m_int_err = -1
        else:
            int_rapix, int_decpix = np.arange(int_pix_ra1, int_pix_ra2), np.arange(int_pix_dec1, int_pix_dec2)
            int_rapixlen, int_decpixlen = int_rapix.shape[0], int_decpix.shape[0]
            # the order of the ra, dec is reversed due to data[vel, dec, ra]
            int_decpixm, int_rapixm = np.meshgrid(int_decpix, int_rapix)
            int_worldarr = np.array(wcs.pixel_to_world_values(np.hstack([
                int_rapixm.reshape(-1, 1), 
                int_decpixm.reshape(-1, 1), 
                np.ones((int_rapixlen*int_decpixlen, 1)) * pix.vel
            ])))
            int_ram, int_decm = int_worldarr[:, 0].reshape((int_decpixlen, int_rapixlen)), int_worldarr[:, 1].reshape((int_decpixlen, int_rapixlen))
            mask_intm_ap = np.square(int_ram - world.ra) + np.square(int_decm - world.dec) <= intm_ap ** 2
            mask_noise = np.all([
                np.square(int_ram - world.ra) + np.square(int_decm - world.dec) >= intn_in ** 2,
                np.square(int_ram - world.ra) + np.square(int_decm - world.dec) <= intn_out ** 2
            ], axis=0)

            def fitfunc(x, Ap, w, C, B, ome, phi):
                period = lambda xi: B * np.sin(ome * xi + phi)  # fit periodic noise
                return np.piecewise(x, 
                    [x < w, x >= w], 
                    [lambda x: (C + Ap) * x + period(x), lambda x: C*x + Ap*w + period(x)]
                )
            
            # measure both side of the signal
            int_sigcents = []   # signal at the center of each integration region
            int_sigapsums = []  # signal at the aperture of each integration region, calculated by sum
            int_sigaps = []     # signal at the aperture of each integration region, calculated by mean
            int_signoises = []  # noise of each integration region
            int_vels = []       # velocity range of each integration region
            for pix_idx in range(1, proper_int_pix_vel_range):
                intsubdata = np.nansum(data[pix.vel - pix_idx:pix.vel + pix_idx, int_pix_dec1:int_pix_dec2, int_pix_ra1:int_pix_ra2], axis=0)
                sigcent = intsubdata[pix.dec - int_pix_dec1, pix.ra - int_pix_ra1]
                sigapsum = np.nansum(intsubdata[mask_intm_ap])
                sigap = np.nanmean(intsubdata[mask_intm_ap])
                signoise = np.nanmean(intsubdata[mask_noise])
                int_vel = wcs.pixel_to_world_values([[pix.ra, pix.dec, pix.vel + pix_idx]])[0][2]
                int_sigcents.append(sigcent)
                int_sigapsums.append(sigapsum)
                int_sigaps.append(sigap)
                int_signoises.append(signoise)
                int_vels.append(int_vel)
            int_sigcents = np.array(int_sigcents)
            int_sigapsums = np.array(int_sigapsums)
            int_sigaps = np.array(int_sigaps)
            int_signoises = np.array(int_signoises)
            int_vels = np.array(int_vels)
            int_snrs = int_sigaps / int_signoises   # not used in this version
            int_vels0 = abs(world.vel - int_vels)
            bounds = np.array([
                    [0, 1e-4],                                          # Ap
                    [0, int_vels0.max()],                               # w
                    [0, 1e-4],                                          # C
                    [0, 1.0],                                           # B
                    [2e-5, 3e-5],                                       # ome
                    [0, 2*np.pi]                                        # phi
                ])
            p, e = curve_fit(fitfunc, int_vels0, int_sigcents, 
                p0=[
                    int_sigcents.max()/int_vels0.max(), 
                    int_vels0.max()/2, 
                    int_sigcents.max()/int_vels0.max(), 
                    0.1, 
                    2.56e-5, 
                    np.pi
                ], 
                bounds=(bounds[:, 0], bounds[:, 1], )
            )
            int_Ap, int_w, int_C, _, _, _ = p
            int_fitcents = fitfunc(int_vels0, *p)

            m_int_cal = int_Ap * int_w
            m_int_fw = int_w
            m_int_Ap = int_Ap
            m_int_snr = int_Ap / int_C
            m_int_err = np.std(int_fitcents - int_sigcents)


        # --------------------------------------------------
        # STEP 5: plot and save
        # --------------------------------------------------
        # 0. create figure and do some preparation
        fig = plt.figure(figsize=(18, 10), dpi=150)
        gs = GridSpec(3, 3)
        fig.suptitle('id: %d ra=%7.3f dec=%7.3f z=%8.6f'%(sid, ra, dec, z))
        fig.clf()
        view_hw, view_hh = float(anaconf['bsl_hw']) / 60, float(anaconf['bsl_hh']) / 60
        
        # 1. plot object radio image (integrated)
        # get sub-datacube and sub-wcs for view
        view_int_vel = float(anaconf['int_vel']) * 1e3 # km/s -> m/s
        view_world_arr = [
            [world.ra - view_hw, world.dec - view_hh, world.vel - view_int_vel], 
            [world.ra + view_hw, world.dec + view_hh, world.vel + view_int_vel]
        ]
        view_pix_arr = np.array(wcs.world_to_pixel_values(view_world_arr), dtype=int)
        view_pix_ra1, view_pix_dec1, view_pix_vel1 = view_pix_arr[0]
        view_pix_ra2, view_pix_dec2, view_pix_vel2 = view_pix_arr[1]
        view_pix_ra1, view_pix_ra2 = max(min(view_pix_ra1, view_pix_ra2), 0), min(max(view_pix_ra1, view_pix_ra2), n_ra)
        view_pix_dec1, view_pix_dec2 = max(min(view_pix_dec1, view_pix_dec2), 0), min(max(view_pix_dec1, view_pix_dec2), n_dec)
        view_pix_vel1, view_pix_vel2 = max(min(view_pix_vel1, view_pix_vel2), 0), min(max(view_pix_vel1, view_pix_vel2), n_vel)
        view_flux = np.mean(np.nan_to_num(data[view_pix_vel1:view_pix_vel2, view_pix_dec1:view_pix_dec2, view_pix_ra1:view_pix_ra2]), axis=0)
        view_subwcs = wcs[:, view_pix_dec1:view_pix_dec2, view_pix_ra1:view_pix_ra2]

        view_ax = fig.add_subplot(gs[0, 0], projection=view_subwcs, slices=('x', 'y', pix.vel))
        view_ax.imshow(view_flux, origin='lower')
        # rect = Quadrangle((world.ra - R, world.dec - R)*units.deg, 2*R*units.deg, 2*R*units.deg,
        #             edgecolor='m', facecolor='none',
        #             transform=view_ax.get_transform('fk5'))
        # view_ax.add_patch(rect)
        view_ax.scatter(world.ra, world.dec, transform=view_ax.get_transform('fk5'), \
            marker='x', color='red', alpha=0.8, label='now')
        view_ax.set_xlabel('RA')
        view_ax.set_ylabel('Dec')
        view_ax.coords.grid(color='black', linestyle='--', alpha=0.6)
        view_ax.set_title('int_view')
        plt.draw()

        # 2. plot object optical image
        # download file first
        opt_img_url = 'https://www.legacysurvey.org/viewer/cutout.fits?ra={:f}&dec={:f}&layer=ls-dr9&pixscale={:d}'.format(world.ra, world.dec, 2)
        opt_img_fpath = 'optimgs/cutout_{:f}_{:f}_{:d}.fits'.format(world.ra, world.dec, 2)
        if not os.path.exists(opt_img_fpath):
            _ = wget.download(opt_img_url, out=opt_img_fpath)
        with fits.open(opt_img_fpath) as optimg:
            optimg_data = optimg[0].data
            optimg_wcs = WCS(optimg[0].header)
        world_arr = [
            [ra - view_hw,  dec - view_hh,  0], 
            [ra,            dec,            0],
            [ra + view_hw,  dec + view_hh,  0]
        ]
        pix_arr = np.array(optimg_wcs.world_to_pixel_values(world_arr), dtype=int)
        pix_ra1, pix_dec1, _     = pix_arr[0]
        pix_ra,  pix_dec,  pix_3 = pix_arr[1]
        pix_ra2, pix_dec2, _     = pix_arr[2]
        opt_n_ra, opt_n_dec, opt_n_3 = optimg_data.shape

        def rgb_precent_scaler(img, pmin, pmax):
            imgmax = np.percentile(img.reshape(-1), pmax)
            imgmin = np.percentile(img.reshape(-1), pmin)
            img1 = (img - imgmin) / (imgmax - imgmin)
            img1[img1 > 1] = 1
            img1[img1 < 0] = 0
            return img1

        pix_ra1, pix_ra2 = max(min(pix_ra1, pix_ra2), 0), min(max(pix_ra1, pix_ra2), opt_n_ra)
        pix_dec1, pix_dec2 = max(min(pix_dec1, pix_dec2), 0), min(max(pix_dec1, pix_dec2), opt_n_dec)
        optimg_ax = fig.add_subplot(gs[0, 2], projection=optimg_wcs, slices=('x', 'y', 0))
        optimg_data = optimg_data.transpose(1, 2, 0)[:, :, ::-1]
        optimg_data_sqrt = np.log(np.abs(optimg_data))
        optimg_data_scale = rgb_precent_scaler(optimg_data_sqrt, 1, 99)
        optimg_ax.imshow(optimg_data_scale, origin='lower', cmap=plt.cm.viridis)
        optimg_ax.scatter(ra, dec, transform=optimg_ax.get_transform('fk5'), marker='o', \
		    s=300, edgecolor='red', facecolor='none', alpha=0.8)
        optimg_ax.grid(color='yellow', linestyle='--', alpha=0.6)
        optimg_ax.set_xlabel('RA')
        optimg_ax.set_ylabel('Dec')
        optimg_ax.set_title('optical image')
        plt.draw()

        # 3. plot object raw spectra and baseline
        bsl_sel_world = np.array(wcs.pixel_to_world_values(bsl_pixarr))
        bsl_sel_ra, bsl_sel_dec = bsl_sel_world[:, 0], bsl_sel_world[:, 1]
        view_world_arr = [
            [world.ra - view_hw, world.dec - view_hh, world.vel], 
            [world.ra,           world.dec          , world.vel],
            [world.ra + view_hw, world.dec + view_hh, world.vel]
        ]
        view_pix_arr = np.array(wcs.world_to_pixel_values(view_world_arr), dtype=int)
        view_pix_ra1, view_pix_dec1, _            = view_pix_arr[0]
        view_pix_ra,  view_pix_dec,  view_pix_vel = view_pix_arr[1]
        view_pix_ra2, view_pix_dec2, _            = view_pix_arr[2]
        view_pix_ra1, view_pix_ra2 = max(min(view_pix_ra1, view_pix_ra2), 0), min(max(view_pix_ra1, view_pix_ra2), n_ra)
        view_pix_dec1, view_pix_dec2 = max(min(view_pix_dec1, view_pix_dec2), 0), min(max(view_pix_dec1, view_pix_dec2), n_dec)
        view_flux = np.nan_to_num(data[view_pix_vel, view_pix_dec1:view_pix_dec2, view_pix_ra1:view_pix_ra2])
        view_subwcs = wcs[:, view_pix_dec1:view_pix_dec2, view_pix_ra1:view_pix_ra2]
        bsl_ax = fig.add_subplot(gs[0, 1], projection=view_subwcs, slices=('x', 'y', pix.vel))
        bsl_ax.imshow(view_flux, vmin=-1e-2, vmax=1e-2, origin='lower')
        bsl_ax.scatter(bsl_sel_ra, bsl_sel_dec, transform=bsl_ax.get_transform('fk5'), \
            marker='x', color='m', alpha=0.8)
        bsl_ax.scatter(world.ra, world.dec, transform=bsl_ax.get_transform('fk5'), \
            marker='x', color='r', alpha=0.8)
        bsl_ax.set_xlabel('RA')
        bsl_ax.set_ylabel('Dec')
        bsl_ax.coords.grid(color='black', linestyle='--', alpha=0.6)
        bsl_ax.set_title('baseline')
        plt.draw()

        # 4. plot object measured spectra with baseline reduced
        x_argsort = np.argsort(x)
        xm_argsort = np.argsort(xm)
        signal_ax1 = fig.add_subplot(gs[1, :])
        signal_ax1.plot(x[x_argsort], y_raw[x_argsort], 'k-', linewidth=0.8, label='raw spectra')
        signal_ax1.plot(x[x_argsort], bsl0[x_argsort], 'r--', linewidth=0.8, label='baseline')
        signal_ax1.set_xlabel('Velocity (m/s)')
        signal_ax1.set_ylabel('Flux Density (Jy)')
        signal_ax1.set_xlim([x.min(), x.max()])
        signal_ax1.grid(True, alpha=0.4)
        signal_ax1.label_outer()
        signal_ax1.legend(loc='upper right', framealpha=0.4)

        signal_ax2 = fig.add_subplot(gs[2, :])
        signal_ax2.plot(x[x_argsort], y[x_argsort], 'k-', linewidth=0.8, alpha=0.5, label='signal')
        signal_ax2.plot(xm[xm_argsort], ym[xm_argsort], 'r-.', linewidth=0.8, alpha=0.8, label='averaged %d'%measure_ma)
        signal_ax2.set_xlabel('Velocity (m/s)')
        signal_ax2.set_ylabel('Flux Density (Jy)')
        signal_ax2.set_xlim([x.min(), x.max()])
        signal_ax2.grid(True, alpha=0.4)
        signal_ax2.label_outer()
        signal_ax2.legend(loc='upper right')

        plt.draw()
        plt.savefig(fig_dir + '/%08d.png'%sid)
        plt.close()

        # 5. plot int measurement
        int_fig = plt.figure(figsize=(7, 3))

        int_ax1 = int_fig.add_subplot(111)
        int_ax1.plot(int_vels0, int_sigcents, 'k-', linewidth=0.8, alpha=0.5, label='signal')
        int_ax1.plot(int_vels0, int_fitcents, 'r-.', linewidth=0.8, alpha=0.8, label='fit')
        int_ax1.vlines(p[1], 0, int_sigcents.max(), linestyles='dashed', linewidth=0.8, alpha=0.5)
        int_ax1.set_xlabel('Velocity (m/s)')
        int_ax1.set_ylabel('Int Flux')
        int_ax1.set_xlim([int_vels0.min(), int_vels0.max()])
        int_ax1.legend(loc='upper left')
        int_ax1.set_title(' '.join(map(lambda d: '{:.2e}'.format(d), p)))

        plt.savefig(mea_dir + '/%08d.png'%sid)
        plt.close()


        # --------------------------------------------------
        # STEP 6: output catalog
        # --------------------------------------------------
        # write to seperated catalog on this step
        # merge catalogs in outer function
        data_summary = [
            [sid,           '{:08d}'], 
            [world.ra,      '{:.4f}'],
            [world.dec,     '{:.4f}'],
            [world.vel,     '{:.3e}'],
            [m_peakvel,     '{:.3e}'],
            [m_w50,         '{:.3e}'],
            [m_w20,         '{:.3e}'],
            [m_int_flux,    '{:.3e}'],
            [m_rms,         '{:.3e}'],
            [m_snr,         '{:.3e}'],
            [m_int_cal,     '{:.3e}'],
            [m_int_fw,      '{:.3e}'],
            [m_int_Ap,      '{:.3e}'],
            [m_int_snr,     '{:.3e}'],
            [m_int_err,     '{:.3e}'],
        ]
        data_summary_str = ','.join(['{:s}'.format(d[1].format(d[0])) for d in data_summary])
        return data_summary_str
