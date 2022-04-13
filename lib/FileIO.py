from os.path import exists as path_exists
from re import search as re_search
from urllib.request import urlopen
from json import loads as json_loads
from pandas import json_normalize


class FitsInfo():
    '''Class to store fits info'''
    def __init__(self, fpath: str):
        self.path = fpath
        self.fpath = fpath.split('/')[-1]
        self.fname = self.fpath[:-5]
        ra_str = re_search(r'RA[0-9]{4}_[0-9]{4}', self.fname).group(0)
        self.ra1 = int(ra_str.split('_')[0][-4:])
        self.ra2 = int(ra_str.split('_')[1])
        dec_str = re_search(r'DEC(\+|\-)[0-9]{6}', self.fname).group(0)
        self.dec = int(dec_str[4:])
        if dec_str[3] == '-':
            self.dec = - self.dec
        type_seq = re_search(r'F?(_H)?(_O)?(_R)?_hifast', self.fname).group(0).split('_')
        if 'O' in type_seq:     # optical frequency
            self.freq_type = 'optical'
        elif 'F' in type_seq:   # radio frequency
            self.freq_type = 'radio'
        else:
            print('No specific type in fits name {:s}, use radio frequency!'.format(self.name))
            self.freq_type = 'radio'
        
    def __str__(self):
        return self.fname



class SDSSCatalog():
    def __init__(self, fits: FitsInfo, dec_with: float=0.50, zmin: float=0.00, zmax: float=0.08):
        self.query_fname = fits.fname
        self.query_fmt = 'http://159.226.170.146/fast/rectsearch.aspx?' + \
            'ramin={:f}&ramax={:f}&decmin={:f}&decmax={:f}&zmin={:f}&zmax={:f}'
        self.ra1, self.ra2 = self.f2s_ra(fits.ra1), self.f2s_ra(fits.ra2)
        dec0 = self.f2s_dec(fits.dec)   # dec of fits image's center
        self.dec1, self.dec2 = dec0 - dec_with / 2.0, dec0 + dec_with / 2.0
        self.zmin, self.zmax = zmin, zmax
        self.path = ''  # will generate when self.download called

    def f2s_ra(self, ra: int):
        '''Convert RA's format from name in fits file to d.ms'''
        ra_h = ra // 100
        ra_m = (ra % 100) / 60.0
        return (ra_h + ra_m) * 15   # hours -> deg
    
    def f2s_dec(self, dec: int):
        '''Convert Dec's format from name in fits file to d.ms'''
        dec_d = dec // 100_00
        dec_m = ((dec % 100_00) // 100) / 60.0
        dec_s = (dec % 100) / 3600.0
        return dec_d + dec_m + dec_s
    
    def download(self, savedir: str, skip_saved: bool=True) -> int:
        '''
        Download SDSS catalog from website and save as csv file.            \\
        Output:                                                             \\
           -2       Null data queried                                       \\
           -1       Failed to download and save                             \\
            0       File exists, skipped                                    \\
            1       Successfully downloaded and saved
        '''
        self.path = '/'.join((savedir, self.query_fname + '.csv'))
        if skip_saved and path_exists(self.path):
            print('SDSS catalog {:s} already exists, skipped!'.format(self.path))
            return 0
        print('Downloading SDSS catalog of *{:s}*'.format(self.query_fname))
        response = urlopen(self.query_fmt.format(\
            self.ra1, self.ra2, self.dec1, self.dec2, self.zmin, self.zmax))
        if response.status != 200:
            print('Failed to download catalog of *{:s}* with status code {:d} {:s}'.format(\
                self.query_fname, response.status, response.reason))
            return -1
        data = response.read()
        if data.decode('utf-8') == r'{}':
            print('Null data queried by {:s}, skipped'.format(self.query_fname))
            return -2
        print('SDSS catalog of *{:s}* queried! Now saving ...'.format(self.query_fname))
        data_dict: dict = json_loads(data)
        data_pd = json_normalize(data_dict['sources'])
        with open(self.path, 'w', newline='') as fout:
            fout.write('#' + ','.join(data_pd.columns) + '\n\n')
            fout.write('#rect search:%f,%f,%f,%f,%f,%f\n'%(\
                self.ra1, self.ra2, self.dec1, self.dec2, self.zmin, self.zmax))
            data_pd.to_csv(fout, index=False, header=False)
        print('SDSS catalog saved to *{:s}*'.format(self.path))
        return 1
    