#!python
from logging import raiseExceptions
from cosmosis.datablock import names, option_section
import healpy as hp
import numpy as np
from astropy.io import fits
from numba import njit
import warnings
import time
import os

@njit(nogil=True)
def _map_shears_weights(she_map, wht_map, gal_pix, gal_she, gal_wht):
    for i, s, w in zip(gal_pix, gal_she, gal_wht):
        she_map[i] += s
        wht_map[i] += w

@njit(nogil=True)
def _map_shears(she_map, wht_map, gal_pix, gal_she):
    for i, s in zip(gal_pix, gal_she):
        she_map[i] += s
        wht_map[i] += 1

def map_shears(she_map, wht_map, gal_lon, gal_lat, gal_she, gal_wht=None):
    nside = hp.get_nside(she_map)
    gal_pix = hp.ang2pix(nside, gal_lon, gal_lat, lonlat=True)
    if gal_wht is None:
        _map_shears(she_map, wht_map, gal_pix, gal_she)
    else:
        _map_shears_weights(she_map, wht_map, gal_pix, gal_she, gal_wht)

def save_fits(name, map):
    if len(map) != 3:
        full   = np.array(map).astype(np.float32)
        nbRows = full.size // 1024
        full   = full.reshape(nbRows, 1024)
        nside  = hp.npix2nside(full.size)

        HDU1 = fits.PrimaryHDU()
        HDU2 = fits.BinTableHDU.from_columns([
        fits.Column(name='TEMPERATURE', format='1024E', unit='unknown ', array=full)
        ])

        hdr = HDU2.header
        hdr.append(('EXTNAME',  'xtension ',   'Name of this binary table extension'),          bottom=True)
        hdr.append(('PIXTYPE',  'HEALPIX ',    'HEALPIX pixelisation'),                         bottom=True)
        hdr.append(('ORDERING', 'RING    ',    'Pixel ordering scheme'),                        bottom=True)
        hdr.append(('COORDSYS', 'C       ',    'Ecliptic, Galactic or Celestial (equatorial)'), bottom=True)
        hdr.append(('NSIDE',    nside,         'nside of the pixel'),                           bottom=True)
        hdr.append(('FIRSTPIX', 0,             'First pixel # (0 based)'),                      bottom=True)
        hdr.append(('LASTPIX',  12*nside**2-1, 'Last pixel # (0 based)'),                       bottom=True)
        hdr.append(('INDXSCHM', 'IMPLICIT',    'Indexing: IMPLICIT or EXPLICIT'),               bottom=True)
        hdr.append(('POLCCONV', 'COSMO   ',    ''),               bottom=True)

        fits.HDUList([HDU1, HDU2]).writeto(name, overwrite=True)
    
    else:
        full   = np.array(map).astype(np.float32)
        nside  = hp.npix2nside(full[0].size)
        nbRows = full[0].size // 1024

        HDU1 = fits.PrimaryHDU()
        HDU2 = fits.BinTableHDU.from_columns([
        fits.Column(name='TEMPERATURE', format='1024E', unit='unknown ', array=full[0].reshape(nbRows, 1024)),
        fits.Column(name='Q_POLARISATION', format='1024E', unit='unknown ', array=full[1].reshape(nbRows, 1024)),
        fits.Column(name='U_POLARISATION', format='1024E', unit='unknown ', array=full[2].reshape(nbRows, 1024))
        ])

        hdr = HDU2.header
        hdr.append(('EXTNAME',  'xtension ',   'Name of this binary table extension'),          bottom=True)
        hdr.append(('PIXTYPE',  'HEALPIX ',    'HEALPIX pixelisation'),                         bottom=True)
        hdr.append(('ORDERING', 'RING    ',    'Pixel ordering scheme'),                        bottom=True)
        hdr.append(('COORDSYS', 'C       ',    'Ecliptic, Galactic or Celestial (equatorial)'), bottom=True)
        hdr.append(('NSIDE',    nside,         'nside of the pixel'),                           bottom=True)
        hdr.append(('FIRSTPIX', 0,             'First pixel # (0 based)'),                      bottom=True)
        hdr.append(('LASTPIX',  12*nside**2-1, 'Last pixel # (0 based)'),                       bottom=True)
        hdr.append(('INDXSCHM', 'IMPLICIT',    'Indexing: IMPLICIT or EXPLICIT'),               bottom=True)
        hdr.append(('POLCCONV', 'COSMO   ',    ''),               bottom=True)

        fits.HDUList([HDU1, HDU2]).writeto(name, overwrite=True)

def setup(options):
    config = {}
    config['out']       = options.get_string(option_section, "out") #shear

    if config['out'] != 'shear':
        raise Exception('Currently only cosmics shear is supported as an output.')

    config['out_mode']  = options.get_string(option_section, "out_mode") #map or pcl

    config['nside'] = options.get_int(option_section, "nside")

    if not np.log2(float(config['nside'])).is_integer():
        raise Exception('nside must be a power of 2.')

    if 'pcl' in config['out_mode']:
        config['lmax'] = options.get_int(option_section, "lmax")

    config['clean']       = options[option_section,  "clean"]

    if type(config['clean']) != bool:
        raise Exception("The clean must be either 'T' or 'F'")
    
    return config

def execute(block, config):
    if block['salmo', 'outStyle'] != '64':
        raise Exception('SALMO outputs are in the incorrect outStyle, please set outStyle=64, so different fields are separated in different .fits files.')

    nside = config['nside']
    vd = block['salmo', 'doVariableDepth']
    lensing = np.array(str(block['salmo', 'doLensing']).split(' '), dtype = str)

    nbLensFields = len(np.where(lensing == '0')[0])
    nbSourceFields = len(np.where(lensing == '1')[0])

    nz = np.array(str(block['salmo', 'nOfZPath']).split(' '), dtype = str)

    mapping = {}
    for i in np.unique(nz):
        mapping[i] = np.where(nz == i)[0]

    joint_fields = list(mapping.values())   

    if str(vd) == '1':
        ndepth = block['salmo', 'nbDepthMaps']
        ntomo = block['salmo', 'nbTomo']
        for i in range(int(ntomo)):
            joint_fields.append(np.array([nbLensFields + nbSourceFields + n + int(ndepth)*i for n in range(int(ndepth))]))

    with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
            joint_fields = np.array(joint_fields)[np.argsort([np.mean(joint_fields[i]) for i in range(len(joint_fields))])]

    print('Making shear maps...')
    tic = time.time()

    #Shear
    tomo = 0
    alm = []
    for group in joint_fields:
        shear_group, num_group = [], []
        for i in group:
            if i >= nbLensFields:
                she = np.zeros(hp.nside2npix(nside), dtype=complex)
                num = np.zeros_like(she, dtype=int)
                file = '{0}_{1}_sample{2}_type{3}.fits'.format(block['salmo', 'outPrefix'], block['salmo', 'runTag'], block['salmo', 'counter'], i)
                print('  Reading {0}...'.format(file))
                data = fits.getdata(file, 1)
                if config['clean']:
                    print('  Deleting {0} to save disk space...'.format(file))
                    os.remove(file)
                map_shears(she, num, data.field(0),  data.field(1), data.field(5) + 1j * data.field(6), gal_wht=None)
                del data
                num_group.append(num)
                shear_group.append(she)
                lens = False
            else:
                if config['clean']:
                    file = '{0}_{1}_sample{2}_type{3}.fits'.format(block['salmo', 'outPrefix'], block['salmo', 'runTag'], block['salmo', 'counter'], i)
                    os.remove(file)
                lens = True
                pass
        if lens == False:
            print('  Joining the following sample types which are all within tomographic bin {1}: {0}'.format(group, tomo+1))
            if 'map' in config['out_mode']:
                filename = '{0}_{1}_sample{2}_tomo{3}_counts+shear.fits'.format(block['salmo', 'outPrefix'], block['salmo', 'runTag'], block['salmo', 'counter'], tomo)
                print('    Saving consolidated count and shear fields for tomographic bin {0} as {1}...'.format(tomo+1, filename))
                counts = np.add.reduce(num_group)
                shear = np.add.reduce(shear_group)
                save_fits(filename, [counts, shear.real, shear.imag])
                del counts
                del shear
            if 'pcl' in config['out_mode']:
                print('  Computing alms for tomographic bin {0}...'.format(tomo+1))
                shear = np.add.reduce(shear_group)
                print('  -----')
                alm.append([hp.sphtfunc.map2alm(shear.real, use_pixel_weights = True), hp.sphtfunc.map2alm(shear.imag, use_pixel_weights = True)])
                del shear
            tomo += 1
    toc = time.time()
    if 'map' in config['out_mode'] and 'pcl' in config['out_mode'] :
        print('Making and saving shear maps plus computing their alms took {0} seconds'.format(round(toc-tic, 3)))

    elif 'map' in config['out_mode']:
        print('Making and saving shear maps took {0} seconds'.format(round(toc-tic, 3)))

    elif 'pcl' in config['out_mode']:
        print('Making shear maps and computing their alms took {0} seconds'.format(round(toc-tic, 3)))

    if 'pcl' in config['out_mode']:
        block['shear_pcl', "is_auto"] = 'True'
        block['shear_pcl', "sample_a"] = 'source'
        block['shear_pcl', "sample_b"] = 'source'
        block['shear_pcl', "nbin"] = tomo
        block['shear_pcl', "nbin_a"] = tomo
        block['shear_pcl', "nbin_b"] = tomo
        block['shear_pcl', "ell"] = np.arange(0, config['lmax']+1)
        if nbSourceFields > 0:
            block['shear_pcl', 'n_density'] = block['salmo', 'n_gal'][nbLensFields:]
            block['shear_pcl', 'sigma_e'] = block['salmo', 'sigma_eps'][nbLensFields:]

        if str(vd) == '1':
            block['shear_pcl', 'a_n_density'] = block['salmo', 'a_n_gal']
            block['shear_pcl', 'b_n_density'] = block['salmo', 'b_n_gal']
            block['shear_pcl', 'a_sigma_e'] = block['salmo', 'a_sigma_eps']
            block['shear_pcl', 'b_sigma_e'] = block['salmo', 'b_sigma_eps']

        print('Calculating angular power spectra...')
        tic = time.time()
        for i in range(tomo):
            for j in range(i+1):
                print('  Getting shear Cls for bin {0} and bin {1}...'.format(i+1, j+1))
                pcl = hp.alm2cl([alm[i][0], alm[i][1]], [alm[j][0], alm[j][1]], lmax = int(config['lmax']))[0] #EE mode
                block['shear_pcl', 'bin_{0}_{1}'.format(i+1,j+1)] = pcl
        toc = time.time()
        print('Calculating angular power spectra took {0} seconds'.format(round(toc-tic, 3)))
    return 0

def cleanup(config):
    pass