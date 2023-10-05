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

    config['in_name'] = options.get_string(option_section, 'in_name')

    config['out']       = options.get_string(option_section, "out") #shear

    if config['out'] != 'shear':
        raise Exception('Currently only cosmics shear is supported as an output.')

    config['out_mode']  = options.get_string(option_section, "out_mode") #map or pcl

    config['out_name'] = options.get_string(option_section, 'out_name')

    config['shear_bias'] = options.get_bool(option_section, 'shear_bias')

    if config['shear_bias']:
        config['shear_bias_section_name'] = options.get_string(option_section, 'shear_bias_section_name')
    
    config['corrected_biases'] = options.get_string(option_section, 'corrected_biases')

    config['nside'] = options.get_int(option_section, "nside")

    if not np.log2(float(config['nside'])).is_integer():
        raise Exception('nside must be a power of 2.')

    if 'pcl' in config['out_mode']:
        config['lmax'] = options.get_int(option_section, "lmax")

    config['doVariableDepth_in'] = options.get_bool(option_section, 'doVariableDepth')

    config['weight_maps'] = options.get_string(option_section, 'weight_maps')

    config['clean']       = options[option_section,  "clean"]

    if type(config['clean']) != bool:
        raise Exception("The clean must be either 'T' or 'F'")
    
    return config

def execute(block, config):
    if block[config['in_name'], 'outStyle'] != '64':
        raise Exception('SALMO outputs are in the incorrect outStyle, please set outStyle=64, so different fields are separated in different .fits files.')

    nside = config['nside']
    vd = block[config['in_name'], 'doVariableDepth']

    nbLensFields = block[config['in_name'], 'nbLensFields']
    nbSourceFields = block[config['in_name'], 'nbSourceFields']

    nz = np.array(str(block[config['in_name'], 'nOfZPath']).split(' '), dtype = str)
    
    weight_map_names = np.array(config['weight_maps'].split(' '), dtype = str)
    
    #Map fields which are based on the same n(z)
    mapping = {}
    _, index = np.unique(nz, return_index=True)
    for i in nz[np.sort(index)]:
        mapping[i] = np.where(nz == i)[0]

    joint_fields = list(mapping.values())   

    if str(vd) == '1': #If vd, then the shear maps are split into depth bins
        ndepth = block[config['in_name'], 'nbDepthMaps']
        ntomo = block[config['in_name'], 'nbTomo']
        for i in range(int(ntomo)):
            joint_fields.append(np.array([nbLensFields + nbSourceFields + n + int(ndepth)*i for n in range(int(ndepth))]))

    with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
            joint_fields = np.array(joint_fields)[np.argsort([np.mean(joint_fields[i]) for i in range(len(joint_fields))])]

    print('Making shear maps...')
    tic = time.time()

    bin_num = len(joint_fields[nbLensFields:]) #As only cosmic shear is supported, the number of bins is the number of source fields

    #Read in sampled shear bias parameters and calibrated m-bias

    if config['shear_bias']:
        biases = block[config['shear_bias_section_name'], 'biases']
        if 'mult' in biases:
            m = np.array(block[config['shear_bias_section_name'], 'mult_bias_random_sample'])
        else:
            m = np.zeros(bin_num)

        if 'add' in biases:
            c1 = np.array(block[config['shear_bias_section_name'], 'add_bias_e1_random_sample'])
            c2 = np.array(block[config['shear_bias_section_name'], 'add_bias_e2_random_sample'])
        else:
            c1, c2 = np.zeros(bin_num), np.zeros(bin_num)

        if 'psf' in biases:
            alpha1 =  np.array(block[config['shear_bias_section_name'], 'psf_bias_e1_random_sample'])
            alpha2 =  np.array(block[config['shear_bias_section_name'], 'psf_bias_e2_random_sample'])
            psf_map_names = np.array(str(block[config['shear_bias_section_name'], 'psf_ell_map_paths']).split(' '), dtype = str)
        else:
            alpha1, alpha2 = np.zeros(bin_num), np.zeros(bin_num)

        if 'mult' in biases:
            m_corrected = np.array(block[config['shear_bias_section_name'], 'mult_bias_mean'])
        else:
            m_corrected = np.zeros(bin_num)


    # Read in and combine shear maps from different fields for each tomographic bin
    tomo = 0
    alm, alm_rand = [], []
    for group in joint_fields:
        pos1_group, pos2_group, e_group = [], [], []
        for i in group:
            if i >= nbLensFields:
                file = '{0}_{1}_sample{2}_type{3}.fits'.format(block[config['in_name'], 'outPrefix'], block[config['in_name'], 'runTag'], block[config['in_name'], 'counter'], i)
                print('  Reading {0}...'.format(file))
                data = fits.getdata(file, 1)

                if config['clean']:
                    print('  Deleting {0} to save disk space...'.format(file))
                    os.remove(file)

                pos1_group.append(data.field(0))
                pos2_group.append(data.field(1))
                e_group.append(data.field(5) + 1j * data.field(6))
                del data

                lens = False
            else:
                if config['clean']:
                    file = '{0}_{1}_sample{2}_type{3}.fits'.format(block[config['in_name'], 'outPrefix'], block[config['in_name'], 'runTag'], block[config['in_name'], 'counter'], i)
                    os.remove(file)
                lens = True
                pass
        
        if lens == False:
            print('  Joining the following sample types which are all within tomographic bin {1}: {0}'.format(group, tomo+1))

            pos1_all = np.concatenate(pos1_group, axis =  None)
            pos2_all = np.concatenate(pos2_group, axis =  None)
            e_all = np.concatenate(e_group, axis =  None)

            #Apply random shear bias to each tomographic bin

            if config['shear_bias']:
                print("   Including shear bias...")
                if 'psf' in biases:
                    print("     Also including PSF shear bias...")
                    psf_ell_e1 = hp.read_map(psf_map_names[0])
                    psf_ell_e2 = hp.read_map(psf_map_names[1])
                    pix = hp.ang2pix(nside, pos1_all, pos2_all, lonlat=True)
                    e_all = ((1+m[tomo])*e_all.real + c1[tomo] + alpha1[tomo]*psf_ell_e1[pix]) + 1j * (((1+m[tomo])/(1+m_corrected[tomo]))*e_all.imag + c2[tomo] + alpha2[tomo]*psf_ell_e2[pix])
                    del psf_ell_e1
                    del psf_ell_e2
                    del pix
                else:
                    e_all = ((1+m[tomo])*e_all.real + c1[tomo]) + 1j * ((1+m[tomo])*e_all.imag + c2[tomo])
            
            #Correct for multiplicative shear bias and/or additive shear bias

            if 'add' in config['corrected_biases']:
                e_all = (e_all.real - np.mean(e_all.real)) + 1j * (e_all.imag - np.mean(e_all.imag))
            
            if 'mult' in config['corrected_biases']:
                e_all *= 1/(1+m_corrected[tomo])

            shear = np.zeros(hp.nside2npix(nside), dtype=complex)
            counts = np.zeros_like(shear, dtype=int)

            map_shears(shear, counts, pos1_all,  pos2_all, e_all, gal_wht=None)

            #Normalise the shear map by the number of galaxies in each pixel
            shear[counts > 0] = np.divide(shear[counts > 0], counts[counts > 0])
            
            weight_maps = []
            for nb in range(len(weight_map_names)):
                weight_maps.append(hp.read_map(weight_map_names[nb]))
            print('     Weighting shear values according to mask weights...')
            weight_maps = np.sum(weight_maps, axis = 0)
            
            shear = np.multiply(shear, weight_maps)
            
            if config['doVariableDepth_in']: #Correct for the absence of a fractional mask in SALMO in the variable depth case
                counts = np.multiply(counts, weight_maps)

            if 'map' in config['out_mode']:
                filename = '{0}_{1}_sample{2}_tomo{3}_counts+shear.fits'.format(block[config['in_name'], 'outPrefix'], block[config['in_name'], 'runTag'], block[config['in_name'], 'counter'], tomo)
                print('    Saving consolidated count and shear fields for tomographic bin {0} as {1}...'.format(tomo+1, filename))
                save_fits(filename, [counts, shear.real, shear.imag])

            #Compress the shear map into pseudo-Cls
            if 'pcl' in config['out_mode']:
                del counts

                #Randomly rotate the shear field to isolate the shape noise
                gal_num = len(pos1_all)
                rand_theta = 2*np.pi*np.random.random_sample(gal_num)
                
                e1_corr = e_all.real*np.cos(rand_theta) - e_all.imag*np.sin(rand_theta)
                e2_corr = e_all.imag*np.cos(rand_theta) + e_all.real*np.sin(rand_theta)
                del e_all

                rand = np.zeros(hp.nside2npix(nside), dtype=complex)
                _ = np.zeros_like(rand, dtype=int)

                map_shears(rand, _, pos1_all, pos2_all,  e1_corr + 1j * e2_corr, gal_wht=None)
                del e1_corr
                del e2_corr
                del pos1_all
                del pos2_all
                rand[_ > 0] = np.divide(rand[_ > 0], _[_ > 0])
                del _

                print('  Computing alms for tomographic bin {0}...'.format(tomo+1))
                print('  -----')
                
                rand = np.multiply(rand, weight_maps)
                
                del weight_maps
                
                alm.append(hp.sphtfunc.map2alm_spin([shear.real, shear.imag], spin = 2, lmax = config['lmax'])) #Compute the alms for the shear field
                alm_rand.append(hp.sphtfunc.map2alm_spin([rand.real, rand.imag], spin = 2, lmax = config['lmax'])) #Compute the alms for the random shear field
                del shear
                del rand

            tomo += 1

    toc = time.time()
    if 'map' in config['out_mode'] and 'pcl' in config['out_mode'] :
        print('Making and saving shear maps plus computing their alms took {0} seconds'.format(round(toc-tic, 3)))

    elif 'map' in config['out_mode']:
        print('Making and saving shear maps took {0} seconds'.format(round(toc-tic, 3)))

    elif 'pcl' in config['out_mode']:
        print('Making shear maps and computing their alms took {0} seconds'.format(round(toc-tic, 3)))

    if 'pcl' in config['out_mode']:
        block[config['out_name'], "is_auto"] = 'True'
        block[config['out_name'], "sample_a"] = 'source'
        block[config['out_name'], "sample_b"] = 'source'
        block[config['out_name'], "nbin"] = tomo
        block[config['out_name'], "nbin_a"] = tomo
        block[config['out_name'], "nbin_b"] = tomo
        block[config['out_name'], "ell"] = np.arange(0, config['lmax']+1)
        block[config['out_name'], 'nOfZPath'] = block[config['in_name'], 'nOfZPath']
        if nbSourceFields > 0:
            block[config['out_name'], 'n_density'] = block[config['in_name'], 'n_gal'][nbLensFields:]
            block[config['out_name'], 'sigma_e'] = block[config['in_name'], 'sigma_eps'][nbLensFields:]
            block[config['out_name'], "nbLensFields"] = block[config['in_name'], "nbLensFields"]
            block[config['out_name'], "nbSourceFields"] = block[config['in_name'], "nbSourceFields"]

        if str(vd) == '1':
            block[config['out_name'], 'a_n_density'] = block[config['in_name'], 'a_n_gal']
            block[config['out_name'], 'b_n_density'] = block[config['in_name'], 'b_n_gal']
            block[config['out_name'], 'a_sigma_e'] = block[config['in_name'], 'a_sigma_eps']
            block[config['out_name'], 'b_sigma_e'] = block[config['in_name'], 'b_sigma_eps']
            block[config['out_name'], 'bin_depth'] = block[config['in_name'], 'bin_depth']

        print('Calculating angular power spectra...')
        tic = time.time()
        for i in range(tomo):
            for j in range(i+1):
                print('     Getting shear Cls for bin {0} and bin {1}...'.format(i+1, j+1))
                pcls = hp.alm2cl(alm[i], alm[j], lmax = int(config['lmax'])) #Compute the pseudo-Cls
                block[config['out_name'], 'bin_{0}_{1}'.format(i+1,j+1)] = pcls[0]
                block[config['out_name'] + '_BB', 'bin_{0}_{1}'.format(i+1,j+1)] = pcls[1]
                block[config['out_name'] + '_EB', 'bin_{0}_{1}'.format(i+1,j+1)] = pcls[2]
                del pcls

                if i == j:
                    pcls_rand = hp.alm2cl(alm_rand[i], alm_rand[j], lmax = int(config['lmax'])) #Compute the pseudo-Cls due to shape noise bias
                    block[config['out_name'] + '_noise', 'bin_{0}_{1}'.format(i+1,j+1)] = pcls_rand[0]
                    block[config['out_name'] + '_noise_BB', 'bin_{0}_{1}'.format(i+1,j+1)] = pcls_rand[1]
                    block[config['out_name'] + '_noise_EB', 'bin_{0}_{1}'.format(i+1,j+1)] = pcls_rand[2]
                    del pcls_rand

        toc = time.time()
        print('Calculating angular power spectra took {0} seconds'.format(round(toc-tic, 3)))
    return 0

def cleanup(config):
    pass