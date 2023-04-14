from os import EX_SOFTWARE
from numba import njit
from cosmosis.datablock import option_section, names, BlockError
from pathlib import Path
import numpy as np
from cosmology import LCDM
import glass.sim
import glass.cosmosis
import glass.matter
import glass.lensing
import glass.galaxies
import glass.observations
import healpy as hp
from interpcl import interpcl
import os
from collections.abc import Iterable
import healpy as hp
from astropy.io import fits
import time

def setup(options):
    config = {}
    
    config['nside'] = options.get_int(option_section, "nside")
    config['nside_out'] = options.get_int(option_section, "nside_out")

    if config['nside_out'] < config['nside']:
        raise Warning('Are you sure that you want to downgrade the resolution of the maps?')

    if not np.log2(float(config['nside'])).is_integer():
        raise Exception('nside must be a power of 2.')

    config['out']       = options.get_string(option_section, "out") #shear
    config['out_mode']  = options.get_string(option_section, "out_mode") #pcl, map, catalogue, salmo

    if 'pcl' in config['out_mode']:
        config['lmax'] = options.get_int(option_section, "lmax")

    try:
        visibility_file = options.get_string(option_section, "visibility_file")
        config['mask_nside'] = options.get_int(option_section, "mask_nside")
        config['mask'] =  hp.read_map(visibility_file)
        print('Reading mask file: {0}'.format(visibility_file))
    except:
        visibility_file = None
        config['mask_nside'] = None
        config['mask'] = None
        print('No visibility file given. Full sky will be computed.')

    if 'salmo' in config['out_mode']:
        config['out_folder'] = options.get_string(option_section, "out_folder")
        config['prefix'] = options.get_string(option_section, "prefix")
        config['runTag'] = options.get_string(option_section, "runTag")
        Path(config['out_folder']).mkdir(parents=True,exist_ok=True)
    else:
        config['n_density']  = options[option_section, 'n_density']
        config['sigma_e']    = options[option_section, 'sigma_e']

        if isinstance(config['sigma_e'], Iterable):
            raise Exception('Currently only support a single value of sigma_e for all bins.')

    config['ia'] = options.get_string(option_section, "ia")

    try:
        config['seed'] = int(options[option_section, "seed"])
        print('-- Setting a fixed seed for Glass with value: {0}'.format(config['seed']))
    except:
        config['seed'] = None

    return config

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

def execute(block, config):
    
    #Create file paths for fits files
    counter = np.random.randint(low=0, high=999999999)
                                             
    new_path = False
    path = '{0}/{1}_sample{2}'.format(config['out_folder'], config['runTag'], counter)
    while new_path is False:
        try:
            Path(path).mkdir(parents=False,exist_ok=False)
            print('Creating {0}...'.format(path))
            Path('{0}/glass_denMap'.format(path)).mkdir(exist_ok=False)
            Path('{0}/glass_lenMap'.format(path)).mkdir(exist_ok=False)
            config['counter'] = counter
            new_path = True
        except:
            counter = np.random.randint(low=0, high=999999999)
            path = '{0}/{1}_sample{2}'.format(config['out_folder'], config['runTag'], counter)

    #Setting up cosmology for the convergence map
    config["h0"]            = block[names.cosmological_parameters, "h0"]
    config["omega_m"]       = block[names.cosmological_parameters, "omega_m"]
    config["omega_k"]       = block[names.cosmological_parameters, "omega_k"]
    config["omega_lambda"]  = block[names.cosmological_parameters, "omega_lambda"]
    config['a_ia']          = block[names.intrinsic_alignment_parameters, "a"]

    cosmo = LCDM(h=config['h0'], Om = config['omega_m'], Ol = config['omega_lambda'], Ok = config['omega_k'])
    
    #Reading pre-computed Cls
    nshell = block['shell_matter', 'nbin']
    redshift_shells = block['shell_matter', 'zlim']
    ell = block['matter_cl', 'ell']
    lmax_in = int(np.max(ell))

    matter_cl = []
    for i in range(nshell):
        for j in range(i+1):
            matter_cl.append(interpcl(ell[:-1], block['matter_cl', 'bin_{0}_{1}'.format(i+1, j+1)][:-1], lmax=lmax_in, dipole=True, monopole=True))

    if config['seed'] != None:
        rng = np.random.default_rng(config['seed'])
    else:
        rng = None

    if 'salmo' in config['out_mode']: #for variable depth
        print('Initialising simulation ...')

        if 'nla' in config['ia']:
            generators = [
            glass.cosmosis.file_matter_cls(np.array(matter_cl), np.array(redshift_shells)),
            glass.matter.lognormal_matter(config['nside'], rng = rng),
            glass.lensing.convergence(cosmo),
            glass.lensing.ia_nla(cosmo, config['a_ia']),
            glass.lensing.shear()]
        else:
            generators = [
            glass.cosmosis.file_matter_cls(np.array(matter_cl), np.array(redshift_shells)),
            glass.matter.lognormal_matter(config['nside'], rng = rng),
            glass.lensing.convergence(cosmo),
            glass.lensing.shear()]
        
        delta = np.zeros((nshell, hp.nside2npix(config['nside'])), dtype=float)
        kappa = np.zeros_like(delta, dtype=float)
        gamma1 = np.zeros_like(delta, dtype=float)
        gamma2 = np.zeros_like(delta, dtype=float)

        #Iterate and map kappa and gamma to a HEALPix map
        i = 0
        for it in glass.sim.generate(generators):
            delta[i] = it['delta']
            kappa[i] = it['kappa']
            gamma1[i] = it['gamma1']
            gamma2[i] = it['gamma2']
            i += 1

        block['glass', 'map_folder'] = config['out_folder']
        block['glass', 'nside'] = config['nside']
        block['glass', 'prefix'] = config['prefix']
        block['glass', 'runTag'] = config['runTag']
        block['glass', 'counter']  =  config['counter']
        if config['seed'] is not None:
            block['glass', 'seed']  =  config['seed']

        for s in range(nshell):
            filename_denMap = '{0}/{3}_sample{1}/glass_denMap/{2}_denMap_{3}_sample{4}_f1z{5}.fits'.format(config['out_folder'], config['counter'], config['prefix'], config['runTag'], config['counter'], s+1)
            filename_lenMap = '{0}/{3}_sample{1}/glass_lenMap/{2}_lenMap_{3}_sample{4}_f2z{5}.fits'.format(config['out_folder'], config['counter'], config['prefix'], config['runTag'], config['counter'], s+1)
            
            if config['nside_out'] != config['nside']:
                print('Saving {0}...'.format(filename_denMap))
                save_fits(filename_denMap, hp.ud_grade(delta[s], config['nside_out']))
                print('Saving {0}...'.format(filename_lenMap))
                save_fits(filename_lenMap, [hp.ud_grade(kappa[s], config['nside_out']), hp.ud_grade(gamma1[s], config['nside_out']), hp.ud_grade(gamma2[s], config['nside_out'])])

            else:
                print('Saving {0}...'.format(filename_denMap))
                save_fits(filename_denMap, delta[s])
                #hp.write_map(filename_denMap, m = delta[s], dtype=np.float32, nest=False, fits_IDL = True, overwrite=True)
                print('Saving {0}...'.format(filename_lenMap))
                save_fits(filename_lenMap, [kappa[s], gamma1[s], gamma2[s]])
                #hp.write_map(filename_lenMap, m = [kappa[s], gamma1[s], gamma2[s]], dtype=[np.float32, np.float32, np.float32], nest=False, fits_IDL = True, overwrite=True)

    else:
        #No variable depth, so we can go directly to catalogues, maps or pseudo-Cls

        #Set up source redshift distributions
        nbin = block['nz_source', 'nbin']
        z = block['nz_source', 'z']
        dndz = []
        for i in range(nbin):
            dndz.append(block['nz_source', 'bin_{0}'.format(i+1)])
        n_arcmin2 = np.array([list(config['n_density'])]).T
        dndz *= n_arcmin2/np.trapz(dndz, z)[..., np.newaxis]

        print('Initialising simulation ...')
        if 'nla' in config['ia']:
            generators = [
            glass.cosmosis.file_matter_cls(np.array(matter_cl), np.array(redshift_shells)),
            glass.observations.vis_constant(config['mask'], config['mask_nside']),
            glass.matter.lognormal_matter(config['nside'], rng = rng),
            glass.lensing.convergence(cosmo),
            glass.lensing.ia_nla(cosmo, config['a_ia']),
            glass.lensing.shear(),
            glass.galaxies.gal_dist_uniform(z, dndz),
            glass.galaxies.gal_ellip_gaussian(float(config['sigma_e'])),
            glass.galaxies.gal_shear_interp(cosmo)]
        else:
            generators = [
            glass.cosmosis.file_matter_cls(np.array(matter_cl), np.array(redshift_shells)),
            glass.observations.vis_constant(config['mask'], config['mask_nside']),
            glass.matter.lognormal_matter(config['nside'], rng = rng),
            glass.lensing.convergence(cosmo),
            glass.lensing.shear(),
            glass.galaxies.gal_dist_uniform(z, dndz),
            glass.galaxies.gal_ellip_gaussian(float(config['sigma_e'])),
            glass.galaxies.gal_shear_interp(cosmo)]

        if 'pcl' in config['out_mode'] or 'map' in config['out_mode']:
            she = np.zeros((*dndz.shape[:-1], hp.nside2npix(config['nside'])), dtype=complex)
            num = np.zeros_like(she, dtype=int)

        if 'cat' in config['out_mode']:
            cat_lon, cat_lat, cat_she, cat_pop = np.empty(0), np.empty(0), np.empty(0, dtype=complex), np.empty(0)

        #Iterate and map the galaxy shears to a HEALPix map
        for it in glass.sim.generate(generators):
            gal_lon, gal_lat = it['gal_lon'], it['gal_lat']
            gal_she = it['gal_she']
            gal_pop = it['gal_pop']

            if 'cat' in config['out_mode']:
                cat_lon = np.append(cat_lon, gal_lon)
                cat_lat = np.append(cat_lat, gal_lat)
                cat_she = np.append(cat_she, gal_she)
                cat_pop = np.append(cat_pop, gal_pop)
            
            if 'pcl' in config['out_mode'] or 'map' in config['out_mode']:
                for i in np.ndindex(*dndz.shape[:-1]):
                    in_bin = (gal_pop == i)
                    map_shears(she[i], num[i], gal_lon[in_bin], gal_lat[in_bin], gal_she[in_bin])

        if 'pcl' in config['out_mode']:
            block['shear_pcl_novd', "is_auto"] = 'True'
            block['shear_pcl_novd', "sample_a"] = 'source'
            block['shear_pcl_novd', "sample_b"] = 'source'
            block['shear_pcl_novd', "nbin"] = nbin
            block['shear_pcl_novd', "nbin_a"] = nbin
            block['shear_pcl_novd', "nbin_b"] = nbin
            block['shear_pcl_novd', "ell"] = np.arange(0, config['lmax']+1)
            block['shear_pcl_novd', "n_density"] = config["n_density"]
            block['shear_pcl_novd', "sigma_e"] = config['sigma_e']

            for i in range(nbin):
                for j in range(i+1):
                    # get the angular power spectra from the galaxy shears
                    pcl = hp.anafast([num[i], she[i].real, she[i].imag], [num[j], she[j].real, she[j].imag], pol=True, lmax=config['lmax'])[1] #EE
                    #anafast output: (nbin(nbin+1)/2, 6, n_ell)
                    #anafast outputs contain TT, EE, BB, TE, EB, BB
                    block['shear_pcl_novd', 'bin_{0}_{1}'.format(i+1,j+1)] = pcl
        
        if 'map' in config['out_mode']:
            block['shear_map_novd', "sample"] = 'source'
            block['shear_map_novd', "nbin"] = nbin
            block['shear_map_novd', "sigma_e"] = config['sigma_e']

            block['galaxy_map_novd', "sample"] = 'lens'
            block['galaxy_map_novd', "nbin"] = nbin
            block['galaxy_map_novd', "n_density"] = config["n_density"]

            for i in range(nbin):
                block['shear_map_novd', 'shear_1_bin_{0}'.format(i+1)] = she[i].real
                block['shear_map_novd', 'shear_2_bin_{0}'.format(i+1)] = she[i].imag
                block['galaxy_map_novd', 'bin_{0}'.format(i+1)] = num[i]
        
        if 'cat' in config['out_mode']:
            block['catalogue_novd', 'lon'] = cat_lon
            block['catalogue_novd', 'lat'] = cat_lat
            block['catalogue_novd', 'shear_1'] = cat_she.real
            block['catalogue_novd', 'shear_2'] = cat_she.imag
            block['catalogue_novd', 'bin'] = cat_pop 

    return 0

def cleanup(config):
    pass