from os import EX_SOFTWARE
from numba import njit
from cosmosis.datablock import option_section, names, BlockError
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

def setup(options):
    config = {}
    
    config['nside'] = options.get_int(option_section, "nside")

    if not np.log2(float(config['nside'])).is_integer():
        raise Exception('nside must be a power of 2.')

    config['out']       = options.get_string(option_section, "out") #shear
    config['out_mode']  = options.get_string(option_section, "out_mode") #pcl, map, catalogue, salmo

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
        
    else:
        config['n_density']  = options[option_section, 'n_density']
        config['sigma_e']    = options[option_section, 'sigma_e']

        if isinstance(config['sigma_e'], Iterable):
            raise Exception('Currently only support a single value of sigma_e for all bins.')

    config['ia'] = options.get_string(option_section, "ia")

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

def execute(block, config):
    
    #Create file paths for fits files
    counter = 0
    new_path = False
    while new_path is False:
            path = '{0}/{1}_sample{2}'.format(config['out_folder'], config['runTag'], counter)
            print(path)
            if os.path.exists(path) == True:
                new_path = False
                counter += 1
            else:
                os.mkdir(path)
                os.mkdir('{0}/glass_denMap'.format(path))
                os.mkdir('{0}/glass_lenMap'.format(path))
                config['counter'] = counter
                new_path = True

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
    lmax = int(config['nside'])

    matter_cl = []
    for i in range(nshell):
        for j in range(i+1):
            matter_cl.append(interpcl(ell[:-1], block['matter_cl', 'bin_{0}_{1}'.format(i+1, j+1)][:-1], lmax=lmax_in, dipole=True, monopole=True))

    if 'salmo' in config['out_mode']: #for variable depth
        print('Initialising simulation ...')
        if 'nla' in config['ia']:
            generators = [
            glass.cosmosis.file_matter_cls(np.array(matter_cl), np.array(redshift_shells)),
            glass.matter.lognormal_matter(config['nside']),
            glass.lensing.convergence(cosmo),
            glass.lensing.ia_nla(cosmo, config['a_ia']),
            glass.lensing.shear()]
        else:
            generators = [
            glass.cosmosis.file_matter_cls(np.array(matter_cl), np.array(redshift_shells)),
            glass.matter.lognormal_matter(config['nside']),
            glass.lensing.convergence(cosmo),
            glass.lensing.shear()]
        
        delta = np.zeros((nshell, hp.nside2npix(config['nside'])), dtype=float)
        kappa = np.zeros_like(delta, dtype=float)
        gamma1 = np.zeros_like(delta, dtype=float)
        gamma2 = np.zeros_like(delta, dtype=float)

        #Iterate and map kappa and gamma to a HEALPix map
        for it in glass.sim.generate(generators):
            delta[i] = it['delta']
            kappa[i] = it['kappa']
            gamma1[i] = it['gamma1']
            gamma2[i] = it['gamma2']

        block['salmo', 'map_folder'] = config['out_folder']
        block['salmo', 'nside'] = config['nside']
        block['salmo', 'prefix'] = config['prefix']
        block['salmo', 'runTag'] = config['runTag']
        block['salmo', 'counter']  =  config['counter']

        for s in range(nshell):
            filename_denMap = '{0}/{3}_sample{1}/glass_denMap/{2}_denMap_{3}_f1z{4}.fits'.format(config['out_folder'], config['counter'], config['prefix'], config['runTag'], s+1)
            filename_lenMap = '{0}/{3}_sample{1}/glass_lenMap/{2}_lenMap_{3}_f2z{4}.fits'.format(config['out_folder'], config['counter'], config['prefix'], config['runTag'], s+1)
            
            print('Saving {0}...'.format(filename_denMap))
            hp.write_map(filename_denMap, m = delta[s], dtype=np.float64, nest=False, fits_IDL = True, overwrite=True)
            print('Saving {0}...'.format(filename_lenMap))
            hp.write_map(filename_lenMap, m = [kappa[s], gamma1[s], gamma2[s]], dtype=[np.float64, np.float64, np.float64], nest=False, fits_IDL = True, overwrite=True)

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
            glass.matter.lognormal_matter(config['nside']),
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
            glass.matter.lognormal_matter(config['nside']),
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
            block['shear_pcl', "is_auto"] = 'True'
            block['shear_pcl', "sample_a"] = 'source'
            block['shear_pcl', "sample_b"] = 'source'
            block['shear_pcl', "nbin"] = nbin
            block['shear_pcl', "nbin_a"] = nbin
            block['shear_pcl', "nbin_b"] = nbin
            block['shear_pcl', "ell"] = np.arange(0, lmax+1)
            block['shear_pcl', "n_density"] = config["n_density"]
            block['shear_pcl', "sigma_e"] = config['sigma_e']

            for i in range(nbin):
                for j in range(i+1):
                    # get the angular power spectra from the galaxy shears
                    pcl = hp.anafast([num[i], she[i].real, she[i].imag], [num[j], she[j].real, she[j].imag], pol=True, lmax=lmax)[1] #EE
                    #anafast output: (nbin(nbin+1)/2, 6, n_ell)
                    #anafast outputs contain TT, EE, BB, TE, EB, BB
                    block['shear_pcl', 'bin_{0}_{1}'.format(i+1,j+1)] = pcl
        
        if 'map' in config['out_mode']:
            block['shear_map', "sample"] = 'source'
            block['shear_map', "nbin"] = nbin
            block['shear_map', "sigma_e"] = config['sigma_e']

            block['galaxy_map', "sample"] = 'lens'
            block['galaxy_map', "nbin"] = nbin
            block['galaxy_map', "n_density"] = config["n_density"]

            for i in range(nbin):
                block['shear_map', 'shear_1_bin_{0}'.format(i+1)] = she[i].real
                block['shear_map', 'shear_2_bin_{0}'.format(i+1)] = she[i].imag
                block['galaxy_map', 'bin_{0}'.format(i+1)] = num[i]
        
        if 'cat' in config['out_mode']:
            block['catalogue', 'lon'] = cat_lon
            block['catalogue', 'lat'] = cat_lat
            block['catalogue', 'shear_1'] = cat_she.real
            block['catalogue', 'shear_2'] = cat_she.imag
            block['catalogue', 'bin'] = cat_pop 

    return 0

def cleanup(config):
    pass
