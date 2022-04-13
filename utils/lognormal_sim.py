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

def setup(options):
    config = {}
    
    config['nside'] = options.get_int(option_section, "nside")

    if not np.log2(float(config['nside'])).is_integer():
        raise Exception('nside must be a power of 2.')

    config['out']       = options.get_string(option_section, "out") #shear
    config['out_mode']  = options.get_string(option_section, "out_mode") #pcl, map

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

    config['n_density']  = options[option_section, 'n_density']
    config['sigma_e']    = options[option_section, 'sigma_e']

    if isinstance(config['sigma_e'], Iterable):
        raise Exception('Currently only support a single value of sigma_e for all bins.')

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
    #Setting up cosmology for the convergence map
    config["h0"]            = block[names.cosmological_parameters, "h0"]
    config["omega_m"]       = block[names.cosmological_parameters, "omega_m"]
    config["omega_k"]       = block[names.cosmological_parameters, "omega_k"]
    config["omega_lambda"]  = block[names.cosmological_parameters, "omega_lambda"]

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

    #Set up source redshift distributions
    nbin = block['nz_source', 'nbin']
    z = block['nz_source', 'z']
    dndz = []
    for i in range(nbin):
        dndz.append(block['nz_source', 'bin_{0}'.format(i+1)])
    n_arcmin2 = np.array([list(config['n_density'])]).T
    dndz *= n_arcmin2/np.trapz(dndz, z)[..., np.newaxis]
    

    print('Initialising simulation ...')
    generators = [
    glass.cosmosis.file_matter_cls(np.array(matter_cl), np.array(redshift_shells)),
    glass.observations.vis_constant(config['mask'], config['mask_nside']),
    glass.matter.lognormal_matter(config['nside']),
    glass.lensing.convergence(cosmo),
    glass.lensing.shear(),
    glass.galaxies.gal_dist_uniform(z, dndz),
    glass.galaxies.gal_ellip_gaussian(float(config['sigma_e'])),
    glass.galaxies.gal_shear_interp(cosmo)]

    she = np.zeros((*dndz.shape[:-1], hp.nside2npix(config['nside'])), dtype=complex)
    num = np.zeros_like(she, dtype=int)
    npix = she.shape[-1]

    # Getting all memory using os.popen()
    total_memory, used_memory, free_memory = map(
        int, os.popen('free -t -m').readlines()[-1].split()[1:])
    
    # Memory usage
    print("GLASS after initialisation: RAM memory %d used:" % (round((used_memory/total_memory) * 100, 2)))

    #Iterate and map the galaxy shears to a HEALPix map
    for it in glass.sim.generate(generators):
        gal_lon, gal_lat = it['gal_lon'], it['gal_lat']
        gal_she = it['gal_she']
        gal_pop = it['gal_pop']

        # Getting all memory using os.popen()
        total_memory, used_memory, free_memory = map(
            int, os.popen('free -t -m').readlines()[-1].split()[1:])
        
        # Memory usage
        print("GLASS shell: RAM memory %d used:" % (round((used_memory/total_memory) * 100, 2)))

        for i in np.ndindex(*dndz.shape[:-1]):
            in_bin = (gal_pop == i)
            map_shears(she[i], num[i], gal_lon[in_bin], gal_lat[in_bin], gal_she[in_bin])
            #gal_pix = hp.ang2pix(config['nside'], gal_lon[in_bin], gal_lat[in_bin], lonlat=True)
            #s = np.argsort(gal_pix)
            #pix, start, count = np.unique(gal_pix[s], return_index=True, return_counts=True)
            #she[i][pix] += list(map(np.sum, np.split(gal_she[in_bin][s], start[1:])W))
            #num[i][pix] += count
    
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

    return 0

def cleanup(config):
    pass