from cosmosis.datablock import option_section, names, BlockError
import numpy as np
import healpy as hp
from scipy.interpolate import interp1d

def setup(options):
    config = {}
    config['mm_mask'] = options.get_string(option_section, "mm_mask")

    config['mm_sampling_prefix'] = options.get_string(option_section, "mm_sampling_prefix")

    config['nside'] = options.get_int(option_section, "nside")

    config['l_max'] = options.get_int(option_section, "l_max")

    config['in_name'] = options.get_string(option_section, 'in_name')
    config['out_name'] = options.get_string(option_section, 'out_name')

    if config['mm_mask'][-4:] != '.npz':
        raise Exception('Only numpy .npz files are supported for the mixing matrix.')
    config['out'] = options.get_string(option_section, "out") #only shear
    return config

def execute(block, config):
    if config['out'] == 'shear':
        nbin = block[config['in_name'], 'nbin']

        ell = block[config['in_name'], 'ell']
        l_min = int(round(np.min(ell), 0))
        l_max = config['l_max']

        print('Reading in shear mixing matrix from the mask: {0}'.format(config['mm_mask']))
        mm = np.load(config['mm_mask'])['arr_0'][l_min:l_max+1, l_min:l_max+1]
           
        block[config['out_name'], "is_auto"] = 'True'
        block[config['out_name'], "sample_a"] = 'source'
        block[config['out_name'], "sample_b"] = 'source'
        block[config['out_name'], "nbin"] = nbin
        block[config['out_name'], "nbin_a"] = nbin
        block[config['out_name'], "nbin_b"] = nbin
        block[config['out_name'], "ell"] = ell
        
        _, pixel_window = hp.pixwin(config['nside'], lmax=l_max, pol=True)

        counter = 0
        for i in range(nbin):
            for j in range(i+1):
                shear_cl = block[config['in_name'], 'bin_{0}_{1}'.format(i+1, j+1)]
                cl_interp = interp1d(ell, shear_cl, fill_value = "extrapolate") #Will only extrapolate to the nearest integer ell
                cl = cl_interp(np.arange(l_min, l_max+1))
                print('     Reading in shear mixing matrix to account for sampling: {0}'.format(config['mm_sampling_prefix'] + '_{0}_{1}.npz'.format(i+1, j+1)))
                mm_sampling = np.load(config['mm_sampling_prefix'] + '_{0}_{1}.npz'.format(i+1, j+1))['arr_0'][l_min:l_max+1, l_min:l_max+1]
                block[config['out_name'], 'bin_{0}_{1}'.format(i+1,j+1)] = pixel_window[l_min:l_max+1]**2 * mm.dot(mm_sampling.dot(cl))
                del mm_sampling
                counter += 1
        
    else:
        raise Exception('Unsupported output.')
    
    return 0

def cleanup(config):
    pass