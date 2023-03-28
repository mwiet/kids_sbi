from cosmosis.datablock import option_section, names, BlockError
import numpy as np
import healpy as hp

def setup(options):
    config = {}
    config['mm_mask'] = options.get_string(option_section, "mm_mask")

    config['mm_sampling_prefix'] = options.get_string(option_section, "mm_sampling_prefix")

    config['nside'] = options.get_int(option_section, "nside")

    config['in_name'] = options.get_string(option_section, 'in_name')
    config['out_name'] = options.get_string(option_section, 'out_name')

    if config['mm'][-4:] != '.npz':
        raise Exception('Only numpy .npz files are supported for the mixing matrix.')
    config['out'] = options.get_string(option_section, "out") #only shear
    return config

def execute(block, config):
    if config['out'] == 'shear':
        print('Reading in shear mixing matrix: {0}'.format(config['mm']))
        mm = np.load(config['mm'])['arr_0']
        nbin = block[config['in_name'], 'nbin']

        ell = block[config['in_name'], 'ell']
        l_min = int(np.min(ell))
        l_max = int(np.max(ell))

        ell_new = ell[(l_max > ell) & (ell >= l_min)]

        if np.all((ell_new[1:] - ell_new[:-1]) == 1.0):
            print('Given Cls are calculated at interger values of ell already.')
        else:
            raise Exception('Given Cls are NOT calculated at interger values of ell.')
            
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
                mm_sampling = np.load(config['mm_sampling_prefix'] + '_{0}_{1}.npz'.format(i+1, j+1))['arr_0']
                block[config['out_name'], 'bin_{0}_{1}'.format(i+1,j+1)] = pixel_window[l_min:l_max+1]**2 * mm[l_min:l_max+1, l_min:l_max+1].dot(mm_sampling[l_min:l_max+1, l_min:l_max+1].dot(shear_cl[l_min:l_max+1]))
                counter += 1
        
    else:
        raise Exception('Unsupported output.')
    
    return 0

def cleanup(config):
    pass