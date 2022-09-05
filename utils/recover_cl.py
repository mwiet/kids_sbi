from cosmosis.datablock import option_section, names, BlockError
import numpy as np

def setup(options):
    config = {}
    config['mm'] = options.get_string(option_section, "mm")

    config['in_name'] = options.get_string(option_section, 'in_name')
    config['out_name'] = options.get_string(option_section, 'out_name')

    if config['mm'][-4:] != '.npz':
        raise Exception('Only numpy .npz files are supported for the mixing matrix.')
    config['out'] = options.get_string(option_section, "out") #only shear
    return config

def execute(block, config):
    if config['out'] == 'shear':
        print('Reading in shear mixing matrix: {0}'.format(config['mm']))
        mm = np.load(config['mm'])['mm_shear']
        nbin = block[config['in_name'], 'nbin']

        block[config['out_name'], "is_auto"] = 'True'
        block[config['out_name'], "sample_a"] = 'source'
        block[config['out_name'], "sample_b"] = 'source'
        block[config['out_name'], "nbin"] = nbin
        block[config['out_name'], "nbin_a"] = nbin
        block[config['out_name'], "nbin_b"] = nbin
        block[config['out_name'], "ell"] = block[config['in_name'], 'ell']
        try:
            block[config['out_name'], "n_density"] = block[config['in_name'], "n_density"]
            block[config['out_name'], "sigma_e"] = block[config['in_name'], "sigma_e"]
        except:
            print('Only variable depth bins given.')
            pass

        try:
            block[config['out_name'], "a_n_density"] = block[config['in_name'], 'a_n_density']
            block[config['out_name'], "b_n_density"] = block[config['in_name'], 'b_n_density']
            block[config['out_name'], "a_sigma_e"] = block[config['in_name'], "a_sigma_e"]
            block[config['out_name'], "b_sigma_e"] = block[config['in_name'], "b_sigma_e"]
        except:
            print('No variable depth bins given.')
            pass

        counter = 0
        for i in range(nbin):
            for j in range(i+1):
                shear_pcl = block[config['in_name'], 'bin_{0}_{1}'.format(i+1, j+1)]
                shear_rec_cl = np.matmul(mm, shear_pcl)
                block[config['out_name'], 'bin_{0}_{1}'.format(i+1,j+1)] = shear_rec_cl
                counter += 1
        
    else:
        raise Exception('Unsupported output.')
    
    return 0

def cleanup(config):
    pass