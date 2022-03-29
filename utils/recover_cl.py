from cosmosis.datablock import option_section, names, BlockError
import numpy as np

def setup(options):
    config = {}
    config['mm'] = options.get_string(option_section, "mm")

    if config['mm'][-4:] != '.npz':
        raise Exception('Only .npz files are supported for the mixing matrix.')
    config['out'] = options.get_string(option_section, "out") #only shear
    return config

def execute(block, config):
    if config['out'] == 'shear':
        mm = np.load(config['mm'])['mm_shear']
        nbin = block['shear_pcl', 'nbin']

        block['shear_rec_cl', "is_auto"] = 'True'
        block['shear_rec_cl', "sample_a"] = 'source'
        block['shear_rec_cl', "sample_b"] = 'source'
        block['shear_rec_cl', "nbin"] = nbin
        block['shear_rec_cl', "nbin_a"] = nbin
        block['shear_rec_cl', "nbin_b"] = nbin
        block['shear_rec_cl', "ell"] = block['shear_pcl', 'ell']
        block['shear_rec_cl', "n_density"] = block['shear_pcl', "n_density"]
        block['shear_rec_cl', "sigma_e"] = block['shear_pcl', "sigma_e"]

        counter = 0
        for i in range(nbin):
            for j in range(i+1):
                shear_pcl = block['shear_pcl', 'bin_{0}_{1}'.format(i+1, j+1)]
                shear_rec_cl = np.matmul(mm, shear_pcl)
                block['shear_rec_cl', 'bin_{0}_{1}'.format(i+1,j+1)] = shear_rec_cl
                counter += 1
        
    else:
        raise Exception('Unsupported output.')
    
    return 0

def cleanup(config):
    pass