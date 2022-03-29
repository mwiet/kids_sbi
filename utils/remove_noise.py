from cosmosis.datablock import option_section, names, BlockError
import numpy as np

def setup(options):
    config = {}
    config['where'] = options.get_string(option_section, "where") #only shear
    config['fsky'] = options[option_section, 'fsky']
    return config

def execute(block, config):
    if config['where'] == 'shear':
        nbin = block['shear_rec_cl', 'nbin']
        arcmin2_in_sphere =  60**6//100/np.pi
        n_density = np.array([list(block['shear_rec_cl', "n_density"])]).T*arcmin2_in_sphere/(4*np.pi)
        sigma_e = float(block['shear_rec_cl', "sigma_e"])
        ell =  block['shear_rec_cl', 'ell']
        area = config['fsky']*4*np.pi

        counter = 0
        for i in range(nbin):
            for j in range(i+1):
                shear_cl = block['shear_rec_cl', 'bin_{0}_{1}'.format(i+1, j+1)]
                noise = np.sqrt(np.pi/(area*ell*n_density[i]*n_density[j]))*sigma_e**2
                shear_cl -= noise
                block['shear_cl', 'bin_{0}_{1}'.format(i+1,j+1)] = shear_cl
                block['shear_cl_noise', 'bin_{0}_{1}'.format(i+1,j+1)] = noise
                counter += 1
                
        block['shear_cl', "is_auto"] = 'True'
        block['shear_cl', "sample_a"] = 'source'
        block['shear_cl', "sample_b"] = 'source'
        block['shear_cl', "nbin"] = nbin
        block['shear_cl', "nbin_a"] = nbin
        block['shear_cl', "nbin_b"] = nbin
        block['shear_cl', "ell"] = ell

        block['shear_cl_noise', "is_auto"] = 'True'
        block['shear_cl_noise', "sample_a"] = 'source'
        block['shear_cl_noise', "sample_b"] = 'source'
        block['shear_cl_noise', "nbin"] = nbin
        block['shear_cl_noise', "nbin_a"] = nbin
        block['shear_cl_noise', "nbin_b"] = nbin
        block['shear_cl_noise', "ell"] = ell
        
    else:
        raise Exception('Unsupported output.')
    
    return 0

def cleanup(config):
    pass