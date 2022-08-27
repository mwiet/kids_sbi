from cosmosis.datablock import option_section, names, BlockError
import numpy as np
import warnings

def setup(options):
    config = {}
    config['where'] = options.get_string(option_section, "where") #only shear
    config['fsky'] = options[option_section, 'fsky']
    return config

def execute(block, config):
    if config['where'] == 'shear':
        nbin = block['shear_rec_cl', 'nbin']
        arcmin2_in_sphere =  60**6//100/np.pi
        try:
            n_density = block['shear_rec_cl', "n_density"]
            sigma_e = block['shear_rec_cl', "sigma_e"]
        except:
            n_density = np.array([])
            sigma_e = np.array([])

        try:
            a_n_density = block['shear_rec_cl', "a_n_density"]
            b_n_density = block['shear_rec_cl', "a_n_density"]
            a_sigma_e = block['shear_rec_cl', "a_sigma_e"]
            b_sigma_e = block['shear_rec_cl', "b_sigma_e"]
            bin_depth = block['salmo', "bin_depth"]
            vd = True
        except:
            pass

        if vd:
            bin_mean = bin_depth[:-1] + 0.5*(bin_depth[1:] - bin_depth[:-1])
            n = np.mean(a_n_density*bin_mean[:, None] + b_n_density, axis = 0)
            s = np.mean(a_sigma_e*bin_mean[:, None] + b_sigma_e, axis = 0)
        
        n_density = np.concatenate((n_density, n))*arcmin2_in_sphere/(4*np.pi)
        sigma_e = np.concatenate((sigma_e, s))
        ell =  block['shear_rec_cl', 'ell']
        area = config['fsky']*4*np.pi
        print("Calculating shot and shape noise...")
        counter = 0
        for i in range(nbin):
            for j in range(i+1):
                shear_cl = block['shear_rec_cl', 'bin_{0}_{1}'.format(i+1, j+1)]
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    noise = np.sqrt(np.pi/(area*ell*n_density[i]*n_density[j]))*sigma_e[i]*sigma_e[j]
                noise[0] = 0
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