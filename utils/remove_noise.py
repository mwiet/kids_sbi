from cosmosis.datablock import option_section, names, BlockError
import numpy as np
import warnings

def setup(options):
    config = {}
    config['where'] = options.get_string(option_section, "where") #only shear
    config['fsky'] = options[option_section, 'fsky']

    config['in_name'] = options.get_string(option_section, 'in_name')
    config['out_name'] = options.get_string(option_section, 'out_name')
    return config

def execute(block, config):
    if config['where'] == 'shear':
        nbin = block[config['in_name'], 'nbin']
        arcmin2_in_sphere =  60**6//100/np.pi
        try:
            n_density = block[config['in_name'], "n_density"]
            sigma_e = block[config['in_name'], "sigma_e"]
        except:
            n_density = np.array([])
            sigma_e = np.array([])

        try:
            a_n_density = block[config['in_name'], "a_n_density"]
            b_n_density = block[config['in_name'], "a_n_density"]
            a_sigma_e = block[config['in_name'], "a_sigma_e"]
            b_sigma_e = block[config['in_name'], "b_sigma_e"]
            bin_depth = block['salmo', "bin_depth"]
            vd = True
        except:
            vd = False
            pass

        if vd:
            bin_mean = bin_depth[:-1] + 0.5*(bin_depth[1:] - bin_depth[:-1])
            n = np.mean(a_n_density*bin_mean[:, None] + b_n_density, axis = 0)
            s = np.mean(a_sigma_e*bin_mean[:, None] + b_sigma_e, axis = 0)
            n_density = np.concatenate((n_density, n))
            sigma_e = np.concatenate((sigma_e, s))
        
        n_density = n_density*arcmin2_in_sphere/(4*np.pi)
        ell =  block[config['in_name'], 'ell']
        print("Calculating shot and shape noise...")
        counter = 0
        for i in range(nbin):
            for j in range(i+1):
                shear_cl = block[config['in_name'], 'bin_{0}_{1}'.format(i+1, j+1)]
                if i == j:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=RuntimeWarning)
                        noise = sigma_e[i]**2/2*n_density[i]
                else:
                    noise = 0
                shear_cl -= noise
                block[config['out_name'], 'bin_{0}_{1}'.format(i+1,j+1)] = shear_cl
                block['{0}_noise'.format(config['out_name']), 'bin_{0}_{1}'.format(i+1,j+1)] = noise
                counter += 1
                
        block[config['out_name'], "is_auto"] = 'True'
        block[config['out_name'], "sample_a"] = 'source'
        block[config['out_name'], "sample_b"] = 'source'
        block[config['out_name'], "nbin"] = nbin
        block[config['out_name'], "nbin_a"] = nbin
        block[config['out_name'], "nbin_b"] = nbin
        block[config['out_name'], "ell"] = ell

        block['{0}_noise'.format(config['out_name']), "is_auto"] = 'True'
        block['{0}_noise'.format(config['out_name']), "sample_a"] = 'source'
        block['{0}_noise'.format(config['out_name']), "sample_b"] = 'source'
        block['{0}_noise'.format(config['out_name']), "nbin"] = nbin
        block['{0}_noise'.format(config['out_name']), "nbin_a"] = nbin
        block['{0}_noise'.format(config['out_name']), "nbin_b"] = nbin
        
    else:
        raise Exception('Unsupported output.')
    
    return 0

def cleanup(config):
    pass