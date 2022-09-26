from cosmosis.datablock import option_section, names, BlockError
import numpy as np
import warnings

def setup(options):
    config = {}
    config['where'] = options.get_string(option_section, "where") #only shear

    config['vd'] =  options.get_bool(option_section, "vd")

    if config['vd']:
        config['n_density'] = options[option_section, "n_density"]
        config['sigma_eps'] = options[option_section, "sigma_eps"] 

    config['in_name'] = options.get_string(option_section, 'in_name')
    config['out_name'] = options.get_string(option_section, 'out_name')
    return config

def execute(block, config):
    if config['where'] == 'shear':
        nbin = block[config['in_name'], 'nbin']
        arcmin2_in_sphere =  60**6//100/np.pi
        try:
            nbSourceFields = block[config['in_name'], "nbSourceFields"]
            n_density = block[config['in_name'], "n_density"]
            sigma_e = block[config['in_name'], "sigma_e"]
            if len(sigma_e) != nbSourceFields:
                print('WARNING: Sigma_eps and nbSourceFields incompatible!')

            nbLensFields = block[config['in_name'], "nbLensFields"]
            nofz_paths = block[config['in_name'], "nOfZPath"]
            nofz_paths = np.array(nofz_paths.split(' '), dtype = str)[nbLensFields:]
            #Make sure each tomographic bin is only included once
            _, index = np.unique(nofz_paths, return_index=True)
            if len(index) != 0:
                n_density = n_density[np.sort(index)]
                sigma_e = sigma_e[np.sort(index)]
            print('Found uniform shape and shot noise values...')
        except:
            nbSourceFields = 0
            n_density = np.array([])
            sigma_e = np.array([])
            print('No uniform shape and shot noise values found.')

        if config['vd']:
            n_density = np.array(config['n_density'])
            sigma_e = np.array(config['sigma_eps'])
        
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
                        noise = (sigma_e[i]**2)/(2*n_density[i])
                    print('Removing ({0}^2)/(2*{1} arcmin^-2) of angular power in {2}-{2}'.format(round(sigma_e[i], 2), round(n_density[i]*4*np.pi/arcmin2_in_sphere, 2), i+1))
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