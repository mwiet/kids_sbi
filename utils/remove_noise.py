from cosmosis.datablock import option_section, names, BlockError
import numpy as np
import warnings

def setup(options):
    config = {}
    config['where'] = options.get_string(option_section, "where") #only shear

    config['in_name'] = options.get_string(option_section, 'in_name')
    config['min_ell_noise'] = options.get_int(option_section, 'min_ell_noise')
    config['out_name'] = options.get_string(option_section, 'out_name')
    return config

def execute(block, config):
    if config['where'] == 'shear':
        nbin = block[config['in_name'], 'nbin']
        
        ell =  block[config['in_name'], 'ell']
        ell_bb =  block[config['in_name'] + '_bb', 'ell']
        print("Subtracting shot and shape noise...")
        counter = 0
        for i in range(nbin):
            for j in range(i+1):
                shear_cl = block[config['in_name'], 'bin_{0}_{1}'.format(i+1, j+1)]
                shear_cl_bb = block[config['in_name']+'_bb', 'bin_{0}_{1}'.format(i+1, j+1)]

                if i == j:
                    noise = block[config['in_name'] + '_noise', 'bin_{0}_{1}'.format(i+1, j+1)]
                    noise_bb = block[config['in_name'] + '_noise_bb', 'bin_{0}_{1}'.format(i+1, j+1)]
                    shear_cl -= np.average(noise[config['min_ell_noise']:], weights = (2*ell[config['min_ell_noise']:]) + 1) #Subtracting the weighted average of the shape noise bias
                    shear_cl_bb -= np.average(noise_bb[config['min_ell_noise']:], weights = (2*ell_bb[config['min_ell_noise']:]) + 1)

                block[config['out_name'], 'bin_{0}_{1}'.format(i+1,j+1)] = shear_cl
                block[config['out_name'] + '_bb', 'bin_{0}_{1}'.format(i+1,j+1)] = shear_cl_bb
                counter += 1
                
        block[config['out_name'], "is_auto"] = 'True'
        block[config['out_name'], "sample_a"] = 'source'
        block[config['out_name'], "sample_b"] = 'source'
        block[config['out_name'], "nbin"] = nbin
        block[config['out_name'], "nbin_a"] = nbin
        block[config['out_name'], "nbin_b"] = nbin
        block[config['out_name'], "ell"] = ell

        block[config['out_name'] + '_bb', "is_auto"] = 'True'
        block[config['out_name'] + '_bb', "sample_a"] = 'source'
        block[config['out_name'] + '_bb', "sample_b"] = 'source'
        block[config['out_name'] + '_bb', "nbin"] = nbin
        block[config['out_name'] + '_bb', "nbin_a"] = nbin
        block[config['out_name'] + '_bb', "nbin_b"] = nbin
        block[config['out_name'] + '_bb', "ell"] = ell_bb
    else:
        raise Exception('Unsupported output.')
    
    return 0

def cleanup(config):
    pass