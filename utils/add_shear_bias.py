from cosmosis.datablock import option_section, names, BlockError
import numpy as np

def setup(options):
    config = {}
    config['section_name'] = options.get_string(option_section, "section_name")
    config['biases'] = options.get_string(option_section, "biases") #multiplicative, additive or psf

    if 'mult' not in config['biases'] and 'add' not in config['biases'] and 'psf' not in config['biases']:
        raise Exception("Biases must be set to one or some of the following: 'multiplicative', 'additive' and/or 'psf'")

    if 'mult' in config['biases']:
        config['mult_bias_mean'] = options[option_section,  'mult_bias_mean']
        config['mult_bias_std'] = options[option_section,  'mult_bias_std']

    if 'add' in config['biases']:
        config['add_bias_mean_e1'] = options[option_section,  'add_bias_mean_e1']
        config['add_bias_std_e1'] = options[option_section,  'add_bias_std_e1']
        config['add_bias_mean_e2'] = options[option_section,  'add_bias_mean_e2']
        config['add_bias_std_e2'] = options[option_section,  'add_bias_std_e2']

    if 'psf' in config['biases']:
        config['psf_bias_mean_e1'] = options[option_section,  'psf_bias_mean_e1']
        config['psf_bias_std_e1'] = options[option_section,  'psf_bias_std_e1']
        config['psf_bias_mean_e2'] = options[option_section,  'psf_bias_mean_e2']
        config['psf_bias_std_e2'] = options[option_section,  'psf_bias_std_e2']
        config['psf_ell_map_paths'] = str(options[option_section,  'psf_ell_map_paths'])

        if len(np.array(str(config['psf_ell_map_paths']).split(' '), dtype = str)) != 2:
            raise Exception('To include the effect of the PSF on the shear bias, two healpix maps must be given: one for e1 and another for e2')

    try:
        config['bias_seed'] = int(options[option_section, "bias_seed"])
        print('-- Setting a fixed seed for the shear bias with value: {0}'.format(config['bias_seed']))
    except:
        config['bias_seed'] = None

    return config

def execute(block, config):
    block[config['section_name'], 'biases'] = config['biases']

    if 'mult' in config['biases']:
        print('Including multiplicative shear bias...')

        block[config['section_name'], 'mult_bias_mean'] = config['mult_bias_mean']
        block[config['section_name'], 'mult_bias_std'] = config['mult_bias_std']
        m_biases = np.zeros(len(config['mult_bias_mean']))
        for i in range(len(config['mult_bias_mean'])):
            print('Source bin {0} has a multiplicative shear bias of m = {1} ± {2}'.format(i+1, config['mult_bias_mean'][i], config['mult_bias_std'][i]))
            np.random.seed(config['bias_seed'])
            m_biases[i] = float(np.random.normal(config['mult_bias_mean'][i], config['mult_bias_std'][i], 1))
            print(' Randomly sampled a multiplicative shear bias of m = {0}'.format(round(m_biases[i], 3)))
        block[config['section_name'], 'mult_bias_random_sample'] = m_biases

    if 'add' in config['biases']:
        print('Including additive shear bias...')

        block[config['section_name'], 'add_bias_mean_e1'] = config['add_bias_mean_e1']
        block[config['section_name'], 'add_bias_mean_e2'] = config['add_bias_mean_e2']
        block[config['section_name'], 'add_bias_std_e1'] = config['add_bias_std_e1']
        block[config['section_name'], 'add_bias_std_e2'] = config['add_bias_std_e2']
        c_biases_e1 = np.zeros(len(config['add_bias_mean_e1']))
        c_biases_e2 = np.zeros(len(config['add_bias_mean_e2']))
        for i in range(len(config['add_bias_mean_e1'])):
            print('Source bin {0} has an additive shear bias in e1 of c = {1} ± {2}'.format(i+1, config['add_bias_mean_e1'][i], config['add_bias_std_e1'][i]))
            np.random.seed(config['bias_seed'])
            c_biases_e1[i] = float(np.random.normal(config['add_bias_mean_e1'][i], config['add_bias_std_e1'][i], 1))
            print(' Randomly sampled an additive shear bias in e1 of c = {0}'.format(round(c_biases_e1[i], 6)))
            print('Source bin {0} has an additive shear bias in e2 of c = {1} ± {2}'.format(i+1, config['add_bias_mean_e2'][i], config['add_bias_std_e2'][i]))
            np.random.seed(config['bias_seed'])
            c_biases_e1[i] = float(np.random.normal(config['add_bias_mean_e2'][i], config['add_bias_std_e2'][i], 1))
            print(' Randomly sampled an additive shear bias in e2 of c = {0}'.format(round(c_biases_e1[i], 6)))
        block[config['section_name'], 'add_bias_e1_random_sample'] = c_biases_e1
        block[config['section_name'], 'add_bias_e2_random_sample'] = c_biases_e2

    if 'psf' in config['biases']:
        print('Including PSF shear bias...')
        
        block[config['section_name'], 'psf_bias_mean_e1'] = config['psf_bias_mean_e1']
        block[config['section_name'], 'psf_bias_mean_e2'] = config['psf_bias_mean_e2']
        block[config['section_name'], 'psf_bias_std_e1'] = config['psf_bias_std_e1']
        block[config['section_name'], 'psf_bias_std_e2'] = config['psf_bias_std_e2']
        block[config['section_name'], 'psf_ell_map_paths'] = config['psf_ell_map_paths']
        psf_biases_e1 = np.zeros(len(config['psf_bias_mean_e1']))
        psf_biases_e2 = np.zeros(len(config['psf_bias_mean_e2']))
        for i in range(len(config['psf_bias_mean_e1'])):
            print('Source bin {0} has a PSF shear bias factor in e1 of alpha = {1} ± {2}'.format(i+1, config['psf_bias_mean_e1'][i], config['psf_bias_std_e1'][i]))
            np.random.seed(config['bias_seed'])
            psf_biases_e1[i] = float(np.random.normal(config['psf_bias_mean_e1'][i], config['psf_bias_std_e1'][i], 1))
            print(' Randomly sampled a PSF shear bias factor in e1 of alpha = {0}'.format(round(psf_biases_e1[i], 3)))
            print('Source bin {0} has a PSF shear bias factor in e2 of alpha = {1} ± {2}'.format(i+1, config['psf_bias_mean_e2'][i], config['psf_bias_std_e2'][i]))
            np.random.seed(config['bias_seed'])
            psf_biases_e2[i] = float(np.random.normal(config['psf_bias_mean_e2'][i], config['psf_bias_std_e2'][i], 1))
            print(' Randomly sampled a PSF shear bias factor in e2 of alpha = {0}'.format(round(psf_biases_e2[i], 3)))
        block[config['section_name'], 'psf_bias_e1_random_sample'] = psf_biases_e1
        block[config['section_name'], 'psf_bias_e2_random_sample'] = psf_biases_e2
        
    return 0

def cleanup(config):
    pass