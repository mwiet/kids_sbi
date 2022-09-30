from cosmosis.datablock import option_section, names, BlockError
import numpy as np

def setup(options):
    config = {}

    config['path'] = options.get_string(option_section, 'path')
    config['runtag'] = options.get_string(option_section, 'runtag')
    config['name'] = options.get_string(option_section, 'name')

    return config

def execute(block, config):

    with open('{0}/{1}/{2}/values.txt'.format(config['path'], config['runtag'], config['name'])) as f:
        lines = f.readlines()

    values = np.array([l.replace("'", "", 2)[:-1].split(' = ') for l in lines]).T

    nbin = int(values[1][np.where(values[0] == 'nbin')[0]].astype(int))
    print('Assuming that the input Cls are autocorrelations...')

    for i in range(nbin):
        for j in range(i+1):
            block[config['name'], "bin_{0}_{1}".format(i+1, j+1)] = np.loadtxt('{0}/{1}/{2}/bin_{3}_{4}.txt'.format(config['path'], config['runtag'], config['name'], i+1, j+1))

    block[config['name'], "is_auto"] = str(values[1][np.where(values[0] == 'is_auto')[0]].astype(str))
    block[config['name'], "sample_a"] = str(values[1][np.where(values[0] == 'sample_a')[0]].astype(str))
    block[config['name'], "sample_b"] = str(values[1][np.where(values[0] == 'sample_b')[0]].astype(str))
    block[config['name'], "nbin"] = nbin
    block[config['name'], "nbin_a"] = int(values[1][np.where(values[0] == 'nbin_a')[0]].astype(int))
    block[config['name'], "nbin_b"] = int(values[1][np.where(values[0] == 'nbin_b')[0]].astype(int))
    block[config['name'], "sep_name"] = str(values[1][np.where(values[0] == 'sep_name')[0]].astype(str))
    block[config['name'], "save_name"] = str(values[1][np.where(values[0] == 'save_name')[0]].astype(str))
    block[config['name'], "ell"] = np.loadtxt('{0}/{1}/{2}/ell.txt'.format(config['path'], config['runtag'], config['name']))

    return 0

def cleanup(config):
    pass