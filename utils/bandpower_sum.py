import numpy as np
from cosmosis.datablock import option_section, names
from cosmosis.datablock.cosmosis_py import errors
from scipy.interpolate import interp1d

def setup(options):
    config = {}
    config['num_bands'] = options.get_int(option_section, "num_bands", default=8)
    config['min_ell'] = options.get_int(option_section, "min_ell", default=1)
    config['max_ell'] = options.get_int(option_section, "max_ell", default=1500)
    config['in_name'] = options.get_string(option_section, 'in_name')
    config['out_name'] = options.get_string(option_section, 'out_name')
    return config

def average_bands(l, cls, n):
    '''
    n is the number of bands
    '''
    band_cutoffs = np.logspace(np.log(min(l)), np.log(max(l)+1), n+1, base = np.e)

    bandpowers, centre_ell = np.zeros(n), np.zeros(n)
    for i in range(n):
        temp_l = l[np.logical_and(l >= band_cutoffs[i], l < band_cutoffs[i+1])]
        temp_cls = cls[np.logical_and(l >= band_cutoffs[i], l < band_cutoffs[i+1])]
        temp_cls *= temp_l * (temp_l + 1)
        bandpowers[i] = np.sum(temp_cls)/(2*np.pi*( band_cutoffs[i+1] - band_cutoffs[i])) #Use Brown et al. 2005 multipole binning scheme
        centre_ell[i] = band_cutoffs[i] + 0.5*(band_cutoffs[i+1] - band_cutoffs[i])
    return bandpowers, centre_ell, band_cutoffs

def execute(block, config):
    input_section = config['in_name']

    # Read minimum and maximum ell value
    l_min = config['min_ell']
    l_max = config['max_ell']

    # Read the input ell.
    ell = block[input_section, "ell"]
    ell_new = ell[(l_max > ell) & (ell >= l_min)]

    if np.all((ell_new[1:] - ell_new[:-1]) == 1.0):
        print('Given Cls are calculated at interger values of ell already.')

    # Read the number of bands
    num_bands = config['num_bands']

    # Loop through bin pairs and see if C_ell exists for all of them
    n_bins = int(block[input_section, 'nbin'])
    theory_bandpowers_stacked, bandpower_ell = np.array([]), np.array([])

    for i in range(n_bins):
        for j in range(i+1):
            theory_cl = block[input_section, 'bin_{0}_{1}'.format(i+1,j+1)]
            theory_cl = theory_cl[(l_max > ell) & (ell >= l_min)]
            band_average, centre_ell, band_cutoffs = average_bands(ell_new, theory_cl, num_bands)
            theory_bandpowers_stacked = np.append(theory_bandpowers_stacked, band_average)
            bandpower_ell = np.append(bandpower_ell, centre_ell)
            block[input_section, 'bandpowers_bin_{0}_{1}'.format(i+1,j+1)] = band_average

    # block[config['out_name'], "bandpowers"] = theory_bandpowers_stacked.reshape(n_bins,-1).T.flatten()
    block[config['out_name'], "bandpowers"] = theory_bandpowers_stacked
    block[config['out_name'], "bandpower_ell"] = bandpower_ell
    block[config['out_name'], "centre_ell"] = centre_ell
    block[config['out_name'], "edge_ell"] = band_cutoffs
    print("Writing bandpowers to its own block")

    return 0

# def clean(config):
#     likelihood_type, n_dim, parameter_section, mu_section, likelihood_name, counter = config
#     print(f"Ran likelihood {counter[0]} times.", flush=True)
