import numpy as np
from cosmosis.datablock import option_section, names
from cosmosis.datablock.cosmosis_py import errors

def setup(options):
    config = {}
    config['num_bands'] = options.get_int(option_section, "num_bands", default=8)
    config['min_ell'] = options.get_double(option_section, "min_ell", default=1.0)
    config['in_name'] = options.get_string(option_section, 'in_name')
    config['out_name'] = options.get_string(option_section, 'out_name')
    return config

def bandpower_integral(x, y):
    '''Calculates the integral for bandpowers based on the composite trapezoidal rule relying on Riemann Sums.
    :param array x: array of x values
    :param array y: array of y values
    :return float: the integral of the bandpowers from the lowest l to max l
    '''
    num_trapz = len(x) - 1
    widths = np.array([x[i+1] - x[i] for i in range(num_trapz)])
    trapz_heights = np.array([y[i] + y[i+1] for i in range(num_trapz)])
    trapz_areas = 0.5 * widths * trapz_heights
    return np.sum(trapz_areas)/(x[-1] - x[0])

def average_bands(l, cls, n):
    '''
    n is the number of bands
    '''
    band_cutoffs = np.logspace(np.log(min(l)), np.log(max(l)+1), n+1, base = np.e)
    
    binned_l = []
    binned_cls = []
    for i in range(n):
        temp_l = l[np.logical_and(l >= band_cutoffs[i], l < band_cutoffs[i+1])]
        temp_cls = cls[np.logical_and(l >= band_cutoffs[i], l < band_cutoffs[i+1])]
        binned_l.append(np.array(temp_l))
        binned_cls.append(np.array(temp_cls))
    
    bandpowers = np.zeros(n)
    for i in range(n):
        bandpowers[i] = bandpower_integral(binned_l[i], binned_cls[i])

    return bandpowers

def execute(block, config):
    input_section = config['in_name']

    # Read minimum ell value
    l_min = config['min_ell']

    # Read the input ell.
    ell = block[input_section, "ell"]
    ell = ell[ell >= l_min] # Filter out the ell = 0 value
    num_ell = len(ell)

    # Read the number of bands
    num_bands = config['num_bands']

    # Loop through bin pairs and see if C_ell exists for all of them
    n_bins = int(block[input_section, 'nbin'])
    bandpower_ell = np.logspace(np.log(min(ell)), np.log(max(ell)+1), num_bands*2+1, base = np.e)[1::2]
    theory_bandpowers_stacked = np.array([])

    for i in range(n_bins):
        for j in range(i+1):
            theory_cl = block[input_section, 'bin_{0}_{1}'.format(i+1,j+1)]
            theory_cl = theory_cl[-num_ell:]
            band_average = average_bands(ell, theory_cl, num_bands)
            theory_bandpowers_stacked = np.append(theory_bandpowers_stacked, band_average)
            block[input_section, 'bandpowers_bin_{0}_{1}'.format(i+1,j+1)] = band_average

    # block[config['out_name'], "bandpowers"] = theory_bandpowers_stacked.reshape(n_bins,-1).T.flatten()
    block[config['out_name'], "bandpowers"] = theory_bandpowers_stacked
    block[config['out_name'], "bandpower_ell"] = bandpower_ell
    print("Writing bandpowers to its own block")

    return 0

# def clean(config):
#     likelihood_type, n_dim, parameter_section, mu_section, likelihood_name, counter = config
#     print(f"Ran likelihood {counter[0]} times.", flush=True)



