import numpy as np
from cosmosis.datablock import option_section, names
from cosmosis.datablock.cosmosis_py import errors

def setup(options):
    config = {}
    config['num_bands'] = options.get_int(option_section, "num_bands", default=8)
    config['min_ell'] = options.get_double(option_section, "min_ell", default=1.0)
    config['in_name'] = options.get_string(option_section, 'in_name')
    config['noisy_in_name'] = options.get_string(option_section, 'noisy_in_name')
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
    cls_per_band = len(l)//n
    assert len(l)%n == 0
    bandpowers = np.zeros(n)
    for i in range(n):
        lower = i * cls_per_band
        upper = i * cls_per_band + cls_per_band
        bandpowers[i] = bandpower_integral(l[lower:upper], cls[lower:upper])

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

    theory_bandpowers_stacked = np.array([])
    noisey_bandpowers_stacked = np.array([])

    for i in range(n_bins):
        for j in range(i+1):
            noisey_cl = block[config['noisy_in_name'], 'bin_{0}_{1}'.format(i+1,j+1)]
            noisey_cl = noisey_cl[-num_ell:]
            theory_cl = block[input_section, 'bin_{0}_{1}'.format(i+1,j+1)]
            theory_cl = theory_cl[-num_ell:]

            noisey_band_average = average_bands(ell, noisey_cl, num_bands)
            band_average = average_bands(ell, theory_cl, num_bands)
            noisey_bandpowers_stacked = np.append(noisey_bandpowers_stacked, noisey_band_average)
            theory_bandpowers_stacked = np.append(theory_bandpowers_stacked, band_average)
            block[input_section, 'bandpowers_bin_{0}_{1}'.format(i+1,j+1)] = band_average

    block[config['out_name'], "noisey_bandpower_cls"] = noisey_bandpowers_stacked.reshape(num_bands,-1).T.flatten() # So the data structure is done in stacks of "l-bands" instead of unique tomographic bins
    block[config['out_name'], "theory_bandpower_cls"] = theory_bandpowers_stacked.reshape(num_bands,-1).T.flatten()
    print("Writing bandpowers to its own block")

    return 0

# def clean(config):
#     likelihood_type, n_dim, parameter_section, mu_section, likelihood_name, counter = config
#     print(f"Ran likelihood {counter[0]} times.", flush=True)



