import numpy as np

from cosmosis.datablock import option_section, names

from cosmosis.datablock.cosmosis_py import errors

def setup(options):
    config = {}
    config['num_bands'] = options.get_int(option_section, "num_bands", default=8)
    config['num_bins'] = options.get_int(option_section, "num_bins", default=5)
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
    input_section = "shear_cl"
    # Read the input ell.
    ell = block[input_section, "ell"]
    # Read the number of bands
    num_bands = config['num_bands']

    # Loop through bin pairs and see if C_ell exists for all of them
    n_bins = config['num_bins']

    theory_bandpowers_stacked = np.array([])

    for i in range(n_bins):
        for j in range(n_bins):
            # Read input c_ell from data block.
            try:
                theory_cl = block[input_section, 'bin_%d_%d' % (j+1, i+1)]
                band_average = average_bands(ell, theory_cl, num_bands)
                theory_bandpowers_stacked = np.append(theory_bandpowers_stacked, band_average)
                block['bandpowers', 'bandpowers_bin_%d_%d' %(j+1, i+1)] = band_average
            except:
                print("Skipping bin_%d_%d as it doesn't exist" % (i+1, j+1))
    block["bandpowers", "theory_bandpower_cls"] = theory_bandpowers_stacked
    print("Writing bandpowers to its own block")

    return 0

# def clean(config):
#     likelihood_type, n_dim, parameter_section, mu_section, likelihood_name, counter = config
#     print(f"Ran likelihood {counter[0]} times.", flush=True)



