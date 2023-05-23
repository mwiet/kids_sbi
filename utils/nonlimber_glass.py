from cosmosis.datablock import option_section, names, BlockError
import levinpower
import numpy as np
from scipy.interpolate import interp1d
import sys

sys.settrace

def gen_log_space(limit, n):
    result = [1]
    if n>1:
        ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
    while len(result)<n:
        next_value = result[-1]*ratio
        if next_value - result[-1] >= 1:
            result.append(next_value)
        else:
            result.append(result[-1]+1)
            ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
    return np.array(list(map(lambda x: round(x)-1, result)), dtype=np.uint64)

def setup(options):
    config = {}
    config["ell_min"]   = options.get_int(option_section, "ell_min")
    config["ell_max"]   = options.get_int(option_section, "ell_max")
    config["ell_limber"]   = options.get_int(option_section, "ell_limber")
    config["ell_nonlimber"]   = options.get_int(option_section, "ell_nonlimber")
    config["n_ell"]     = options.get_int(option_section, "n_ell")

    config["shell_input"] = options.get_string(option_section, "shell_input") #Should always be "shell_matter"
    
    config["output_section"] = 'matter_cl'
    
    return config

def execute(block, config):
    z_distance      = block[names.distances, 'z']
    a_distance      = block[names.distances, 'a']
    chi_distance    = np.multiply(block[names.distances, 'd_a'], 1+z_distance)

    if z_distance[1] < z_distance[0]:
        z_distance      = z_distance[::-1].copy()
        a_distance      = a_distance[::-1].copy()
        chi_distance    = chi_distance[::-1].copy()

    config["h0"]            = block[names.cosmological_parameters, "h0"]
    config["ombh2"]         = block[names.cosmological_parameters, "ombh2"]
    config["omch2"]         = block[names.cosmological_parameters, "omch2"]
    config["omega_m"]       = block[names.cosmological_parameters, "omega_m"]
    config["omega_k"]       = block[names.cosmological_parameters, "omega_k"]
    config["omega_lambda"]  = block[names.cosmological_parameters, "omega_lambda"]  

    config["w"]         = block[names.cosmological_parameters, "w"]
    config["wa"]        = block[names.cosmological_parameters, "wa"]
    config["cs2_de"]    = block[names.cosmological_parameters, "cs2_de"]

    config["chi_max"]   = chi_distance.max()
    config["a_of_chi"]  = interp1d(chi_distance, a_distance)
    config["chi_of_z"]  = interp1d(z_distance, chi_distance)
    chi_cl  = config["chi_of_z"](z_distance)

    c = float(np.sqrt(block[names.cosmological_parameters, "cs2_de"]))*299792.4580 #km/s

    z_interp    = interp1d(chi_distance, z_distance, bounds_error = False, fill_value = "extrapolate")

    try:
        z_pk    = block[names.matter_power_nl, "z"]
        k_pk    = block[names.matter_power_nl, "k_h"]*config["h0"]
        p_k     = block[names.matter_power_nl, "p_k"]/config["h0"]**3
    except:
        print('Could not find non-Linear matter power spectrum. Using linear power spectrum instead...')
        z_pk    = block[names.matter_power_lin, "z"]
        k_pk    = block[names.matter_power_lin, "k_h"]*config["h0"]
        p_k     = block[names.matter_power_lin, "p_k"]/config["h0"]**3

    power_spectrum = p_k.flatten()

    if len(power_spectrum) != len(z_pk)*len(k_pk):
        raise Exception("Power spectrum dimensions are inconsistent.")


    number_count = block[config["shell_input"], "nbin"]
    zlim      = block[config["shell_input"], "zlim"]

    kernels = []
    for i in range(number_count):
            z_range = z_distance[((zlim[i+1] >= z_distance) & (z_distance > zlim[i]))]
            chi_range = chi_distance[((zlim[i+1] >= z_distance) & (z_distance > zlim[i]))]
            w           = config["w"] + config["wa"] * np.divide(z_range, z_range+1)
            E_of_z      = np.sqrt(config['omega_m']*(1+z_range)**3 + config['omega_k']*(1+z_range)**2 + config['omega_lambda']*(1+z_range)**(3*(1+w)))
            k = np.zeros(len(z_distance))
            k[((zlim[i+1] >= z_distance) & (z_distance > zlim[i]))] = np.divide(chi_range**2, E_of_z)
            k /= np.trapz(k, z_distance)
            kernels.append(k)
    kernels     = np.array(kernels)
    
    ell = list(map(int, gen_log_space(config["ell_max"], int(config["n_ell"])) + config["ell_min"]))

    print('Launching LevinPower...')
    lp = levinpower.LevinPower(False, number_count = int(number_count),
                            z_bg = z_distance, chi_bg = chi_distance,
                            chi_cl = chi_cl, kernel = kernels.T,
                            k_pk = k_pk, z_pk = z_pk, pk = power_spectrum, boxy = True)

    lp.set_parameters(ELL_limber = config["ell_limber"], ELL_nonlimber = config["ell_nonlimber"], 
                    max_number_subintervals =20, minell = int(config["ell_min"]), maxell = int(config["ell_max"]),
                    N_nonlimber = 40, N_limber = 100, Ninterp = 600)


    print('Computing angular power spectra...')
    Cl_gg, Cl_gs, Cl_ss = lp.compute_C_ells(ell)

    Cl_gg = np.array(Cl_gg)

    block["matter_cl", "is_auto"] = 'True'
    block["matter_cl", "sample_a"] = 'matter'
    block["matter_cl", "sample_b"] = 'matter'
    block["matter_cl", "nbin"] = number_count
    block["matter_cl", "nbin_a"] = number_count
    block["matter_cl", "nbin_b"] = number_count
    block["matter_cl", "sep_name"] = "ell"
    block["matter_cl", "save_name"] = ""
    block["matter_cl", "ell"] = ell

    idx_ls = np.array([[i, j] for i in range(0, number_count) for j in range(0, i+1)]) #Generates bin permutations such that i>=j
    idx_sl = np.array([[i, j+i] for i in range(0, number_count) for j in range(0, number_count-i)]) #Generates bin permuatations such that i<=j
    new_order = [np.where((idx_sl == pair).all(axis=1))[0][0] for pair in idx_ls[:, [1, 0]]]

    Cl_gg_reordered = Cl_gg[new_order] #Reorders Cls s.t. bins are ordered following i>=j

    counter = 0
    for i in range(0, number_count):
        for j in range(0, i+1):
            c_ell = Cl_gg_reordered[counter]
            block["matter_cl", 'bin_{0}_{1}'.format(i+1,j+1)] = c_ell
            counter += 1
    return 0

def cleanup(config):
    pass