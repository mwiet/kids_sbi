from cosmosis.datablock import option_section, names, BlockError
import levinpower
import numpy as np
from scipy.integrate import simps
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
    config["ell_min"]   = options.get_double(option_section, "ell_min")
    config["ell_max"]   = options.get_double(option_section, "ell_max")
    config["n_ell"]     = options.get_int(option_section, "n_ell")
    
    config["data_sets"] = options.get_string(option_section, "data_sets") #source or lens
    config["data_sets"] = config["data_sets"].split()

    try:
        config["source_input"] = options.get_string(option_section, "source_input")
    except BlockError:
        if config["data_sets"][0] == 'SOURCE' or config["data_sets"][1] == 'SOURCE':
            raise Exception('If SOURCE sample provided, it is necessary to specify the source_input.')
        else:
            config["source_input"] = None
            pass
    try:
        config["lens_input"] = options.get_string(option_section, "lens_input")
    except BlockError:
        if config["data_sets"][0] == 'LENS' or config["data_sets"][1] == 'LENS':
            raise Exception('If LENS sample provided, it is necessary to specify the lens_input.')
        else:
            config["lens_input"] = None
            pass

    if config["data_sets"][0] == "LENS" or config["data_sets"][1] == "LENS":
        try:
            config["clustering_bias"] = options[option_section, "clustering_bias"]
        except Exception:
            print("Linear clustering bias for each tomographic lens bin is required.")

    config["output_section"] = options.get_string(option_section, "output_section") #shear_cl, galaxy_cl, galaxy_shear_cl
    
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

    np.savez("npz/growth.npz", block[names.growth_parameters, "fsigma_8"])
    np.savez("npz/growth_z.npz", block[names.growth_parameters, "z"])

    c = float(np.sqrt(block[names.cosmological_parameters, "cs2_de"]))*299792.4580 #km/s

    if z_distance[0] == 0.0:
        z_distance[0] = (z_distance[0] + z_distance[1])/2.
        chi_distance[0] = z_distance[0]*c/(config['h0']*100) #only for small z
        if chi_distance[0] > chi_distance[1]:
            raise Exception('Smallest z value too large.')

    z_interp    = interp1d(chi_distance, z_distance, bounds_error = False, fill_value = "extrapolate")
    z_max       = z_interp(config['chi_max'])

    np.savez('npz/z_distance.npz', z_distance)
    np.savez('npz/chi_distance.npz', chi_distance)

    z_pk    = block[names.matter_power_nl, "z"]
    k_pk    = block[names.matter_power_nl, "k_h"]*config["h0"]
    p_k     = block[names.matter_power_nl, "p_k"]/config["h0"]**3

    p_k_linear = block[names.matter_power_lin, "p_k"]/config["h0"]**3

    np.savez('npz/p_k.npz', p_k)
    np.savez('npz/k_pk.npz', k_pk)
    np.savez('npz/z_pk.npz', z_pk)

    power_spectrum = p_k.flatten()
    linear_power_spectrum = p_k_linear.flatten()

    if len(power_spectrum) != len(z_pk)*len(k_pk):
        raise Exception("Power spectrum dimensions are inconsistent.")
    
    if config["source_input"] != None:
        number_count_sh = block[config["source_input"], "nbin"]
    else:
        number_count_sh = 0
    
    if config["lens_input"] != None:
        number_count = block[config["lens_input"], "nbin"]
    else:
        number_count = 0

    kernel = []
    for i in range(number_count + number_count_sh):
        if i < number_count:
            p_lens      = block[config["lens_input"], "bin_{0}".format(i+1)]
            z_lens      = block[config["lens_input"], "z"]
            p_lens_int  = interp1d(z_lens, p_lens, bounds_error = False, fill_value = "extrapolate")

            np.savez('npz/p_lens_n_{0}.npz'.format(i+1), p_lens)
            np.savez('npz/z_lens_n_{0}.npz'.format(i+1), z_lens)

            w           = config["w"] + config["wa"] * np.divide(z_distance, z_distance+1)
            H_of_z      = 100*config['h0']*np.sqrt(config['omega_m']*(1+z_distance)**3 + config['omega_k']*(1+z_distance)**2 + config['omega_lambda']*(1+z_distance)**(3*(1+w)))
            kernel.append(float(config["clustering_bias"][i])*np.multiply(H_of_z/c, p_lens_int(z_distance)))

        elif i >= number_count:
            p_source = block[config["source_input"], "bin_{0}".format(i-number_count+1)]
            z_source = block[config["source_input"], "z"]

            p_interp = interp1d(z_source, p_source, fill_value = "extrapolate")
            p_source_n = p_interp(z_distance)
            p_source_n = p_source_n/np.trapz(p_source_n, z_distance)

            np.savez('npz/p_source_n_{0}.npz'.format(i-number_count+1), p_source_n)
            np.savez('npz/z_source_n_{0}.npz'.format(i-number_count+1), z_distance)
            
            chi_cl  = config["chi_of_z"](z_distance)
            norm        = 1.0/np.trapz(p_source_n[z_max >= z_distance], x = z_distance[z_max >= z_distance])

            kernel_i = []
            count = 0
            for z in z_distance:
                z_integrant = z_distance[(z_max >= z_distance) & (z_distance >= z)]
                w           = config["w"] + config["wa"] * np.divide(z, z+1)
                H_of_z      = 100*config['h0']*np.sqrt(config['omega_m']*(1+z)**3 + config['omega_k']*(1+z)**2 + config['omega_lambda']*(1+z)**(3*(1+w)))
                chi         = config["chi_of_z"](z)
                chi_prime   = chi_cl[(config['chi_max'] >= chi_cl) & (chi_cl >= chi)]
                product     = np.multiply(p_source_n[(z_max >= z_distance) & (z_distance >= z)], chi_prime - chi)
                ratio       = np.divide(product, chi_prime)
                prefactor   = 1.5*config['omega_m']*((100*config['h0'])**2)/(c**2)
                kernel_i.append(prefactor*norm*chi*(1+z)*np.trapz(ratio, x = z_integrant))#integral from chi to chi_max of the source redshift distribution * (chi' - chi)/chi' w.r.t. chi'
                count+=1
            #for chi in chi_cl:
            #    chi_integrant   = chi_cl[(config['chi_max'] >= chi_cl) & (chi_cl >= chi)]
            #    z               = float(z_distance[chi_cl == chi][0])
            #    z_integrant = z_distance[(z_max >= z_distance) & (z_distance >= z)]
            #    w               = config["w"] + config["wa"] * np.divide(z_integrant, z_integrant+1)
            #    H_of_z          = 100*config['h0']*np.sqrt(config['omega_m']*(1+z_integrant)**3 + config['omega_k']*(1+z_integrant)**2 + config['omega_lambda']*(1+z_integrant)**(3*(1+w)))
            #    product         = np.multiply(p_source_n[(z_max >= z_distance) & (z_distance >= z)], chi_integrant - chi)
            #    ratio           = np.divide(product, chi_integrant)
            #    prefactor       = 1.5*config['omega_m']*((100*config['h0'])**2)/(c**2)
            #    kernel_i.append(prefactor*norm*chi*(1+z)*np.trapz(np.multiply(ratio, H_of_z/c), x = chi_integrant))
            kernel.append(kernel_i)
    kernels     = np.array(kernel)
    np.savez('kernels', kernels)
    np.savez('chi_cl', chi_cl)
    ell = list(map(int, gen_log_space(config["ell_max"], int(config["n_ell"])) + config["ell_min"]))

    print(np.shape(chi_cl))
    print('Launching Levin...')
    lp = levinpower.LevinPower(False, number_count = int(number_count),
                            z_bg = z_distance, chi_bg = chi_distance,
                            chi_cl = chi_cl, kernel = kernels.T,
                            k_pk = k_pk, z_pk = z_pk, pk = power_spectrum, boxy = True)

    kernel_4 = []
    for chi_i in chi_cl:
        k = lp.kernels(chi_i, 4)
        kernel_4.append(k)

    np.savez("kernel_4", np.array(kernel_4))
    
    #lp.init_splines(background_z, background_chi,
    #            chi_kernels, kernels.T, k_pk, z_pk, power_spectrum)
    print('Computing angular power spectra...')
    Cl_gg, Cl_gs, Cl_ss = lp.compute_C_ells(ell)

    Cl_ss = np.array(Cl_ss)
    Cl_gg = np.array(Cl_gg)
    Cl_gs = np.array(Cl_gs)

    if "shear_cl" in str(config["output_section"]):
        block["shear_cl", "is_auto"] = 'True'
        block["shear_cl", "sample_a"] = config["data_sets"][0]
        block["shear_cl", "sample_b"] = config["data_sets"][1]
        block["shear_cl", "nbin"] = number_count_sh
        block["shear_cl", "nbin_a"] = number_count_sh
        block["shear_cl", "nbin_b"] = number_count_sh
        block["shear_cl", "sep_name"] = "ell"
        block["shear_cl", "save_name"] = ""
        block["shear_cl", "ell"] = np.array(ell)

        idx_ls = np.array([[i, j] for i in range(0, number_count_sh) for j in range(0, i+1)]) #Generates bin permutations such that i>=j
        idx_sl = np.array([[i, j+i] for i in range(0, number_count_sh) for j in range(0, number_count_sh-i)]) #Generates bin permuatations such that i<=j
        new_order = [np.where((idx_sl == pair).all(axis=1))[0][0] for pair in idx_ls[:, [1, 0]]]
        Cl_ss_reordered = Cl_ss[new_order] #Reorders Cls s.t. bins are ordered following i>=j
        np.savez('shear_cl', Cl_ss_reordered)

        counter = 0
        for i in range(0, number_count_sh):
            for j in range(0, i+1):
                c_ell = Cl_ss_reordered[counter]
                block["shear_cl", 'bin_{0}_{1}'.format(i+1,j+1)] = c_ell
                counter += 1

    if "galaxy_cl" in str(config["output_section"]):
        block["galaxy_cl", "is_auto"] = 'True'
        block["galaxy_cl", "sample_a"] = config["data_sets"][0]
        block["galaxy_cl", "sample_b"] = config["data_sets"][1]
        block["galaxy_cl", "nbin"] = number_count
        block["galaxy_cl", "nbin_a"] = number_count
        block["galaxy_cl", "nbin_b"] = number_count
        block["galaxy_cl", "sep_name"] = "ell"
        block["galaxy_cl", "save_name"] = ""
        block["galaxy_cl", "ell"] = ell

        idx_ls = np.array([[i, j] for i in range(0, number_count) for j in range(0, i+1)]) #Generates bin permutations such that i>=j
        idx_sl = np.array([[i, j+i] for i in range(0, number_count) for j in range(0, number_count-i)]) #Generates bin permuatations such that i<=j
        new_order = [np.where((idx_sl == pair).all(axis=1))[0][0] for pair in idx_ls[:, [1, 0]]]

        Cl_gg_reordered = Cl_gg[new_order] #Reorders Cls s.t. bins are ordered following i>=j
        np.savez('galaxy_cl', Cl_gg_reordered)

        counter = 0
        for i in range(0, number_count):
            for j in range(0, i+1):
                c_ell = Cl_gg_reordered[counter]
                block["galaxy_cl", 'bin_{0}_{1}'.format(i+1,j+1)] = c_ell
                counter += 1

    if "galaxy_shear_cl" in str(config["output_section"]):
        block["galaxy_shear_cl", "is_auto"] = 'False'
        block["galaxy_shear_cl", "sample_a"] = config["data_sets"][0]
        block["galaxy_shear_cl", "sample_b"] = config["data_sets"][1]
        block["galaxy_shear_cl", "nbin"] = number_count + number_count_sh
        block["galaxy_shear_cl", "nbin_a"] = number_count
        block["galaxy_shear_cl", "nbin_b"] = number_count_sh
        block["galaxy_shear_cl", "sep_name"] = "ell"
        block["galaxy_shear_cl", "save_name"] = ""
        block["galaxy_shear_cl", "ell"] = ell

        np.savez('galaxy_shear_cl', Cl_gs)

        counter = 0
        for i in range(0, number_count):
            for j in range(0, number_count_sh):
                c_ell = Cl_gs[counter]
                block["galaxy_shear_cl", 'bin_{0}_{1}'.format(i+1,j+1)] = c_ell
                counter += 1
    return 0

def cleanup(config):
    pass