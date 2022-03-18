from cosmosis.datablock import option_section, names, BlockError
import numpy as np
from scipy.special import jv

def setup(options):
    config = {}
    config['data_sets'] = options.get_string(option_section, "data_sets") #source or lens
    config['data_sets'] = config['data_sets'].split()
    if not config['data_sets']:
        raise RuntimeError(
            "Option data_sets empty; please set the option data_sets=name1 name2 etc and I will search the fits file for nz_name2, nz_name2, etc.")

    config['nz_shape'] = options.get_string(option_section, "nz_shape") #tophat, gaussian, dirac
    config['bin_mean'] = options[option_section, "bin_mean"] #tophat: midpoint, gaussian: mean, dirac: mean
    if config['nz_shape'] == 'tophat' or config['nz_shape'] == 'tophat_like' or config['nz_shape'] == 'gaussian':
        try:
            config['bin_width'] = options[option_section, "bin_width"] #tophat: half-width, gaussian: std. dev.
        except:
            raise Exception('If nz_shape == tophat or gaussian, please provide a half-width or standard deviation for each bin.')
    
    config['resolution'] = options.get_int(option_section, "resolution")

    return config

def execute(block, config):
    bin_mean = np.array(config['bin_mean'].split(','), dtype = np.float64)
    bin_width = np.array(config['bin_width'].split(','), dtype = np.float64)

    z_max = bin_mean[-1] + 5*bin_width[-1] #arbitrary maximum approx. 5*sigma away from the mean of the last bin

    z = np.linspace(0, float(z_max), int(config['resolution']))

    data = {}
    for data_set in config['data_sets']:
        name = "nz_" + data_set.lower()
        nz = []
        if config['nz_shape'] == 'tophat':
            zlim = [bin_mean[i] - bin_width[i] for i in range(len(bin_mean))]
            zlim = zlim + [bin_mean[-1] + bin_width[-1]]
            for i in range(0, len(bin_mean)):
                n_of_z = np.zeros(len(z))
                n_of_z[(bin_mean[i] + bin_width[i] > z) & (z > bin_mean[i] - bin_width[i])] = np.ones(len(n_of_z[(bin_mean[i] + bin_width[i] > z) & (z > bin_mean[i] - bin_width[i])]))
                n_of_z[(bin_mean[i] + bin_width[i] == z) & (z == bin_mean[i] - bin_width[i])] = 0.5*np.ones(len(n_of_z[(bin_mean[i] + bin_width[i] == z) & (z == bin_mean[i] - bin_width[i])]))
                #n_of_z = 4 * (jv(1, 2*bin_width[i]*(z - bin_mean[i])))**2 / (2*bin_width[i]*(z - bin_mean[i]))**2
                n_of_z = n_of_z/np.trapz(n_of_z, x = z)
                print('Bin {0} between z = {1} and z = {2} with total prob. of {3}'.format(i+1, round(bin_mean[i] - bin_width[i], 2), round(bin_mean[i] + bin_width[i], 2), round(np.trapz(n_of_z, x = z), 2)))
                nz.append(n_of_z)
        
        elif config['nz_shape'] == 'tophat_like':
            zlim = z_max
            for i in range(0, len(bin_mean)):
                n_of_z = (1/(bin_width[i]*np.sqrt(2*np.pi))) * np.exp(-0.5*((z - bin_mean[i])/bin_width[i])**30)
                n_of_z = n_of_z/np.trapz(n_of_z, x = z)
                nz.append(n_of_z)
        
        elif config['nz_shape'] == 'gaussian':
            zlim = z_max
            for i in range(0, len(bin_mean)):
                n_of_z = (1/(bin_width[i]*np.sqrt(2*np.pi))) * np.exp(-0.5*((z - bin_mean[i])/bin_width[i])**2)
                n_of_z = n_of_z/np.trapz(n_of_z, x = z)
                nz.append(n_of_z)
        
        elif config['nz_shape'] == 'dirac':
            zlim = z_max
            for i in range(0, len(bin_mean)):
                n_of_z = np.zeros(len(z))
                n_of_z[int((np.abs(z - bin_mean[i])).argmin())] = 1
                n_of_z = n_of_z/np.trapz(n_of_z, x = z)
                nz.append(n_of_z)

        data[name] = (z, np.array(nz))
    
    for name, data in list(data.items()):
        z, nz = data
        nbin = len(nz)
        ns = len(z)
        block[name, "nbin"] = nbin
        block[name, "nz"] = ns
        block[name, "z"] = z
        block[name, "zlim"] = zlim
        for i, n in enumerate(nz):
            block[name, "bin_{0}".format(i + 1)] = n
    return 0