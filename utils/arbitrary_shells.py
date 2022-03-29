from cosmosis.datablock import option_section, names, BlockError
import numpy as np

def setup(options):
    config = {}
    config['data_sets'] = options.get_string(option_section, "data_sets") #matter or convergence
    config['data_sets'] = config['data_sets'].split()
    if not config['data_sets']:
        raise RuntimeError(
            "Option data_sets empty; please set the option data_sets=name1 name2 etc and I will search the fits file for shell_name2, shell_name2, etc.")

    if 'matter' in config['data_sets']:
        try:
            config['matter_shell_shape'] = options.get_string(option_section, "matter_shell_shape") #tophat, gaussian, dirac
            config['matter_shell_mean'] = options[option_section, "matter_shell_mean"] #tophat: midpoint, gaussian: mean, dirac: mean
            if config['matter_shell_shape'] == 'tophat' or config['matter_shell_shape'] == 'gaussian':
                try:
                    config['matter_shell_width'] = options[option_section, "matter_shell_width"] #tophat: half-width, gaussian: std. dev.
                except:
                    raise Exception('If matter_shell_shape == tophat or gaussian, please provide a half-width or standard deviation for each bin.')
        except:
            raise Exception('To define matter shells, please provide a matter_shell_shape and a matter_shell_mean.')
    
    if 'convergence' in config['data_sets']:
        try:
            config['convergence_shell_shape'] = options.get_string(option_section, "convergence_shell_shape") #tophat, gaussian, dirac
            config['convergence_shell_mean'] = options[option_section, "convergence_shell_mean"] #tophat: midpoint, gaussian: mean, dirac: mean
            if config['convergence_shell_shape'] == 'tophat' or config['convergence_shell_shape'] == 'gaussian':
                try:
                    config['convergence_shell_width'] = options[option_section, "convergence_shell_width"] #tophat: half-width, gaussian: std. dev.
                except:
                    raise Exception('If convergence_shell_shape == tophat or gaussian, please provide a half-width or standard deviation for each bin.')
        except:
            raise Exception('To define convergence shells, please provide a convergence_shell_shape and a convergence_shell_mean.')
    
    
    config['resolution'] = options.get_int(option_section, "resolution")

    return config

def execute(block, config):
    data = {}
    for data_set in config['data_sets']:
        name = "shell_" + data_set.lower()
        shell = []
        if name == 'shell_matter':
            bin_mean = np.array(config['matter_shell_mean'].split(','), dtype = np.float64)
            bin_width = np.array(config['matter_shell_width'].split(','), dtype = np.float64)
            z_max = bin_mean[-1] + 5*bin_width[-1] #arbitrary maximum approx. 5*sigma away from the mean of the last bin
            z = np.linspace(0, float(z_max), int(config['resolution']))
            if config['matter_shell_shape'] == 'tophat':
                zlim = [bin_mean[i] - bin_width[i] for i in range(len(bin_mean))]
                zlim = zlim + [bin_mean[-1] + bin_width[-1]]
                for i in range(0, len(bin_mean)):
                    n_of_z = np.zeros(len(z))
                    n_of_z[(bin_mean[i] + bin_width[i] > z) & (z > bin_mean[i] - bin_width[i])] = np.ones(len(n_of_z[(bin_mean[i] + bin_width[i] > z) & (z > bin_mean[i] - bin_width[i])]))
                    n_of_z[(bin_mean[i] + bin_width[i] == z) & (z == bin_mean[i] - bin_width[i])] = 0.5*np.ones(len(n_of_z[(bin_mean[i] + bin_width[i] == z) & (z == bin_mean[i] - bin_width[i])]))
                    n_of_z = n_of_z/np.trapz(n_of_z, x = z)
                    print('Shell {0} between z = {1} and z = {2} with total prob. of {3}'.format(i+1, round(bin_mean[i] - bin_width[i], 2), round(bin_mean[i] + bin_width[i], 2), round(np.trapz(n_of_z, x = z), 2)))
                    shell.append(n_of_z)
            
            elif config['matter_shell_shape'] == 'gaussian':
                zlim = z_max
                for i in range(0, len(bin_mean)):
                    n_of_z = (1/(bin_width[i]*np.sqrt(2*np.pi))) * np.exp(-0.5*((z - bin_mean[i])/bin_width[i])**2)
                    n_of_z = n_of_z/np.trapz(n_of_z, x = z)
                    shell.append(n_of_z)
            
            elif config['matter_shell_shape'] == 'dirac':
                zlim = z_max
                for i in range(0, len(bin_mean)):
                    n_of_z = np.zeros(len(z))
                    n_of_z[int((np.abs(z - bin_mean[i])).argmin())] = 1
                    n_of_z = n_of_z/np.trapz(n_of_z, x = z)
                    shell.append(n_of_z)

            data[name] = (z, np.array(shell))
        
        elif name == 'shell_convergence':
            bin_mean = np.array(config['convergence_shell_mean'].split(','), dtype = np.float64)
            bin_width = np.array(config['convergence_shell_width'].split(','), dtype = np.float64)
            z_max = bin_mean[-1] + 5*bin_width[-1] #arbitrary maximum approx. 5*sig
            if config['convergence_shell_shape'] == 'tophat':
                zlim = [bin_mean[i] - bin_width[i] for i in range(len(bin_mean))]
                zlim = zlim + [bin_mean[-1] + bin_width[-1]]
                for i in range(0, len(bin_mean)):
                    n_of_z = np.zeros(len(z))
                    n_of_z[(bin_mean[i] + bin_width[i] > z) & (z > bin_mean[i] - bin_width[i])] = np.ones(len(n_of_z[(bin_mean[i] + bin_width[i] > z) & (z > bin_mean[i] - bin_width[i])]))
                    n_of_z[(bin_mean[i] + bin_width[i] == z) & (z == bin_mean[i] - bin_width[i])] = 0.5*np.ones(len(n_of_z[(bin_mean[i] + bin_width[i] == z) & (z == bin_mean[i] - bin_width[i])]))
                    n_of_z = n_of_z/np.trapz(n_of_z, x = z)
                    print('Bin {0} between z = {1} and z = {2} with total prob. of {3}'.format(i+1, round(bin_mean[i] - bin_width[i], 2), round(bin_mean[i] + bin_width[i], 2), round(np.trapz(n_of_z, x = z), 2)))
                    shell.append(n_of_z)
            
            elif config['convergence_shape'] == 'gaussian':
                zlim = z_max
                for i in range(0, len(bin_mean)):
                    n_of_z = (1/(bin_width[i]*np.sqrt(2*np.pi))) * np.exp(-0.5*((z - bin_mean[i])/bin_width[i])**2)
                    n_of_z = n_of_z/np.trapz(n_of_z, x = z)
                    shell.append(n_of_z)
            
            elif config['convergence_shape'] == 'dirac':
                zlim = z_max
                for i in range(0, len(bin_mean)):
                    n_of_z = np.zeros(len(z))
                    n_of_z[int((np.abs(z - bin_mean[i])).argmin())] = 1
                    n_of_z = n_of_z/np.trapz(n_of_z, x = z)
                    shell.append(n_of_z)

            data[name] = (z, np.array(shell))
        
    for name, data in list(data.items()):
        z, shell = data
        nbin = len(shell)
        ns = len(z)
        block[name, "nbin"] = nbin
        block[name, "shell"] = ns
        block[name, "z"] = z
        block[name, "zlim"] = zlim
        for i, n in enumerate(shell):
            block[name, "bin_{0}".format(i + 1)] = n
    return 0

def cleanup(config):
    pass