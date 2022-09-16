from cosmosis.datablock import option_section, names, BlockError
import numpy as np
from scipy.interpolate import interp1d

# Apply photo-z distribution bias, b, such that n(z) -> n(z-b)

def setup(options):
    config = {}
    config['doVariableDepth'] =   str(options[option_section,  'doVariableDepth'])
    if config['doVariableDepth'] not in ['0', '1']:
        raise Exception('doVariableDepth must be either 0 or 1: 0 = no, 1 = yes')

    config['nOfZPath'] = options.get_string(option_section, 'nOfZPath')

    nz = np.array(config['nOfZPath'].split(' '), dtype = str)
    if not all(np.array(list(map(lambda x: x[-5:], nz))) == 'ascii'):
        raise Exception('nOfZPath must be a space separated sequence of .ascii files.')

    config['nbTomo'] = options.get_int(option_section, 'nbTomo')

    if config['doVariableDepth'] == '1':
        config['N_depth'] = options.get_int(option_section, 'N_depth')

    else: 
        if config['nbTomo'] != len(np.unique(nz)):
            raise Exception('nbTomo and the given number of unique nOfZPaths is incompatible.')

    return config

def execute(block, config):
    block['shift_nz', 'input_paths'] = config['nOfZPath']
    nz = np.array(config['nOfZPath'].split(' '), dtype = str)

    new_files = []
    if config['doVariableDepth'] == '0':
        nz_unique = np.unique(nz)
        mapping = {}
        for i, file in enumerate(nz_unique):
            mapping[file] = np.where(nz == file)[0]

            n = np.loadtxt(file)
            bias = block['nofz_shifts', 'bias_{0}'.format(i+1)]

            print('Shifting source redshift distribution {0} by a delta_z of {1}...'.format(i+1, bias))

            interp = interp1d(n.T[0] - bias, n.T[1], fill_value="extrapolate")
            new_file = file[:-6] + '_deltaz_{0}.ascii'.format(bias)
            new_files.append(new_file)
            
            print('Saving {0}...'.format(new_file))

            np.savetxt(new_file, np.array([n.T[0], interp(n.T[0])]).T)

        repeated_fields = list(mapping.values())
        paths = np.repeat(np.array(new_files), list(map(len, repeated_fields)))

    else:
        counter = 0
        for i in range(config['nbTomo']):
            for j in range(config['N_depth']):
                file = nz[counter]
                n = np.loadtxt(file)
                bias = block['nofz_shifts', 'bias_{0}'.format(i+1)]

                print('Shifting source redshift distribution {0} for depth bin {1} by a delta_z of {2}...'.format(i+1, j+1, bias))

                interp = interp1d(n.T[0] - bias, n.T[1], fill_value="extrapolate")
                new_file = file[:-6] + '_deltaz_{0}.ascii'.format(bias)
                new_files.append(new_file)

                print('Saving {0}...'.format(new_file))
                
                np.savetxt(new_file, np.array([n.T[0], interp(n.T[0])]).T)
                counter += 1
        paths = np.array(new_files)       
    
    block['shift_nz', 'paths'] = ' '.join(paths)

    return 0

def cleanup(config):
    pass