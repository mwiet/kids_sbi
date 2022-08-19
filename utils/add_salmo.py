#!python
import numpy as np
import re
from cosmosis.datablock import names, option_section
import subprocess as spc
import warnings

def salmo_format(string):
    m1 = np.array(string.split(' '), dtype = str)
    m2 = np.arange(len(m1))
    m3 = np.zeros(m1.shape[0] + m2.shape[0], dtype=m1.dtype)
    m3[::2] = m2
    m3[1::2] = m1
    print(m3)
    paths = re.sub(r"' '", ' ', str(m3))
    paths = re.sub(r"'", '', str(paths))
    paths = re.sub("\n", '', str(paths))
    paths = re.sub("  ", ' ', str(paths))
    new_string = '"{0}"'.format(paths[1:-1])
    print(new_string)
    return new_string

def setup(options):

    config = {}

    config['build_path'] = options.get_string(option_section, 'build_path')

    try:
        config['config_file_path'] = options.get_string(option_section, 'config_file_path')
        print('Taking all paramaters within {0} at face value, so any other survey parameter specified in the cosmosis ini file will be disregarded.')
    except:
        config['config_file_path'] = None
    
    config['seed'] = options.get_string(option_section, 'seed') #'random' or int

    if config['config_file_path'] == None:
        #Selection function params
        config['nbTypes'] = str(options[option_section,  'nbTypes'])
        try:
            int(config['nbTypes'])
        except:
            raise Exception('nbTypes must be a positive integer')
        
        try:
            config['maskPath'] = options.get_string(option_section, 'maskPath')
        except:
            config['maskPath'] = ''

        if len(np.array(config['maskPath'].split(' '), dtype = str)) != int(config['nbTypes']):
                raise Exception('maskPath must have a length of nbTypes')

        try:
            config['nOfZPath'] = options.get_string(option_section, 'nOfZPath')
        except:
            config['nOfZPath'] = ''

        if len(np.array(config['nOfZPath'].split(' '), dtype = str)) != int(config['nbTypes']):
                raise Exception('nOfZPath must have a length of nbTypes')
        
        try:
            config['n_gal'] = options[option_section,  'n_gal'] #float array with n_gal for each tomographic bin
        except:
            config['n_gal'] = ''

        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            if config['n_gal'] != '':
                if len(config['n_gal']) != int(config['nbTypes']):
                    raise Exception('n_gal must have nbType number of values')
                else:
                    config['n_gal'] = str(config['n_gal'])[1:-1]
        
        #Lensing and output params
        config['doNoise'] =  str(options[option_section,  'doNoise'])
        if config['doNoise'] not in ['0', '1', '2']:
            raise Exception('doNoise must be an integer between 0 and 2: 0 = no, 1 = yes, 2 = output both')

        config['doWgt'] =  str(options[option_section,  'doWgt'])
        if config['doWgt'] not in ['0', '1']:
            raise Exception('doWgt must be either 0 or 1: 0 = no, 1 = yes')
        
        config['signConv'] =  str(options[option_section,  'signConv'])
        if config['signConv'] not in ['0', '1']:
            raise Exception('signConv must be either 0 or 1: 0 = athena, 1 = treecorr')
        
        try:
            config['sigma_eps'] = options[option_section,  'sigma_eps'] #float array with the ellipticity dispersion, sigma_eps^2 = 0.5 * (<epsilon_1^2> + <epsilon_2^2>) = sigma_kappa^2
        except:
            config['sigma_eps'] = ''
        
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            if config['sigma_eps'] != '':
                if len(config['sigma_eps']) != int(config['nbTypes']):
                    raise Exception('sigma_eps must have nbType number of values')
                else:
                    config['sigma_eps'] = str(config['sigma_eps'])[1:-1]
        
        try:
            config['doLensing'] = options[option_section,  'doLensing']
        except:
            config['doLensing'] = ''

        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            if  config['doLensing'] != '':
                if len(config['doLensing']) != int(config['nbTypes']):
                    raise Exception('doLensing must have nbType number of values')     
                else:
                    config['doLensing'] = str(config['doLensing'][1:-1])

        config['outPrefix'] = options.get_string(option_section, 'outPrefix')
        config['outStyle'] =  str(options[option_section,  'outStyle'])
        if config['outStyle'] not in ['0', '1', '64']:
            raise Exception('outStyle must be either 1, 2 or 64: 1 = all galaxies in one file, 2 = two files, one w/ lensing, one w/o, 64 = one file per type')

        #Variable depth
        config['doVariableDepth'] =   str(options[option_section,  'doVariableDepth'])
        if config['doVariableDepth'] not in ['0', '1']:
            raise Exception('doVariableDepth must be either 0 or 1: 0 = no, 1 = yes')

        if config['doVariableDepth'] == '1':
            config['nbDepthMaps'] =  str(options[option_section,  'nbDepthMaps'])
            try:
                int(config['nbDepthMaps'])
            except:
                raise Exception('nbDepthMaps must be a positive integer')

            config['depthMapPath'] = options.get_string(option_section, 'depthMapPath')
            if len(np.array(config['depthMapPath'].split(' '), dtype = str)) != int(config['nbDepthMaps']):
                raise Exception('depthMapPath must have a length of nbDepthMaps')

            config['N_depth'] =  str(options[option_section, 'N_depth'])
            try:
                int(config['N_depth'])
            except:
                raise Exception('N_depth must be a positive integer')

            config['bin_depth'] =  options[option_section, 'bin_depth']
            if len(config['bin_depth']) != int(config['N_depth'])+1:
                raise Exception('bin_depth must have N_depth+1 values')
            else:
                config['bin_depth'] = str(config['bin_depth'])[1:-1]

            config['nbTomo'] =  str(options[option_section,  'nbTomo'])
            try:
                int(config['nbTomo'])
            except:
                raise Exception('nbTomo must be a positive integer')
            
            config['a_n_gal'] =  options[option_section,  'a_n_gal']
            if len(config['a_n_gal']) != int(config['nbTomo']):
                raise Exception('a_n_gal must have nbTomo values')
            else:
                config['a_n_gal'] = str(config['a_n_gal'])[1:-1]
            
            config['b_n_gal'] = options[option_section,  'b_n_gal']
            if len(config['b_n_gal']) != int(config['nbTomo']):
                raise Exception('b_n_gal must have nbTomo values')
            else:
                config['b_n_gal'] = str(config['b_n_gal'])[1:-1]

            config['a_sigma_eps'] =  options[option_section,  'a_sigma_eps']
            if len(config['a_sigma_eps']) != int(config['nbTomo']):
                raise Exception('a_sigma_eps must have nbTomo values')
            else:
                config['a_sigma_eps'] = str(config['a_sigma_eps'])[1:-1]

            config['b_sigma_eps'] =  options[option_section,  'b_sigma_eps']
            if len(config['b_sigma_eps']) != int(config['nbTomo']):
                raise Exception('b_sigma_eps must have nbTomo values')
            else:
                config['b_sigma_eps'] = str(config['b_sigma_eps'])[1:-1]

            config['VD_nOfZPath'] = options.get_string(option_section, 'VD_nOfZPath')

            if len(np.array(config['VD_nOfZPath'].split(' '), dtype = str)) != int(config['nbTomo'])*int(config['N_depth']):
                raise Exception('VD_nOfZPath must have a length of nbTomo*N_depth')
        
    return config

def execute(block, config):

    if config['config_file_path'] != None:
        spc.run(['./salmo', str(config['config_file_path']), '3', 'seed={0}'.format(config['seed']),
        'nside={0}'.format(block['salmo', 'nside']),
        'N_z_map={0}'.format(block['shell_matter', 'nbin']),
        'bin_z_map={0}'.format(block['shell_matter', 'zlim']),
        'denPrefix={0}/{1}_sample{2}/glass_denMap/{3}_denMap'.format(block['salmo', 'map_folder'], block['salmo', 'runTag'], block['salmo', 'counter'], block['salmo', 'prefix']),
        'lenPrefix={0}/{1}_sample{2}/glass_lenMap/{3}_lenMap'.format(block['salmo', 'map_folder'], block['salmo', 'runTag'], block['salmo', 'counter'], block['salmo', 'prefix']),
        'runTag=_{0}'.format(block['salmo', 'runTag'])
        ], cwd=config['build_path'], text = True)
    
    else:
        block['salmo', 'seed'] = config['seed']
        block['salmo', 'nbTypes'] = config['nbTypes']
        block['salmo', 'maskPath'] = config['maskPath']
        block['salmo', 'nOfZPath'] = config['nOfZPath']
        block['salmo', 'n_gal'] = config['n_gal']
        block['salmo', 'doNoise'] = config['doNoise']
        block['salmo', 'doWgt'] = config['doWgt']
        block['salmo', 'signConv'] = config['signConv']
        block['salmo', 'sigma_eps'] = config['sigma_eps']
        block['salmo', 'doLensing'] = config['doLensing']
        block['salmo', 'outPrefix'] = config['outPrefix']
        block['salmo', 'outStyle'] = config['outStyle']
        block['salmo', 'doVariableDepth'] = config['doVariableDepth']

        mp = np.array(config['maskPath'].split(' '), dtype = str)
        maskPath = ['maskPath="{0} {1}"'.format(i, mp[i]) for i in range(len(mp))]

        nz = np.array(config['nOfZPath'].split(' '), dtype = str)
        nOfZPath = ['nOfZPath="{0} {1}"'.format(i, nz[i]) for i in range(len(nz))]

        if config['doVariableDepth'] == '0':
            spc.run(['./salmo', 'default', '3', 'seed={0}'.format(config['seed']),
            'verbose=0', 'nside={0}'.format(block['salmo', 'nside']),
            'N_z_map={0}'.format(block['shell_matter', 'nbin']),
            'bin_z_map={0}'.format(block['shell_matter', 'zlim']),
            'denPrefix={0}/{1}_sample{2}/glass_denMap/{3}_denMap'.format(block['salmo', 'map_folder'], block['salmo', 'runTag'], block['salmo', 'counter'], block['salmo', 'prefix']),
            'lenPrefix={0}/{1}_sample{2}/glass_lenMap/{3}_lenMap'.format(block['salmo', 'map_folder'], block['salmo', 'runTag'], block['salmo', 'counter'], block['salmo', 'prefix']),
            'runTag=_{0}'.format(block['salmo', 'runTag']),
            'nbTypes={0}'.format(config['nbTypes']),
            'n_gal={0}'.format(config['n_gal']),
            'doNoise={0}'.format(config['doNoise']),
            'doWgt={0}'.format(config['doWgt']),
            'signConv={0}'.format(config['signConv']),
            'sigma_eps={0}'.format(config['sigma_eps']),
            'doLensing={0}'.format(config['doLensing']),
            'outPrefix={0}'.format(config['outPrefix']),
            'outStyle={0}'.format(config['outStyle']),
            'doVariableDepth={0}'.format(config['doVariableDepth'])
            ] + maskPath + nOfZPath, cwd=config['build_path'], text = True)
        else:
            block['salmo', 'nbDepthMaps'] = config['nbDepthMaps']
            block['salmo', 'depthMapPath'] = config['depthMapPath']
            block['salmo', 'N_depth'] = config['N_depth']
            block['salmo', 'bin_depth'] = config['bin_depth']
            block['salmo', 'nbTomo'] = config['nbTomo']
            block['salmo', 'a_n_gal'] = config['a_n_gal']
            block['salmo', 'b_n_gal'] = config['b_n_gal']
            block['salmo', 'a_sigma_eps'] = config['a_sigma_eps']
            block['salmo', 'b_sigma_eps'] = config['b_sigma_eps']
            block['salmo', 'VD_nOfZPath'] = config['VD_nOfZPath']

            dm = np.array(config['depthMapPath'].split(' '), dtype = str)
            depthMapPath = ['depthMapPath="{0} {1}"'.format(i, dm[i]) for i in range(len(dm))]

            vd = np.array(config['VD_nOfZPath'].split(' '), dtype = str)
            VD_nOfZPath = ['VD_nOfZPath="{0} {1}"'.format(i, vd[i]) for i in range(len(vd))]

            spc.run(['./salmo', 'default', '3', 'seed={0}'.format(config['seed']),
            'verbose=0', 'nside={0}'.format(block['salmo', 'nside']),
            'N_z_map={0}'.format(block['shell_matter', 'nbin']),
            'bin_z_map={0}'.format(str(block['shell_matter', 'zlim'])[1:-1]),
            'denPrefix={0}/{1}_sample{2}/glass_denMap/{3}_denMap'.format(block['salmo', 'map_folder'], block['salmo', 'runTag'], block['salmo', 'counter'], block['salmo', 'prefix']),
            'lenPrefix={0}/{1}_sample{2}/glass_lenMap/{3}_lenMap'.format(block['salmo', 'map_folder'], block['salmo', 'runTag'], block['salmo', 'counter'], block['salmo', 'prefix']),
            'runTag=_{0}'.format(block['salmo', 'runTag']),
            'nbTypes={0}'.format(config['nbTypes']),
            'n_gal={0}'.format(config['n_gal']),
            'doNoise={0}'.format(config['doNoise']),
            'doWgt={0}'.format(config['doWgt']),
            'signConv={0}'.format(config['signConv']),
            'sigma_eps={0}'.format(config['sigma_eps']),
            'doLensing={0}'.format(config['doLensing']),
            'outPrefix={0}'.format(config['outPrefix']),
            'outStyle={0}'.format(config['outStyle']),
            'doVariableDepth={0}'.format(config['doVariableDepth']),
            'nbDepthMaps={0}'.format(config['nbDepthMaps']),
            'N_depth={0}'.format(config['N_depth']),
            'bin_depth={0}'.format(config['bin_depth']),
            'nbTomo={0}'.format(config['nbTomo']),
            'a_n_gal={0}'.format(config['a_n_gal']),
            'b_n_gal={0}'.format(config['b_n_gal']),
            'a_sigma_eps={0}'.format(config['a_sigma_eps']),
            'b_sigma_eps={0}'.format(config['b_sigma_eps'])
            ] + maskPath + nOfZPath + depthMapPath + VD_nOfZPath, cwd=config['build_path'], text = True)
    return 0

def cleanup(config):
    pass
    