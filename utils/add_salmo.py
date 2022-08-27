#!python
import numpy as np
from cosmosis.datablock import names, option_section
import subprocess as spc
import warnings
import shutil

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
                    config['doLensing'] = str(config['doLensing'])[1:-1]

        config['outPrefix'] = options.get_string(option_section, 'outPrefix')
        config['outStyle'] =  str(options[option_section,  'outStyle'])
        if config['outStyle'] not in ['0', '1', '64']:
            raise Exception('outStyle must be either 1, 2 or 64: 1 = all galaxies in one file, 2 = two files, one w/ lensing, one w/o, 64 = one file per type')
        
        if config['outStyle'] != '64':
            raise Warning ('If you do not set outStyle = 64, the .fits outputs will not be compatible with other modules in KCAP_GLASS.')
        
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

            config['nbTomo'] =  str(options[option_section,  'nbTomo'])
            try:
                int(config['nbTomo'])
            except:
                raise Exception('nbTomo must be a positive integer')
            
            config['a_n_gal'] =  options[option_section,  'a_n_gal']
            if len(config['a_n_gal']) != int(config['nbTomo']):
                raise Exception('a_n_gal must have nbTomo values')
            
            config['b_n_gal'] = options[option_section,  'b_n_gal']
            if len(config['b_n_gal']) != int(config['nbTomo']):
                raise Exception('b_n_gal must have nbTomo values')

            config['a_sigma_eps'] =  options[option_section,  'a_sigma_eps']
            if len(config['a_sigma_eps']) != int(config['nbTomo']):
                raise Exception('a_sigma_eps must have nbTomo values')

            config['b_sigma_eps'] =  options[option_section,  'b_sigma_eps']
            if len(config['b_sigma_eps']) != int(config['nbTomo']):
                raise Exception('b_sigma_eps must have nbTomo values')

            config['VD_nOfZPath'] = options.get_string(option_section, 'VD_nOfZPath')

            if len(np.array(config['VD_nOfZPath'].split(' '), dtype = str)) != int(config['nbTomo'])*int(config['N_depth']):
                raise Exception('VD_nOfZPath must have a length of nbTomo*N_depth')
    
    config['clean']       = options[option_section,  "clean"]

    if type(config['clean']) != bool:
        raise Exception("The clean must be either 'T' or 'F'")
        
    return config

def execute(block, config):

    if config['config_file_path'] != None:
        spc.run(['./salmo', str(config['config_file_path']), '3', 'seed={0}'.format(config['seed']),
        'nside={0}'.format(block['salmo', 'nside']),
        'N_z_map={0}'.format(block['shell_matter', 'nbin']),
        'bin_z_map={0}'.format(block['shell_matter', 'zlim']),
        'denPrefix={0}/{1}_sample{2}/glass_denMap/{3}_denMap'.format(block['salmo', 'map_folder'], block['salmo', 'runTag'], block['salmo', 'counter'], block['salmo', 'prefix']),
        'lenPrefix={0}/{1}_sample{2}/glass_lenMap/{3}_lenMap'.format(block['salmo', 'map_folder'], block['salmo', 'runTag'], block['salmo', 'counter'], block['salmo', 'prefix']),
        'runTag=_{0}_sample{1}'.format(block['salmo', 'runTag'], block['salmo', 'counter'])
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
        block['salmo', 'denPrefix'] = '{0}/{1}_sample{2}/glass_denMap/{3}_denMap'.format(block['salmo', 'map_folder'], block['salmo', 'runTag'], block['salmo', 'counter'], block['salmo', 'prefix'])
        block['salmo', 'lenPrefix'] = '{0}/{1}_sample{2}/glass_lenMap/{3}_lenMap'.format(block['salmo', 'map_folder'], block['salmo', 'runTag'], block['salmo', 'counter'], block['salmo', 'prefix'])

        mp = np.array(config['maskPath'].split(' '), dtype = str)
        maskPath = ['maskPath="{0} {1}"'.format(i, mp[i]) for i in range(len(mp))]

        nz = np.array(config['nOfZPath'].split(' '), dtype = str)
        nOfZPath = ['nOfZPath="{0} {1}"'.format(i, nz[i]) for i in range(len(nz))]

        if config['doVariableDepth'] == '0':
            spc.run(['./salmo', 'default', '3', 'seed={0}'.format(config['seed']),
            'verbose=0', 'nside={0}'.format(block['salmo', 'nside']),
            'N_z_map={0}'.format(block['shell_matter', 'nbin']),
            'bin_z_map={0}'.format(block['shell_matter', 'zlim']),
            'denPrefix={0}'.format(block['salmo', 'denPrefix']),
            'lenPrefix={0}'.format(block['salmo', 'lenPrefix']),
            'runTag=_{0}_sample{1}'.format(block['salmo', 'runTag'], block['salmo', 'counter']),
            'nbTypes={0}'.format(config['nbTypes']),
            'n_gal={0}'.format(str(config['n_gal'])[1:-1]),
            'doNoise={0}'.format(config['doNoise']),
            'doWgt={0}'.format(config['doWgt']),
            'signConv={0}'.format(config['signConv']),
            'sigma_eps={0}'.format(str(config['sigma_eps'])[1:-1]),
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
            'denPrefix={0}'.format(block['salmo', 'denPrefix']),
            'lenPrefix={0}'.format(block['salmo', 'lenPrefix']),
            'runTag=_{0}_sample{1}'.format(block['salmo', 'runTag'], block['salmo', 'counter']),
            'nbTypes={0}'.format(config['nbTypes']),
            'n_gal={0}'.format(str(config['n_gal'])[1:-1]),
            'doNoise={0}'.format(config['doNoise']),
            'doWgt={0}'.format(config['doWgt']),
            'signConv={0}'.format(config['signConv']),
            'sigma_eps={0}'.format(str(config['sigma_eps'])[1:-1]),
            'doLensing={0}'.format(config['doLensing']),
            'outPrefix={0}'.format(config['outPrefix']),
            'outStyle={0}'.format(config['outStyle']),
            'doVariableDepth={0}'.format(config['doVariableDepth']),
            'nbDepthMaps={0}'.format(config['nbDepthMaps']),
            'N_depth={0}'.format(config['N_depth']),
            'bin_depth={0}'.format(str(config['bin_depth'])[1:-1]),
            'nbTomo={0}'.format(config['nbTomo']),
            'a_n_gal={0}'.format(str(config['a_n_gal'])[1:-1]),
            'b_n_gal={0}'.format(str(config['b_n_gal'])[1:-1]),
            'a_sigma_eps={0}'.format(str(config['a_sigma_eps'])[1:-1]),
            'b_sigma_eps={0}'.format(str(config['b_sigma_eps'])[1:-1])
            ] + maskPath + nOfZPath + depthMapPath + VD_nOfZPath, cwd=config['build_path'], text = True)

    if config['clean']:
        print('Deleting input files from GLASS for sample {0}...'.format(block['salmo', 'counter']))
        shutil.rmtree('{0}/{1}_sample{2}'.format(block['salmo', 'map_folder'], block['salmo', 'runTag'], block['salmo', 'counter']))
        return 0

def cleanup(config):
    pass
    