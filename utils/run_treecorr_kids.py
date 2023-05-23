from cosmosis.datablock import option_section, names, BlockError
import numpy as np
import subprocess as spc
import os
import glob
import time

def setup(options):
    config = {}
    config['build_path'] = options.get_string(option_section, 'build_path') #/share/splinter/maxivonw/Cat_to_Obs_K1000_P1/Calc_2pt_Stats
    config['in_name'] = options.get_string(option_section, 'in_name') #salmo or salmo_novd
    config['in_path'] = options.get_string(option_section, 'in_path') #"/share/data1/maxivonw/kcap_out/salmo"

    config['vd'] = options.get_bool(option_section, 'vd')

    config['mockname'] = options[option_section,  'mockname']
    config['patches'] = options.get_string(option_section,  'patches') # 'N' and/or 'S', or 'ALL'
    config['patches'] = np.array(config['patches'].split(' '), dtype = str)

    config['tomo_bin_limits'] = options[option_section,  'tomo_bin_limits'].astype(str) #e.g. "6 0.1 0.3 0.5 0.7 0.9 1.2 2.0" for 6 tomographic bins with the subsequent limits
    config['n_tomo'] = (len(config['tomo_bin_limits']) - 2)
    config['tomo_bin_limits'][0] = config['tomo_bin_limits'][0][:-2]
    config['tomo_bin_limits']  = ' '.join(config['tomo_bin_limits'])

    config['theta_bin_range'] = options[option_section,  'theta_bin_range'].astype(str)  #e.g. "300 0.1 300" for 300 bins between theta of 0.1 and 300
    config['theta_bin_range'][0] = config['theta_bin_range'][0][:-2]
    config['theta_bin_range']  = ' '.join(config['theta_bin_range'])
    
    config['theta_rebin'] = options.get_bool(option_section, 'theta_rebin')

    if config['theta_rebin']:
        config['theta_rebin_range'] = options[option_section,  'theta_bin_range'].astype(str) #e.g. "9 0.1 300" for 9 bins between theta of 0.1 and 300
        config['theta_rebin_range'][0] = config['theta_rebin_range'][0][:-2]
        config['theta_rebin_range']  = ' '.join(config['theta_rebin_range'])
    
    config['theta_rebin_range_cosebis'] = options[option_section,  'theta_rebin_range_cosebis'].astype(str) 

    config['out_path'] = options.get_string(option_section, 'out_path') #"/share/data1/maxivonw/kcap_out/treecorr"

    config['clean'] = options.get_bool(option_section, 'clean')

    return config

def execute(block, config):

    if  config['vd']:
        n_sourcefield = block[config['in_name'], 'nbSourceFields_vd']
    else:
        n_sourcefield = block[config['in_name'], 'nbSourceFields']

    if int(config['n_tomo']*len(config['patches'])) != int(n_sourcefield):
        print('WARNING: Input number of source fields ({0}) does not match the number of fields simulated ({1})'.format(config['n_tomo'], n_sourcefield))

    print('Found {0} different source fields...'.format(n_sourcefield))

    runid = '_{0}_sample{1}'.format(block[config['in_name'], 'runTag'], block[config['in_name'], 'counter'])

    for patch in config['patches']:
        print('Running treecorr for {0} patch(es)...'.format(patch))
        for i in range(config['n_tomo']):
            for j in range(i+1):
                tic = time.time()
                spc.run(['./doall_calc2pt.sh',
                    '-d', config['in_path'],
                    '-m', 'XI', 
                    '-u', "{0} {1} {1}".format(config['mockname'], runid),
                    '-o', config['out_path'],
                    '-p', patch,
                    '-t', config['theta_bin_range'],
                    '-n', config['tomo_bin_limits'],
                    '-i', str(i+1),
                    '-j', str(j+1)
                    ], cwd=config['build_path'], text = True)
                toc = time.time()
                print('XI of bins {0}-{1} in patch {2} took {3} seconds'.format(i+1, j+1, patch, round(toc-tic, 2)))


    if config['theta_rebin']:
        if 'ALL' not in config['patches']:
            print('Combining corr. function patches...')
            tic = time.time()
            for i in range(config['n_tomo']):
                for j in range(i+1):
                    spc.run(['./doall_calc2pt.sh',
                        '-d', config['in_path'],
                        '-m', 'COMBINEXI',
                        '-u', "{0} {1} {1}".format(config['mockname'], runid),
                        '-o', config['out_path'],
                        '-t', config['theta_bin_range'],
                        '-n', config['tomo_bin_limits'],
                        '-i', str(i+1),
                        '-j', str(j+1)
                        ], cwd=config['build_path'], text = True)
            toc = time.time()
            print('Combining patches took {0} seconds'.format(round(toc-tic, 2)))


        print('Rebinning corr. functions for {0} patch(es)...'.format(patch))
        tic = time.time()
        for i in range(config['n_tomo']):
                for j in range(i+1):
                    spc.run(['./doall_calc2pt.sh',
                        '-m', 'REBINXI',
                        '-u', "{0} {1} {1}".format(config['mockname'], runid),
                        '-o', config['out_path'],
                        '-p', 'ALL',
                        '-t', config['theta_bin_range'],
                        '-n', config['tomo_bin_limits'],
                        '-i', str(i+1),
                        '-j', str(j+1),
                        '-x', config['theta_rebin_range']
                        ], cwd=config['build_path'], text = True)
        toc = time.time()
        print('Rebining patch {0} took {1} seconds'.format(patch, round(toc-tic, 2)))

    # print('Getting cosmic shear bandpowers, Pkk, for {0} patch(es)...'.format(patch))
    # tic = time.time()
    # for i in range(config['n_tomo']):
    #         for j in range(i+1):
    #             spc.run(['./doall_calc2pt.sh',
    #                 '-m', 'Pkk',
    #                 '-u', "{0} {1} {1}".format(config['mockname'], runid),
    #                 '-o', config['out_path'],
    #                 '-p', 'ALL',
    #                 '-t', config['theta_bin_range'],
    #                 '-n', config['tomo_bin_limits'],
    #                 '-i', str(i+1),
    #                 '-j', str(j+1)
    #                 ], cwd=config['build_path'], text = True)
    # toc = time.time()
    # print('Getting cosmic shear bandpowers, Pkk, for patch {0} took {1} seconds'.format(patch, round(toc-tic, 2)))

    # print('Getting cosmic shear COSEBIS for {0} patch(es)...'.format(patch))
    # tic = time.time()
    # for i in range(config['n_tomo']):
    #         for j in range(i+1):
    #             spc.run(['./doall_calc2pt.sh',
    #                 '-m', 'COSEBIS',
    #                 '-u', "{0} {1} {1}".format(config['mockname'], runid),
    #                 '-o', config['out_path'],
    #                 '-p', 'ALL',
    #                 '-t', config['theta_bin_range'],
    #                 '-n', config['tomo_bin_limits'],
    #                 '-i', str(i+1),
    #                 '-j', str(j+1)
    #                 ], cwd=config['build_path'], text = True)
    # toc = time.time()
    # print('Getting cosmic shear COSEBIS for patch {0} took {1} seconds'.format(patch, round(toc-tic, 2)))

    if config['clean']:
        print('Deleting input files from SALMO for sample {0} of run {1}...'.format(block[config['in_name'], 'counter'], block[config['in_name'], 'runTag']))
        filelist = glob.glob('{0}_{1}_sample{2}*.fits'.format(block[config['in_name'], 'outPrefix'], block[config['in_name'], 'runTag'], block[config['in_name'], 'counter']))
        for file in filelist:
            try:
                os.remove(file)
            except:
                print('Error while deleting file: ', file)
    return 0

def cleanup(config):
    pass