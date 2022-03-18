from os import EX_SOFTWARE
from cosmosis.datablock import option_section, names, BlockError
import numpy as np
from glass.matter import lognormal_matter
from glass.lensing import lognormal_convergence, shear_from_convergence, shear_catalog, shapes_from_lensing
from glass.random import generate_random_fields
from glass.observations import visibility_from_file
from glass.galaxies import positions_from_galaxies, gaussian_galaxies_shapes
from glass.util.random import uniform_over_sphere
import healpy as hp
from interpcl import interpcl

def setup(options):
    config = {}
    
    config['nside'] = options.get_int(option_section, "nside")

    if not np.log2(float(config['nside'])).is_integer():
        raise Exception('nside must be a power of 2.')

    config['out']       = options.get_string(option_section, "out") #shear, galaxy, galaxy_shear
    config['out_mode']  = options.get_string(option_section, "out_mode") #pcl, catalogue
    try:
        config["reg_gaussian_cls"] = options.get_bool(option_section, "reg_gaussian_cls")
    except:
        config["reg_gaussian_cls"] = False

    if ('shear' in config['out'] and 'galaxy' in config['out']) or 'galaxy_shear' in config['out']:
        config["fields"] = np.array(['matter', 'convergence'])
    elif 'galaxy' in config['out']:
        config["fields"] = np.array(['matter'])
    elif 'shear' in config['out']:
        config["fields"] = np.array(['convergence'])

    config['random_fields'] = {}

    try:
        visibility_file = options.get_string(option_section, "visibility_file")
        config['mask'] =  visibility_from_file(visibility_file, nside=config['nside'])
        print('Reading mask file: {0}'.format(visibility_file))
    except:
        visibility_file = None
        config['mask'] = None
        print('No visibility file given. Full sky will be computed.')

    config['n_density']  = options.get_double(option_section, "n_density") #0.01 for test
    config['sigma_e']    = options.get_double(option_section, "sigma_e")

    return config

def execute(block, config):
    if 'matter' in config['fields']:
        nbin = block['galaxy_cl', 'nbin']
        config['random_fields']['matter'] = lognormal_matter(nbin)
    if 'convergence' in config['fields']:
        redshift_bins = block['nz_source', 'zlim']
        config['random_fields']['convergence'] =  lognormal_convergence(redshift_bins)

    nbin_g = block['galaxy_cl', 'nbin']
    nbin_s = block['shear_cl', 'nbin']
    ell_sh = block['shear_cl', 'ell']
    ell_g  = block['galaxy_cl', 'ell']
    ell_gs = block['galaxy_shear_cl', 'ell']

    lmax = int(np.max([np.max(ell_sh), np.max(ell_g), np.max(ell_gs)]))

    cls = {}
    if 'galaxy' in config['out']:
        for i in range(0, nbin_g):
            for j in range(0, i+1):
                cl = interpcl(ell_g, block['galaxy_cl', 'bin_{0}_{1}'.format(i+1,j+1)], lmax=lmax, dipole=True, monopole=False)
                cls[tuple(['matter[{0}]'.format(i), 'matter[{0}]'.format(j)])] = cl
    if 'shear' in config['out']:
        for i in range(0, nbin_s):
            for j in range(0, i+1):
                cl = interpcl(ell_sh, block['shear_cl', 'bin_{0}_{1}'.format(i+1,j+1)], lmax=lmax, dipole=True, monopole=False)
                cls[tuple(['convergence[{0}]'.format(i), 'convergence[{0}]'.format(j)])] = cl
    if 'galaxy_shear' in config['out']:
        for i in range(0, nbin_g):
            for j in range(0, nbin_s):
                cl = interpcl(ell_gs, block['galaxy_shear_cl', 'bin_{0}_{1}'.format(i+1,j+1)], lmax=lmax, dipole=True, monopole=False)
                cls[tuple(['matter[{0}]'.format(i), 'convergence[{0}]'.format(j)])] = cl         

#    cls = {
#        ('matter[0]', 'matter[0]'): ...,
#        ('convergence[0]', 'matter[0]'): ...,
#    }

    print('Generating random fields ...')
    if config['reg_gaussian_cls']:
        fields, cls_new = generate_random_fields(config['nside'], cls, config['random_fields'], allow_missing_cls=False, return_cls=True)
    else:
        fields = generate_random_fields(config['nside'], cls, config['random_fields'], allow_missing_cls=False, return_cls=False)

    if 'shear' in config['out'] or 'galaxy_shear' in config['out']:
        convergence = fields['convergence']
        print('Calculating shear field ...')
        shear = shear_from_convergence(convergence) #slow

    if 'galaxy' in config['out'] or 'galaxy_shear' in config['out']:
        matter = fields['matter']
        galaxies = matter #add bias here

    if config['mask'] != None:
        convergence =  np.multiply(convergence,  config['mask'])
        galaxies = np.multiply(galaxies,  config['mask'])

    #Integrate over the shells x source distributions
    #Sigma_eps dependent uncertainty term

    print('Generating outputs ...')
    if config['out_mode'] == 'pcl':

        if 'galaxy' in config['out']:
            for i in range(0, nbin_g):
                for j in range(0, i+1):
                    pcl = hp.sphtfunc.anafast(galaxies[i], galaxies[j], lmax = 4*config['nside'])
                    block['galaxy_pcl', 'bin_{0}_{1}'.format(i+1,j+1)] = pcl
                    block['galaxy_pcl', "is_auto"] = block["galaxy_cl", "is_auto"]
                    block['galaxy_pcl', "sample_a"] = block["galaxy_cl", "is_auto"]
                    block['galaxy_pcl', "sample_b"] = block["galaxy_cl", "sample_b"]
                    block['galaxy_pcl', "nbin"] = block["galaxy_cl", "nbin"]
                    block['galaxy_pcl', "nbin_a"] = block["galaxy_cl", "nbin_a"]
                    block['galaxy_pcl', "nbin_b"] = block["galaxy_cl", "nbin_b"]
                    block['galaxy_pcl', "sep_name"] = block["galaxy_cl", "sep_name"]
                    block['galaxy_pcl', "save_name"] = block["galaxy_cl", "save_name"]
                    block['galaxy_pcl', "ell"] = np.arange(0, 4*config['nside'] + 1)
        
        if 'shear' in config['out']:
            for i in range(0, nbin_s):
                for j in range(0, i+1):
                    pcl = hp.sphtfunc.anafast(shear[i], shear[j], lmax=4*config['nside'])
                    block['shear_pcl', 'bin_{0}_{1}'.format(i+1,j+1)] = pcl
                    block['shear_pcl', "is_auto"] = block["shear_cl", "is_auto"]
                    block['shear_pcl', "sample_a"] = block["shear_cl", "is_auto"]
                    block['shear_pcl', "sample_b"] = block["shear_cl", "sample_b"]
                    block['shear_pcl', "nbin"] = block["shear_cl", "nbin"]
                    block['shear_pcl', "nbin_a"] = block["shear_cl", "nbin_a"]
                    block['shear_pcl', "nbin_b"] = block["shear_cl", "nbin_b"]
                    block['shear_pcl', "sep_name"] = block["shear_cl", "sep_name"]
                    block['shear_pcl', "save_name"] = block["shear_cl", "save_name"]
                    block['shear_pcl', "ell"] = np.arange(0, 4*config['nside'] + 1)
        
        if 'galaxy_shear' in config['out']:
            for i in range(0, nbin_g):
                for j in range(0, nbin_s):
                    pcl = hp.sphtfunc.anafast(galaxies[i], shear[j], lmax=4*config['nside'])
                    block['galaxy_shear_pcl', 'bin_{0}_{1}'.format(i+1,j+1)] = pcl
                    block['galaxy_shear_pcl', 'bin_{0}_{1}'.format(i+1,j+1)] = pcl
                    block['galaxy_shear_pcl', "is_auto"] = block["galaxy_shear_cl", "is_auto"]
                    block['galaxy_shear_pcl', "sample_a"] = block["galaxy_shear_cl", "is_auto"]
                    block['galaxy_shear_pcl', "sample_b"] = block["galaxy_shear_cl", "sample_b"]
                    block['galaxy_shear_pcl', "nbin_a"] = block["galaxy_shear_cl", "nbin_a"]
                    block['galaxy_shear_pcl', "nbin_b"] = block["galaxy_shear_cl", "nbin_b"]
                    block['galaxy_shear_pcl', "sep_name"] = block["galaxy_shear_cl", "sep_name"]
                    block['galaxy_shear_pcl', "save_name"] = block["galaxy_shear_cl", "save_name"]
                    block['galaxy_shear_pcl', "ell"] = np.arange(0, 4*config['nside'] + 1)

    elif config['out_mode'] == 'catalogue':
        if 'matter' in config["fields"]:
            positions = positions_from_galaxies(galaxies, mean_density = config['n_density'], visibility = config['mask']) #ra, dec, bin columns
        else:
            if config['mask'] is not None:
                fsky = np.mean(config['mask'], axis=-1)
            else:
                fsky = 1
            mean_density = np.array(config['n_density'], copy=False, subok=True)
            expected = mean_density*60**6/(100*np.pi)
            visible = fsky*expected
            observed = np.random.poisson(visible, size=nbin_s)
            total = observed.sum()
            catalog = np.empty(total, dtype=[('bin', int), ('ra', float), ('dec', float)])
            count = 0
            for i, n in np.ndenumerate(observed):
                if len(i) == 1:
                    i = i[0]
                while n > 0:
                    # how many galaxies to sample at once
                    k = 10000
                    # propose positions
                    ra, dec = uniform_over_sphere(k)
                    # get acceptance probability
                    ipix = hp.ang2pix(config['nside'], ra, dec, lonlat=True)
                    if config['mask'] is not None:
                        p = config['mask'][ipix]
                    else:
                        p = 1
                    # accept or reject values
                    a = np.nonzero(np.random.uniform(0, 1, size=k) < p)[0][:n]
                    # number of galaxies sampled
                    k = len(a)
                    # add to catalog
                    page = catalog[count:count+k]
                    page['bin'] = i
                    page['ra'] = ra[a]
                    page['dec'] = dec[a]
                    # galaxies added
                    count += k
                    n -= k
            positions = catalog

        shapes_int = gaussian_galaxies_shapes(positions, sigma_e = config['sigma_e']) #e1, e2

        shear_cat = shear_catalog(positions, shear, convergence) #g1, g2

        shape_cat = shapes_from_lensing(shapes_int, shear_cat) #e1, e2
        
        block['catalogue', 'bin'] = [x[0] for x in positions]
        block['catalogue', 'ra'] = [x[1] for x in positions]
        block['catalogue', 'dec'] = [x[2] for x in positions]
        block['catalogue', 'e1'] = [x[0] for x in shape_cat]
        block['catalogue', 'e2'] = [x[1] for x in shape_cat]
        block['catalogue', 'g1'] = [x[0] for x in shear_cat]
        block['catalogue', 'g2'] = [x[1] for x in shear_cat]

    if config['reg_gaussian_cls']:
        print('Saving Regularised Gaussian Cls ...')
        if 'galaxy' in config['out']:
            block["galaxy_cl_reg_gaussian", "is_auto"] = block["galaxy_cl", "is_auto"]
            block["galaxy_cl_reg_gaussian", "sample_a"] = block["galaxy_cl", "is_auto"]
            block["galaxy_cl_reg_gaussian", "sample_b"] = block["galaxy_cl", "sample_b"]
            block["galaxy_cl_reg_gaussian", "nbin"] = block["galaxy_cl", "nbin"]
            block["galaxy_cl_reg_gaussian", "nbin_a"] = block["galaxy_cl", "nbin_a"]
            block["galaxy_cl_reg_gaussian", "nbin_b"] = block["galaxy_cl", "nbin_b"]
            block["galaxy_cl_reg_gaussian", "sep_name"] = block["galaxy_cl", "sep_name"]
            block["galaxy_cl_reg_gaussian", "save_name"] = block["galaxy_cl", "save_name"]
            block["galaxy_cl_reg_gaussian", "ell"] = block["galaxy_cl", "ell"]
            for i in range(0, nbin_g):
                for j in range(0, i+1):
                    block['galaxy_cl_reg_gaussian', 'bin_{0}_{1}'.format(i+1,j+1)] = cls_new['reg_gaussian_cls'][tuple(['matter[{0}]'.format(j), 'matter[{0}]'.format(i)])]
        if 'shear' in config['out']:
            block["shear_cl_reg_gaussian", "is_auto"] = block["shear_cl", "is_auto"]
            block["shear_cl_reg_gaussian", "sample_a"] = block["shear_cl", "is_auto"]
            block["shear_cl_reg_gaussian", "sample_b"] = block["shear_cl", "sample_b"]
            block["shear_cl_reg_gaussian", "nbin"] = block["shear_cl", "nbin"]
            block["shear_cl_reg_gaussian", "nbin_a"] = block["shear_cl", "nbin_a"]
            block["shear_cl_reg_gaussian", "nbin_b"] = block["shear_cl", "nbin_b"]
            block["shear_cl_reg_gaussian", "sep_name"] = block["shear_cl", "sep_name"]
            block["shear_cl_reg_gaussian", "save_name"] = block["shear_cl", "save_name"]
            block["shear_cl_reg_gaussian", "ell"] = block["shear_cl", "ell"]
            for i in range(0, nbin_g):
                for j in range(0, i+1):
                    block['shear_cl_reg_gaussian', 'bin_{0}_{1}'.format(i+1,j+1)] = cls_new['reg_gaussian_cls'][tuple(['convergence[{0}]'.format(j), 'convergence[{0}]'.format(i)])]
        if 'galaxy_shear' in config['out']:
            block["galaxy_shear_cl_reg_gaussian", "is_auto"] = block["galaxy_shear_cl", "is_auto"]
            block["galaxy_shear_cl_reg_gaussian", "sample_a"] = block["galaxy_shear_cl", "is_auto"]
            block["galaxy_shear_cl_reg_gaussian", "sample_b"] = block["galaxy_shear_cl", "sample_b"]
            block["galaxy_shear_cl_reg_gaussian", "nbin_a"] = block["galaxy_shear_cl", "nbin_a"]
            block["galaxy_shear_cl_reg_gaussian", "nbin_b"] = block["galaxy_shear_cl", "nbin_b"]
            block["galaxy_shear_cl_reg_gaussian", "sep_name"] = block["galaxy_shear_cl", "sep_name"]
            block["galaxy_shear_cl_reg_gaussian", "save_name"] = block["galaxy_shear_cl", "save_name"]
            block["galaxy_shear_cl_reg_gaussian", "ell"] = block["galaxy_shear_cl", "ell"]
            for i in range(0, nbin_g):
                for j in range(0, nbin_s):
                    block['galaxy_shear_cl_reg_gaussian', 'bin_{0}_{1}'.format(i+1,j+1)] = cls_new['reg_gaussian_cls'][tuple(['matter[{0}]'.format(i), 'convergence[{0}]'.format(j)])]

    return 0

def cleanup(config):
    pass