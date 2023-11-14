import numpy as np

from cosmosis.datablock import option_section, names
from cosmosis.datablock.cosmosis_py import errors

def setup(options):
    like_name = options.get_string(option_section, "like_name")
    input_section_name = options.get_string(option_section, "input_section_name", default="likelihood")
    input_theory_name = options.get_string(option_section, "input_theory_name", default="theory")
    data_vector_path = options.get_string(option_section, "data_vector", default="data_vector")
    cov_path = options.get_string(option_section, "cov")
    
    vary_s8_cov = options.get_bool(option_section, "vary_s8_cov", default=False)
    s8_cov_paths = options.get_string(option_section, "s8_cov_paths", default="").split(' ')
    stepsize = options.get_double(option_section, "s8_cov_stepsize", default=0.01)
    fiducial_s8 = options.get_double(option_section, "fiducial_s8", default=0.759)
    
    assert len(s8_cov_paths) == 0 or len(s8_cov_paths) == 4, "Only one s8 cov path allowed"
    
    if vary_s8_cov == True:
        base_cov_trace = np.trace(np.loadtxt(cov_path))
        s8_covs = np.array([
            np.trace(np.loadtxt(s8_cov_paths[0]))/base_cov_trace,
            np.trace(np.loadtxt(s8_cov_paths[1]))/base_cov_trace,
            np.trace(np.loadtxt(s8_cov_paths[2]))/base_cov_trace,
            np.trace(np.loadtxt(s8_cov_paths[3]))/base_cov_trace])
        stencil_coeffs = np.array([1/12, -2/3, 2/3, -1/12])
        s8_cov_grad = np.dot(stencil_coeffs, s8_covs)/stepsize
        s8_cov_const = 1 - s8_cov_grad*fiducial_s8
    else:
        s8_cov_grad = 1.0
        s8_cov_const = 0.0
    
    
    
    data_vector = np.loadtxt(data_vector_path)
    cov = np.loadtxt(cov_path)
    inv_cov = np.linalg.inv(cov)
    return like_name, input_section_name, input_theory_name, data_vector, cov, inv_cov, vary_s8_cov, s8_cov_grad, s8_cov_const

def execute(block, config):
    like_name, input_section_name, input_theory_name, data_vector, cov, inv_cov, vary_s8_cov, s8_cov_grad, s8_cov_const = config
    s8_val = block[names.cosmological_parameters, "s_8"]
    # s8_val = block[cosmo, "s8"]
    
    if vary_s8_cov == True:
        cov = cov*(s8_cov_grad*s8_val + s8_cov_const)
        inv_cov = np.linalg.inv(cov)
        
    mu = block[input_section_name, input_theory_name]
        
    r = data_vector - mu
    chi2 = float(r @ inv_cov @ r)
    ln_like = -0.5*chi2

    block[names.data_vector, like_name+"_CHI2"] = chi2
    block[names.likelihoods, like_name+"_LIKE"] = ln_like

    return 0

def clean(config):
    pass
