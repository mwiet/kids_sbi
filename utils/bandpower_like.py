import numpy as np

from cosmosis.datablock import option_section, names
from cosmosis.datablock.cosmosis_py import errors

def setup(options):
    like_name = options.get_string(option_section, "like_name")
    input_section_name = options.get_string(option_section, "input_section_name", default="likelihood")
    input_theory_name = options.get_string(option_section, "input_theory_name", default="theory")
    data_vector_path = options.get_string(option_section, "data_vector", default="data_vector")
    cov_path = options.get_string(option_section, "cov")
    do_noise = options.get_bool(option_section, "do_noise", default=True)
    data_vector = np.loadtxt(data_vector_path)
    cov = np.loadtxt(cov_path)
    inv_cov = np.linalg.inv(cov)
    return like_name, input_section_name, input_theory_name, data_vector, cov, inv_cov, do_noise

def execute(block, config):
    like_name, input_section_name, input_theory_name, data_vector, cov, inv_cov, do_noise = config
    if do_noise is True:
        mu = np.random.multivariate_normal(block[input_section_name, input_theory_name], cov)
    else:
        mu = block[input_section_name, input_theory_name]
        
    r = data_vector - mu
    chi2 = float(r @ inv_cov @ r)
    ln_like = -0.5*chi2

    block[names.data_vector, like_name+"_CHI2"] = chi2
    block[names.likelihoods, like_name+"_LIKE"] = ln_like

    return 0

def clean(config):
    pass
