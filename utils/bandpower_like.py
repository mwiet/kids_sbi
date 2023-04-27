import numpy as np

from cosmosis.datablock import option_section, names
from cosmosis.datablock.cosmosis_py import errors

def setup(options):
    like_name = options.get_string(option_section, "like_name")
    input_section_name = options.get_string(option_section, "input_section_name", default="likelihood")
    intput_theory_name = options.get_string(option_section, "input_theory_name", default="theory")
    data_vector = np.loadtxt(options.get_string(option_section, "data_vector", default="data_vector"))
    inv_covariance = np.loadtxt(options.get_string(option_section, "inv_covariance", default="inv_covariance"))
    return like_name, input_section_name, intput_theory_name, data_vector, inv_covariance

def execute(block, config):
    like_name, input_section_name, input_theory_name, data_vector, inv_covariance = config

    mu = block[input_section_name, intput_theory_name]
    r = data_vector - mu

    chi2 = float(r @ inv_covariance @ r)
    ln_like = -0.5*chi2

    block[names.data_vector, like_name+"_CHI2"] = chi2
    block[names.likelihoods, like_name+"_LIKE"] = ln_like

    return 0

def clean(config):
    pass
