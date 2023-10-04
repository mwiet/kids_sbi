import numpy as np

from cosmosis.datablock import option_section, names
from cosmosis.datablock.cosmosis_py import errors

def setup(options):
    input_section_name = options.get_string(option_section, "input_section_name")
    input_theory_name = options.get_string(option_section, "input_data_name")
    output_section_name = options.get_string(option_section, "output_section_name")
    
    data_path = options.get_string(option_section, "data_file")
    inv_cov_path = options.get_string(option_section, "inv_cov_file")
    
    data = np.loadtxt(data_path)
    inv_cov = np.loadtxt(inv_cov_path)
    
    return input_section_name, input_data_name, output_section_name, data, inv_cov

def execute(block, config):
    input_section_name, input_data_name, output_section_name, data, inv_cov = config

    block[output_section_name, 'theory'] = block[input_section_name, input_theory_name]
    block[output_section_name, 'data'] = data
    block[output_seciton_name, 'inv_covariance'] = inv_cov

    return 0

def clean(config):
    pass