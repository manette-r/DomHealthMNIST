import numpy as np
import pandas as pd

from parse_model_args_dHMNIST import ModelArgs
import health_MNIST_v2_generate as hMNIST
import domHealth_MNIST_generate as dHMNIST
import change_longitudinal_structure as chg_long_struc

"""
Code to generate the DomHealth MNIST data.

This code is based on the health MNIST dataset. 
It simulates multiple domains, as well as temporal irregularity and heterogeneity.
"""


if __name__ == "__main__":
    """
    Run command: python domHealth_MNIST.py --f=path_to_config-file.txt 
    """

    # Create parser and set variables
    opt = ModelArgs().parse_options()
    for key in opt.keys():
        print('{:s}: {:s}'.format(key, str(opt[key])))
    locals().update(opt)

    base_name = opt['base_file_name']
    opt['source'] = opt['destination']

    # Create a variation of health MNIST with medical intervention 
    opt_hMNIST = {}
    opt_hMNIST_keys = ['source_MNIST', 'destination', 'num_3', 'num_6', 'dim_image', 'nb_halted', 'nb_slow']
    for key in opt.keys():
        if key in opt_hMNIST_keys : 
            opt_hMNIST[key] = opt.get(key)

    opt_hMNIST['data_file_name'] = 'hMNIST_v2_'+str(base_name)+'_data.csv'
    opt_hMNIST['labels_file_name'] = 'hMNIST_v2_'+str(base_name)+'_label.csv'
    hMNIST.main(opt_hMNIST)

    final_file_name = ['domHealthMNIST_'+str(base_name)+'_data.csv', 'domHealthMNIST_'+str(base_name)+'_label.csv', 'domHealthMNIST_'+str(base_name)+'_mask.csv']

    # Creating opt for domHealthMNIST 
    opt_dHMNIST = {}
    opt_dHMNIST_keys = ['source', 'destination', 'dim_image', 'domain_distrib_file_name', 'is_various_domain', 'domain_shifters', 'name_dataset_separation', 'nb_dataset_separation']
    for key in opt.keys():
        if key in opt_dHMNIST_keys : 
            opt_dHMNIST[key] = opt.get(key)

    opt_dHMNIST['data_file_name'] = 'hMNIST_v2_'+str(base_name)+'_data.csv'
    opt_dHMNIST['labels_file_name'] = 'hMNIST_v2_'+str(base_name)+'_label.csv'
    opt_dHMNIST['mask_file_name']= None 
    # Change in the longitudinal structure 
    if opt['change_long_structure']:
    
        domain_file_name = ['domain_'+str(base_name)+'_data.csv', 'domain_'+str(base_name)+'_label.csv', 'domain_'+str(base_name)+'_mask.csv']
        opt_dHMNIST['data_file_name_result'] = domain_file_name[0]
        opt_dHMNIST['labels_file_name_result'] = domain_file_name[1]
        opt_dHMNIST['mask_file_name_result'] = domain_file_name[2]

        # Apply domain transformations 
        dHMNIST.main(opt_dHMNIST)


        # Creating opt for change_long_structure 
        opt_struc = {}
        opt_struc_keys = ['source', 'destination', 'target', 'col_id', 'col_time', 'list_timepoint_to_remove', 'nb_timepoint_to_remove', 'seed']
        for key in opt.keys():
            if key in opt_struc_keys : 
                opt_struc[key] = opt.get(key)

        opt_struc['data_file_name'] = domain_file_name[0]
        opt_struc['labels_file_name'] = domain_file_name[1]
        opt_struc['mask_file_name'] = domain_file_name[2]


        opt_struc['data_file_name_result'] = final_file_name[0]
        opt_struc['labels_file_name_result'] = final_file_name[1]
        opt_struc['mask_file_name_result'] = final_file_name[2]

        # Removing follows-up
        chg_long_struc.main(opt_struc)
    else : 

        opt_dHMNIST['data_file_name_result'] = final_file_name[0]
        opt_dHMNIST['labels_file_name_result'] = final_file_name[1]
        opt_dHMNIST['mask_file_name_result'] = final_file_name[2]

        # Apply domain transformations 
        dHMNIST.main(opt_dHMNIST)

