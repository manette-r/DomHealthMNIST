import argparse
import numpy as np
import pandas as pd
from copy import deepcopy
import domHealth_MNIST_generate as h_MNIST_domain
import os
import ast



#For example : --source=./result --destination=./result  --labels_file_name=domHealthMNIST_various_label.csv --data_file_name=domHealthMNIST_various_data.csv --target=dataset --list_timepoint_to_remove="[[0, 2, 4, 6, 8, 10, 12, 14, 16, 18], [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],[10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]" 
def parse_arguments():
    """
    Parse the command line arguments

    :return: parsed arguments object (2 arguments)
    """

    parser = argparse.ArgumentParser(description='Enter configuration for generating data')
    parser.add_argument('--source', type=str, default='./result', help='Path to Health-MNIST image root directory')
    parser.add_argument('--destination', type=str, default='./result', help='Path to save the generated dataset')
    parser.add_argument('--data_file_name', type=str, default='health_MNIST_data.csv',
                        help='File name of data', required=True)
    parser.add_argument('--data_file_name_result', type=str, default='domHealth_MNIST_data_irregular.csv',
                        help='Result file name of data')
    parser.add_argument('--mask_file_name', type=str, default=None,
                        help='CSV file name of a previous Health MNIST generated data mask')
    parser.add_argument('--mask_file_name_result', type=str, default='domHealth_MNIST_mask_irregular.csv',
                        help='CSV file name of a previous Health MNIST generated data mask')
    parser.add_argument('--labels_file_name', type=str, default='health_MNIST_label.csv',
                        help='File name of labels', required=True)
    parser.add_argument('--labels_file_name_result', type=str, default='domHealth_MNIST_label_irregular.csv',
                        help='Result file name of labels')
    parser.add_argument('--target', type=str, default='all',
                        help='String all means it will have the same effect on all the dataset, else precise a column name to distinguish different effect you want to apply')
    parser.add_argument('--col_id', type=str, default='subject',
                        help='Name of the column containing IDs')
    parser.add_argument('--col_time', type=str, default='time_age',
                        help='Name of the column containing timepoint/date')
    parser.add_argument('--list_timepoint_to_remove', type=str, default='[]',
                        help='If target is \'all\' then it is a list of integers corresponding to the position of timepoints you want to remove. Else it is a list of multiple list of timepoint to remove depending on the dataset')
    parser.add_argument('--nb_timepoint_to_remove', type=str, default='[0]', 
                        help='List of number of timepoints to remove randomly in a patient\'s longitudinal data.')
    parser.add_argument('--seed', type=int, default=None, 
                        help='Number of seed to use in numpy random method.')
    return vars(parser.parse_args())



def remove_timepoint(data_f, label_f, mask_f, cols, list_id = [], list_tp = [], nb_id_removing = 0, nb_tp_removing = 0, seed = None):
    '''
    Removing timepoint for a selection of patients

    :param data_f: dataframe with the same index as label_f
    :param label_f: dataframe with id and the time/date column of the follow-up
    :param mask_f: dataframe with the same index as label_f, can also be None 
    :param cols: list of string for column names we need [id,date] 
    :param list_id: list id we must keep 
    :param list_tp: list of index timepoint we must remove sorted by importance, example [0,2] means for all the patients we must removed their first and third timepoint (if there are at least 2 timepoints left)
    :param nb_id_removing: number of id to remove randomly, if < 1 we consider it is a pourcentage of element to remove, if it is >= 1 we consider it is the number of elemnt to remove 
    :param nb_tp_removing: number of timepoint to remove randomly, if < 1 we consider it is a pourcentage of element to remove, if it is >= 1 we consider it is the number of elemnt to remove 
    :param seed: seed to shuffle np.random method  

    :return: label_targets_final, data_targets_final, mask_targets_final
    '''

    if list_id == [] and list_tp == [] and nb_id_removing == 0 and nb_tp_removing == 0 :
        print('\nNo change is asked in the parameters of remove_timepoint method.')
        return data_f, label_f, mask_f
    
    #selection of ids 
    target_ids = select_ids(deepcopy(label_f[cols[0]].unique()), list_id, nb_id_removing, seed)
    
    #selecting index in both dataframes
    label_targets = label_f[label_f[cols[0]].isin(target_ids)]
    data_targets = data_f.loc[label_targets.index]
    mask_targets = mask_f.loc[label_targets.index] 

    #no timepoint removing 
    if list_tp == [] and nb_tp_removing == 0 : 
        return label_targets, data_targets, mask_targets
    
    #change and store one target id at a time 
    label_targets_list = []
    data_targets_list = []
    mask_targets_list = []

    for target_id in target_ids:
        rows_label_target, rows_data_target, rows_mask_target = get_filtered_rows(data_targets, label_targets, mask_targets, target_id, cols, list_tp, nb_tp_removing,seed)
        label_targets_list.append(rows_label_target)
        data_targets_list.append(rows_data_target)
        mask_targets_list.append(rows_mask_target)

    # Final concatenation outside the loop
    label_targets_final = pd.concat(label_targets_list)
    data_targets_final = pd.concat(data_targets_list)
    mask_targets_final = pd.concat(mask_targets_list)
    
    return label_targets_final, data_targets_final, mask_targets_final



def get_filtered_rows(data_targets, label_targets, mask_targets, target_id, cols, list_tp, nb_tp_removing,col_time,seed=None):
    '''
    Changing longitudinal structure for a patient (by removing follow-ups)

    :param data_targets: dataframe
    :param label_targets: dataframe, must have columns cols 
    :param mask_targets: dataframes 
    :param cols: list of string for column names we need [id,date] 
    :param list_tp: list of integers (index timepoint we must remove sorted by importance)
    :param nb_tp_removing: number of timepoint to remove randomly, if < 1 we consider it is a pourcentage of element to remove, if it is >= 1 we consider it is the number of element to remove 
    :param seed: seed to shuffle np.random method

    :return: rows_label_target, rows_data_target, rows_mask_target
    '''
    rows_label_target = deepcopy(label_targets[label_targets[cols[0]] == target_id])
    rows_data_target = deepcopy(data_targets.loc[rows_label_target.index])
    rows_mask_target = deepcopy(mask_targets.loc[rows_label_target.index])

    nb_tt_tp = len(rows_label_target)

    #when to stop to let at least 2 timepoints
    i_stop = nb_tt_tp-2
    
    if nb_tt_tp < 2 : 
        raise Exception("Every patient must have at least 2 follow-ups, it is not the case for ",target_id," patient.")
    
    #we can remove at least 1 timepoint 
    elif nb_tt_tp != 2 : 

        if list_tp != [] :
            #sort by time column
            rows_label_target.sort_values(by=cols[1])
            #keep tp index that are correct 
            list_tp_buffer = []

            #verify that list_tp values are correct 
            max_tp = max(list_tp)
            if max_tp >= nb_tt_tp : 
                for i in range(len(list_tp)):
                    if list_tp[i] < nb_tt_tp :
                        list_tp_buffer.append(list_tp[i])
            else : 
                list_tp_buffer = deepcopy(list_tp)

            #we remove the extra tp of the index list
            if len(list_tp_buffer) > i_stop : 
                list_tp_buffer = list_tp_buffer[:i_stop-1]#because it starts at 0 
            
            #remove timepoint from both dataframes  
            rows_label_target = rows_label_target.drop(index=rows_label_target.iloc[list_tp_buffer].index)
            rows_data_target = rows_data_target.drop(index=rows_data_target.iloc[list_tp_buffer].index)
            rows_mask_target = rows_mask_target.drop(index=rows_mask_target.iloc[list_tp_buffer].index)
        
        #we remove randomly 
        else : 
            tp_list = rows_label_target[cols[1]].to_numpy()
            np.random.seed(seed)
            np.random.shuffle(tp_list)

            #we take a pourcentage 
            if nb_tp_removing <1:
                nb_tp_removing = int(nb_tp_removing*len(tp_list))

            else : # >= 1 
                nb_tp_removing = int(nb_tp_removing)

            #we remove the extra  tp of the index list
            nb_tp_removing_buffer = nb_tp_removing
            if nb_tp_removing_buffer > i_stop : 
                nb_tp_removing_buffer = i_stop

            #remove timepoints randomly 
            tp_list_to_remove = tp_list[:nb_tp_removing_buffer]

            boolean_to_keep = ~rows_label_target[cols[1]].isin(tp_list_to_remove)
            rows_label_target = rows_label_target[boolean_to_keep]
            rows_data_target = rows_data_target[boolean_to_keep]
            rows_mask_target = rows_mask_target[boolean_to_keep]

    return rows_label_target, rows_data_target, rows_mask_target

             


def select_ids(all_id, list_element, nb_element_removing, seed):
    '''
    Return id list from a list or by randomly picking

    :param all_id: list of all id
    :param list_element: list of id we want to keep 
    :param nb_element_removing: number of element to remove randomly, if < 1 we consider it is a pourcentage of element to remove, if it is >= 1 we consider it is the number of elemnt to remove 
    :param seed: seed to shuffle np.random method

    :return: Return id list from a list or by randomly picking
    '''
    #No selection, all patients are taken 
    if list_element == [] and nb_element_removing == 0 : 
        return all_id
    #A selected list of patients is asked 
    elif list_element != [] : 

        #make sure ids exist in the dataset 
        list_element_exist = []
        for element in list_element : 
            if element in all_id :
                list_element_exist.append(element)
            else : 
                print("Element ", element, " does not exist in the original dataset. ")

        return list_element_exist
    
    #Randomly selection according to nb_element_removing 
    elif nb_element_removing != 0 :

        #shuffle all patients 
        np.random.seed(seed=seed)
        np.random.shuffle(all_id)
        
        #we take a pourcentage 
        if nb_element_removing <1:
            nb_ids = int(nb_element_removing*len(all_id))

        else : # >= 1 
            nb_ids = int(nb_element_removing)

            #make sure we have enough patients (at least one must stay)
            if nb_ids >= len(all_id) : 
                raise Exception('\n Too much element must be removed, there is only ', len(all_id), ' elements in the dataset.')
        
        #selection 
        return all_id[:nb_ids]
    

def verify_list(idx_list_tp_remove_str, number_list_tp_remove_str):
    '''
    Verify if the timepoints lists to remove are in the good format
    '''
    try : 
        idx_list_tp_remove = ast.literal_eval(idx_list_tp_remove_str)

    except Exception as e :
        raise Exception("Problem of str translation for timepoints list to remove args (list_timepoint_to_remove)", e)
    if not isinstance(idx_list_tp_remove, list) : 
        raise Exception('Argument list_timepoint_to_remove is not a list.')
    try : 
        nb_list_tp_remove = ast.literal_eval(number_list_tp_remove_str)
    except Exception as e :
        raise Exception("Problem of str translation for timepoints list to remove args (nb_timepoint_to_remove)", e)
    if not isinstance(nb_list_tp_remove, list) : 
        raise Exception('Argument nb_timepoint_to_remove is not a list.')
    return idx_list_tp_remove, nb_list_tp_remove


def main(opt):
    """
    Main method.

    :param opt: parsed arguments
    :type opt: dictionnary
    """
    source = opt['source']
    data_file_name = opt['data_file_name']
    labels_file_name = opt['labels_file_name']
    mask_file_name = opt['mask_file_name']

    if os.path.exists(source+'/'+data_file_name) and os.path.exists(source+'/'+labels_file_name) and os.path.exists(source+'/'+mask_file_name):
        print("Files found.")
    else:
        raise Exception("One of the file not found :", source+'/'+labels_file_name," or ", source+'/'+data_file_name, " or ", source+'/'+mask_file_name)


    labels_file = pd.read_csv(source+'/'+labels_file_name)
    data_file = pd.read_csv(source+'/'+data_file_name)
    mask_file = pd.read_csv(source+'/'+mask_file_name)

    if len(data_file) != len(labels_file):
        raise Exception('\nLabels and data files don\'t have the same number of rows.')

    col_target = opt['target']

    if not(opt['col_id'] in labels_file) or not(opt['col_time'] in labels_file): 
        raise Exception('One of the column between ', opt['col_id'],' and ',opt['col_time'], ' does not exist in the labels file.') 
    cols_id_time = [opt['col_id'], opt['col_time']]

    #transform str list into a list type 
    idx_list_tp_remove, nb_list_tp_remove = verify_list(opt['list_timepoint_to_remove'], opt['nb_timepoint_to_remove'])

    # either choose idx_list_tp_remove or nb_list_tp_remove but not both 
    if len(idx_list_tp_remove) != 0 and len(nb_list_tp_remove) != 0:
        if len(nb_list_tp_remove) == 1 and nb_list_tp_remove[0] != 0 :
            raise Exception('Choose between list_timepoint_to_remove or nb_timepoint_to_remove.')

    #if removing is apply on all the dataset 
    if col_target == 'all' : 
        #we only give the first element of nb_list_tp_remove
        final_label, final_data, final_mask = remove_timepoint(data_file, labels_file,mask_file, cols_id_time, list_tp = idx_list_tp_remove, nb_tp_removing=nb_list_tp_remove[0], seed=opt['seed'])

    #if removing is apply by using a specific column in the dataset
    else : 
        if not(col_target in labels_file) : 
            raise Exception('The column ', col_target, 'does not exist in the labels file.') 
        #all ids in a list 
        targets_list = labels_file[col_target].unique()

        datasets_result_label, datasets_result_data, datasets_result_mask = [], [], []

        #Verify if we have the good amount of tp_remove 
        nb_targets = len(targets_list)
        if idx_list_tp_remove != [] and len(idx_list_tp_remove) < nb_targets :
            raise Exception('Not enought values, '+str(nb_targets)+' targets but only '+str(len(idx_list_tp_remove))+' for list_timepoint_to_remove.')
        elif idx_list_tp_remove == [] and  len(nb_list_tp_remove) < nb_targets : 
            raise Exception('Not enought values, '+str(nb_targets)+' targets but only '+str(len(nb_list_tp_remove))+' for nb_timepoint_to_remove.')
        
        #Process one target (=patient) at a time 
        for i in range(nb_targets) : 

            target = targets_list[i]
            dataset_label = deepcopy(labels_file[labels_file[col_target] == target])
            dataset_data = deepcopy(data_file.loc[dataset_label.index])  
            dataset_mask = deepcopy(mask_file.loc[dataset_label.index]) 

            # print("\nFor target ", target, " with list_tp_remove : ",str(idx_list_tp_remove[i]), ', it will impact ',len(dataset_data),' rows.')     
            if idx_list_tp_remove != [] :
                result_label, result_data, result_mask = remove_timepoint(dataset_data, dataset_label, dataset_mask, cols_id_time, list_tp = idx_list_tp_remove[i], seed=opt['seed'])
            else : 
                result_label, result_data, result_mask = remove_timepoint(dataset_data, dataset_label, dataset_mask, cols_id_time, nb_tp_removing=nb_list_tp_remove[i], seed=opt['seed'])

            datasets_result_label.append(result_label)
            datasets_result_data.append(result_data)
            datasets_result_mask.append(result_mask)

        #concat and save the global dataset
        final_label = pd.concat(datasets_result_label)
        final_data = pd.concat(datasets_result_data)
        final_mask = pd.concat(datasets_result_mask)

        final_label = final_label.reset_index(drop=True)
        final_data = final_data.reset_index(drop=True)
        final_mask = final_mask.reset_index(drop=True)

    h_MNIST_domain.save_data(source+'/'+opt['data_file_name_result'], source+'/'+opt['labels_file_name_result'], deepcopy(final_data), deepcopy(final_label), deepcopy(final_mask), source+'/'+opt['mask_file_name_result'])


if __name__ == '__main__':
    opt = parse_arguments()
    for key in opt.keys():
        print('{:s}: {:s}'.format(key, str(opt[key])))
    locals().update(opt)

    main(opt)
