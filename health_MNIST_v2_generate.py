import os
import glob
import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid
from scipy import ndimage
import matplotlib.pyplot as plt
import argparse


"""
Code to generate the Health MNIST data.

This code manipulates the original MNIST images as described in the L-VAE paper.
"""

def parse_arguments():
    """
    Parse the command line arguments
    :return: parsed arguments object (2 arguments)
    """

    parser = argparse.ArgumentParser(description='Enter configuration for generating data')
    parser.add_argument('--source', type=str, default='./trainingSet', help='Path to MNIST image root directory')
    parser.add_argument('--destination', type=str, default='./data', help='Path to save the generated dataset')
    parser.add_argument('--num_3', type=int, default=50, help='Number of unique instances for digit 3')
    parser.add_argument('--num_6', type=int, default=50, help='Number of unique instances for digit 6')
    parser.add_argument('--data_file_name', type=str, default='health_MNIST_data.csv',
                        help='File name of generated data')
    parser.add_argument('--labels_file_name', type=str, default='health_MNIST_label.csv',
                        help='File name of generated labels')
    parser.add_argument('--dim_image', type=int, default=3888, help='Number of pixels in an image')
    parser.add_argument('--nb_halted', type=int, default=0, help='Number of patient intervention with halted result')
    parser.add_argument('--nb_slow', type=int, default=0, help='Number of patient intervention with slow result')
    parser.add_argument('--nb_back', type=int, default=0, help='Number of patient intervention with going back result')
    parser.add_argument('--after_b_intervention', type=str, default='H', help='Trajectory of the patient after a going back intervention, H=Halted, S=Slow, D=Disease')


    return vars(parser.parse_args())

def create_data_file(path, open_str):
    if os.path.exists(path):
        os.remove(path)
    return open(path, open_str)

def write_label_file_header(label_file, intervention=False):
    if intervention : 
        cols = ['subject', 'digit', 'angle', 'disease', 'disease_time', 'gender', 'time_age', 'location', 'intervention']
    else : 
        cols = ['subject', 'digit', 'angle', 'disease', 'disease_time', 'gender', 'time_age', 'location']

    df = pd.DataFrame.from_dict({}, orient='index', columns=cols)
    df.to_csv(label_file, index=False)

def save_data(data_file, label_file, rotated_MNIST, label_dict, intervention = False):

    # save rotated MNIST
    if os.path.exists(data_file) : 
        df_past = pd.read_csv(data_file)
        df_past.columns = df_past.columns.astype(int)

        rotated_MNIST = pd.concat([df_past,rotated_MNIST], ignore_index=True)
        rotated_MNIST = rotated_MNIST.reset_index(drop=True)

    rotated_MNIST.to_csv(data_file, index=False)

    if intervention : 
        cols = ['subject', 'digit', 'angle', 'disease', 'disease_time', 'gender', 'time_age', 'location', 'intervention']
    else : 
        cols = ['subject', 'digit', 'angle', 'disease', 'disease_time', 'gender', 'time_age', 'location']

    df = pd.DataFrame.from_dict(label_dict, orient='index', columns=cols)

    # save labels
    df.to_csv(label_file, index=False, header=False)


def img_to_rgb(img):
    '''
    Transform an image with grayscale to rgb, every pixel will host a tuple instead of a single value 
    
    :param img: ndarray 
    '''
    #img is in grayscale 
    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)
    else:
        if not (len(img.shape) == 3 and img.shape[2] == 3):
            raise Exception("Unexpected shape ",img.shape)

    return img 

if __name__ == '__main__':
    opt = parse_arguments()
    for key in opt.keys():
        print('{:s}: {:s}'.format(key, str(opt[key])))
    locals().update(opt)



    digit_mod = {'3': opt['num_3'], '6': opt['num_6']}
    sick_prob = 0.5  # probability of instance being sick
    sample_index = 0
    subject_index = 0
    label_dict = {}
    gender = 0
    dim_line = opt['dim_image'] #3888 because 36*36*3

    #verify that intervention numbers are coherents
    halted_patient = opt['nb_halted']
    slow_patient = opt['nb_slow']
    going_back_patient = opt['nb_back']
    tt_intervention = halted_patient+slow_patient#+going_back_patient not available for now
    if tt_intervention > sick_prob*(opt['num_3']+opt['num_6']) :
        raise print('The amount of medical intervention ('+str(tt_intervention)+') is over than the number of patients having a disease.')
    
    list_interventions = ['H']*halted_patient + ['S']*slow_patient #+ ['B']*going_back_patient not available for now
    np.random.shuffle(list_interventions)
    idx_intervention = 0

    # Trajectory after Going back intervention 
    traj_back = opt['after_b_intervention'].upper()
    if not (traj_back == 'H' or traj_back == 'S' or traj_back == 'D') : 
        raise print('Please enter a correct trajectory type for going back intervention : H, S or D')
    
    # 20 time points
    time_age = np.arange(0, 20)
    time_points = np.arange(-9, 11)

    # accumulate digits
    rotated_MNIST_columns = list(range(dim_line))
    rotated_MNIST = pd.DataFrame(columns=rotated_MNIST_columns) 
    path_data_file = os.path.join(opt['destination'], opt['data_file_name'])
    if os.path.exists(path_data_file):
        os.remove(path_data_file)

    label_file = create_data_file(os.path.join(opt['destination'], opt['labels_file_name']), "a")
    write_label_file_header(label_file, tt_intervention>0) 

    for digit in digit_mod.keys():
        print("Creating instances of digit {}".format(digit))

        # read in the files
        data_path = os.path.join(opt['source'], digit)
        files = glob.glob('{}/*.jpg'.format(data_path))

        # Assume requested files less than total available!
        for i in range(digit_mod[digit]):

            original_image = plt.imread(files[i])
            original_image_pad = np.pad(original_image, ((4, 4), (4, 4)), 'constant')

            if digit == '3':
                gender = 0
            else:
                gender = 1

            # decide on sickness
            sick_var = np.random.binomial(1, sick_prob)

            # irrelevant location
            loc_var = np.random.binomial(1, 0.5)

            # introduce some noise
            rotations = np.random.normal(0, 2, len(time_points))

            #list with intervention tuples when it happens 
            interventions = [None]*len(time_age)
            #creating a list because it can change with intervention 
            sick_var_list = [sick_var]*len(time_age)

            # define rotation for each instance
            if sick_var:

                #if interventions 
                if idx_intervention < len(list_interventions) : 

                    intervention_type = list_interventions[idx_intervention]

                    #greater than disease_time = 0 because there is no sickness before it (margin at 3)
                    buffer_interventions = np.arange(12,19)
                    np.random.shuffle(buffer_interventions)
                    timepoint_intervention = buffer_interventions[0]

                    for i_tp in np.arange(timepoint_intervention,20):
                        interventions[i_tp] = (timepoint_intervention, intervention_type)

                    # simulate disease effect
                    rotations_before_intervention = 45 * sigmoid(time_points[:timepoint_intervention])

                    if intervention_type == 'H' : 
                        rotations_after_intervention = [0]*len(time_points[timepoint_intervention:])
                        sick_var_list[timepoint_intervention:] = [0]*(len(sick_var_list)-timepoint_intervention)
                    elif intervention_type == 'S':
                          rotations_after_intervention = 25 * sigmoid(time_points[timepoint_intervention:])
                    
                    rotations += (list(rotations_before_intervention)+list(rotations_after_intervention))
                    idx_intervention+=1

                else : 
                    # simulate disease effect
                    rotations += 45 * sigmoid(time_points)                    
            else:

                # baseline rotation for non-sick
                rotations += 5


            for idx, rotation in enumerate(rotations):

                # rotate an instance
                img = ndimage.rotate(original_image_pad, angle=rotation, reshape=False)

                # diagonal shift the image
                img = ndimage.shift(img, shift=idx/10)

                #changing into rgb if it is not already 
                if dim_line == 3888 : 
                    img = img_to_rgb(img)

                if tt_intervention > 0:
                    label_dict[sample_index] =\
                        [subject_index, digit, rotation, sick_var_list[idx], time_points[idx], gender, time_age[idx], loc_var, interventions[idx]]
                elif sick_var == 1: 
                    label_dict[sample_index] =\
                        [subject_index, digit, rotation, sick_var, time_points[idx], gender, time_age[idx], loc_var]
                elif sick_var == 0:
                    label_dict[sample_index] = [subject_index, digit, rotation, sick_var, 'nan', gender,
                                                time_age[idx], loc_var]

                rotated_MNIST.loc[len(rotated_MNIST)] = img.flatten()
                sample_index += 1

            subject_index += 1

            if i%200 == 199:
                print("Instance no {} for digit {}".format(i+1, digit))

                save_data(path_data_file, label_file,
                          rotated_MNIST, label_dict, tt_intervention>0)
                
                rotated_MNIST = pd.DataFrame(columns=rotated_MNIST_columns)
                label_dict = {}
        
        save_data(path_data_file, label_file,
                  rotated_MNIST, label_dict, tt_intervention>0)
        
        rotated_MNIST = pd.DataFrame(columns=rotated_MNIST_columns)
        label_dict = {}

    print('Saved! Number of samples: {}'.format(sample_index))
