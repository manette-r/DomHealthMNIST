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

    return vars(parser.parse_args())

def create_data_file(path, open_str):
    if os.path.exists(path):
        os.remove(path)
    return open(path, open_str)

def write_label_file_header(label_file):
    df = pd.DataFrame.from_dict({}, orient='index',
                                columns=['subject', 'digit', 'angle', 'disease',
                                         'disease_time', 'gender',
                                         'time_age', 'location'])
    df.to_csv(label_file, index=False)

def save_data(data_file, label_file, rotated_MNIST, label_dict):

    # save rotated MNIST
    if os.path.exists(data_file) : 
        df_past = pd.read_csv(data_file)
        df_past.columns = df_past.columns.astype(int)

        rotated_MNIST = pd.concat([df_past,rotated_MNIST], ignore_index=True)
        rotated_MNIST = rotated_MNIST.reset_index(drop=True)

    rotated_MNIST.to_csv(data_file, index=False)

    df = pd.DataFrame.from_dict(label_dict, orient='index',
                                columns=['subject', 'digit', 'angle', 'disease',
                                         'disease_time', 'gender',
                                         'time_age', 'location'])

    # save labels
    df.to_csv(label_file, index=False, header=False)

# Transform an image with grayscale to rgb, every pixel will host a tuple instead of a single value 
    # - img : ndarray 
def img_to_rgb(img):
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

    digit_mod = {'3': num_3, '6': num_6}
    sick_prob = 0.5  # probability of instance being sick
    sample_index = 0
    subject_index = 0
    label_dict = {}
    gender = 0
    dim_line = opt['dim_image'] #3888 because 36*36*3

    # 20 time points
    time_age = np.arange(0, 20)
    time_points = np.arange(-9, 11)

    # accumulate digits
    rotated_MNIST_columns = list(range(dim_line))
    rotated_MNIST = pd.DataFrame(columns=rotated_MNIST_columns) 
    path_data_file = os.path.join(destination, data_file_name)
    if os.path.exists(path_data_file):
        os.remove(path_data_file)

    label_file = create_data_file(os.path.join(destination, labels_file_name), "a")
    write_label_file_header(label_file) 

    for digit in digit_mod.keys():
        print("Creating instances of digit {}".format(digit))

        # read in the files
        data_path = os.path.join(source, digit)
        files = glob.glob('{}/*.jpg'.format(data_path))

        # Assume requested files less than total available!
        for i in range(digit_mod[digit]):

            original_image = plt.imread(files[i])
            original_image_pad = np.pad(original_image, ((4, 4), (4, 4)), 'constant')

            # decide on sickness
            sick_var = np.random.binomial(1, sick_prob)

            # irrelevant location
            loc_var = np.random.binomial(1, 0.5)

            # introduce some noise
            rotations = np.random.normal(0, 2, len(time_points))

            # define rotation for each instance
            if sick_var:

                # simulate disease effect
                rotations += 45 * sigmoid(time_points)
            else:

                # baseline rotation for non-sick
                rotations += 5

            if digit == '3':
                gender = 0
            else:
                gender = 1

            for idx, rotation in enumerate(rotations):

                # rotate an instance
                img = ndimage.rotate(original_image_pad, angle=rotation, reshape=False)

                # diagonal shift the image
                img = ndimage.shift(img, shift=idx/10)

                #changing into rgb if it is not already 
                if dim_line == 3888 : 
                    img = img_to_rgb(img)

                if sick_var == 1:
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
                          rotated_MNIST, label_dict)
                
                rotated_MNIST = pd.DataFrame(columns=rotated_MNIST_columns)
                label_dict = {}
        
        save_data(path_data_file, label_file,
                  rotated_MNIST, label_dict)
        
        rotated_MNIST = pd.DataFrame(columns=rotated_MNIST_columns)
        label_dict = {}

    print('Saved! Number of samples: {}'.format(sample_index))
