import os
import glob
import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import argparse
from copy import deepcopy
import cv2
import seaborn as sns
import math
import ast

#Parsing arguments from a command line
#For example : domHealth_MNIST_generate.py --source=./result --destination=./result --data_file_name=health_MNIST_data.csv --labels_file_name=health_MNIST_label.csv --domain_shifters=[[],[('scale',0.5)],[('scale',0.7)],[('scale',1.5)]] --is_various_domain=False
def parse_arguments():
    """
    Parse the command line arguments
    :return: parsed arguments object (2 arguments)
    """

    parser = argparse.ArgumentParser(description='Enter configuration for generating data')
    parser.add_argument('--source', type=str, default='./result', help='Path to Health-MNIST image root directory')
    parser.add_argument('--destination', type=str, default='./result', help='Path to save the generated dataset')
    parser.add_argument('--data_file_name', type=str, default='health_MNIST_data.csv',
                        help='File name of Health MNIST generated data', required=True)
    parser.add_argument('--data_file_name_result', type=str, default='domHealth_MNIST_data.csv',
                        help='File name of Health MNIST generated data with domain variations')
    parser.add_argument('--mask_file_name', type=str, default=None,
                        help='File name of a previous Health MNIST generated data mask with domain variations')
    parser.add_argument('--mask_file_name_result', type=str, default='domHealth_MNIST_mask.csv',
                        help='File name of Health MNIST generated data mask with domain variations')
    parser.add_argument('--labels_file_name', type=str, default='health_MNIST_label.csv',
                        help='File name of Health MNIST generated labels', required=True)
    parser.add_argument('--labels_file_name_result', type=str, default='domHealth_MNIST_label.csv',
                        help='File name of Health MNIST generated labels with domain informations')
    parser.add_argument('--domain_distrib_file_name', type=str, default='domain_distrib_health_MNIST.png',
                        help='File name to save plot of domain distributions for health MNIST')
    parser.add_argument('--is_various_domain', type=bool, default=False,
                        help='Give a command value if digit instances from one subject can be associated to various domains, else don\'t specify it in the command line : if there is only one domain for a subject')
    parser.add_argument('--domain_shifters', type=str, default="[[],[('scale',0.5),('colour','FF0000')],[('scale',0.7),('colour','0D00FF')],[('scale',1.5),('colour','00FF04')]]",
                        help='string of a list of tuples lists (string, list/value)')#les couleurs rouge, bleu et vert en hexadecimale
    parser.add_argument('--name_dataset_separation', type=str, default='dataset',
                        help='Give a new column name to split the global dataset into small ones, if is_various_domain=False dataset will be split by its domain value, else is it randomly split')
    parser.add_argument('--nb_dataset_separation', type=int, default=3,
                        help='If is_various_domain=False dataset will be split by number of total domain values, else is it randomly split in your number choice')
    parser.add_argument('--dim_image', type=int, default=3888, help='Number of pixels in an image')
    return vars(parser.parse_args())


def split_array(len_array_target, nb_domains):

    #have the same number for each domain 
    len_domain = int(len_array_target/nb_domains)
    len_domains = [len_domain]*nb_domains
    
    #if each domain cannot have the same amount, spread it 
    rest = len_array_target-len_domain*nb_domains
    for i in range(rest):
        len_domains[i] = len_domains[i]+1

    #convert into index 
    index_split = []
    start = 0
    end = 0
    for len_domain in len_domains : 
        end = start+len_domain
        index_split.append((start,end-1))
        start = end

    return index_split


# Spliting data into domains according to instructions given 
    # - is_various_domain : boolean True if digit instances from one subject can be associated to various domains, False if there is only one domain for a subject
    # - nb_domains : number of domains we want to split (int)
    # - labels_file: dictionnary containing all relative informations to digit
def split_into_domains(is_various_domain,nb_domains, labels_file):

    labels_file['domain'] = None 

    #split by timepoints : it means that one subject can be assigned to various domains 
    if is_various_domain : 
        #Shuffle timepoint with a copy of subject and disease_time
        # buffer_labels = deepcopy(labels_file[['subject','disease_time']]) 
        # buffer_labels = buffer_labels.sample(frac=1).reset_index(drop=True)

        # domains_split = np.array_split(buffer_labels, nb_domains)
        # for i,d in enumerate(domains_split) : 
        #     for _,row in d.iterrows() : 
        #         subject = row['subject']
        #         disease_time = row['disease_time']
        #         #find the exact timepoint 
        #         conditions = (labels_file['subject'] == subject) & (labels_file['disease_time'] == disease_time)
        #         labels_file.loc[conditions, 'domain'] = i
        
        index_split = split_array(len(labels_file), nb_domains)
        index_list = labels_file.index.tolist()
        np.random.seed(42)
        np.random.shuffle(index_list)

        for i in range(len(index_split)):
            start, end = index_split[i]
            for j in range(start, (end+1)):
                index_target = index_list[j]
                labels_file.loc[index_target,'domain'] = i

    #Split by subject 
    else : 
        #Shuffle timepoint with a copy of subject and disease_time
        np.random.seed(42)  
        buffer_subjects = labels_file['subject'].unique()
        np.random.shuffle(buffer_subjects)

        # domains_split = np.array_split(buffer_subjects, nb_domains)
        # for i,d in enumerate(domains_split) :
            
        #     labels_file.loc[labels_file['subject'].isin(d),'domain'] = i
        
        index_split = split_array(len(buffer_subjects), nb_domains)
       
        for i in range(len(index_split)):
            start, end = index_split[i]
            for j in range(start, (end+1)):
                subject_target = buffer_subjects[j]
                labels_file.loc[labels_file['subject'] == subject_target,'domain'] = i

                # print(labels_file[labels_file['subject'] == subject_target,'domain'])


    print(labels_file['domain'].value_counts())

def inverted_colour(image): 
    img = image.astype(np.uint8) 
    img = 255 -img 
    return img 


def stretched_digit(image,new_shape, dim_line=3888, rescale = 0.5):
    '''
    Stretched a digit, it changes its form and its colour intensity.

    :param image: image in (36x36) or (36x36x3)
    :param new_shape: tuple with height and width
    :param dim_line: 1296 or 3888
    '''

    if dim_line == 3888 :
        h, w, c = image.shape
    else : 
        h,w = image.shape

    new_h, new_w = new_shape 


    scaled = cv2.resize(image.astype("uint8"), (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mini_scaled = scale_digit(scaled, rescale, dim_line)
    result = np.zeros_like(image)

    # offsets pour centrer
    y_offset = (h - new_h) // 2
    x_offset = (w - new_w) // 2

    # si l'image est trop grande → crop centré
    if new_h > h or new_w > w:
        y_start = max((new_h - h) // 2, 0)
        x_start = max((new_w - w) // 2, 0)

        mini_scaled = mini_scaled[y_start:y_start + h, x_start:x_start + w]

        y_offset = 0
        x_offset = 0

    # coordonnées finales
    y1 = y_offset
    y2 = y_offset + mini_scaled.shape[0]
    x1 = x_offset
    x2 = x_offset + mini_scaled.shape[1]

    result[y1:y2, x1:x2] = mini_scaled

    return result 


# Changing the digit scale but keeping the initial image dimension 
# image :  ndarray shape x,x,x
# scale_factor : a float 
def scale_digit(image, scale_factor, dim_line = 3888):
    # Get the original dimensions

    if dim_line == 3888 :
        h, w, c = image.shape
    else : 
        h,w = image.shape

    new_h, new_w = max(1, int(h * scale_factor)), max(1, int(w * scale_factor))

    # Resize the image (zoom or shrink)
    scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create a blank canvas with the original size
    result = np.zeros_like(image)

    # Compute offset to center the digit
    y_offset = max((h - new_h) // 2, 0)
    x_offset = max((w - new_w) // 2, 0)

    # Get the region to paste
    y1 = y_offset
    y2 = y_offset + scaled.shape[0]
    x1 = x_offset
    x2 = x_offset + scaled.shape[1]

    # Handle cropping if the scaled image is larger than the canvas
    scaled_cropped = scaled[:h, :w]  # just in case

    # Place scaled image at the center
    result[y1:y2, x1:x2] = scaled_cropped[:y2 - y1, :x2 - x1]

    return result

#Change a hexadedcimal colour to a decimal colour 
    #- hex_colour : str of 6 carac (without #)
def hex_to_decimal_colour(hex_colour):
    return tuple(int(hex_colour[i:i+2], 16) for i in (0, 2, 4))

# Verify if the instructions and their values are in the good format 
    # - domain_shifters : a list of element, an element is a list of tuple (str, value) the value depends on the instructions 
def verify_instructions(domain_shifters_str): 

    #understand contents str into list 
    #exemple [[], [('scale',0.5), ('colour','FF0000')], [('scale',0.7), ('colour','0D00FF')], [('scale',1.5), ('colour','00FF04')]]
    try : 
        domain_shifters = ast.literal_eval(domain_shifters_str)
    except Exception as e :
        raise Exception("Problem of str translation for domain_shifters args", e)
    
    
    #verify contents 
    for i in range(len(domain_shifters)):
        instructions = domain_shifters[i]

        for instruction, value in instructions :
            if instruction=='colour':
                #value has to be a string of length 6 juste with letter and number 
                if not isinstance(value, str):
                    raise Exception("The colour ",value," is not a string.")
                if len(str(value)) != 6 :
                    raise Exception("The length of colour ",value," is not good, it has to be in hexadecimal, so length 6.")
            elif instruction=='blur' or instruction=='scale':
                #float > 0 
                ex = Exception("For ",instruction, "instruction the value has to be higher than 0 but you gave value=",value)
                try : 
                    if not (value > 0) : 
                        raise ex
                #in case its not a number 
                except : 
                    raise ex
            elif instruction=='stretched':
                #tuple (new_h, new_w)
                ex = Exception("For ",instruction, "instructionit must be a tuple of two integers.",value)
                try : 
                    buffer_h, buffer_w = value 
                    if not (buffer_h > 0) or not (buffer_w > 0): 
                        raise ex
                except : 
                    raise ex
            elif instruction=='inverted':
                #pas de condition sur i_value 
                ex = Exception(value)#ce sert a rien 
            elif instruction=='missing':
                #0 <= float < 100  
                ex = Exception("For ",instruction, "instruction the value has to be a percentage of missing pixels in range [0, 100] but you gave value=",value)
                try : 
                    if not (0 <= value and value <100) : 
                        raise ex
                #in case its not a number 
                except : 
                    raise ex
            else : 
                raise Exception('\nDomain shifter instruction ',instruction,' doesnt exist. /!\\')
            
    return domain_shifters

#Change image colour by applying a filter 
def apply_filter(pixels, base_rgb):

    # Convert pixels to a NumPy array for efficient processing
    pixels = np.array(pixels)
    # Normalize the pixel values to the range [0, 1]
    normalized_pixels = pixels / 255.0
    # Apply the base RGB filter
    filtered_pixels = normalized_pixels * (np.array(base_rgb) / 255.0)
    # Ensuring that they are within the valid range [0, 1]
    filtered_pixels = np.clip(filtered_pixels, 0, 1)

    # Convert back to the range [0, 255] and to uint8 type
    filtered_pixels = (filtered_pixels * 255).astype(np.uint8)
    # plt.imshow(filtered_pixels)
    # plt.show()

    return filtered_pixels

#Erase some values pixels depending on the given value missing_frac
#Input : 
    # - img :image en 36x36 ou 36x36x3
    # - missing_frac : pourcentage of pixels to erase 
    # - dim_line : 1296 ou 3888 selon les dimensions de l'image 
#Return : 
    # the masked image and its mask 
def missing_pixels(img, missing_frac, dim_line):

    if dim_line == 3888 : 
        h, w, c = img.shape
    else : 
        h,w = img.shape

    # Create mask of shape (H, W)
    mask = np.random.choice([0, 1], size=(h, w), p=[missing_frac, 1-missing_frac]) #jai remis ce quil y avait dans l'original

    # Expand mask for RGB channels
    if dim_line == 3888 : 
        mask = np.repeat(mask[:, :, np.newaxis], c, axis=2)

    # 0 implies missing, 1 implies observed
    masked_data = np.multiply(img.copy(), mask)
    masked_data = masked_data.astype(np.uint8)

    return masked_data, mask

# Applied changes according to domain value 
    #- domain_shifters : list of tuples lists (string, list/value)
    #- labels_file : dataframe with relative informations for each digit instances as its domain 
    #- health_MNIST : digit instances (images) on which the shifters are applied 
    #- mask_df : dataframe with two columns (label_idx and mask) 
def shift_instances(domain_shifters, labels_file, health_MNIST,mask_df, dim_line = 3888):

    #for each domain, there are instructions of how to change data to create different domains 
    for i in range(len(domain_shifters)) : 

        #take only rows corresponding to the actual domain
        index_rows = labels_file[labels_file['domain'] == i].index
        
        actual_domain_instances = health_MNIST.loc[index_rows]


        domain_instructions = domain_shifters[i]

        for idx in index_rows :
            try : 
                img_line = np.array(actual_domain_instances.loc[idx].to_list())

                if dim_line == 3888:
                    img = img_line.reshape((36, 36, 3))  
                else : 
                    img = img_line.reshape((36, 36))  
            except : 
                raise Exception('\n ',idx,'. /!\\')
                        
            #see all instructions tuples for one domain
            for instruction,i_value in domain_instructions :   
                
                if instruction=='colour':
                    img = apply_filter(img, hex_to_decimal_colour(i_value))
                elif instruction=='blur':
                    img = gaussian_filter(img, sigma=i_value)
                elif instruction=='scale':
                    img = scale_digit(img, i_value, dim_line)
                elif instruction=='stretched':
                    img = stretched_digit(img,i_value, dim_line)
                elif instruction=='inverted':
                    img = inverted_colour(img)
                elif instruction=='missing':
                    missing_frac = i_value/100
                    img, mask = missing_pixels(img, missing_frac, dim_line)
                    mask_df.loc[idx] = np.reshape(mask.astype(np.uint8), (dim_line,))
                else : 
                    raise Exception('\nDomain shifter instruction ',instruction,' doesnt exist. /!\\')
                
            #make sure every image pixel has the same type 
            if img.dtype not in [np.uint8, np.uint16]:
                img = img.astype(np.uint8)  
            health_MNIST.loc[idx] = np.reshape(img, (dim_line,))

            
# Delete rows with the columns names in their rows 
    # - df : dataframe 
    # - element : column name targeted 
def df_delete_rows(df, element):
    index_list = []
    if element in df.columns.tolist():

        for i in range(len(df)):
            if df[element].iloc[i] == element : 
                index_list.append(i)
        print(len(index_list)," rows to delete.")
        df = df.drop(index_list).reset_index(drop=True)
        df[element].apply(lambda x : print("data") if element == x else x)
    return df

# Translate a str (list of numbers) into a ndarray 
    # - s : string 
def strList_to_numberList(s):

    # Delete space and [ ]
    s = s.strip()[1:-1]

    numbers = s.split(',')
    
    result = []
    for nb in numbers:
        
        if nb != '':
            try:
                # Convert into an int or float (uint8 and float32 work with openCV)
                nb_buffer = np.uint8(float(nb)) if '.' in nb else np.uint8(nb)
                result.append(nb_buffer)
            except ValueError as ve:
                print(ve)
    
    return np.array(result)

# Almost same method as in health_MNIST_v2_generate.py but rotated_MNIST and mask are now a dataframes  
    # - data_file : path 
    # - label_file : path 
    # - rotated_MNIST : dataframe 
    # - labels : datframe  
def save_data(data_file, label_file, rotated_MNIST, labels, mask_df, mask_path = None):
    if len(rotated_MNIST) != len(labels):
        print("BAD MATCH rotated_MNIST.shape ",len(rotated_MNIST)," labels ", len(labels))
        print(rotated_MNIST)
    else : 
        print("Good match rotated_MNIST.shape ",len(rotated_MNIST)," labels ", len(labels))

    # #data saving 
    rotated_MNIST.to_csv(data_file,index=False)

    #mask saving 
    if mask_path != None : 
        mask_df.to_csv(mask_path, index=False)

    #we don't keep the file column 
    if 'file' in labels.columns :
        labels = labels.drop(columns=['file'])
    # save labels
    labels.to_csv(label_file, index=False)   



# Plot all the timepoints ordered by the age for a subject 
    # - health_mnist : dataframe 
    # - labels_file : dict
    # - subject : subject_id 
def plot_timepoints_per_subject(health_mnist, labels_file, subject, dim_line = 3888):

    #filter labels and data 
    subject_rows = labels_file[labels_file['subject']==subject]
    idx_list = subject_rows.index
    subject_rows = pd.concat([subject_rows,health_mnist.loc[idx_list]], axis=1)
    nb_timepoints = len(subject_rows)

    #order by date with disease_time col
    subject_rows = subject_rows.sort_values(by='time_age', ascending=True)

    # Plotting every timepoints 

    #organizing cols and rows depending on timepoints number (pair or odd)   
    cols = int(nb_timepoints/2) if int(nb_timepoints/2)<=4 else 4 #4 colonnes car le max timepoint est 20
    rows = math.ceil(nb_timepoints/cols)

    fig, axs = plt.subplots(nrows=rows,ncols=cols,constrained_layout=True)
    

    #we don't want to see axes except if they are called 
    for ax in fig.axes:
        ax.set_visible(False)

    #title figure 
    if subject_rows['disease'].iloc[0] == 0 :
        disease_subtitle = "is not sick."
    elif subject_rows['disease'].iloc[0] == 1 : 
        disease_subtitle = "is sick."
    else :
        disease_subtitle = "is weird." 

    title_subject = 'Subject '+str(subject)+', with '+str(nb_timepoints)+' timepoints, '+disease_subtitle
    fig.suptitle(title_subject)
    
    
    nrows, ncols = 0,0
    for i in range(nb_timepoints):
        # print('ncols=',ncols, ' et nrows=', nrows)
        img = np.array(subject_rows.iloc[i].to_list())
    
        if dim_line == 3888 :
            img = img.reshape((36, 36, 3))
        else : 
            img = img.reshape(36,36)
        
        axs[nrows, ncols].imshow(img)
        axs[nrows, ncols].set_title('age ='+str(subject_rows['time_age'].iloc[i]), fontsize=8)
        axs[nrows, ncols].set_visible(True)

        if ncols == cols-1 :  
            nrows += 1
            ncols = 0
        else :
            ncols += 1
    
    plt.show()

#Plot a line of images with a different changement 
    # - path_data : path of the data file made from health_MNIST_v2_generate.py
def plot_domain_examples(path_data, dim_line = 3888):

    data_file = pd.read_csv(path_data)

    #just take one image 
    img_line = np.array(data_file.loc[0].to_list())

    if dim_line == 3888:
        img = img_line.reshape((36, 36, 3))
    else : 
        img = img_line.reshape((36, 36))

    #store images with every changement possible 
    images = []
    titles = []
    images.append(img)
    titles.append('Original')
    images.append(apply_filter(img, hex_to_decimal_colour('FF0000')))
    titles.append('Colour')
    images.append(gaussian_filter(img, 1.2))
    titles.append('Blur')
    images.append(scale_digit(img, 0.7, dim_line))
    titles.append('Scale')
    images.append(stretched_digit(img,(100,36), dim_line))
    titles.append('Stretched')
    images.append(inverted_colour(img))
    titles.append('Inverted colour')
    masked_data, mask = missing_pixels(img, 25/100, dim_line)
    images.append(masked_data)
    titles.append('Missing pixels')

    nb_images = len(images)

    fig, axs = plt.subplots(nrows=1,ncols=nb_images,constrained_layout=True)


    ncols = 0
    for i in range(nb_images):
        
        axs[ncols].imshow(images[i])
        axs[ncols].set_title(titles[i], fontsize=8)

        ncols +=1

    plt.show()

# Separate the global dataset into small ones, if is_various_domain=False dataset will be split by domain values, else is it randomly split 
    # - col_name : name of the column to distinguish datasets
    # - labels_file : dataframe 
    # - is_various_domain = boolean given in the args command 
    # - nb_datasets : number of smaller datasets we will create 
    # - seed : seed for random method 
def dataset_separation(col_name, labels_file, is_various_domain, nb_datasets, seed = 42):

    if is_various_domain : 

        #verify if col_name already exists 
        if not(col_name in labels_file.columns) :
            
            np.random.seed(seed)  
            buffer_subjects = labels_file['subject'].unique()
            np.random.shuffle(buffer_subjects)

            index_split = split_array(len(buffer_subjects), nb_datasets)

            for i in range(len(index_split)):
                start, end = index_split[i]
                for j in range(start, (end+1)):
                    subject_target = buffer_subjects[j]
                    labels_file.loc[labels_file['subject'] == subject_target,col_name] = i
    else : 
        domains_existent = labels_file['domain'].unique()
        for dom_exists in domains_existent : 
            labels_file.loc[labels_file['domain'] == dom_exists,col_name] = dom_exists


if __name__ == '__main__':
    opt = parse_arguments()
    for key in opt.keys():
        print('{:s}: {:s}'.format(key, str(opt[key])))
    locals().update(opt)

    #verify instructions 
    domain_shifters = verify_instructions(opt['domain_shifters'])

    #data_file and labels_file with normal Health MNIST (without pixels missing)
    source = opt['source']
    data_file_name = opt['data_file_name']
    labels_file_name = opt['labels_file_name']
    dim_image = opt['dim_image']

    data_file = pd.read_csv(source+'/'+data_file_name)
    labels_file = pd.read_csv(source+'/'+labels_file_name)

    nb_domains = len(domain_shifters)
        

    if nb_domains<2: 
        raise Exception('\nDomain shifter has less than 2 domains. /!\\')

    if len(data_file) != len(labels_file):
        raise Exception('\nLabels and data files don\'t have the same number of rows.')

    #load or create mask dataframe 
    if opt['mask_file_name'] != None : 
        mask_df = pd.read_csv(source+'/'+opt['mask_file_name'])
    else : 
        #create an independant mask for every row with an image containing only 1 (observed value) 
        mask_df =pd.DataFrame(np.ones(data_file.shape, dtype=np.uint8))


    #splitting and add domain column 
    split_into_domains(opt['is_various_domain'],nb_domains, labels_file)

    #apply filter to each domain 
    shift_instances(domain_shifters, labels_file, data_file, mask_df, dim_image)

    #dataset split 
    col_dataset = opt['name_dataset_separation']
    nb_datasets = None 
    if opt['is_various_domain'] : 
        nb_datasets = opt['nb_dataset_separation']

    dataset_separation(col_dataset, labels_file, opt['is_various_domain'], nb_datasets)

    #save 
    path_data = str(opt['destination'])+'\\'+opt['data_file_name_result']
    path_labels = str(opt['destination'])+'\\'+opt['labels_file_name_result']
    path_mask = str(opt['destination'])+'\\'+opt['mask_file_name_result']

    save_data(path_data, path_labels, deepcopy(data_file), deepcopy(labels_file), deepcopy(mask_df), path_mask)


