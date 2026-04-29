# DomHealthMNIST
---

This synthetic dataset was designed for the study of continual learning on longitudinal data; it simulates multiple domains, as well as temporal irregularity and heterogeneity.
 
## Overview 
---

DomHealth MNIST is a synthetic extension of the Health MNIST dataset (Ramchandran et al.,2021). Both classification and regression are possible with this dataset. 
The first step is to create this original dataset.

We developed two approaches to introduce irregularity: simulating medical interventions and removing timepoints. 

- In Health MNIST, disease progression is simulated by rotating the digit. To make this synthetic dataset more realistic, we propose to include medical interventions. These interventions simulate changes in disease progression, slowing, halting, or even reversing the trajectory, potentially improving a patient’s condition. This enables the generation of more complex disease trajectories, which better reflect real-world clinical scenarios.

- Originally, a patient has 20 timepoints, but you can modify it by removing images. This step creates distinct longitudinal structures for each subject, dataset, or domain. As in real-life applications, a patient can miss an appointment, and we will still use their data. 

DomHealth MNIST allows easier manipulation of domains with multiple proposed transformations: 
- colours 
- blur 
- digit scale 
- missing pixels
- stretching digit
- inverting colour



## 1. Original Health MNIST 
---

We need to create the original Health MNIST from *Ramchandran, S., Tikhonov, G., Kujanpää, K., Koskinen, M., & Lähdesmäki, H. (2021). Longitudinal Variational Autoencoder. Proceedings of the Twenty Fourth International Conference on Artificial Intelligence and Statistics (AISTATS)*. Original Health MNIST includes longitudinal data with 20 timepoints/follow-ups per subject. 

Each MNIST digit is a subject; variations in MNIST digits create a series of 20 images. 

Translation (to the right corner): age growing
 
Rotation: disease progression 

Half of the dataset is sick, and the other half is healthy; a random rotation, instead of the disease progression, is applied for all of them. 
Only the sick subject has a value for ```disease_time```.
Time can be measured with the subject's age growing, in ```time_age``` variable.

Variable to forecast : 
- For classification, the disease status of the subject variable ```disease``` (sick or healthy)
- For regression, the rotation angle variable ```angle```

<img src="images/subject_example.png" alt="Subject from original Health MNIST example" width="500"/>

It returns two CSV files, a label, and a data file (containing an image in each row). The mask file is not created at this point; if needed, it can be automatically created in the second part. 

### Downloading MNIST digits 
---

- Download MNIST images with *(36x36)* dimension and unzip archive from [here](https://www.dropbox.com/s/j80vfwcqqu3vmnf/trainingSet.tar?dl=0)

### Generating experiment data
---

- Comparing to the original Health MNIST, we followed their creation guidelines except for some details. Our result dataset doesn't have missing pixels; this step will be possible during the domain splitting. Moreover, you can choose the image dimensions between *(36x36)* or *(36x36x3)*. We are now able to process colour variations because we add three 3 channels.

- Before you, make sure that the result folder exists. 

- To create the dataset, run (it is a slightly different version of the original Python file):
```python health_MNIST_v2_generate.py --source=./trainingSet --destination=./result --num_3=10 --num_6=10 --data_file_name=health_MNIST_data.csv --labels_file_name=health_MNIST_label.csv --dim_image=1296```

```num_3``` and ```num_6``` can go to 1 normally up to 4137 (depending on the number of digits in the trainingSet folder). 

The argument ```--dim_image``` indicates the dimension of the data processed, i.e., image as a line (1296 or 3888).

## 2. Irregularity and domain shift to create DomHealth-MNIST


### Synthetic medical intervention
---

*This step must be completed before the domain shift.*

Different trajectories result from this event. 
A patient's disease progression (i.e., digit rotation) can be halted or slowed. 

The patient's condition can be improved by going back in time with an ulterior rotation. This results in an improvement, but not the disappearance, of the disease. 


### Splitting the dataset into domains 
---

- Files created are named *domHealth_MNIST_data* for data, *domHealth_MNIST_mask* for masks and *domHealth_MNIST_label* for labels by default. 

Change occurred: 
- In the new label file, a column ```domain``` is added with a numeric number corresponding to each domain and a column ```dataset``` (by default, but you can choose its name with the argument ```--name_dataset_separation```) split the dataset into smaller ones (choose the number with argument ```--nb_dataset_separation```). 
- In the new data file, images are changed following the instructions given. 
- The dimension of the data processed can also be in 3D, i.e., image as a line 1296 or 3888 for the 3D. 
- In order to create irregular longitudinal data, it is possible to remove some follow-ups with the python file ```change_longitudinal_structure.py```.



#### How to give instructions? 
---

Give a list in ```domain_shifters``` variable, every element of this list corresponds to a domain. Each element is a list of instructions for a domain. An instruction is a tuple *(title_instruction, value)*. For example, ```domain_shifters=[[],[('scale',0.5)],[('scale',0.7), ('color','FF0000')],[('scale',1.5)]]```

In the example, there are 4 domains : 
- Domain 0 has no change 
- Domain 1 has a different scale with smaller digit 
- Domain 2 also has a different scale **and** a different colour 
- Domain 3 has bigger digit 

As previously mentioned, to split into different domains, we can manipulate colour, blur, scale, the number of missing pixels, the extent of stretching, and the inverted colour.

<img src="images/different_domains.png" alt="Domains example" width="700"/> 


| Changement  | title_intrusction | Values      |
|-------------|-------------------|-------------|
| colour    | *colour*       | hexadecimal without the prefix #   |
| blur | *blur*        | 0 < value   |
| digit scale | *scale*        |  0 < value  |
| missing pixels | *missing*        | 0 <= value <= 100   |
|stretched digit | *stretched*        | a series of two numbers [value_height, value_width], 0 < value_height and 0 < value_width |
| inverted colour | *inverted*        | 0 |


We assign a domain to every data point following your instructions. The boolean variable ```is_various_domain``` indicates how domains are distributed : 
- Every follow-up of the subject has the same domain
- Follow-ups from the same subject can be associated with different domains; a subject is made from various domains

<img src="images/domain_by_subjectORtimepoint.png" alt="domain by subject or timepoint" width="500"/> 

#### Generating final experiment data with the domains splitter 
---

The 25% missing pixels are not applied automatically; you have to specify it in the instructions for all domains if you want it.

 
- To create the dataset, run :
```python domHealth_MNIST_generate.py --source=./result --destination=./result --data_file_name=health_MNIST_data.csv --labels_file_name=health_MNIST_label.csv --domain_shifters="[[],[('scale',0.5)],[('scale',0.7)],[('scale',1.5)]]" --is_various_domain=true --nb_dataset_separation=4```

If you want ```is_various_domain=False```, don't specify it in the command line, otherwise any response is considered as a True value. 

### Removing follow-ups 
---

*You have to do this step after the domains splitter because you need a mask file.* 

Removing follow-ups/images changes the structure of the longitudinal data, to have patients having a different number of follow-ups and not the same timepoints. 

With the ```--target``` argument, you can apply a general instruction (by default) or one for each dataset/domain by specifying a column name (for example, *domain* or *dataset*).

Two instruction ways to remove follow-ups exist, the ```--list_timepoint_to_remove``` argument lets you specify a list of follow-ups you want to remove and the ```--nb_timepoint_to_remove``` argument stores a number of random timepoints you want to remove for each subject. 

- For example, to create an alternative dataset where all the pair numbers of follow-ups are removed, run : 
```python change_longitudinal_structure.py --source=./result --destination=./result  --labels_file_name=domHealthMNIST_various_label.csv --data_file_name=domHealthMNIST_various_data.csv --target=dataset --list_timepoint_to_remove="[[0, 2, 4, 6, 8, 10, 12, 14, 16, 18], [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],[10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]" ``` 



## 3. Domains visualisation 
---

After training a basic VAE on 4000 images from domain 0, we tested it to obtain latent representations of data from different domains. 

We trained a basic VAE on 4000 images from domain 0 as single data, not as longitudinal data. Then, we tested the VAE to obtain latent representations of data from different domains. 


<img src="images/latent_space_different_scaling.png" alt="Latent representations of DomHealth-MNIST data from different domains" width="500"/> 
