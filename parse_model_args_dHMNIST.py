import argparse
import ast


"""
Helper code for loading parameters from parameter file or from command line
"""

class LoadFromFile (argparse.Action):
    """
    Read parameters from config file
    """
    def __call__ (self, parser, namespace, values, option_string = None):
        with values as f:
            parser.parse_args(f.read().splitlines(), namespace)


class ModelArgs:
    """
    Runtime parameters for the L-VAE model
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Enter configuration arguments for the model')
        self.parser.add_argument('--f', type=open, action=LoadFromFile)
        self.parser.add_argument('--source_MNIST', type=str, default='./trainingSet', help='Path to MNIST image root directory')
        self.parser.add_argument('--destination', type=str, default='./data', help='Path to save the generated dataset')
        self.parser.add_argument('--num_3', type=int, default=50, help='Number of unique instances for digit 3')
        self.parser.add_argument('--num_6', type=int, default=50, help='Number of unique instances for digit 6')
        self.parser.add_argument('--base_file_name', type=str, default='', help='Base of the file name')
        self.parser.add_argument('--dim_image', type=int, default=3888, help='Number of pixels in an image')
        self.parser.add_argument('--nb_halted', type=int, default=0, help='Number of patient intervention with halted result')
        self.parser.add_argument('--nb_slow', type=int, default=0, help='Number of patient intervention with slow result')
        # self.parser.add_argument('--nb_back', type=int, default=0, help='Number of patient intervention with going back result')
        # self.parser.add_argument('--after_b_intervention', type=str, default='H', help='Trajectory of the patient after a going back intervention, H=Halted, S=Slow, D=Disease')
        self.parser.add_argument('--domain_distrib_file_name', type=str, default='domain_distrib_health_MNIST.png',
                            help='File name to save plot of domain distributions for health MNIST')
        self.parser.add_argument('--is_various_domain', type=bool, default=False,
                            help='Give a command value if digit instances from one subject can be associated to various domains, else don\'t specify it in the command line : if there is only one domain for a subject')
        self.parser.add_argument('--domain_shifters', type=str, default="[[],[('scale',0.5),('colour','FF0000')],[('scale',0.7),('colour','0D00FF')],[('scale',1.5),('colour','00FF04')]]",
                            help='string of a list of tuples lists (string, list/value)')
        self.parser.add_argument('--name_dataset_separation', type=str, default='dataset',
                            help='Give a new column name to split the global dataset into small ones, if is_various_domain=False dataset will be split by its domain value, else is it randomly split')
        self.parser.add_argument('--nb_dataset_separation', type=int, default=3,
                            help='If is_various_domain=False dataset will be split by number of total domain values, else is it randomly split in your number choice')
        self.parser.add_argument('--target', type=str, default='all',
                            help='String all means it will have the same effect on all the dataset, else precise a column name to distinguish different effect you want to apply')
        self.parser.add_argument('--col_id', type=str, default='subject',
                            help='Name of the column containing IDs')
        self.parser.add_argument('--col_time', type=str, default='time_age',
                            help='Name of the column containing timepoint/date')
        self.parser.add_argument('--list_timepoint_to_remove', type=str, default='[]',
                            help='If target is \'all\' then it is a list of integers corresponding to the position of timepoints you want to remove. Else it is a list of multiple list of timepoint to remove depending on the dataset')
        self.parser.add_argument('--nb_timepoint_to_remove', type=str, default='[0]', 
                            help='List of number of timepoints to remove randomly in a patient\'s longitudinal data.')
        self.parser.add_argument('--seed', type=int, default=None, 
                            help='Number of seed to use in numpy random method.')
        self.parser.add_argument('--change_long_structure', type=bool, default=False, help='True if you want to change the longitudinal structure')
                
    def parse_options(self):
        opt = vars(self.parser.parse_args())
        return opt




def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
