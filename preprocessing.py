import os
import sys
import pandas as pd
# import tensorflow as ts
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import json
from optparse import OptionParser
# add the current folder to the python PATH
# to make sure that you can load something
thisdir = os.path.realpath(os.path.dirname(__file__))
if not thisdir in sys.path:
    sys.path.append(thisdir)

import preprocessing_config


# Things the preprocessing needs to do
# - add labels to data -> config?
# - weight events?

def init_config_file(path):
        with open(path) as conf:
            variable_conf = json.load(conf)
        
        return variable_conf.get("variables_to_process")

def load_data(input_folder:str, variables_list):
    """
    function to load parquet files as pd.DataFrame. 
    Args:

    input_folder (str):  path to folder containing 'data_vbf*.parquet' files
    variables_list (list): list of all variables to get from the parquet files 

    Return:
    pd_data : the pandas dataframe containing all variables from all files with the name data_resolved 
    for the given variables_list and the given input_folder
    """ 
    # source_paths ist eine Liste mit den Pfaden zu den files, die du öffnest möchstest
    wildcard = os.path.join(input_folder, "data_vbf*.parquet")
    source_paths = glob(wildcard)
    cols = variables_list

    # hier öffnest du jetzt alle diese files. Mit *pd.read_parquet* wird ein einzelnes parquet file geladen. 
    # Die geöffneten files sammelst du dann in einer Liste, die du mit *pd.concat* konkatinierst, sprichst an einander hängst
    # Am Ende hast du dann also ein pandas dataframe, das alle Informationen beinhaltet
    pd_data = pd.concat([
                    pd.read_parquet(
                        file_path, 
                        columns=cols
                    )
                    for file_path in source_paths
                ])
    return pd_data

def get_label(label_config, directory_name):

    # get name of directory containing the parquet files
    basename = os.path.basename(directory_name)

    # if basename contains any key of the label config, return
    # the corresponding label value
    # if not, throw error
    
    matches = list(filter(lambda x: x in basename, list(label_config.keys())))
    # if we have more than one match, the key is bad
    if len(matches) == 1:
        return label_config[matches[0]]
    else:
        print("I have not programmed this path yet!")
        # hier musst du if len(matches) == 0 und  if len(matches) >1 (oder nutze elif und else)
        exit(0)
        # now find the match


def selection_events(df):
    """
    select which events to use in the network

    Args: 
    df: Concatenated dataframe with all signals and backgrounds

    Return:
    reduced_df: the reduced dataframe with only the selected events
    """
    reduced_df = df
    return reduced_df

def add_columns(df):
    """
    Add columns needed for the training 

    Args: 
    df: the reduced dataframe with only the selected events for signal and background

    Return:
    reduced_df_with_added_columns: the reduced dataframe with only the selected events and the added columns
    """
    reduced_df_with_added_columns = df
    return reduced_df_with_added_columns

def change_features(df):
    """
    Change the features to improve the training (normalisation...)

    Args: 
    df:  the reduced dataframe with only the selected events and the added columns

    Return:
    final_df: the preprocessed dataframe
    """
    final_df = df
    return final_df


def parse_arguments():
    parser = OptionParser()

    parser.add_option("-o", "--output-dir",
                        help="path to folder where preprocessed data should be stored in",
                        dest="output_dir",
                        type="str",
                        default="preprocessed_data",
                        metavar="path/to/output/folder"
    )

    options, args = parser.parse_args()
    return options, args


def main(*args, **kwargs):
    print("hello Maja")
    print(preprocessing_config.label_config)
    label_config = preprocessing_config.label_config
    input_directories = preprocessing_config.input_directories
    thisdir = os.path.realpath(os.path.dirname(__file__))
    variables_list = init_config_file(os.path.join(thisdir,"preprocessed_variables.json"))

    conc_dataframe = pd.DataFrame()
    for directory in input_directories:
        tmp_dataframe = load_data(directory, variables_list)
        label = get_label(label_config=label_config, directory_name=directory)
        tmp_dataframe["labels"] = label*np.ones(tmp_dataframe.shape[0])
        conc_dataframe = pd.concat([conc_dataframe, tmp_dataframe])

    class_weights = np.zeros(len(conc_dataframe))

    for label_class in preprocessing_config.label_classes:
        size_label_class = len(conc_dataframe[conc_dataframe["labels"]==label_class])
        label_class_position = (conc_dataframe["labels"]==label_class).to_numpy()
        one_class_weights=float(label_class_position/(size_label_class))
        class_weights = class_weights + one_class_weights

    conc_dataframe["class_weights"] = class_weights
    reduced_dataframe = selection_events(conc_dataframe)
    reduced_dataframe_with_added_columns = add_columns(reduced_dataframe)
    final_dataframe = change_features(reduced_dataframe_with_added_columns)

    print(final_dataframe)
    # prepare to write preprocessed data
    output_dir = kwargs.get("output_dir")
    if output_dir:
        # write data
        final_dataframe.to_parquet(os.path.join(output_dir,"preprocessed_data.parquet"), engine="pyarrow")
        print(f"writing data to folder {output_dir}")
    else:
        print("cannot write data: no output dir defined!")
    #final_dataframe.to_parquet
if __name__ == '__main__':
    options, args = parse_arguments()
    main(*args, **vars(options))

