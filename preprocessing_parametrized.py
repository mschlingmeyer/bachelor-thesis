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

import preprocessing_parametrized_config
from tqdm import tqdm
from IPython import embed

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
    wildcard = os.path.join(input_folder, "*.parquet")
    source_paths = glob(wildcard)
    cols = variables_list

    # hier öffnest du jetzt alle diese files. Mit *pd.read_parquet* wird ein einzelnes parquet file geladen.
    # Die geöffneten files sammelst du dann in einer Liste, die du mit *pd.concat* konkatinierst, sprichst an einander hängst
    # Am Ende hast du dann also ein pandas dataframe, das alle Informationen beinhaltet

    path_bar = tqdm(source_paths)
    path_bar.set_description("Loading file")
    for file_path in path_bar:
        pd_data = pd.concat([
                        pd.read_parquet(
                            file_path,
                            columns=cols
                    )
                    for file_path in path_bar
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

def multiply_weights(df, weights_list):
    """
    multiply all the columns which names are given within a list and give the sum over all events back
    """
    multiplied_array=np.ones(df.shape[0])
    for weight_name in weights_list:
        multiplied_array=multiplied_array*df[weight_name].to_numpy()
    return sum(multiplied_array)

def calculate_signal_prob(df, weights_list):
    """
    gives an array with the probability for an event from the signal samples to be from a sample with
    an underlying specific kappa_lambda value, given the weights columns used in weights_list
    """
    signal_events=df[df["labels"] == 1]
    kl0_events=df[df["kappa_lambda"] == 0]
    kl1_events=df[df["kappa_lambda"] == 1]
    kl2p45_events=df[df["kappa_lambda"] == 2.45]
    kl5_events=df[df["kappa_lambda"] == 5]

    sum_signal_event_weights=multiply_weights(signal_events, weights_list)
    sum_kl0_event_weights=multiply_weights(kl0_events, weights_list)
    sum_kl1_event_weights=multiply_weights(kl1_events, weights_list)
    sum_kl2p45_event_weights=multiply_weights(kl2p45_events, weights_list)
    sum_kl5_event_weights=multiply_weights(kl5_events, weights_list)

    signal_probability = np.array([float(sum_kl0_event_weights/sum_signal_event_weights),
                                  float(sum_kl1_event_weights/sum_signal_event_weights),
                                  float(sum_kl2p45_event_weights/sum_signal_event_weights),
                                  float(sum_kl5_event_weights/sum_signal_event_weights),
                                  ])
    print("signal_probability for weights_list", weights_list, "is", signal_probability)
    return signal_probability

def change_flag(kl_value_array, signal_probability):
    """
    look at the returned kl_value from the parameter_config. If kl_value is 9999 (flag),
    change it to a random value with the same distribution as signal
    """
    kl_value_array[np.where(kl_value_array==9999.)]=np.random.choice(np.array([0,1,2.45,5]), size=len(np.where(kl_value_array==9999.)[0]), replace=True, p=signal_probability)
    return kl_value_array

def get_kl(parameter_config, directory_name):
    # get name of directory containing the parquet files
    basename = os.path.basename(directory_name)

    # if basename contains any key of the label config, return
    # the corresponding label value
    # if not, throw error

    matches = list(filter(lambda x: x in basename, list(parameter_config.keys())))
    # if we have more than one match, the key is bad
    if len(matches) == 1:
        return parameter_config[matches[0]]
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

def add_class_weights(df):
    """
    Add class weights to mitigate difference in occurance between classes.
    Final weight for each class is 1/(occurance of class)

    Args:
    df: input pandas.DataFrame with data and class labels in columns 'label'

    Return:
    pandas.DataFrame: final data frame including new column 'class_weights'
    """
    # count occurance of values in column 'labels'
    label_counts = df["labels"].value_counts()

    # initialize array for class weights with same dimension as labels column
    class_weights = np.zeros_like(df["labels"], dtype="float64")
    for label, count in zip(label_counts.index, label_counts):
        class_weights += np.where(df["labels"] == label, 1./count, 0.)

    df["class_weights"] = class_weights
    return df

def add_columns(df):
    """
    Add columns needed for the training

    Args:
    df: the reduced dataframe with only the selected events for signal and background

    Return:
    reduced_df_with_added_columns: the reduced dataframe with only the selected events and the added columns
    """
    reduced_df_with_added_columns = add_class_weights(df)

    # add different signal weights
    start = 1
    end=7
    for weight in np.logspace(start=start, stop=end, num=end-start+1, base=10, dtype="float32"):
        reduced_df_with_added_columns[f"signal_weight_{int(weight)}"] = np.where(reduced_df_with_added_columns["labels"] == 1, weight, 1.)

    bkg_df = reduced_df_with_added_columns[reduced_df_with_added_columns["labels"] == 0]
    sig_df = reduced_df_with_added_columns[reduced_df_with_added_columns["labels"] == 1]
    bkg_weight = np.sum(bkg_df["class_weights"]*bkg_df["plot_weight"]*bkg_df["lumi_weight"])
    sig_weight = np.sum(sig_df["class_weights"]*sig_df["plot_weight"]*sig_df["lumi_weight"])
    ratio = bkg_weight/sig_weight if not any(x == 0 for x in [bkg_weight, sig_weight]) else 1.
    reduced_df_with_added_columns["weight_equalize_sig_bkg"] = np.where(reduced_df_with_added_columns["labels"] == 1, ratio, 1.)
    reduced_df_with_added_columns["weight_global"] = 1e12
    # embed()
    return reduced_df_with_added_columns

def change_features(df):
    """
    Change the features to improve the training (normalisation...)

    Args:
    df:  the reduced dataframe with only the selected events and the added columns

    Return:
    final_df: the preprocessed dataframe
    """
    # print("df[kappa_lambda]", df["kappa_lambda"])
    # weights_list = ["class_weights", "plot_weight", "lumi_weight", "weight_equalize_sig_bkg"]
    weights_list = ["class_weights"]
    signal_prob = calculate_signal_prob(df, weights_list)

    kappa_lambda_values = df["kappa_lambda"].to_numpy()
    kappa_lambda_values = change_flag(kappa_lambda_values, signal_prob)
    # print("kappa_lambda_values", kappa_lambda_values, print(len(np.where(kappa_lambda_values!=0))))

    df["kappa_lambda"] = kappa_lambda_values
    # print("after change flag",df["kappa_lambda"])
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
    print(preprocessing_parametrized_config.label_config)
    label_config = preprocessing_parametrized_config.label_config
    input_directories = preprocessing_parametrized_config.input_directories
    parameter_config = preprocessing_parametrized_config.parameter_config
    thisdir = os.path.realpath(os.path.dirname(__file__))
    variables_list = init_config_file(os.path.join(thisdir,"preprocessed_variables.json"))

    conc_dataframe = pd.DataFrame()
    for directory in input_directories:
        tmp_dataframe = load_data(directory, variables_list)
        label = get_label(label_config=label_config, directory_name=directory)
        tmp_dataframe["labels"] = label*np.ones(tmp_dataframe.shape[0])
        kl_value = get_kl(parameter_config=parameter_config, directory_name=directory)
        tmp_dataframe["kappa_lambda"] = kl_value*np.ones(tmp_dataframe.shape[0])
        conc_dataframe = pd.concat([conc_dataframe, tmp_dataframe])

    reduced_dataframe = selection_events(conc_dataframe)
    reduced_dataframe_with_added_columns = add_columns(reduced_dataframe)
    final_dataframe = change_features(reduced_dataframe_with_added_columns)

    print(final_dataframe)
    # prepare to write preprocessed data
    output_dir = kwargs.get("output_dir")
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # write data
        final_path = os.path.join(output_dir,"preprocessed_parametrized_data.parquet")
        print(f"write preprocessed data to {final_path}")
        final_dataframe.to_parquet(final_path, engine="pyarrow")
        print(f"writing data to folder {output_dir}")
    else:
        print("cannot write data: no output dir defined!")
    #final_dataframe.to_parquet
if __name__ == '__main__':
    options, args = parse_arguments()
    main(*args, **vars(options))

