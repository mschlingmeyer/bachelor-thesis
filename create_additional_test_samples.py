import os
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from preprocessing_parametrized import load_data, get_label, init_config_file, get_kl
import preprocessing_multiclass_config_v2 as preprocessing_config_multiclass
import preprocessing_config as preprocessing_config_binary
import create_additional_test_samples_config as test_samples_config
from IPython import embed

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
    class_weights = np.ones_like(df["labels"], dtype="float64")/int(label_counts)
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
    reduced_df_with_added_columns["weight_equalize_sig_bkg"] = np.ones_like(reduced_df_with_added_columns["labels"])
    reduced_df_with_added_columns["weight_global"] = 1e12
    # embed()
    return reduced_df_with_added_columns


def main(*args, **kwargs):
    network_type = kwargs.get("network_type")
    # parametrized = kwargs.get("parametrized")
    if network_type == "binary":
        label_config = preprocessing_config_binary.label_config
    else:
        label_config = preprocessing_config_multiclass.label_config
    input_directories = test_samples_config.input_directories
    parameter_config = test_samples_config.parameter_config
    thisdir = os.path.realpath(os.path.dirname(__file__))
    variables_list = init_config_file(os.path.join(thisdir, "preprocessed_variables.json"))

    conc_dataframe = pd.DataFrame()
    for directory in input_directories:
        tmp_dataframe = load_data(directory, variables_list)
        label = get_label(label_config=label_config, directory_name=directory)
        tmp_dataframe["labels"] = label*np.ones(tmp_dataframe.shape[0])
        kl_value = get_kl(parameter_config=parameter_config, directory_name=directory)
        tmp_dataframe["kappa_lambda"] = kl_value*np.ones(tmp_dataframe.shape[0])
        tmp_dataframe = add_columns(tmp_dataframe)
        conc_dataframe = pd.concat([conc_dataframe, tmp_dataframe])

    final_dataframe = conc_dataframe

    print(final_dataframe)
    # prepare to write preprocessed data
    output_dir = kwargs.get("output_dir")
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # write data
        if network_type == "binary":
            parquet_file_name="signal_samples_binary"
        else:
            parquet_file_name="signal_samples_multiclass"
        final_path = os.path.join(output_dir, parquet_file_name+".parquet")
        print(f"write preprocessed data to {final_path}")
        final_dataframe.to_parquet(final_path, engine="pyarrow")
        print(f"writing data to folder {output_dir}")
    else:
        print("cannot write data: no output dir defined!")


def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument("-t", "--network_type",
        help="type of network for which the data is prepared, binary or multiclass",
        choices=['binary', 'multiclass'],
        default="binary"
    )

    # parser.add_argument("-p", "parametrized",
    #     help="boolean to express if the network is poarametrized or not",
    #     type=bool,
    #     default=False
    # )

    parser.add_argument("-o", "--output-dir",
                        help="path to folder where preprocessed data should be stored in",
                        type=str,
                        default=os.path.join("f-r-Training", "additional_test_samples"),
                        metavar="path/to/output/folder"
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    main(**vars(args))
