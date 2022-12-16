import os
import sys
import shap
import numpy as np
import pandas as pd
from IPython import embed

import matplotlib.pyplot as plt
import qu_tf_tutorial
from qutf import data
import tensorflow as tf
from tensorflow import keras
import Keras_Sachen_test
# import tensorflow_datasets as tfds
from argparse import ArgumentParser
import importlib
import dnn_utils


HOMEPATH=os.path.realpath(os.path.dirname(__file__))

label_dict = {
    "true_bjet1_pt": r'Generator p$_{T}$ (b-Jet 1)',
    "true_bjet2_pt": r'Generator p$_{T}$ (b-Jet 2)'
}

input_dict = {
    "bH_pt": r'p$_{T}$ (H$_{bb}$)',
    "bH_e": r'Energie (H$_{bb}$)',
    "bH_eta": r'$\eta$ (H$_{bb}$)',
    "bH_phi": r'$\phi$ (H$_{bb}$)',
    "bjet1_pt": r'p$_{T}$ (b-Jet 1)',
    "bjet2_pt": r'p$_{T}$ (b-Jet 2)',
    "bjet1_e": r'Energie (b-Jet 1)',
    "bjet2_e": r'Energie (b-Jet 2)',
    "bjet1_eta": r'$\eta$ (b-Jet 1)',
    "bjet2_eta": r'$\eta$ (b-Jet 2)',
    "bjet1_phi": r'$\phi$ (b-Jet 1)',
    "bjet2_phi": r'$\phi$ (b-Jet 2)',
    "bjet2_HHbtag": r'HHbtag (b-Jet 2)',
    "bjet1_HHbtag": r'HHbtag (b-Jet 1)',
    "bjet2_bID": r'bID (b-Jet 2)',
    "bjet1_bID": r'bID (b-Jet 1)',
    "HT20": r'HT20',
    "dau1_pt": r'p$_{T}$ (Tau 1)',
    "dau2_pt": r'p$_{T}$ (Tau 2)',
    "tauH_mass": r'm (H$_{\tau\tau}$)',
    "tauH_pt": r'p$_{T}$ (H$_{\tau\tau}$)',
    "HH_mass": r'm (HH)',
    "HH_pt": r'p$_{T}$ (HH)',
}

def create_plot(
    shap_values,
    test_all_features,
    plot_type,
    feature_names,
    class_names,
    max_display,
    outname):
    fig, ax = plt.subplots(figsize=(20, 8))
    shap.summary_plot(shap_values, test_all_features, plot_type=plot_type,
        feature_names = [input_dict.get(x, x) for x in feature_names],
        class_names = [label_dict.get(x, x) for x in class_names],
        max_display = max_display, show = False)
    
    for ext in "png pdf".split():
        final_outpath = f"{outname}_{plot_type}.{ext}"
        plt.savefig(final_outpath)
    plt.close()

def split_features_from_labels(*args, **kwargs):
    return None, None, None

def calculate_shap_values(
    model, 
    train_data, 
    test_data,
    label_names,
    input_features=None,
    input_features_conti=None,
    input_features_disc=None,
    n_events=100,
    outpath=os.path.join(HOMEPATH, "shap_plots"),
    outname="shap_plots",
    draw_violin=True
    ):
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    if input_features == None and not any(x == None for x in [input_features_conti, input_features_disc]):
        train_conti, train_disc, train_labels = split_features_from_labels(
            dataset=train_data,
            columns_input_conti=input_features_conti,
            columns_input_disc=input_features_disc,
            columns_output=label_names
            )
        test_conti, test_disc, test_labels = split_features_from_labels(
            dataset=test_data,
            columns_input_conti=input_features_conti,
            columns_input_disc=input_features_disc,
            columns_output=label_names
            )
        # concatenate input features for correct parse through to TensorFlow model
        train_all_features = train_conti[:n_events].copy()
        for c in train_disc.columns:
            train_all_features[c] = train_disc[c][:n_events]
        test_all_features = test_conti[:n_events].copy()
        for c in test_disc.columns:
            test_all_features[c] = test_disc[c][:n_events]
        
        input_features_all = input_features_conti+input_features_disc
    else:
        train_all_features = train_data
        test_all_features = test_data
        input_features_all = input_features
    
    try:
        # shap_train = tfds.as_dataframe(train_all_features).drop(label_names)
        # shap_test = tfds.as_dataframe(test_all_features).drop(label_names)
        shap_train = train_data[:n_events]
        shap_test = test_data[:n_events]
        max_display = len(shap_train.columns)

        shap_train = shap.sample(shap_train)
        shap_test = shap.sample(shap_test)
        explainer = shap.KernelExplainer(model, shap_train)
        shap_values = explainer.shap_values(shap_test, nsamples=n_events)
    except Exception as e:
        print(e)
        print("start debug shell")
        from IPython import embed; embed()
    
    # plot summary plot
    print(f"drawing bar plot with {len(train_all_features)} events")
    create_plot(
        shap_values=shap_values,
        test_all_features=shap_test,
        plot_type="bar",
        feature_names=input_features_all,
        class_names=label_names,
        max_display=max_display,
        outname=os.path.join(outpath, outname))
    
    if draw_violin:
        
        try:
            for i, label in enumerate(label_names):
                print(f"drawing violin plot with {len(train_all_features)} events for class '{label}'")
                create_plot(
                    shap_values=shap_values[i],
                    test_all_features=shap_test,
                    plot_type="violin",
                    feature_names=input_features_all,
                    class_names=[label],
                    max_display=max_display,
                    outname=os.path.join(outpath, f"{outname}_{label}"))
        except Exception as e:
            print(e)
            print("start IPython shell")
            from IPython import embed; embed()


def main(dnn_folders, **kwargs):
    variable_config_path = kwargs.get("variable_config_path")
    input_files = kwargs.get("input_data", None)
    hyperparameters = kwargs.get("hyperparameters", None)
    dnn_architectures = dict()
    if hyperparameters:
        dirname = os.path.dirname(hyperparameters)
        if not dirname in sys.path:
            sys.path.append(dirname)
        modulename = os.path.basename(hyperparameters)
        modulename = ".".join(modulename.split(".")[:-1])
        hyperpars = importlib.import_module(modulename)
        dnn_architectures = hyperpars.dnn_architectures
    
    
    for dnn_folder in dnn_folders:
        current_dir = dnn_folder
        if current_dir.endswith(os.path.sep):
            current_dir=os.path.dirname(current_dir)
        prefix = os.path.basename(current_dir)
        if prefix[:10]=="binary_dnn":
            prefix=prefix[11:]
        if prefix[:14]=="multiclass_dnn":
            prefix=prefix[15:]
        hyperparameters = dnn_architectures.get(prefix, dict())
        print("hyperparameters", prefix, hyperparameters)

        data_handler = data.DataHandler(
            variable_config_path=variable_config_path,
            file_paths=input_files,
            #parametrized=True,
            **hyperparameters
        )
    # first, load data
    # data_handler = data.DataHandler(variable_config_path=os.path.join(HOMEPATH, "variables.json"))
    
    # for model_path in dnn_models:
        model = tf.keras.models.load_model(
            dnn_folder,
            custom_objects={
                "custom_loss_function" : dnn_utils.custom_loss_function
                }
            )
        if dnn_folder.endswith(os.path.sep):
            dnn_folder = os.path.dirname(dnn_folder)        
        training_nr = os.path.basename(dnn_folder)
        input_features_df = data_handler.input_features
        nevents = len(input_features_df)
        train_valid_events = int(np.floor(nevents*(data_handler.train_percentage + data_handler.validation_percentage)))
        train_events = int(np.floor(nevents*(data_handler.train_percentage)))
        train_input = input_features_df[:train_events]
        test_input = input_features_df[train_valid_events:]
        # train_input = data_handler.train_data
        # test_input = data_handler.test_data
        calculate_shap_values(
            model = model,
            train_data=train_input,
            test_data=test_input,
            input_features=data_handler.input_features_list,
            label_names=data_handler.labels.columns,
            outname=f"{training_nr}_shap",
            # draw_violin=True
        )

def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument("--hyperparameters", "-p", type=str,
        help="path to config file for hyper parameters",
        default=os.path.join(HOMEPATH, "hyperparameters.py"),
        dest="hyperparameters"
    )
    parser.add_argument("-i", "--input-data",
        help="path to input data to be used for the evaluation",
        nargs="*",
        metavar="path/to/preprocessed_data.parquet",
        type=str,
    )
    parser.add_argument("dnn_folders", nargs="+",
        help="paths to folders containing dnn models to be evaluated",
        metavar="path/to/dnn_folders",
        type=str
    )
    parser.add_argument("-v", "--variable_config_path",
        help="path to config .json file containing input variables",
        type=str,
        metavar="path/to/variable_config.json",
        default=os.path.join(HOMEPATH, "variables.json"),
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()
    main(**vars(args))