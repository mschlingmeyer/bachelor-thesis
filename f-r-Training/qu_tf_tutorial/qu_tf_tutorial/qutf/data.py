# coding: utf-8

"""
Helpers for handling the input pd_data.
"""

import os
from random import shuffle
import shutil

import numpy as np
import six

import pyarrow
import pandas as pd
import tensorflow as tf
from glob import glob
import json

from . import util, specs

from IPython import embed

# limit memory consumption on GPU
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=1024*2)]
        )
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(f"{len(gpus)} Physical GPUs,\t {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)
        print("start debug shell")
        embed()


thisdir = os.path.realpath(os.path.dirname(__file__))
class DataHandler(object):
    def __init__(
        self,
        file_paths=None,
        variable_config_path=os.path.join(thisdir, "variables.json"),
        # Parameter in def data_split
        train_percentage = 0.7,
        validation_percentage=0.2,
        SHUFFLE_BUFFER=123456,
        random_state=1,
        BATCH_SIZE=128,
        label_names=["labels"],
        training_weight_names=["class_weights"]
        ):
        super().__init__()

        # define member variables
        self.input_features = None
        self.labels = None
        
        self.label_names = label_names
        self.training_weight_names = training_weight_names

        self.normed_labels=None
        self.label_means = None
        self.label_stds = None

        self.train_data = None
        self.validation_data = None
        self.test_data = None
        self.variable_config = None
        self.train_percentage = train_percentage
        self.validation_percentage= validation_percentage
        self.SHUFFLE_BUFFER=SHUFFLE_BUFFER
        self.input_features_list = []
        self.gen_features_list = []
        self.file_paths = []
        self.random_state = random_state
        
        if not file_paths:
            self.file_paths = [os.path.join(thisdir, "..", "..", "..", "..", "preprocessed_data", "preprocessed_data.parquet")]
        else:
            self.file_paths = file_paths
        
        print("loading information from config file")
        # load information about variables from config
        self.init_config_file(variable_config_path)
        
        # embed()
        print("loading data")
        # load data from parquet files
        self.load_data()

        # embed()
        print("transforming labels")
        # transform the labels for later training
        # self.transform_labels()

        # embed()
        print("creating tf datasets")
        self.create_tf_datasets(BATCH_SIZE=BATCH_SIZE)

        print("done with init")

    def number_of_train_batches(self):
        trainlen = len(self.train_data)
        # vallen = len(self.validation_data)
        # testlen = len(self.test_data)  
        return trainlen  

    def init_config_file(self, path):
        with open(path) as conf:
            variable_conf = json.load(conf)
        
        self.input_features_list = variable_conf.get("training_features")

    def _check_pd_dataset_file(self, kind, index):
        # validate the kind
        kinds = ["train", "valid", "test"]
        if kind not in kinds:
            raise ValueError("unknown pd_dataset kind '{}', must be one of {}".format(
                kind, ",".join(kinds)))

        # validate the index
        if not (0 <= index < specs.n_files[kind]):
            raise ValueError("pd_dataset '{}' has no file index {}".format(kind, index))

        return "{}_{}.npz".format(kind, index)


    def provide_file(self, kind, index):
        """
        Returns the absolute file path of a file of an input pd_dataset given by *kind* ("train", "valid",
        "test", or "challenge") and its *index*. When the file is not locally accessible, it is
        downloaded from the public CERNBox directory.
        """
        # get the name of the file
        file_name = self._check_pd_dataset_file(kind, index)

        # make sure the user pd_data directory exists
        #Specs wegmachen oder auch umschreiben?
        if not os.path.exists(specs.pd_data_dir):
            os.makedirs(specs.pd_data_dir)

        # when it's already in the user pd_data directy, return the full path
        file_path = "/nfs/dust/cms/user/wittmath/root_to_parquet/SKIM_ggF_BulkGraviton_m300/output_*.parquet"
        #os.path.join(specs.pd_data_dir, file_name)
        if os.path.exists(file_path):
            return file_path

    def load_pd_data_to_pandas(
        self, 
        kind, 
        start_file=0, 
        stop_file=-1, 
        ):
        """
        Loads a certain *kind* of pd_dataset ("train", "valid", or "test") and returns the
        four-vectors of the jet constituents, the true jet four-vector and the truth label in a 3-tuple.
        When *kind* is *challenge*, the two latter elements of the returned tuple are *None*.
        Internally, each pd_dataset consists of multiple files whose arrays are automatically concatenated.
        For faster prototyping and testing, *start_file* (included) and *stop_file* (first file that is
        *not* included) let you define the range of files to load.
        """
        # fix the file range if necessary
        if stop_file < 0:
            stop_file = specs.n_files[kind] - 1
        cols = self.input_features_list+self.label_names + self.training_weight_names
        source_paths = self.file_paths
        try:
            pd_data = pd.concat([
                pd.read_parquet(
                    file_path, 
                    columns=cols
                )
                for file_path in source_paths
            ])
        except Exception as e:
            print(e)
            print("open debug shell")
            embed()
        return pd_data

        #DeltaR-Matching
    def apply_delta_r_selection(self, pd_data, var1, var2):
        deltaR_mask = (pd_data[var1] <= 0.4) & (pd_data[var2] <= 0.4)
        pd_data = pd_data[~deltaR_mask]
        deltaR_mask = (pd_data[var1] > 0.4) & (pd_data[var2] > 0.4)
        pd_data = pd_data[~deltaR_mask]
        return pd_data

    def data_preparation(self, pd_data, variable_config = "variables.json", shuffle_random_state=None):
        
        # first, shuffle the input data
        if shuffle_random_state == None:
            pd_data = pd_data.sample(frac=1).reset_index(drop=True)
        else:
            pd_data = (pd_data.sample(frac=1, random_state=shuffle_random_state)
                        .reset_index(drop=True))
        
        # filtere falsche pT-Werte heraus (sind auf -1 gesetzt)
        # pd_data_mask = (pd_data["bjet1_pt"] >= 0) & (pd_data["bjet2_pt"] >= 0)
        # pd_data = pd_data[pd_data_mask]

        # filtere, dass es mindestens zwei bJet-Kandidaten gibt (bjetscand > 1) 
        pd_data_cut = (pd_data["nbjetscand"] > 1)
        pd_data = pd_data[pd_data_cut]

        # DeltaR-Columns werden eingefügt
        pd_data["deltaR_det1_gen1"] = util.deltaR(dataset=pd_data, eta1="bjet1_eta", phi1="bjet1_phi", eta2 = "m_genB1_eta", phi2 = "m_genB1_phi")
        pd_data["deltaR_det1_gen2"] = util.deltaR(dataset=pd_data, eta1="bjet1_eta", phi1="bjet1_phi", eta2 = "m_genB2_eta", phi2 = "m_genB2_phi")
        pd_data["deltaR_det2_gen1"] = util.deltaR(dataset=pd_data, eta1="bjet2_eta", phi1="bjet2_phi", eta2 = "m_genB1_eta", phi2 = "m_genB1_phi")
        pd_data["deltaR_det2_gen2"] = util.deltaR(dataset=pd_data, eta1="bjet2_eta", phi1="bjet2_phi", eta2 = "m_genB2_eta", phi2 = "m_genB2_phi")
        
        # Filter Daten mit Maske raus
        pd_data = self.apply_delta_r_selection(pd_data=pd_data, var1="deltaR_det1_gen1", var2="deltaR_det1_gen2")
        pd_data = self.apply_delta_r_selection(pd_data=pd_data, var1="deltaR_det2_gen1", var2="deltaR_det2_gen2")
        
        # zuordnung zu bjet1 oder bjet2
        pd_data["true_bjet1_pt"] = np.where(pd_data["deltaR_det1_gen1"] < 0.4, pd_data["m_genB1_pt"], pd_data["m_genB2_pt"])
        pd_data["true_bjet2_pt"] = np.where(pd_data["deltaR_det2_gen1"] < 0.4, pd_data["m_genB1_pt"], pd_data["m_genB2_pt"])
        

        # extrahiere gen_features aus pd_framework
        self.gen_data = pd.concat([pd_data.pop(x) for x in self.gen_features_list], axis=1)
        
        # extrahiere target aus pd_framework
        self.labels = pd.concat([pd_data.pop(f"true_bjet{i}_pt") for i in range(1,3)], axis=1)
            #gen_data["matched_gen_bjet1_pt"] = target_data["true_bjet1_pt"].copy()
            #try:
            #     pd_data["matched_gen_bjet1_pt"] = target_data["true_bjet1_pt"].copy()
            # except:
            #     from IPython import embed; embed()
            # normalize to detector-level pt

        self.labels["true_bjet1_pt"] /= pd_data["bjet1_pt"]
        self.labels["true_bjet2_pt"] /= pd_data["bjet2_pt"]
        # target_data["true_bjet1_pt"] /= pd_data["bjet1_pt"]
        # test_input_features = pd.DataFrame()

        return pd_data

    def create_tf_datasets(self, BATCH_SIZE=128):
        # with tf.device('/cpu:0'):
        tf_input_features = tf.data.Dataset.from_tensor_slices(self.input_features.to_numpy())
        # tf_labels = tf.data.Dataset.from_tensor_slices(dict(labels))
        tf_labels = tf.data.Dataset.from_tensor_slices(self.labels.to_numpy())
        if len(self.training_weight_names) != 0:
            tf_class_weights = tf.data.Dataset.from_tensor_slices(self.class_weights.to_numpy())
            final_weights = tf_class_weights.map(lambda x: tf.math.reduce_prod(x))
            numeric_dict_ds = tf.data.Dataset.zip((tf_input_features, tf_labels, final_weights))
        else:
            numeric_dict_ds = tf.data.Dataset.zip((tf_input_features, tf_labels))
        print("generated final tf Dataset for training!")

        self.split_dataset(numeric_dict_ds, SHUFFLE_BUFFER=len(self.input_features), BATCH_SIZE=BATCH_SIZE)

        # return input_features, mean, std, train_data, validation_data, test_data

    def labels_to_onehot(self, labels):
        labels = labels.astype(np.int32)
        onehot = np.zeros((labels.shape[0], labels.max() + 1), dtype=np.float32)
        onehot[np.arange(labels.shape[0]), labels] = 1
        return onehot

    # def preprocess_input_features(self, input_features=None):
    #     if input_features == None:
    #         input_features = self.input_features
    #     inputs = {}
    #     for name, column in input_features.items():
    #         inputs[name] = tf.keras.Input(
    #             shape=(1,), name=name, dtype=tf.float32)
    #     return inputs

    def preprocess_input_features(self, input_features=None):
        if input_features == None:
            input_features = self.input_features
        inputs = {}
        for name, column in input_features.items():
            inputs[name] = tf.keras.Input(
                shape=(1,), name=name, dtype=tf.float32)

        inputs = tf.keras.Input(shape=(len(input_features.columns),), dtype=tf.float32)
        return inputs

    def load_data(self, shuffle_random_state=None):
        if shuffle_random_state == None:
            shuffle_random_state = self.random_state
        pd_data = self.load_pd_data_to_pandas(kind="train", stop_file=4)
        
        
        if shuffle_random_state == None:
            pd_data = pd_data.sample(frac=1).reset_index(drop=True)
        else:
            pd_data = (pd_data.sample(frac=1, random_state=shuffle_random_state)
                        .reset_index(drop=True))
        self.labels = pd_data[self.label_names]
        if len(self.training_weight_names) != 0:
            self.class_weights = pd_data[self.training_weight_names]
        # input_features= self.data_preparation(
        #     pd_data,
        #     shuffle_random_state=shuffle_random_state
        # )
        
        self.input_features = pd_data.drop(columns=self.label_names+self.training_weight_names)
        # columns = [
        #     "deltaR_det1_gen1",
        #     "deltaR_det1_gen2",
        #     "deltaR_det2_gen1",
        #     "deltaR_det2_gen2",
        # ]
        # self.input_features = input_features.drop(columns=columns)

    def split_dataset(
        self,
        numeric_dict_ds,
        SHUFFLE_BUFFER=None,
        BATCH_SIZE=128
    ):  
        gpu_device = tf.device('/gpu:0' if tf.config.list_physical_devices('GPU') else '/cpu:0')
        cpu_device = tf.device('/cpu:0')
        if SHUFFLE_BUFFER==None:
            SHUFFLE_BUFFER=self.SHUFFLE_BUFFER
        numeric_dict_batches = numeric_dict_ds
        # nevents = len(list(numeric_dict_batches))
        nevents = numeric_dict_batches.cardinality().numpy()
        train_size = np.floor(self.train_percentage*nevents)  
        valid_size = np.floor(self.validation_percentage*nevents)
        # test_size =  np.floor((1-train_percentage-validation_percentage)*nevents)

        print(f"have {nevents} events")
        print(f"train size: {train_size}")
        print(f"validation size: {valid_size}")
        # print(f"test size: {test_size}")

        # with device:
        train_data = numeric_dict_batches.take(train_size)
        testval_data = numeric_dict_batches.skip(train_size)
        val_data = testval_data.take(valid_size)
        rest = testval_data.skip(valid_size)
        test_data = rest.take(-1)


        self.train_data = train_data.shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)
        self.validation_data = val_data.batch(BATCH_SIZE)
        self.test_data = test_data.batch(BATCH_SIZE)

    def transform_labels(self, labels=None):
        if labels == None:
            labels = self.labels
        label_mean = np.mean(labels)
        label_std = np.std(labels)

        for c in labels.columns:
            labels[c] = (labels[c] - label_mean[c]) / label_std[c]

        self.normed_labels = labels
        self.label_means = label_mean
        self.label_stds = label_std

    def get_normed_labels_and_means(self):
        if len(self.transform_labels) == 0:
            self.transform_labels()
        
        return self.transform_labels, self.label_means, self.label_stds