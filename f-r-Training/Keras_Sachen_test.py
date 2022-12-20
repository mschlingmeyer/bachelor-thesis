from calendar import EPOCH
import os
from posixpath import split
import sys
#from qu_tf_tutorial import qutf
from qutf import data, util, specs, plots
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import livelossplot as llp
import pandas as pd
from dnn_utils import EarlyStopping
from dnn_model_definition import dnn_model_architecture
import hyperparameters as hyperparams
import create_dnn_plots as dnnplt
import dnn_utils
from optparse import OptionParser
from IPython import embed

thisdir = os.path.realpath(os.path.dirname(__file__))
mlp_style_path = os.path.join(thisdir, 'plot.mplstyle')
mlp_other_style=os.path.join(thisdir, 'plot.copy.mplstyle')
plt.style.use(mlp_style_path)
# Loss-Funktion

def stack_dict(inputs, fun=tf.stack):
  values = []
  for key in sorted(inputs.keys()):
    values.append(tf.cast(inputs[key], tf.float32))

  return fun(values, axis=-1)


def restore_checkpoint(model, checkpoint_dir):
    model.load_weights(checkpoint_dir)


def create_model_callbacks(
    name, 
    number_training_samples,
    decay_steps,
    decay_rate,
    initial_learning_rate,
    early_stopping_criteria=None,
    early_stopping_patience=None,
    checkpoint_frequency=1,
    **kwargs
):

    callbacks = []
    speicherort = f'./qu_tf_tutorial/checkpoints/{name}' + "_cp-{epoch:03d}.ckpt"

    # number_training_samples = len(train_data)
    
    steps_per_execution = number_training_samples * checkpoint_frequency


    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        speicherort,
        monitor='mse',
        verbose=1,
        save_best_only=False,
        save_weights_only=False,
        mode='auto',
        # save_freq=steps_per_execution,
        save_freq=number_training_samples,
    ))

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate
    )

    callbacks.append(
        tf.keras.callbacks.LearningRateScheduler(
            lr_schedule, verbose=1
        ),
        # tf.keras.callbacks.EarlyStopping
    )
    if not all(x == None for x in [early_stopping_criteria, early_stopping_patience]):
        callbacks.append(
            EarlyStopping(
                value=early_stopping_criteria,
                patience=early_stopping_patience
            )
        )

    with plt.style.context(mlp_other_style):
        callbacks.append(llp.PlotLossesKerasTF(
            outputs=[llp.outputs.MatplotlibPlot(cell_size=(8, 4))]
            )
        )
    return callbacks

def main(*args, **kwargs):
    # setup device for tensorfor
    device = tf.device('/gpu:0' if tf.config.list_physical_devices('GPU') else '/cpu:0')  

    # load hyperparameters for training
    this_architecture = kwargs.get("dnn_architecture")

    dnn_archs = hyperparams.dnn_architectures

    hp = dnn_archs.get(this_architecture)
    if hp == None:
        raise KeyError(f"Could not find '{this_architecture}' in hyperparameters.py!")

    model_architecture = kwargs.get("model_architecture") 
    create_model = dnn_model_architecture.get(model_architecture, None)
    if not create_model:
        raise ValueError(f"could not load function '{model_architecture}' from dnn_model_architecture!")
    # train_variables = "bjet1_pt bjet1_eta bjet1_phi bjet1_e".split()
   
    EPOCHS = hp.get("epochs", 50)
    data_handler = data.DataHandler(
        variable_config_path=os.path.join(thisdir, "variables.json"),
        BATCH_SIZE=hp.get("BATCH_SIZE", 128),
        file_paths = args,
        training_weight_names = hp.get("training_weight_names", []),
        parametrized = hp.get("parametrized", False),
        multiclass=hp.get("multiclass", False)
    )
    # input_features, mean, std, train_data, validation_data, test_data = qutf.data.load_model_inputs(BATCH_SIZE=BATCH_SIZE)
    # check_input_target(numeric_dict_ds, mean=mean, std=std, data_name="mother")
    inputs = data_handler.preprocess_input_features()
    model = create_model(
        inputs=inputs,
        number_of_labels=len(np.unique(data_handler.labels.values)),
        **hp)
    from IPython import embed; embed()
    #restore_checkpoint(model, './qu_tf_tutorial/checkpoints/_cp-010.ckpt')
    
    # check_input_target(train_data, mean=mean, std=std, data_name="train")
    # check_input_target(validation_data, mean=mean, std=std, data_name="validation")
    # check_input_target(test_data, mean=mean, std=std, data_name="test")

    # Zeigt Anzahl aller wichtiger Sachen usw.
    model.summary()
    callbacks = create_model_callbacks(
        name=this_architecture,
        number_training_samples=data_handler.number_of_train_batches(),
        **hp)

    print("start fit")
    with device:
    # with tf.device("/cpu:0"):
        fit_history = model.fit(
                data_handler.train_data, 
                validation_data=data_handler.validation_data, 
                epochs=EPOCHS,
                verbose=1,
                callbacks=callbacks
        )

    dnnplt.create_plot("dnn_plots", this_architecture, suffix="plots", style=mlp_other_style)   


    dnn_output_path = os.path.join(thisdir, "dnn_models")
    if not os.path.exists(dnn_output_path):
        os.makedirs(dnn_output_path)
    final_path = os.path.join(dnn_output_path, f"{model_architecture}_{this_architecture}")
    model.save(final_path)

    # dnnplt.create_dnn_plots(model=model, prefix=this_architecture, outpath="plots")


def parse_arguments():
    usage="python %prog [options]"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "-d", "--dnn_architecture",
        help = " ".join("""
                name of the dnn architecture to train.
                The name should be the same is in
                'hyperparameters.py'
            """.split()),
        type="str",
        default="baseline",
        dest="dnn_architecture"
    )

    parser.add_option(
        "-m", "--model-architecture",
        help = " ".join("""
                name of the dnn _model_ architecture to train.
                The name should be the same is in
                'dnn_model_definition.py'
            """.split()),
        type="str",
        default="binary_dnn",
        dest="model_architecture"
    )

    options, args = parser.parse_args()

    return options, args
if __name__ == '__main__':
    options, args = parse_arguments()
    main(*args, **vars(options))
