import tensorflow as tf
import livelossplot as llp
import pandas as pd
from dnn_utils import EarlyStopping, custom_loss_function

def create_model(
    inputs, 
    n_hidden_layers=3, 
    n_nodes_per_layer=100, 
    activation="relu", 
    l2_norm = 1e-3,
    dropout_rate = None,
    **kwargs
):
    # x = stack_dict(inputs, fun=tf.concat)
    N = inputs.shape[1]*n_nodes_per_layer+((n_hidden_layers-1)*n_nodes_per_layer**2)+n_nodes_per_layer*2
    x = inputs
    # from IPython import embed; embed()
    # x = tf.keras.Input(shape=(len(inputs.columns),))
    # normalizer.adapt( stack_dict(dict(input_features)))
    a = tf.keras.layers.BatchNormalization(axis=-1)(x)
    # a = x
    regulizer = tf.keras.regularizers.L2(l2_norm / N)
    for i in range(n_hidden_layers):
        a = tf.keras.layers.Dense(
            n_nodes_per_layer,
            use_bias=True,
            kernel_regularizer=regulizer)(a)
        a = tf.keras.layers.BatchNormalization(axis=-1)(a)
        if activation == "prelu":
            a = tf.keras.layers.PReLU()(a)
        else:
            a = tf.keras.layers.Activation(activation)(a)
        if dropout_rate:
            a = tf.keras.layers.Dropout(rate=dropout_rate)
    # x = tf.keras.layers.Dense(10, activation='tanh')(x)
    y = tf.keras.layers.Dense(2, activation='linear')(a)

    model = tf.keras.Model(inputs=inputs, outputs=y)

    # Model durchfuehren mit Loss-Funktion und Metriken
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=custom_loss_function,
                metrics=[
                    tf.keras.metrics.MeanSquaredError(),
                    # tf.keras.metrics.KLDivergence(),
                    tf.keras.metrics.MeanAbsoluteError()
                    ],
                run_eagerly=False)
    return model


def create_binary_model(
    inputs, 
    n_hidden_layers=3, 
    n_nodes_per_layer=100, 
    activation="relu", 
    l2_norm = 1e-3,
    dropout_rate = None,
    **kwargs
):
    # x = stack_dict(inputs, fun=tf.concat)
    N = inputs.shape[1]*n_nodes_per_layer+((n_hidden_layers-1)*n_nodes_per_layer**2)+n_nodes_per_layer*2
    x = inputs
    # from IPython import embed; embed()
    # x = tf.keras.Input(shape=(len(inputs.columns),))
    a = tf.keras.layers.BatchNormalization(axis=-1)(x)
    regulizer = tf.keras.regularizers.L2(l2_norm / N)
    for i in range(n_hidden_layers):
        if i==0:
            a = tf.keras.layers.Dense(
                n_nodes_per_layer,
                use_bias=True,
                kernel_regularizer=regulizer)(a)
            #a = tf.keras.layers.BatchNormalization(axis=-1)(a) 
        else:
            a = tf.keras.layers.Dense(
                n_nodes_per_layer,
                use_bias=True,
                kernel_regularizer=regulizer)(a)
            #a = tf.keras.layers.BatchNormalization(axis=-1)(a) 
        a = tf.keras.layers.Activation(activation)(a)
        a = tf.keras.layers.BatchNormalization(axis=-1)(a) 
    y = tf.keras.layers.Dense(1, activation='sigmoid')(a)

    model = tf.keras.Model(inputs=inputs, outputs=y)

    # Model durchfuehren mit Loss-Funktion und Metriken
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=custom_loss_function,
                metrics=[
                    # tf.keras.metrics.MeanSquaredError(),
                    # tf.keras.metrics.KLDivergence(),
                    # tf.keras.metrics.MeanAbsoluteError(),
                    tf.keras.metrics.Accuracy(),
                    tf.keras.metrics.BinaryAccuracy(),
                    tf.keras.metrics.BinaryCrossentropy(),
                    ],
                run_eagerly=False)
    return model

def create_multiclass_model(
    inputs, 
    n_hidden_layers=3, 
    n_nodes_per_layer=100, 
    activation="relu", 
    l2_norm = 1e-3,
    number_of_labels=1,
    dropout_rate = None,
    **kwargs
):
    # x = stack_dict(inputs, fun=tf.concat)
    N = inputs.shape[1]*n_nodes_per_layer+((n_hidden_layers-1)*n_nodes_per_layer**2)+n_nodes_per_layer*2
    x = inputs
    # from IPython import embed; embed()
    # x = tf.keras.Input(shape=(len(inputs.columns),))
    a = tf.keras.layers.BatchNormalization(axis=-1)(x)
    regulizer = tf.keras.regularizers.L2(l2_norm / N)
    for i in range(n_hidden_layers):
        if i==0:
            a = tf.keras.layers.Dense(
                n_nodes_per_layer,
                use_bias=True,
                kernel_regularizer=regulizer)(a)
            #a = tf.keras.layers.BatchNormalization(axis=-1)(a) 
        else:
            a = tf.keras.layers.Dense(
                n_nodes_per_layer,
                use_bias=True,
                kernel_regularizer=regulizer)(a)
            #a = tf.keras.layers.BatchNormalization(axis=-1)(a) 
        a = tf.keras.layers.Activation(activation)(a)
        a = tf.keras.layers.BatchNormalization(axis=-1)(a) 
    y = tf.keras.layers.Dense(number_of_labels, activation='softmax')(a)

    model = tf.keras.Model(inputs=inputs, outputs=y)

    # Model durchfuehren mit Loss-Funktion und Metriken
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=[
                    # tf.keras.metrics.MeanSquaredError(),
                    tf.keras.metrics.KLDivergence(),
                    # tf.keras.metrics.MeanAbsoluteError(),
                    tf.keras.metrics.CategoricalAccuracy(),
                    # tf.keras.metrics.BinaryAccuracy(),
                    tf.keras.metrics.CategoricalCrossentropy(),
                    ],
                run_eagerly=False)
    return model

dnn_model_architecture = {
    "base": create_model,
    "binary_dnn": create_binary_model,
    "multiclass_dnn": create_multiclass_model,
}