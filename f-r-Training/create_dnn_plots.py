import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import matplotlib.pyplot as plt
import numpy as np
# from qu_tf_tutorial import qutf
from qutf import data, util, specs, plots
import tensorflow as tf
from tensorflow import keras
import Keras_Sachen_test
from IPython import embed
import dnn_utils
from argparse import ArgumentParser
import importlib

thisdir = os.path.realpath(os.path.dirname(__file__))
mlp_style_path = os.path.join(thisdir, "plot.mplstyle")
# Parameter für Histogramm relative Abweichung
bins = 100

# Funktion zum Speichern der Plots
def create_plot(outpath, fig_name, suffix="lossfunction", style=mlp_style_path):
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    with plt.style.context(style):
        for ext in "png pdf".split():
            outname = os.path.join(outpath, f"{fig_name}_{suffix}.{ext}")
            print(f"saving {outname}")
            plt.savefig(outname,  bbox_inches='tight')
        plt.close('all')


def histogramm2d(x_axes, y_axes, x_beschriftung, y_beschriftung, range, title):
    a = [-5, 200]
    b = [-5, 200]
    fig, axs = plt.subplots(figsize=(20,10), sharey=True)
    sache = axs.hist2d(x_axes, y_axes, bins=80, range = range)
    fig.colorbar(sache[3], ax=axs)
    axs.set_title(f'{title}')
    axs.set_xlabel(f'{x_beschriftung}')
    axs.set_ylabel(f'{y_beschriftung}')
    # Winkelhalbierende mit rein
    with plt.style.context(mlp_style_path):
        axs.plot(a,b, color='red')

def histogramm2d_ohne(x_axes, y_axes, x_beschriftung, y_beschriftung, range, title):
    a = [-5, 200]
    b = [-5, 200]
    fig, axs = plt.subplots(figsize=(20,10), sharey=True)
    sache = axs.hist2d(x_axes, y_axes, bins=80, range = range)
    fig.colorbar(sache[3], ax=axs)
    axs.set_title(f'{title}')
    axs.set_xlabel(f'{x_beschriftung}')
    axs.set_ylabel(f'{y_beschriftung}')

def other_plots(
    thing1,
    range,
    title,
    xlabel,
    legends=[]
    ): 
    with plt.style.context(mlp_style_path):
        fig, axs = plt.subplots(figsize=(20,10))
        axs.hist(thing1, bins=50, histtype='step', density=True, range=range)
        # axs.hist(thing2, bins=50, histtype='step', density=True, range=range)
        axs.set_title(f'{title}')
        axs.set_xlabel(f'{xlabel}')
        axs.set_ylabel('Normierte Anzahl')
        axs.legend(list(reversed(legends)))

def input_plots(
    thing1,
    title,
    xlabel,
    legends=[]
    ): 
    with plt.style.context(mlp_style_path):
        fig, axs = plt.subplots(figsize=(20,10))
        axs.hist(thing1, bins=50, histtype='step', density=True)
        # axs.hist(thing2, bins=50, histtype='step', density=True, range=range)
        axs.set_title(f'{title}')
        axs.set_xlabel(f'{xlabel}')
        axs.set_ylabel('Normierte Anzahl')
        axs.legend(list(reversed(legends)))

def plot_distributions(
    value_dict,
    range,
    title,
    xlabel,
    ylabel="Normierte Anzahl",
    bins=50
):
    legends = []
    with plt.style.context(mlp_style_path):
        fig, axs = plt.subplots(figsize=(20,10))
        for label in value_dict:
            weights = value_dict[label].get("weights", None)
            values = value_dict[label]["values"]
            histtype = value_dict[label].get("histtype", "step")
            style = value_dict[label].get("style", None)
            legends.append(label)

            axs.hist(
                values,
                bins=bins,
                histtype=histtype,
                density=True,
                ls=style,
                weights=weights,
                range=range,
            )
        
        axs.set_title(f'{title}')
        axs.set_xlabel(f'{xlabel}')
        axs.set_ylabel(f'{ylabel}')
        # axs.legend(list(reversed(legends)))
        axs.legend(legends) 

def other_plots_two(
    thing1,
    thing2,
    range,
    title,
    xlabel,
    legends=[],
    weights=None,
    ): 
    with plt.style.context(mlp_style_path):
        fig, axs = plt.subplots(figsize=(20,10))
        weights1 = None
        weights2 = None
        if weights and isinstance(weights, list):
            weights1 = weights[0]
            weights2 = weights[1]
        axs.hist(
            thing1,
            bins=50,
            histtype='step',
            density=True,
            weights=weights1,
            range=range,
        )
        axs.hist(
            thing2,
            bins=50,
            histtype='step',
            density=True,
            ls="dashed",
            range=range,
            weights=weights2,
        )
        axs.set_title(f'{title}')
        axs.set_xlabel(f'{xlabel}')
        axs.set_ylabel('Normierte Anzahl')
        # axs.legend(list(reversed(legends)))
        axs.legend(legends)

def calculate_theta(eta):
    return 2*tf.math.atan(tf.math.exp(-eta))

def calculate_p_from_pt(jet):
    """
    calculates magnitude of momentum from transverse component.
    
    jet(tf.Tensor): Jet with following component (in order!): (nevents, (pT, Phi, Eta))
    """
    sin_the = tf.math.sin(calculate_theta(jet[:, 2]))
    return jet[:, 0]/sin_the

def calculate_angle_metric(jet1, jet2):
    # load angles
    theta1 = calculate_theta(jet1[:, 2])
    theta2 = calculate_theta(jet2[:, 2])
    phi1 = jet1[:, 1]
    phi2 = jet2[:, 1]

    # calculate trigonometric functions
    sin_the1 = tf.math.sin(theta1)
    cos_the1 = tf.math.cos(theta1)
    
    sin_the2 = tf.math.sin(theta2)
    cos_the2 = tf.math.cos(theta2)
    
    sin_phi1 = tf.math.sin(phi1)
    cos_phi1 = tf.math.cos(phi1)
    
    sin_phi2 = tf.math.sin(phi2)
    cos_phi2 = tf.math.cos(phi2)

    # calculate final angular metric
    c = sin_the1*cos_phi1*sin_the2*cos_phi2
    c += sin_the1*sin_phi1*sin_the2*sin_phi2
    c += cos_the1*cos_the2
    return c

# p ist gleich pt/sin(theta)
def calculate_higgs_mass(jet1, jet2, mass_b=tf.constant( 0, dtype=tf.float32)):
    
    p1 = calculate_p_from_pt(jet1)
    p2 = calculate_p_from_pt(jet2)
    
    c = calculate_angle_metric(jet1, jet2)
    higgs_mass = tf.math.sqrt(2*p1*p2*(1-c)+2*tf.math.square(mass_b))
    return tf.reshape(higgs_mass, [-1, 1])

# invariante Masse berechnen
def invariant_mass(e1, e2, p1, p2, c):
    return tf.math.sqrt((e1+e2)**2 - (p1**2 + p2**2 + 2*p1*p2*c))

def confusion_matrix_plot(
    confusion_matrix, 
    output_name, 
    colormap="viridis", 
    change_tick_labels=False, 
    labels=["signal", "background"],
):
    with plt.style.context(mlp_style_path):
        #here kommt das Kommentar der Methode, siehe txt Datei
        fig, ax = plt.subplots()
        plt.imshow(np.array(confusion_matrix), cmap=plt.get_cmap(colormap),
                    interpolation='nearest')
        width, height = confusion_matrix.shape
        for x in range(width):
            for y in range(height):
                ax.annotate(r"$\bf{{{:.2f}}}$".format(confusion_matrix[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')
        plt.xticks(range(width), range(width))
        plt.yticks(range(height), range(height))
        embed()
        if change_tick_labels:
            plt.xticks(range(width), labels)
            plt.yticks(range(height), labels)
        #plt.title('confusion matrix', fontsize=11)
        plt.xlabel(r'Predicted label')
        plt.ylabel(r'True label')
        plt.colorbar()
        fig.tight_layout()
        plt.text(
            0.99,
            1.01,
            r'$\mathbf{CMS}$ $\mathit{Private}$ $\mathit{Work}$',
            horizontalalignment='right',
            verticalalignment='bottom',
            transform=ax.transAxes
        ) 
        
        create_plot( "dnn_plots" , output_name, suffix="")

def roc_curves_plots(
    predictions,
    labels,
    output_name,
    sample_weight=None,
    ax_legend=None,):
    """
    Calculate and plot the ROC-curve for the given model predictions and labels

    Args:
    predictions: the model predictions
    labels: the ground truth values
    """
    try:
        with plt.style.context(mlp_style_path):
            from sklearn.metrics import roc_curve, roc_auc_score
            fpr, tpr, thresholds = roc_curve(
                labels, 
                predictions, 
                pos_label=1, 
                sample_weight=sample_weight
            )

            fig, ax = plt.subplots()
            plt.plot(fpr, tpr, 'r')
            ax.axline((0.5, 0.5), slope=1)
            plt.text(0.3, 0.98, 'A.u.c.={:.5f}'.format(roc_auc_score(labels, predictions)),
                    horizontalalignment='left',
                    verticalalignment='top',
                    transform=ax.transAxes)
            #plt.title('ROC curve', fontsize=11)
            plt.xlabel(r'FPR')
            plt.ylabel(r'TPR')
            plt.text(
                0.99,
                1.01,
                r'$\mathbf{CMS}$ $\mathit{Private}$ $\mathit{Work}$',
                horizontalalignment='right',
                verticalalignment='bottom',
                transform=ax.transAxes
            )
            if ax_legend:
                ax.legend(ax_legend)

            create_plot("dnn_plots", output_name, suffix="") 
    except Exception as e:
        print("error in 'roc_curves_plot':")
        print(e)
        print("start debugging shell")
        embed()   

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

        mean = data_handler.label_means
        std = data_handler.label_stds
        test_data = data_handler.test_data

        nevents = len(test_data)
        train_valid_events = int(np.floor(nevents*(data_handler.train_percentage + data_handler.validation_percentage)))
        input_feature_list = data_handler.input_features.columns
        
        model = tf.keras.models.load_model(
            dnn_folder,
            custom_objects={
                "custom_loss_function" : dnn_utils.custom_loss_function
            }
        )
        # create_dnn_plots(model, std, mean, test_data, prefix)
        if hyperparameters.get("multiclass", False):
            create_dnn_plots_multiclass(
                model=model, 
                std=std, 
                mean=mean,
                test_data=test_data,
                prefix=prefix,
                input_feature_list=list(input_feature_list),
                label_map=hyperparameters.get("label_map", None),
            )
        else:
            create_dnn_plots_binary(
            model=model, 
            std=std, 
            mean=mean,
            test_data=test_data,
            prefix=prefix,
            input_feature_list=list(input_feature_list),
            label_map=hyperparameters.get("label_map", None),
        )


def build_jet_vector(pt, phi, eta):
    tensor_pt = tf.reshape(tf.convert_to_tensor(pt), [-1,1])
    tensor_phi = tf.reshape(tf.convert_to_tensor(phi), [-1,1])
    tensor_eta = tf.reshape(tf.convert_to_tensor(eta), [-1,1])
    return tf.concat([tensor_pt, tensor_phi, tensor_eta], axis=-1)

def filter_values(tensor, value=4):
    mask = (tf.abs(tensor) < value)
    mask = tf.cast(mask, tf.float32)
    print(f"mask valid for {tf.reduce_mean(mask)*100}% of events")
    return mask*tensor

def load_input_data(tf_data):
    inputs = list()
    y = list()
    weights = None
    # check dimensions of input data
    input_dim = len(tf_data.element_spec)
    # if there are no weights, the dimension is 2
    if input_dim == 2:
        for input_vars, truth in tf_data.as_numpy_iterator():
            inputs.append(input_vars)
            y.append(truth)
        inputs = np.vstack(inputs)
        y = np.vstack(y)
    elif input_dim == 3:
        weights = list()
        for input_vars, truth, w in tf_data.as_numpy_iterator():
            inputs.append(input_vars)
            y.append(truth)
            weights.append(w)
        inputs = np.asarray(inputs)
        y = np.asarray(y)
        weights = np.asarray(weights)
    print(inputs,inputs.shape)
    return inputs, y, weights

# fuer Maja: nur sinnvoll um zu gucken wie die Methoden genutzt werden
def create_dnn_plots_multiclass(
    model,
    std,
    mean,
    test_data,
    input_feature_list,
    prefix="baseline",
    outpath="plots",
    label_map=None
):

    #print("test data", test_data)
    pred_vector = model.predict(test_data)
    #print(pred_vector.shape)
    weights = None
    y = None

    test_data = test_data.unbatch()

    
    try:
        inputs, y, weights = load_input_data(test_data)

        # get index of predicted class
        predicted_class_index = np.argmax(pred_vector, axis=1)
        true_class_index = np.argmax(y, axis=1)

        # loop through the predicted classes
        for cls_index in np.unique(predicted_class_index):
            # slice the prediction vector to the current class
            slice_mask = predicted_class_index == cls_index
            
            # select probabilities for events that are categorised as 
            # class 'cls_index'
            current_predictions = (np.take_along_axis(
                pred_vector, np.expand_dims(predicted_class_index, axis=-1), axis=-1)
                .squeeze(axis=-1)
            )[slice_mask]
            if isinstance(weights, np.ndarray):
                current_weights = weights[slice_mask]
            else:
                current_weights = None
            current_truth = true_class_index[slice_mask]
            # embed()
            # other_plots_two(
            #     thing1=pred_vector.flatten()[mask_sig],
            #     thing2=pred_vector.flatten()[mask_bkg],
            #     xlabel="Network Output",
            #     title="",
            #     range=[0, 1],
            #     legends=["Signal", "Background"],
            #     weights=[weights[mask_sig], weights[mask_bkg]] if weights else None,
            # )
            if isinstance(current_weights, np.ndarray):
                value_dict = {
                    label_map.get(str(truth), str(truth)) if label_map else str(truth): {
                        "values": current_predictions[current_truth == truth],
                        "weights": current_weights[current_truth == truth],
                    }
                    for truth in np.unique(current_truth)
                }
            else:
                value_dict = {
                label_map.get(str(truth), str(truth)) if label_map else str(truth): {
                    "values": current_predictions[current_truth == truth],
                    "weights": None,
                }
                for truth in np.unique(current_truth)
            }

            plot_distributions(
                value_dict=value_dict,
                xlabel="Network Output",
                title="",
                range=[0, 1],
            )

            create_plot(
                os.path.join(thisdir, "dnn_plots"),
                f"output_distributions_cls_{cls_index}",
                suffix=prefix
            )
            roc_curves_plots(
                current_predictions,
                np.where(current_truth == cls_index, 1, 0),
                prefix + f"_roc_curves_cls_{cls_index}",
                sample_weight=current_weights,
                ax_legend=[r"{} vs Rest".format(label_map[str(cls_index)])] if label_map else None,
            )
            # predictions_class_labels = np.where(pred_vector.flatten() > 0.5, 1, 0)
        from sklearn.metrics import confusion_matrix
        if label_map:
            true_class_index = translate_labels(label_map=label_map,value_array = true_class_index)
            predicted_class_index = translate_labels(label_map=label_map,value_array = predicted_class_index)
        matrix = confusion_matrix(
            true_class_index,
            predicted_class_index,
            sample_weight=weights,
            normalize="true", # or normalize="pred" depending on what you want
            # 
        ) 
        confusion_matrix_plot(
            matrix, 
            prefix + "_confusion_matrix",
            change_tick_labels = label_map is not None,
            labels=[label_map[x] for x in sorted(label_map.keys())] if label_map else None,
        )  # , colormap="jet", change_tick_labels=True)

    except Exception as e:
        print("error during readout of data!")
        print(e)
        print("start debug shell")
        embed()

    


    print("input_feature_list", input_feature_list, len(input_feature_list))
    if "kappa_lambda" in input_feature_list:
        print("this is a parametrized network")
        kl0_indices=np.where((y==1) & (inputs[:,-1]==0))
        kl1_indices=np.where((y==1) & (inputs[:,-1]==1))
        kl2p45_indices=np.where((y==1) & (inputs[:,-1]==2.45))
        kl5_indices=np.where((y==1) & (inputs[:,-1]==5))
        bkg_indices=np.where(y==0)

        make_parametrized_plots(model, inputs[kl0_indices].copy(), y[kl0_indices], weights[kl0_indices],
                                inputs[bkg_indices].copy(), y[bkg_indices], weights[bkg_indices], 0, prefix, outpath)
        make_parametrized_plots(model, inputs[kl1_indices].copy(), y[kl1_indices], weights[kl1_indices],
                                inputs[bkg_indices].copy(), y[bkg_indices], weights[bkg_indices], 1, prefix, outpath)
        make_parametrized_plots(model, inputs[kl2p45_indices].copy(), y[kl2p45_indices], weights[kl2p45_indices],
                                inputs[bkg_indices].copy(), y[bkg_indices], weights[bkg_indices], 2.45, prefix, outpath)
        make_parametrized_plots(model, inputs[kl5_indices].copy(), y[kl5_indices], weights[kl5_indices],
                                inputs[bkg_indices].copy(), y[bkg_indices], weights[bkg_indices], 5, prefix, outpath)

# fuer Maja: nur sinnvoll um zu gucken wie die Methoden genutzt werden
def create_dnn_plots_binary(
    model,
    std,
    mean,
    test_data,
    input_feature_list,
    prefix="baseline",
    outpath="plots",
    label_map=None

):

    #print("test data", test_data)
    pred_vector = model.predict(test_data)
    #print(pred_vector.shape)
    weights = None
    y = None

    test_data = test_data.unbatch()

    try:
        inputs, y, weights = load_input_data(test_data)
        y = y.squeeze(-1)
        pred_vector = pred_vector.squeeze(-1)
        #embed()
        mask_0 = np.ones_like(y)
        mask_1 = np.ones_like(y)
        if label_map:
            int_class_identifier = np.array(list(label_map.keys()), dtype=np.int)
            sorted_identifiers = np.sort(int_class_identifier)
            # this would be nicer in a single array
            mask_0 = y == sorted_identifiers[0]
            mask_1 = y == sorted_identifiers[1]

        # mask_sig = y == 1.
        # mask_bkg = y == 0.
        other_plots_two(
            thing1=pred_vector[mask_0],
            thing2=pred_vector[mask_1],
            xlabel="Network Output",
            title="",
            range=[0, 1],
            legends=[label_map[str(x)] for x in sorted_identifiers] if label_map else None,
            weights=[weights[mask_0], weights[mask_1]],
        )

        create_plot(os.path.join(thisdir, "dnn_plots"), "output_distributions", suffix=prefix)
    
    except Exception as e:
        print("error during readout of data!")
        print(e)
        print("start debug shell")
        embed()

    roc_curves_plots(
        pred_vector.flatten(),
        y,
        prefix + "_roc_curves",
        sample_weight=weights
    )
    predictions_class_labels = np.where(pred_vector.flatten() > 0.5, 1, 0)

    from sklearn.metrics import confusion_matrix
    matrix = confusion_matrix(
        y,
        predictions_class_labels,
        sample_weight=weights,
        normalize="true" # or normalize="pred" depending on what you want
    ) 

    confusion_matrix_plot(matrix, prefix + "_confusion_matrix",
        change_tick_labels = label_map is not None,
        labels=[label_map[x] for x in sorted(label_map.keys())] if label_map else None,
    )  # , colormap="jet", change_tick_labels=True)

    print("input_feature_list", input_feature_list, len(input_feature_list))
    if "kappa_lambda" in input_feature_list:
        print("this is a parametrized network")
        kl0_indices=np.where((y==1) & (inputs[:,-1]==0))
        kl1_indices=np.where((y==1) & (inputs[:,-1]==1))
        kl2p45_indices=np.where((y==1) & (inputs[:,-1]==2.45))
        kl5_indices=np.where((y==1) & (inputs[:,-1]==5))
        bkg_indices=np.where(y==0)

        make_parametrized_plots(model, inputs[kl0_indices].copy(), y[kl0_indices], weights[kl0_indices],
                                inputs[bkg_indices].copy(), y[bkg_indices], weights[bkg_indices], 0, prefix, outpath)
        make_parametrized_plots(model, inputs[kl1_indices].copy(), y[kl1_indices], weights[kl1_indices],
                                inputs[bkg_indices].copy(), y[bkg_indices], weights[bkg_indices], 1, prefix, outpath)
        make_parametrized_plots(model, inputs[kl2p45_indices].copy(), y[kl2p45_indices], weights[kl2p45_indices],
                                inputs[bkg_indices].copy(), y[bkg_indices], weights[bkg_indices], 2.45, prefix, outpath)
        make_parametrized_plots(model, inputs[kl5_indices].copy(), y[kl5_indices], weights[kl5_indices],
                                inputs[bkg_indices].copy(), y[bkg_indices], weights[bkg_indices], 5, prefix, outpath)


def translate_labels(label_map, value_array):
    """small function to translate integer class identifiers in *value_array* 
    into corresponding labels as defined in *label_map*.

    Args:
        label_map (dict):   Dictionary containing the integer class identifiers
                            as keys and the corresponding labels as values.
                            Format should be e.g.
                            {
                                '0': "Signal",
                                '1': "Background",
                            }
        value_array (np.ndarray):   Array containing the integer class 
                                    identifiers to translate. 


    Returns:
        np.ndarray: numpy Array containing the translated labels
    """
    # first, create an array containing the integer class identifiers 
    # defined in label_map
    dict_keys = np.array(list(label_map.keys()), dtype=np.int)
    # sort the keys
    sort_idx = np.argsort(dict_keys)
    # identify the position of a given integer class identifier specified in 
    # value_array in the label_map
    idx = np.searchsorted(dict_keys,value_array,sorter = sort_idx)
    # finally, extract the labels for the class identifiers
    out = np.asarray(list(label_map.values()))[sort_idx][idx]
    return out

def make_parametrized_plots(model, signal,
                            signal_labels,
                            signal_weights,
                            background,
                            background_labels,
                            background_weights,
                            kl_value,
                            prefix,
                            outpath):
    signal[:,-1]=kl_value
    background[:,-1]=kl_value
    prediction_background = model.predict(background)
    prediction_signal = model.predict(signal)

    other_plots_two(
        thing1=prediction_signal,
        thing2=prediction_background,
        xlabel="Network Output",
        title="",
        range=[0, 1],
        legends=["Signal", "Background"],
        weights=[signal_weights, background_weights],
        )

    create_plot(os.path.join(thisdir, "dnn_plots"), "output_distributions_kl_{}".format(kl_value), suffix=prefix)

    roc_curves_plots(
        np.concatenate([prediction_signal, prediction_background]),
        np.concatenate([signal_labels, background_labels]),
        prefix + "_roc_curves_kl_{}".format(kl_value),
        sample_weight=np.concatenate([signal_weights, background_weights])
    )
    predictions_class_labels = np.where(np.concatenate([prediction_signal, prediction_background]) > 0.5, 1, 0)

    from sklearn.metrics import confusion_matrix
    matrix = confusion_matrix(
        np.concatenate([signal_labels, background_labels]),
        predictions_class_labels,
        sample_weight=np.concatenate([signal_weights, background_weights]),
        normalize="true" # or normalize="pred" depending on what you want
    )

    confusion_matrix_plot(matrix, prefix + "_confusion_matrix_kl_{}".format(kl_value))  # , colormap="jet", change_tic


def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument("--hyperparameters", "-p", type=str,
        help="path to config file for hyper parameters",
        default=os.path.join(thisdir, "hyperparameters.py"),
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
        default=os.path.join(thisdir, "variables.json"),
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    main(**vars(args))
