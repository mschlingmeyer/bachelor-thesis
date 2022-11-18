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

def other_plots_two(
    thing1,
    thing2,
    range,
    title,
    xlabel,
    legends=[]
    ): 
    with plt.style.context(mlp_style_path):
        fig, axs = plt.subplots(figsize=(20,10))
        axs.hist(thing1, bins=50, histtype='step', density=True, range=range)
        axs.hist(thing2, bins=50, histtype='step', density=True, ls="dashed", range=range)
        axs.set_title(f'{title}')
        axs.set_xlabel(f'{xlabel}')
        axs.set_ylabel('Normierte Anzahl')
        axs.legend(list(reversed(legends)))

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

def confusion_matrix_plot(confusion_matrix, output_name, colormap="viridis", change_tick_labels=False, xtick_labels=["signal", "background"], ytick_labels=["signal", "background"]):
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
    if change_tick_labels:
        plt.xticks(range(width), xtick_labels)
        plt.yticks(range(height), ytick_labels)
    #plt.title('confusion matrix', fontsize=11)
    plt.xlabel(r'Predicted label', fontsize=20)
    plt.ylabel(r'True label', fontsize=20)
    plt.colorbar()
    fig.tight_layout()
    plt.text(0.99,1.01,r'$\mathbf{CMS}$ $\mathit{Private}$ $\mathit{Work}$',horizontalalignment='right',verticalalignment='bottom',transform=ax.transAxes) 
    
    create_plot( "dnn_plots" , output_name, suffix="")

def roc_curves_plots(predictions, labels, output_name):
    """
    Calculate and plot the ROC-curve for the given model predictions and labels

    Args:
    predictions: the model predictions
    labels: the ground truth values
    """
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, thresholds=roc_curve(labels, predictions, pos_label=1)

    fig, ax = plt.subplots()
    plt.plot(fpr, tpr, 'r')
    ax.axline((0.5, 0.5), slope=1)
    plt.text(0.04, 0.98, 'A.u.c.={:.5f}'.format(roc_auc_score(labels, predictions)),
             horizontalalignment='left',
             verticalalignment='top',
             transform=ax.transAxes)
    #plt.title('ROC curve', fontsize=11)
    plt.xlabel(r'FPR', fontsize=11)
    plt.ylabel(r'TPR', fontsize=11)
    plt.text(0.99,1.01,r'$\mathbf{CMS}$ $\mathit{Private}$ $\mathit{Work}$',horizontalalignment='right',verticalalignment='bottom',transform=ax.transAxes)

    create_plot("dnn_plots", output_name, suffix="")    

def main(dnn_folders=sys.argv[1:]):
    data_handler = data.DataHandler(variable_config_path=os.path.join(thisdir, "variables.json"))
    mean = data_handler.label_means
    std = data_handler.label_stds
    test_data = data_handler.test_data

    nevents = len(test_data)
    train_valid_events = int(np.floor(nevents*(data_handler.train_percentage + data_handler.validation_percentage)))
    input_feature_list = data_handler.input_features.columns
    for dnn_folder in dnn_folders:
        current_dir = dnn_folder
        if current_dir.endswith(os.path.sep):
            current_dir=os.path.dirname(current_dir)
        prefix = os.path.basename(current_dir)
        model = tf.keras.models.load_model(dnn_folder, custom_objects={"custom_loss_function":dnn_utils.custom_loss_function})
        # create_dnn_plots(model, std, mean, test_data, prefix)
        create_dnn_plots(
            model=model, 
            std=std, 
            mean=mean,
            test_data=test_data,
            prefix=prefix,
            input_feature_list=list(input_feature_list),    
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



# fuer Maja: nur sinnvoll um zu gucken wie die Methoden genutzt werden
def create_dnn_plots(
    model,
    std,
    mean,
    test_data,
    input_feature_list,
    prefix="baseline",
    outpath="plots"  
):

    #print("test data", test_data)

    pred_vector = model.predict(test_data)
    #print(pred_vector.shape)
    y = np.concatenate([y for x, y in test_data], axis=0)
    test_data = test_data.unbatch()
    #print(y,y.shape)


 
    plt.hist(pred_vector.flatten()[y==1.])
    
    plt.hist(pred_vector.flatten()[y==0.])
    
   

    roc_curves_plots(pred_vector.flatten(),y, prefix + "_roc_curves")
    predictions_class_labels = np.where(pred_vector.flatten() > 0.5, 1, 0)

    from sklearn.metrics import confusion_matrix
    matrix = confusion_matrix(y, predictions_class_labels, sample_weight=None, normalize="true") # or normalize="pred" depending on what you want

    confusion_matrix_plot(matrix, prefix + "_confusion_matrix")  # , colormap="jet", change_tick_labels=True)

    # pred_vector = model.predict(test_data)



if __name__ == '__main__':
    main()