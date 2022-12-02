import numpy as np
from copy import deepcopy
dnn_architectures = {
    "baseline":
    {
        # Parameter in Keras-Sachen_test.py
        # Parameter in def main
        # "batch_size" : 1000,
        "epochs" : 50,
        "checkpoint_frequency": 50,
        "early_stopping_criteria": 0.1,
        "early_stopping_patience": 30,

        # Parameter in def main im lr_schedule
        # learning rate normalerweise zwischen 1e-3 und 1e-5
        "initial_learning_rate":1e-4,
        "decay_steps":100000,
        "decay_rate":0.9,

        # Parameter in create_model
        # activation kann auch prelu sein
        # zwischen 2 und 5 layer, zwischen 50 und 500 nodes
        "n_hidden_layers":2,
        "n_nodes_per_layer":250,
        "l2_norm":1000,
        "activation": "relu",
        

        # Parameter in data.py
        "BATCH_SIZE":500,
    },
  "Alice":
    {
        # Parameter in Keras-Sachen_test.py
        # Parameter in def main
        # "batch_size" : 1000,
        "epochs" : 150,
        "checkpoint_frequency": 50,
        "early_stopping_criteria": 0.1,
        "early_stopping_patience": 30,

        # Parameter in def main im lr_schedule
        # learning rate normalerweise zwischen 1e-3 und 1e-5
        "initial_learning_rate":1e-4,
        "decay_steps":1000,
        "decay_rate":0.4,

        # Parameter in create_model
        # activation kann auch prelu sein
        # zwischen 2 und 5 layer, zwischen 50 und 500 nodes
        "n_hidden_layers":2,
        "n_nodes_per_layer":250,
        "l2_norm":1000,
        "activation": "relu",
        

        # Parameter in data.py
        "BATCH_SIZE":500,  
    },

  "Bob":
    {
        # Parameter in Keras-Sachen_test.py
        # Parameter in def main
        # "batch_size" : 1000,
        "epochs" : 150,
        "checkpoint_frequency": 50,
        "early_stopping_criteria": 0.1,
        "early_stopping_patience": 30,

        # Parameter in def main im lr_schedule
        # learning rate normalerweise zwischen 1e-3 und 1e-5
        "initial_learning_rate":1e-3,
        "decay_steps":1000,
        "decay_rate":0.4,

        # Parameter in create_model
        # activation kann auch prelu sein
        # zwischen 2 und 5 layer, zwischen 50 und 500 nodes
        "n_hidden_layers":4,
        "n_nodes_per_layer":250,
        "l2_norm":1000,
        "activation": "relu",
        

        # Parameter in data.py
        "BATCH_SIZE":500,  
    },

     "Charles":
    {
        # Parameter in Keras-Sachen_test.py
        # Parameter in def main
        # "batch_size" : 1000,
        "epochs" : 150,
        "checkpoint_frequency": 50,
        "early_stopping_criteria": 0.1,
        "early_stopping_patience": 30,

        # Parameter in def main im lr_schedule
        # learning rate normalerweise zwischen 1e-3 und 1e-5
        "initial_learning_rate":1e-3,
        "decay_steps":1000,
        "decay_rate":0.4,

        # Parameter in create_model
        # activation kann auch prelu sein
        # zwischen 2 und 5 layer, zwischen 50 und 500 nodes
        "n_hidden_layers":4,
        "n_nodes_per_layer":400,
        "l2_norm":1e-3,
        "activation": "relu",
        

        # Parameter in data.py
        "BATCH_SIZE":500,  
    },

}

dnn_architectures["Doris"] = deepcopy(dnn_architectures["Charles"])
dnn_architectures["Doris"]["training_weight_names"] = ["class_weights"]

dnn_architectures["Doris_with_plot_weight"] = deepcopy(dnn_architectures["Doris"])
dnn_architectures["Doris_with_plot_weight"]["training_weight_names"] = ["class_weights", "plot_weight"]

dnn_architectures["Doris_plot_and_lumi_weight"] = deepcopy(dnn_architectures["Doris"])
dnn_architectures["Doris_plot_and_lumi_weight"]["training_weight_names"] = ["class_weights", "plot_weight", "lumi_weight"]
dnn_architectures["Doris_plot_and_lumi_weight"]["epochs"] = 800

dnn_architectures["Doris_equalize_bkg_sig"] = deepcopy(dnn_architectures["Doris_plot_and_lumi_weight"])
dnn_architectures["Doris_equalize_bkg_sig"]["training_weight_names"] += ["weight_equalize_sig_bkg"]


dnn_architectures["Doris_equalize_bkg_sig_v2"] = deepcopy(dnn_architectures["Doris_plot_and_lumi_weight"])
dnn_architectures["Doris_global_weight"] = deepcopy(dnn_architectures["Doris_plot_and_lumi_weight"])
dnn_architectures["Doris_global_weight"]["training_weight_names"] += ["weight_global"]
start = 1
end=7
for weight in np.logspace(start=start, stop=end, num=end-start+1, base=10, dtype="int"):
    colname = f"Doris_sig_{weight}"
    dnn_architectures[colname] = deepcopy(dnn_architectures["Doris_equalize_bkg_sig"])
    dnn_architectures[colname]["training_weight_names"] += [f"signal_weight_{int(weight)}"]


# so wie new, nur andere lr
# dnn_architectures["new_lr_lower"] = dnn_architectures["new"]

# dnn_architectures["new_lr_lower"]["initial_learning_rate"] = 1e-5

# so wie ohne_prelu, nur andere lr
# dnn_architectures["ohne_prelu_lr_lower"] = dnn_architectures["ohne_prelu"]

# dnn_architectures["ohne_prelu_lr_lower"]["initial_learning_rate"] = 1e-5

# so wie ohne_prelu, nur andere lr und epochen
# dnn_architectures["ohne_prelu_mehr_epochen"]=dnn_architectures["ohne_prelu_lr_lower"]

# dnn_architectures["ohne_prelu_mehr_epochen"]["epochs"] = 50

# so wie new, nur andere lr und epochen
# dnn_architectures["new_mehr_epochen"]=dnn_architectures["new_lr_lower"]

# dnn_architectures["new_mehr_epochen"]["epochs"] = 50

# wie mit_prelu, nur andere lr und epochen
# dnn_architectures["mit_prelu_erweitert"] = dnn_architectures["mit_prelu"]

# dnn_architectures["mit_prelu_erweitert"]["initial_learning_rate"] = 1e-5
# dnn_architectures["mit_prelu_erweitert"]["epochs"] = 50