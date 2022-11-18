# coding: utf-8

"""
Helpful utilities.
"""

import functools

import six
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tqdm.notebook import tqdm


# print function with auto-flush
print_ = functools.partial(six.print_, flush=True)


E, PX, PY, PZ = range(4)

# calculate deltaR and deltaPhi vectorized


def deltaR2( dataset, eta1, phi1, eta2=None, phi2=None):
    """Take either 4 arguments (eta,phi, eta,phi) or two objects that have 'eta', 'phi' methods)"""
    if(eta2 == None and phi2 == None):
        return deltaR2(eta1.eta(),eta1.phi(), phi1.eta(), phi1.phi())
    de = dataset[eta1] - dataset[eta2]
    dp = deltaPhi(dataset, phi1, phi2)
    return de*de + dp*dp

def deltaR( **args ):
    return np.sqrt( deltaR2(**args) )

def deltaPhi( dataset, p1, p2):
    '''Computes delta phi, handling periodic limit conditions.'''
    
    newcol = dataset[p1] - dataset[p2]
    debug_count = 0
    while (newcol > np.pi).any():
        newcol= np.where(newcol>np.pi, newcol - 2*np.pi, newcol)
        debug_count += 1
        if debug_count == 10:
            print(f"looped in >np.pi condition {debug_count} times!")
    debug_count = 0

    while (newcol < -np.pi).any():
        newcol=np.where(newcol<-np.pi, newcol + 2*np.pi, newcol)
        debug_count += 1
        if debug_count == 10:
            print(f"looped in < -np.pi condition {debug_count} times!")
            from IPython import embed; embed()
    return newcol


# file download helper
def download(src, dst, bar=None):
    import wget
    return wget.download(src, out=dst, bar=bar)


def calculate_accuracy(labels, predictions):
    predicteds_top = np.argmax(predictions, axis=-1) == 1
    labels_top = labels[:, 1] == 1
    return (predicteds_top == labels_top).mean()


def training_loop(dataset_train, dataset_valid, model, loss_fns, optimizer, learning_rate,
                  max_steps=10000, log_every=10, validate_every=100, stack_energy=False): 
    # store the best model, identified by the best validation accuracy
    best_model = None

    # metrics to update during training
    metrics = dict(
        step=0, step_val=0,
        acc_train=0., acc_valid=0., acc_valid_best=0.,
        auc_train=0., auc_valid=0., auc_valid_best=0.,
    )
    for name in loss_fns:
        for kind in ["train", "valid"]:
            metrics[f"loss_{name}_{kind}"] = 0.
    
    # progress bar format
    fmt = ["{percentage:3.0f}% {bar} Step: {pfx[0][step]}/{total}, Validations: {pfx[0][step_val]}"]
    for name in loss_fns:
        fmt.append(f"Loss '{name}': {{pfx[0][loss_{name}_train]:.4f}} | {{pfx[0][loss_{name}_valid]:.4f}}")
    fmt.append("Accuracy: {pfx[0][acc_train]:.4f} | {pfx[0][acc_valid]:.4f} | {pfx[0][acc_valid_best]:.4f}")
    fmt.append("ROC AUC: {pfx[0][auc_train]:.4f} | {pfx[0][auc_valid]:.4f} | {pfx[0][auc_valid_best]:.4f}")
    fmt.append("(loss format: 'last train | last valid', metric format: 'last train | last valid | best valid')")
    fmt = " --- ".join(fmt).replace("pfx", "postfix")

    # helper to update metrics
    def update_metrics(bar, kind, step, labels, predictions, losses):
        # calculate accuracy and roc auc
        acc = calculate_accuracy(labels.numpy(), predictions.numpy())
        auc = roc_auc_score(labels[:, 1], predictions[:, 1])
        # update bar data
        metrics["step"] = step + 1
        metrics[f"acc_{kind}"] = acc
        metrics[f"auc_{kind}"] = auc
        for name, loss in losses.items():
            metrics[f"loss_{name}_{kind}"] = loss
        # validation specific
        if kind == "valid":
            metrics["step_val"] += 1
            metrics["acc_valid_best"] = max(metrics["acc_valid_best"], acc)
            metrics["auc_valid_best"] = max(metrics["auc_valid_best"], auc)
            # return True when this was the best validation step
            return acc == metrics["acc_valid_best"]
    
    # start the loop
    with tqdm(total=max_steps, bar_format=fmt, postfix=[metrics]) as bar:
        for step, (c_vectors, true_vectors, labels) in enumerate(dataset_train):
            if step >= max_steps:
                print(f"{max_steps} steps reached, stopping training")
                break
                
            # when stack_energy is set (see end of the exercise)
            # stack the true energy on top of the labels
            if stack_energy:
                labels = tf.concat([labels, true_vectors[:, :E + 1]], axis=-1)

            # do a train step
            with tf.GradientTape() as tape:
                # get predictions
                predictions = model(c_vectors, training=True)
                # compute all losses and combine them into the total loss
                losses = {
                    name: loss_fn(labels, predictions)
                    for name, loss_fn in loss_fns.items()
                }
                loss = tf.add_n(list(losses.values()))
            # get and propagate gradients
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # logging
            do_log = step % log_every == 0
            if do_log:
                update_metrics(bar, "train", step, labels, predictions, losses)

            # validation
            do_validate = step % validate_every == 0
            if do_validate:
                c_vectors_valid, true_vectors_valid, labels_valid = next(iter(dataset_valid))
                if stack_energy:
                    labels_valid = tf.concat([labels_valid, true_vectors_valid[:, :E + 1]], axis=-1)
                predictions_valid = model(c_vectors_valid, training=False)
                losses_valid = {
                    name: loss_fn(labels_valid, predictions_valid)
                    for name, loss_fn in loss_fns.items()
                }
                is_best = update_metrics(bar, "valid", step, labels_valid, predictions_valid, losses_valid)
                
                # store the best model
                if is_best:
                    best_model = tf.keras.models.clone_model(model)
            
            bar.update()

        # else:
        #     log("dataset exhausted, stopping training")

    print("validation metrics of the best model:")
    print(f"Accuracy: {metrics['acc_valid_best']:.4f}")
    print(f"ROC AUC : {metrics['auc_valid_best']:.4f}")
    
    return best_model, metrics