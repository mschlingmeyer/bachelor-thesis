import tensorflow as tf
from tensorflow import keras

class EarlyStopping(keras.callbacks.Callback):
    ''' custom implementation of early stopping
        with options for
            - stopping when val/train loss difference exceeds a percentage threshold
            - stopping when val loss hasnt increased for a set number of epochs '''

    def __init__(
        self,
        monitor = "loss",
        value = None,
        min_epochs = 20,
        stopping_epochs = None,
        patience = 10,
        verbose = 1,
        restore_best_weights = True,
    ):
        super(keras.callbacks.Callback, self).__init__()
        self.val_monitor = "val_"+monitor
        self.train_monitor = monitor
        self.patience = patience
        self.n_failed = 0

        self.stopping_epochs = stopping_epochs
        self.best_epoch = 0
        self.best_validation = 999.
        self.best_model_weights = None
        self.restore_best_weights = restore_best_weights
        self.min_epochs = min_epochs
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs = {}):
        current_val = logs.get(self.val_monitor)
        if epoch == 0:
            self.best_validation = current_val
        current_train = logs.get(self.train_monitor)

        if current_val is None or current_train is None:
            warnings.warn("Early stopping requires {} and {} available".format(
                self.val_monitor, self.train_monitor), RuntimeWarning)

        # check loss by percentage difference
        if self.value:
            if (current_val-current_train)/(current_train) > self.value and epoch > self.min_epochs:
                self.n_failed += 1
                if self.verbose > 0:
                    print("\nEpoch {}: early stopping threshold reached".format(epoch))
                    print(f"\nCriteria met {self.n_failed}/{self.patience} epochs")
                if self.n_failed > self.patience:
                    if self.best_model_weights and self.restore_best_weights:
                        print(f"Setting weights back to epoch {self.best_epoch}")
                        self.model.set_weights(self.best_model_weights)
                    self.model.stop_training = True
            else:
                self.n_failed=0
                if current_val < self.best_validation:
                    self.best_validation = current_val
                    self.best_epoch = epoch
                    self.best_model_weights = self.model.get_weights()

        # check loss by validation performance increase
        if self.stopping_epochs:
            if self.best_epoch + self.stopping_epochs < epoch and epoch > self.min_epochs:
                if self.verbose > 0:
                    print("\nValidation loss has not decreased for {} epochs".format( epoch - self.best_epoch ))
                self.model.stop_training = True
    
def custom_loss_wrapper(weights):
    def loss(true, pred):
        sum_weights = tf.reduce_sum(weights) * tf.cast(tf.shape(pred)[0], tf.float32)
        resid = tf.sqrt(tf.reduce_sum(weights * tf.square(true - pred)))
        return resid/sum_weights
    return loss

def custom_loss_function(y_true, y_pred):
   bce = tf.keras.losses.BinaryCrossentropy()
   return bce(y_true, y_pred)
