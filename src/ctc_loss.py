import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class CTCLossLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        labels, y_pred, input_length, label_length = inputs
        return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

