import tensorflow as tf

def build_autoencoder(input_dim: int, bottleneck_dim: int = 16) -> tf.keras.Model:
    """
    Returns an autoencoder Model:
      - input_dim: number of features
      - bottleneck_dim: size of the central compressed layer
    """
    # Input layer
    inputs = tf.keras.Input(shape=(input_dim,), name="input_layer")

    # Encoder
    x = tf.keras.layers.Dense(64, activation="relu", name="encoder_1")(inputs)
    x = tf.keras.layers.Dense(32, activation="relu", name="encoder_2")(x)
    z = tf.keras.layers.Dense(bottleneck_dim, activation="relu", name="bottleneck")(x)

    # Decoder
    x = tf.keras.layers.Dense(32, activation="relu", name="decoder_1")(z)
    x = tf.keras.layers.Dense(64, activation="relu", name="decoder_2")(x)
    outputs = tf.keras.layers.Dense(input_dim, activation="linear", name="output_layer")(x)

    # Build the model
    autoencoder = tf.keras.Model(inputs=inputs, outputs=outputs, name="autoencoder")
    return autoencoder
