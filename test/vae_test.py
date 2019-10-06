import os
import sys
import warnings
sys.path.insert(0, os.getcwd())
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Machine Learning Libraries
from sklearn.datasets import *
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam
from tensorflow.keras.backend import get_value

from vae import *

def test_breast_cancer():
    """binary classification test
    Reference:
        https://www.programcreek.com/python/example/104690/sklearn.datasets.load_breast_cancer
    """
    X, y = load_breast_cancer(return_X_y=True)
    assert len(X) == len(y)
    input_shape = X[0].shape
    original_dim = np.prod(X[0].shape)
    enable_mse = False
    print("\n--------------------------------------")
    print("%d samples" % len(X))
    print("input_shape:", input_shape)
    print("--------------------------------------")
    scaler = MinMaxScaler()
    scaler.fit(X)
    transformed_X = scaler.transform(X)
    enable_graph = False

    vae, encoder, decoder, vae_loss = build_vae(input_shape, enable_mse=enable_mse,
                                        enable_graph=enable_graph)
    vae.compile(optimizer=Adam(0.0002, 0.5), loss=vae_loss)
    vae.fit(transformed_X, transformed_X, batch_size=128, epochs=300)

    # Test
    reconstruct_X = vae.predict(transformed_X)
    z_mean, z_log_var, _ = encoder.predict(transformed_X)
    print(transformed_X)
    print(reconstruct_X)
    loss_instance = vae_loss(y_pred=reconstruct_X, y_true=transformed_X)
    print(get_value(loss_instance))

def test_mnist():
    enable_mse = False

    # MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    image_size = x_train.shape[1]
    original_dim = image_size * image_size
    x_train = np.reshape(x_train, [-1, original_dim])
    x_test = np.reshape(x_test, [-1, original_dim])
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    data = (x_test, y_test)

    # network parameters
    input_shape = (original_dim, )
    batch_size = 128
    epochs = 50
    enable_graph = False

    vae, encoder, decoder, vae_loss = build_vae(input_shape, enable_mse=enable_mse,
                                        enable_graph=enable_graph)
    models = (encoder, decoder)

    vae.compile(optimizer='adam', loss=vae_loss)
    vae.summary()
    if enable_graph:
        plot_model(vae, to_file='vae_mlp.png', show_shapes=True)

    # train the variational autoencoder
    # feed the same tensor for X and y:
    # https://github.com/tensorflow/tensorflow/issues/21894
    vae.fit(x_train, x_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, x_test))
    vae.save_weights('vae_mlp_mnist.h5')

    plot_results(models,
                 data,
                 batch_size=batch_size,
                 model_name="vae_mlp")
