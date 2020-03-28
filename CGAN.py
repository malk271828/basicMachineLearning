#------------------------------------------------------------------------------
# Python 3.5
# @author Masato Tsuchiya
#------------------------------------------------------------------------------
import unittest
import sys
import numpy as np
from enum import Enum
import warnings
from collections.abc import Iterable
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
import time

# Machine Learning Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from keras.models import Sequential, Model
from keras.layers import *
from keras import optimizers
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras import backend as K
from keras import losses

# Other 3rd party modules
from pysnooper import snoop

# Original APIs
sys.path.insert(0, 'Common')
from GoF.AbstractFactory import AbstractFactory

# Data params
data_mean = 4
data_stddev = 1.25

def extract(v):
    return v.data.storage().tolist()

def stats(d):
    return [np.mean(d), np.std(d)]

# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def build_keras_generator(kwargs):
    """
        :param latent_dim: shape of latent space
        :type latent_dim: 1 dimensional tuple including integers
        :return: generator model
        :ref: https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/
            https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-1-dimensional-function-from-scratch-in-keras/
    """
    GRAPHVIZ_ENABLE = False
    input_shape  = kwargs["input_shape"]
    num_class = kwargs["num_class"]
    hidden_size  = kwargs["hidden_size"]
    output_shape = kwargs["output_shape"]
    latent_dim = kwargs["latent_dim"]
    model_type = kwargs["model_type"]
    verbose = kwargs["verbose"]

    if type(input_shape) == tuple:
        input_shape = input_shape[0]
    if type(output_shape) == tuple:
        output_shape = output_shape[0]

    if not isinstance(input_shape, Iterable):
        input_shape = (input_shape,)

    model = Sequential()
    if model_type=="linear":
        """
        https://androidkt.com/linear-regression-model-in-keras/
        """
        model.add(Dense(units=hidden_size,
                        activation="relu",
                        input_shape=input_shape))
        model.add(Dense(units=output_shape,
                        activation='relu'))
        if verbose > 0:
            model.summary()
        return model
    elif model_type=="2dcnn":
        model.add(Dense(units=32, input_shape=input_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(units=64))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(units=128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(units=np.prod(input_shape), activation='tanh'))
        model.add(Reshape(target_shape=input_shape))
        return model
    elif model_type=="lstm":
        model.add(LSTM(32, input_shape=input_shape, return_sequences=True, stateful=False, name="LSTM_Generator"))
        model.add(Dense(output_shape, activation='relu'))
    elif model_type=="vae":
        # https://keras.io/examples/variational_autoencoder/
        original_dim = np.prod(input_shape)
        intermediate_dim = latent_dim

        # VAE model = encoder + decoder
        # build encoder model
        inputs = Input(shape=input_shape, name='encoder_input')
        x = Dense(intermediate_dim, activation='relu')(inputs)
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

        # instantiate encoder model
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        encoder.summary()
        if GRAPHVIZ_ENABLE:
            plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

        # build decoder model
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        x = Dense(intermediate_dim, activation='relu')(latent_inputs)
        outputs = Dense(units=np.prod(input_shape), activation='sigmoid')(x)

        # instantiate decoder model
        decoder = Model(latent_inputs, outputs, name='decoder')
        decoder.summary()
        if GRAPHVIZ_ENABLE:
            plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

        # instantiate VAE model
        outputs = decoder(encoder(inputs)[2])
        vae = Model(inputs, outputs, name='vae_mlp')

        # define reconstruction loss
        # https://towardsdatascience.com/advanced-keras-constructing-complex-custom-losses-and-metrics-c07ca130a618
        reconstruction_loss = losses.binary_crossentropy(inputs, outputs)
        reconstruction_loss *= original_dim
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)

        return vae
    else:
        raise ValueError("unknown model")

    # build conditional inputs layer
    noise = Input(shape=input_shape)
    cInput= Input(shape=(1, ))
    model_input = multiply([noise, cInput])
    generator_out = model(model_input)
    out = Model(inputs=[noise, cInput], outputs=generator_out)

    if verbose > 0:
        out.summary()
        if verbose > 1:
            plot_model(out, to_file='generator.png', show_shapes=True)
    return out

def build_keras_discriminator(kwargs):
    """
        @return discriminator model
    """
    input_shape  = kwargs["input_shape"]
    num_class = kwargs["num_class"]
    hidden_size  = kwargs["hidden_size"]
    output_shape = kwargs["output_shape"]
    latent_dim = kwargs["latent_dim"]
    model_type = kwargs["model_type"]
    verbose = kwargs["verbose"]

    if type(output_shape) == tuple:
        output_shape = output_shape[0]

    model = Sequential()
    if model_type=="linear":
        model = Sequential()
        model.add(Dense(units=hidden_size, activation="relu", kernel_initializer="he_uniform", input_shape=input_shape))
        model.add(Dense(units=num_class, activation='softmax'))
        if verbose > 0:
            model.summary()
        return model
    elif model_type=="1dcnn":
        model.add(Conv1D(filters=32, kernel_size=3, input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Conv1D(32,3))
        model.add(Activation('relu'))
        model.add(MaxPool1D(pool_size=2))
        model.add(Conv1D(64,3))
        model.add(Activation('relu'))
        model.add(MaxPool1D(pool_size=2))
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(Dropout(1.0))
        model.add(Dense(num_class, activation='softmax'))
        return model
    elif model_type=="2dcnn":
        model.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Conv2D(32, kernel_size=(3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(Dropout(1.0))
        model.add(Dense(num_class, activation='softmax'))
        return model
    elif model_type=="lstm":
        model.add(LSTM(32, input_shape=input_shape, stateful=False, name="LSTM_Discriminator"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='relu'))
    elif model_type=="vae":
        # VAE model = encoder + decoder
        # build encoder model
        inputs = Input(shape=input_shape, name='encoder_input')
        x = Dense(intermediate_dim, activation='relu')(inputs)
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

        # instantiate encoder model
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        encoder.summary()
        plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

        # build decoder model
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        x = Dense(intermediate_dim, activation='relu')(latent_inputs)
        outputs = Dense(units=np.prod(input_shape), activation='sigmoid')(x)

        # instantiate decoder model
        decoder = Model(latent_inputs, outputs, name='decoder')
        decoder.summary()
        plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

        # instantiate VAE model
        outputs = decoder(encoder(inputs)[2])
        model = Model(inputs, outputs, name='vae_mlp')
    else:
        raise ValueError("unknown model")

    latent = Input(shape=input_shape)
    cinput = Input(shape=(1,))
    model_input = Concatenate([latent, cinput])
    validity = model(model_input)

    return Model([latent, cinput], validity)

class adversarialTrainer:
    def __init__(self, kwargs):
        self.timeSteps = 50
        self.num_classes = 30
        self.batch_size = kwargs["batch_size"]
        self.img_shape = kwargs["input_shape"]
        self.lr = kwargs["learning_rate"]
        self.decay = 6e-8

    def _iterate_minibatches(self, inputs, targets, batchsize, shuffle=False):
        assert inputs.shape[0] == targets.shape[0]
        if shuffle:
            indices = np.arange(inputs.shape[0])
            np.random.shuffle(indices)
        for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]

    def _buildAdversarialModel(self, Generator, Discriminator):

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        Discriminator.compile(loss=["mean_squared_error"],
            optimizer=optimizer,
            metrics=['accuracy'])

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=self.img_shape) # (batch_size,) + input_size
        cInput = Input(shape=(1,)) # (batch_size,) + (1,)
        img = Generator([noise, cInput]) 

        # For the combined model we will only train the generator
        Discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = Discriminator([img, cInput])

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        combined = Model([noise, cInput], valid)
        combinediscriminator.compile(loss=['binary_crossentropy'], optimizer=optimizer)

        return combined

    @snoop()
    def fit(self, X, y, generator, discriminator, epochs=3000):

        # build adversarial model
        self.combined = self._buildAdversarialModel(generator, discriminator)
        self.combinediscriminator.summary()
        plot_model(self.combined, to_file='model.png', show_shapes=True)

        X_train, y_train = X, y

        # Adversarial ground truths
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))

        for epoch in range(epochs):
            for batch in self._iterate_minibatches(X, y, self.batch_size, shuffle=True):
                imgs, labels = batch
                print(imgs.shape, labels.shape)

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Sample noise as generator input
                noise = np.random.normal(0.999, 1.001, ((self.batch_size,) + self.img_shape))

                # Generate a half batch of new images
                gen_imgs = generator.predict([noise, labels])
                print("gen_imgs:", np.shape(gen_imgs))
                if epoch % 100 == 0:
                    print(np.shape(labels), np.shape(gen_imgs))
                    #visualize(labels[0], imgs[0], gen_imgs[0])

                # Train the discriminator
                d_loss_real = discriminator.train_on_batch([imgs, labels], valid)
                d_loss_fake = discriminator.train_on_batch([gen_imgs, labels], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Train Generator
                # ---------------------

                # Condition on labels
                sampled_labels = np.random.randint(0, 10, (self.batch_size, 1))

                # Train the generator
                g_loss = self.combinediscriminator.train_on_batch([noise, sampled_labels], valid)

                # Plot the progress
                print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))


class CGANFactory(AbstractFactory):
    def __init__(self, input_shape: tuple,
                       output_shape: tuple,
                       num_class: int,
                       hidden_size: int,
                       learningRate: float,
                       optimBetas: float,
                       batchSize: int,
                       timeSteps: int,
                       generator_type: str,
                       discriminator_type: str):
        """
        Abstract Factory Class for Conditional Generative Adversarial Network

        How to use
        ----------
        To obtain a gan family product, invoke a createProductFamily() method
        with the argument specifying target backend system: ["keras"/"pytorch"]
        E.generator.
            generator, discriminator, trainer = ganfactory.createProductFamily("pytorch")
        """
        self.learningRate = learningRate
        self.optimBetas = optimBetas
        self.batchSize = batchSize
        self.timeSteps = timeSteps

        # register constructor
        self.registerConstructor("keras", build_keras_generator, 
                                {
                                    "input_shape": input_shape,
                                    "num_class" : num_class,
                                    "hidden_size": hidden_size,
                                    "output_shape": output_shape,
                                    "latent_dim": 32,
                                    "model_type": generator_type,
                                    "verbose": 1
                                })
        self.registerConstructor("keras", build_keras_discriminator,
                                {
                                    "input_shape": input_shape,
                                    "num_class" : num_class,
                                    "hidden_size": hidden_size,
                                    "output_shape": output_shape,
                                    "latent_dim": 32,
                                    "model_type": discriminator_type,
                                    "verbose": 1
                                })
        self.registerConstructor("keras", adversarialTrainer,
                                {
                                    "input_shape": input_shape,
                                    "batch_size": batchSize,
                                    "learning_rate": learningRate
                                })

        self.registerConstructor("pytorch", PytorchGenerator,
                                {
                                    "input_shape": input_shape,
                                    "num_class" : num_class,
                                    "hidden_size": hidden_size,
                                    "output_shape": output_shape,
                                    "latent_dim": 32,
                                    "model_type": generator_type
                                })
        self.registerConstructor("pytorch", PytorchDiscriminator,
                                {
                                    "input_shape": input_shape,
                                    "num_class" : num_class,
                                    "hidden_size": hidden_size,
                                    "output_shape": output_shape,
                                    "latent_dim": 32,
                                    "model_type": discriminator_type
                                })
        self.registerConstructor("pytorch", PytorchAdversarialTrainer,
                                {
                                    "learningRate": learningRate,
                                    "optimBetas": optimBetas,
                                })
        return

# A summary of network can be displayed by printing API
class PytorchGenerator(nn.Module):
    def __init__(self, kwargs):
        super(PytorchGenerator, self).__init__()

        # extract arguments from kwargs
        self.model_type = kwargs["model_type"]
        input_shape = kwargs["input_shape"]
        output_shape = kwargs["output_shape"]
        hidden_size = kwargs["hidden_size"]
        if type(input_shape) == tuple:
            input_shape = input_shape[0]
        if type(output_shape) == tuple:
            output_shape = output_shape[0]

        if self.model_type == "linear":
            self.map1 = nn.Linear(input_shape, hidden_size)
            self.dropout = nn.Dropout(p=0.4)
            self.batchnorm1 = nn.BatchNorm1d(hidden_size)
            self.map2 = nn.Linear(hidden_size, hidden_size)
            self.map3 = nn.Linear(hidden_size, output_shape)
            self.softmax = nn.Softmax()
        elif self.model_type == "2dcnn":
            image_size = 8
            self.layer1 = nn.Sequential(
                nn.ConvTranspose2d(hidden_size, image_size * 8,
                                kernel_size=4, stride=1),
                nn.BatchNorm2d(image_size * 8),
                nn.ReLU(inplace=True))

            self.layer2 = nn.Sequential(
                nn.ConvTranspose2d(image_size * 8, image_size * 4,
                                kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(image_size * 4),
                nn.ReLU(inplace=True))

            self.layer3 = nn.Sequential(
                nn.ConvTranspose2d(image_size * 4, image_size * 2,
                                kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(image_size * 2),
                nn.ReLU(inplace=True))

            self.layer4 = nn.Sequential(
                nn.ConvTranspose2d(image_size * 2, image_size,
                                kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(image_size),
                nn.ReLU(inplace=True))

            self.last = nn.Sequential(
                nn.ConvTranspose2d(image_size, 1, kernel_size=4,
                                stride=2, padding=1),
                nn.Tanh())
        else:
            raise Exception("Unknown generator type: {0}".format(kwargs["model_type"]))

    def forward(self, x):
        if self.model_type == "linear":
            x = F.relu(self.map1(x))
            x = self.batchnorm1(x)
            x = F.relu(self.map2(x))
            x = self.dropout(x)
            x = F.relu(self.map3(x))
        elif self.model_type == "2dcnn":
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.last(x)
        return x

class PytorchDiscriminator(nn.Module):
    def __init__(self, kwargs):
        super(PytorchDiscriminator, self).__init__()

        # extract arguments from kwargs
        self.model_type = kwargs["model_type"]
        input_shape = kwargs["input_shape"]
        output_shape = kwargs["output_shape"]
        hidden_size = kwargs["hidden_size"]
        if type(input_shape) == tuple:
            input_shape = input_shape[0]
        if type(output_shape) == tuple:
            output_shape = output_shape[0]

        if self.model_type == "linear":
            self.map1 = nn.Linear(input_shape, kwargs["hidden_size"])
            self.map2 = nn.Linear(kwargs["hidden_size"], kwargs["hidden_size"])
            self.map3 = nn.Linear(kwargs["hidden_size"], 1)
            self.dropout = nn.Dropout(p=0.4)
            self.batchnorm1 = nn.BatchNorm1d(kwargs["hidden_size"])
            self.softmax = nn.Softmax()
        elif self.model_type == "2dcnn":
            image_size = 8
            self.layer1 = nn.Sequential(
                nn.ConvTranspose2d(hidden_size, image_size * 8,
                                kernel_size=4, stride=1),
                nn.BatchNorm2d(image_size * 8),
                nn.ReLU(inplace=True))

            self.layer2 = nn.Sequential(
                nn.ConvTranspose2d(image_size * 8, image_size * 4,
                                kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(image_size * 4),
                nn.ReLU(inplace=True))

            self.layer3 = nn.Sequential(
                nn.ConvTranspose2d(image_size * 4, image_size * 2,
                                kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(image_size * 2),
                nn.ReLU(inplace=True))

            self.layer4 = nn.Sequential(
                nn.ConvTranspose2d(image_size * 2, image_size,
                                kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(image_size),
                nn.ReLU(inplace=True))

            self.last = nn.Sequential(
                nn.ConvTranspose2d(image_size, 1, kernel_size=4,
                                stride=2, padding=1),
                nn.Tanh())
        else:
            raise Exception("Unknown discriminator type: {0}".format(kwargs["model_type"]))

    def forward(self, x):
        if self.model_type == "linear":
            x = F.relu(self.map1(x))
            x = self.batchnorm1(x)
            x = F.relu(self.map2(x))
            x = F.relu(self.map3(x))
            x = self.dropout(x)
            return self.softmax(x)
        elif self.model_type == "2dcnn":
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.last(out)
        return out

class PytorchAdversarialTrainer(nn.Module):
    """
    Reference
    ---------
    https://github.com/YutaroOgawa/pytorch_advanced/blob/master/5_gan_generation/5-1-2_DCGAN.ipynb
    """
    def __init__(self, kwargs):
        self.learningRate = kwargs["learningRate"]
        self.optimBetas = kwargs["optimBetas"]
        self.batchSize = 100
        self.timeSteps = 10

    def train_model(self, generator, discriminator, dataloader, num_epochs=300):
        # GPUが使えるかを確認
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("使用デバイス：", device)

        # 最適化手法の設定
        g_lr, d_lr = 0.0001, 0.0004
        beta1, beta2 = 0.0, 0.9
        g_optimizer = torch.optim.Adam(generator.parameters(), g_lr, [beta1, beta2])
        d_optimizer = torch.optim.Adam(discriminator.parameters(), d_lr, [beta1, beta2])

        # 誤差関数を定義
        criterion = nn.BCEWithLogitsLoss(reduction='mean')

        # パラメータをハードコーディング
        z_dim = 20
        mini_batch_size = 64

        # ネットワークをGPUへ
        generator.to(device)
        discriminator.to(device)

        generator.train()  # モデルを訓練モードに
        discriminator.train()  # モデルを訓練モードに

        # ネットワークがある程度固定であれば、高速化させる
        torch.backends.cudnn.benchmark = True

        # 画像の枚数
        num_train_imgs = len(dataloader.dataset)
        batch_size = dataloader.batch_size

        # イテレーションカウンタをセット
        iteration = 1
        logs = []

        # epochのループ
        for epoch in range(num_epochs):

            # 開始時刻を保存
            t_epoch_start = time.time()
            epoch_g_loss = 0.0  # epochの損失和
            epoch_d_loss = 0.0  # epochの損失和

            print('-------------')
            print('Epoch {}/{}'.format(epoch, num_epochs))
            print('-------------')
            print('（train）')

            # データローダーからminibatchずつ取り出すループ
            for imges, y in dataloader:

                # --------------------
                # 1. Discriminatorの学習
                # --------------------
                # ミニバッチがサイズが1だと、バッチノーマライゼーションでエラーになるのでさける
                if imges.size()[0] == 1:
                    continue

                # GPUが使えるならGPUにデータを送る
                imges = imges.to(device)

                # 正解ラベルと偽ラベルを作成
                # epochの最後のイテレーションはミニバッチの数が少なくなる
                mini_batch_size = imges.size()[0]
                label_real = torch.full((mini_batch_size,), 1).to(device)
                label_fake = torch.full((mini_batch_size,), 0).to(device)

                # 真の画像を判定
                d_out_real = discriminator(imges)

                # 偽の画像を生成して判定
                input_z = torch.randn(mini_batch_size, z_dim).to(device)
                input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
                fake_images = generator(input_z)
                d_out_fake = discriminator(fake_images)

                # 誤差を計算
                d_loss_real = criterion(d_out_real.view(-1), label_real)
                d_loss_fake = criterion(d_out_fake.view(-1), label_fake)
                d_loss = d_loss_real + d_loss_fake

                # バックプロパゲーション
                g_optimizer.zero_grad()
                d_optimizer.zero_grad()

                d_loss.backward()
                d_optimizer.step()

                # --------------------
                # 2. Generatorの学習
                # --------------------
                # 偽の画像を生成して判定
                input_z = torch.randn(mini_batch_size, z_dim).to(device)
                input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
                fake_images = generator(input_z)
                d_out_fake = discriminator(fake_images)

                # 誤差を計算
                g_loss = criterion(d_out_fake.view(-1), label_real)

                # バックプロパゲーション
                g_optimizer.zero_grad()
                d_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

                # --------------------
                # 3. 記録
                # --------------------
                epoch_d_loss += d_loss.item()
                epoch_g_loss += g_loss.item()
                iteration += 1

            # epochのphaseごとのlossと正解率
            t_epoch_finish = time.time()
            print('-------------')
            print('epoch {} || Epoch_D_Loss:{:.4f} ||Epoch_G_Loss:{:.4f}'.format(
                epoch, epoch_d_loss/batch_size, epoch_g_loss/batch_size))
            print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
            t_epoch_start = time.time()

        return generator, discriminator

    def train(self, generator,
                    discriminator,
                    dataloader,
                    epochs: int = 3000,
                    sample_interval: int = 100) -> nn.Module:
        """
        Returns
        -------
        returns an instance of self.
        """
        criterion = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
        d_optimizer = optim.Adam(discriminator.parameters(), lr=self.learningRate, betas=self.optimBetas)
        g_optimizer = optim.Adam(generator.parameters(), lr=self.learningRate, betas=self.optimBetas)

        d_steps = 1
        g_steps = 4

        for epoch in range(epochs):
            for X_train, y_train in dataloader:
                for d_index in range(d_steps):
                    # 1. Train D on real+fake
                    discriminator.zero_grad()

                    #  1A: Train D on real
                    d_real_data = Variable(torch.Tensor(X_train))
                    d_real_decision = discriminator(d_real_data) # "preprocess" was removed
                    d_real_error = criterion(d_real_decision, Variable(torch.ones(self.batchSize, 1)))  # ones = true
                    d_real_error.backward() # compute/store gradients, but don't change params

                    #  1B: Train D on fake
                    noise = np.random.normal(0.999, 1.001, (self.batchSize, self.timeSteps, 1))
                    d_gen_input = Variable(torch.Tensor(noise))
                    d_fake_data = generator(d_gen_input).detach()  # detach to avoid training G on these labels
                    d_fake_decision = discriminator(d_fake_data) # "preprocess" was removed
                    d_fake_error = criterion(d_fake_decision, Variable(torch.zeros(self.batchSize, 1)))  # zeros = fake
                    d_fake_error.backward()
                    d_optimizer.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()

                for g_index in range(g_steps):
                    # 2. Train G on D's response (but DO NOT train D on these labels)
                    generator.zero_grad()

                    noise = np.random.normal(0.999, 1.001, (self.batchSize, self.timeSteps, 1))
                    gen_input = Variable(torch.Tensor(noise))
                    g_fake_data = generator(gen_input)
                    dg_fake_decision = discriminator(g_fake_data)
                    g_error = criterion(dg_fake_decision, Variable(torch.ones(self.batchSize, 1)))  # we want to fool, so pretend it's all genuine

                    g_error.backward()
                    g_optimizer.step()  # Only optimizes G's parameters

                if epoch % 100 == 0:
                    print("%s: D: %s/%s G: %s (Real: %s, Fake: %s) " % (epoch,
                                                                        extract(d_real_error)[0],
                                                                        extract(d_fake_error)[0],
                                                                        extract(g_error)[0],
                                                                        stats(extract(d_real_data)),
                                                                        stats(extract(d_fake_data))))
        return self
