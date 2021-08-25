import time

import tensorflow as tf
import numpy as np

from model.generator import Generator
from model.discriminator import Discriminator
from model.utils import ModelConfig
from preprocessing.dataset import read_dataset
from PIL import Image

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

# pylint: disable=no-member
def gen_logistic_non_sat_loss(fake_scores):
    return tf.reduce_mean(tf.nn.softplus(-fake_scores))


def disc_logistic_loss(fake_scores, real_scores):
    return tf.reduce_mean(tf.nn.softplus(fake_scores) + tf.nn.softplus(-real_scores))


def disc_logistic_loss_r1(fake_scores, real_scores, cfg, disc_real_score_gradients=None):
    loss = disc_logistic_loss(fake_scores, real_scores)
    if disc_real_score_gradients is not None:
        per_example_gradient_penalty = cfg.gamma * 0.5 * \
            tf.reduce_sum(tf.square(disc_real_score_gradients), axis=[1, 2, 3])
        gradient_penalty = tf.reduce_mean(per_example_gradient_penalty)
        loss += gradient_penalty
    return loss
def saveimage(image,epoch):
    image_array=np.concatenate([image[0],image[1],image[2],image[3]], axis=1)
    image_array=np.rint(image_array).clip(0, 255).astype(np.uint8)
    image=Image.fromarray(image_array)
    image.save('C:/Users/Yunji/PycharmProjects/python2/generated_images/flowers/image_at_epoch_{:07d}.jpg'.format(epoch))

def train_step(real_images,
               real_labels,
               generator,
               discriminator,
               gen_optimizer,
               disc_optimizer,
               cfg,
               labels=None,
               disc_regularization=True):

    latents = tf.random.normal([real_images.shape[0], cfg.dlatent_size])
    if cfg.labels_size > 0:
        fake_labels = np.zeros(real_labels.shape, dtype=np.float32)
        fake_labels[np.arange(cfg.batch_size), np.random.randint(10, size=cfg.batch_size)] = 1.0
    else:
        fake_labels = []

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        fake_images = generator(latents, labels=fake_labels, training=True)
        fake_scores = discriminator(fake_images, fake_labels)

        if disc_regularization:
            with tf.GradientTape() as inner_tape:
                inner_tape.watch(real_images)
                real_scores = discriminator(real_images, labels=real_labels)
                disc_real_score_gradients = inner_tape.gradient(tf.reduce_sum(real_scores), real_images)
                disc_loss = disc_logistic_loss_r1(fake_scores, real_scores, cfg, disc_real_score_gradients)
        else:
            real_scores = discriminator(real_images, labels=real_labels)
            disc_loss = disc_logistic_loss_r1(fake_scores, real_scores, cfg)

        gen_loss = gen_logistic_non_sat_loss(fake_scores)

    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

    return gen_loss, disc_loss


def train_model(config_path, data_path):

    cfg = ModelConfig(config_path)

    discriminator = Discriminator(cfg)
    generator = Generator(cfg)

    gen_optimizer = tf.keras.optimizers.Adam(
        learning_rate=cfg.generator_base_learning_rate,
        beta_1=cfg.generator_beta_1,
        beta_2=cfg.generator_beta_2,
        epsilon=1e-8)

    disc_optimizer = tf.keras.optimizers.Adam(
        learning_rate=cfg.discriminator_base_learning_rate,
        beta_1=cfg.discriminator_beta_1,
        beta_2=cfg.discriminator_beta_2,
        epsilon=1e-8)

    generator.load_weights(
        "C:/Users/Yunji/PycharmProjects/python2/checkpoints/flowers/models/generator_0005_epochs")
    discriminator.load_weights(
        "C:/Users/Yunji/PycharmProjects/python2/checkpoints/flowers/models/discriminator_0005_epochs")

    dataset = read_dataset(data_path, cfg)


    # Initialize metrics
    gen_loss = tf.keras.metrics.Metric
    start_time = time.time()
    num_epochs = 5
    num_images_before = 27120*num_epochs
    num_minibatch = 6780*num_epochs


    for example in dataset:

        disc_regularization = (num_minibatch % cfg.disc_reg_intervall == 0)
        gen_loss, disc_loss = train_step(real_images=example['data'],
                                         real_labels=example['label'],
                                         generator=generator,
                                         discriminator=discriminator,
                                         gen_optimizer=gen_optimizer,
                                         disc_optimizer=disc_optimizer,
                                         cfg=cfg,
                                         disc_regularization=disc_regularization)

        num_minibatch = num_minibatch+1
        num_images = num_minibatch * cfg.batch_size
        num_epochs = num_minibatch//6780

        # Save checkpoint
        if (num_images % 54240) < cfg.batch_size:
            generator.save_weights(
                "C:/Users/Yunji/PycharmProjects/python2/checkpoints/flowers/models/generator_{:04d}_epochs".format(num_epochs))
            discriminator.save_weights(
                "C:/Users/Yunji/PycharmProjects/python2/checkpoints/flowers/models/discriminator_{:04d}_epochs".format(num_epochs))

        if (num_images % 27120) < cfg.batch_size:
            random_input = tf.random.normal([cfg.batch_size, cfg.latent_size])
            fake_images_batch = generator(random_input, False)
            saveimage(fake_images_batch,num_epochs)
            epoch_time = (time.time() - start_time)*27120/((num_images - num_images_before)*60)
            print('epoch {} minibatch {} images {} gen loss {:.4f} disc loss {:.4f}'
                  ' minutes per batch {:.2f}'.format(num_epochs, num_minibatch, num_images, gen_loss, disc_loss, epoch_time))
            num_images_before = num_images
            start_time = time.time()

        if (num_images % (800 * 27120)) < cfg.batch_size:
            # Save final state if not already done
            if not (num_images % cfg.checkpoint_intervall_kimg) < cfg.batch_size:
                generator.save_weights(
                    "C:/Users/Yunji/PycharmProjects/python2/checkpoints/flowers/models/generator_{:04d}_num_images".format(num_epochs))
                discriminator.save_weights(
                    "C:/Users/Yunji/PycharmProjects/python2/checkpoints/flowers/models/discriminator_{:04d}_num_images".format(num_epochs))
            break

if __name__ == "__main__":
    train_model('C:/Users/Yunji/PycharmProjects/python2/config/flowers.yaml', 'C:/Users/Yunji/PycharmProjects/python2/outdata/flowers.tfrecords')


