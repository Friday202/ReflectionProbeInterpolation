from preprocess_data import get_train_data, convert_array_to_img_and_display, get_half_train_data, get_least_data, get_center_data
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Sequential
import tensorflow as tf
from keras.callbacks import ModelCheckpoint


def create_model():

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(256, 256, 3)))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))

    model.add(MaxPooling2D((2, 2), padding='same'))

    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    # model.summary()
    return model


def train(model, x, y, epochs):
    model.fit(x, y, epochs=epochs, shuffle=False, callbacks=[checkpoint], batch_size=8)
    # model.save(r'C:\Data\Models\model.model')


def load_model():
    model = tf.keras.models.load_model(r'C:\Data\Models\model.model')
    return model


def load_model_smaller():
    model = tf.keras.models.load_model(r'C:\Data\Models\model.smaller_model')
    return model


def load_model_least():
    model = tf.keras.models.load_model(r'C:\Data\Models\model.least')
    return model


def load_model_center():
    model = tf.keras.models.load_model(r'C:\Data\Models\model.center')
    return model


if __name__ == '__main__':

    start_over = False
    batch = False

    # Get train data
    GT_images, bi_linear_images = get_center_data()

    print(len(GT_images))

    if len(GT_images) != len(bi_linear_images):
        print("Error ground truth images must be of the same size as bi-linear images!")
        exit(1)

    # Create the checkpoint callback
    checkpoint = ModelCheckpoint(r'C:\Data\Models\model.center', monitor='loss', verbose=1, save_best_only=False, mode='min')

    if start_over:
        model = create_model()
    else:
        # model = load_model()
        # model = load_model_smaller()
        # model = load_model_least()
        model = load_model_center()

    # Train on half of database firstly then on the other

    train(model, bi_linear_images, GT_images, 3000000)

    # if batch:
    #     train(model, bi_linear_images[:2450], GT_images[:2450], 100)
    # else:
    #     train(model, bi_linear_images[2450:], GT_images[2450:], 100)

