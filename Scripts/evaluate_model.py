from model_training import load_model
from preprocess_data import get_train_data, convert_array_to_img_and_display
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Sequential
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import time


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

    # Get data
    GT_images, bi_linear_images = get_train_data()

    # img_n = 2300+35
    img_n = 2300 - 70*2 +35+10
    # img_n = 100-2

    # model = load_model_smaller()
    # model = load_model()
    # model = load_model_least()
    model = load_model_center()

    t = time.time()
    predictions = model.predict(bi_linear_images[img_n].reshape(1, 256, 256, 3))
    print("Time took to predict: " + str(time.time() - t))

    img_interpolated = convert_array_to_img_and_display(bi_linear_images[img_n])
    img_GT = convert_array_to_img_and_display(GT_images[img_n])
    img_predicted = convert_array_to_img_and_display(predictions[0].reshape(256, 256, 3))

    # Visualize results
    fig, axes = plt.subplots(1, 3, figsize=(10, 10))

    axes[0].imshow(img_GT)
    axes[1].imshow(img_interpolated)
    axes[2].imshow(img_predicted)

    # Add a title for each subplot
    axes[0].set_title('Ground truth')
    axes[1].set_title('Input: interpolated image')
    axes[2].set_title('Output: cleaned image')

    plt.show()