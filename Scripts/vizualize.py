import numpy as np
from preprocess_data import get_Delaunay_triangles, create_bi_linear_img, get_imgs_and_locations, convert_to_256x256, convert_array_to_img_and_display
from evaluate_model import load_model_center
import tqdm
import cv2

# We are moving on the Y axis and X axis stays the same
current_location = np.array([178.57142857142867, -2550.0])  # is start location

distance = 2550 + 820
time = 6
fps = 60
amount_of_frames = fps * time
step = distance / amount_of_frames
print(amount_of_frames)
print(step)

all_imgs, locations = get_imgs_and_locations()
triangles, img_indexes = get_Delaunay_triangles(locations)

model = load_model_center()

for i in tqdm.tqdm(range(int(amount_of_frames))):
    # Creates a bi linear image on current location given triangles, img indexes for all images => are references
    bi_linear_img = create_bi_linear_img(current_location, triangles, img_indexes, all_imgs)
    current_location[1] += step

    img_for_model = convert_to_256x256(bi_linear_img)

    predicted_img = model.predict(img_for_model.reshape(1, 256, 256, 3))

    img_to_save = convert_array_to_img_and_display(predicted_img[0].reshape(256, 256, 3), False)

    bgr_image = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)

    # Specify the file path and name to save the image
    file_path = r'..\NRG\Results\image' + str(i) +'.png'

    # Save the image to disk using OpenCV
    cv2.imwrite(file_path, bgr_image)



