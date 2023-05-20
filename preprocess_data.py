import matplotlib.pyplot as plt
import csv
from scipy.spatial import Delaunay
import math
import numpy as np
import cv2
import tqdm
import pickle


def get_data():

    all_imgs, locations = get_imgs_and_locations(False)

    # List of triangles and their coordinates
    triangles, img_indexes = get_Delaunay_triangles(locations)

    bi_linear_imgs = []

    for i in tqdm.tqdm(range(len(all_imgs))):
        bi_linear_imgs.append(create_bi_linear_img(locations[i], triangles, img_indexes, all_imgs))
        # convert_array_to_img_and_display(bi_linear_imgs[i])

    return all_imgs, bi_linear_imgs


def get_imgs_and_locations(display=False):

    images = []

    for img in range(4900):
        hdr_img = cv2.imread(f'C:\Data\HDR\Level_0_{img}.hdr')
        hdr_img = cv2.cvtColor(hdr_img, cv2.COLOR_BGR2RGB)
        if display:
            # Optionally display the HDR image
            cv2.imshow('HDR Image', hdr_img)
            cv2.waitKey(0)
            print(img)
        images.append(hdr_img)

    # Convert to numpy array
    data = np.array(images)

    # Normalize the images to [0, 1]
    data = data.astype('float32') / 255.

    with open(r'C:\Data\Locations\Level_0_locations.csv') as file:
        reader = csv.reader(file)
        locations = []
        for row in reader:
            locations.append([float(x) for x in row[1:]])

    return data, locations


def get_Delaunay_triangles(locations):
    # Same as in engine
    step = 70

    # Create a list of x and y coordinates
    x_coords = [x for x, y in locations]
    y_coords = [y for x, y in locations]

    # Locations of reflection spheres in paint
    ref_sphere_locations_from_img_x = [110, 306, 0, 207, 85, 307]
    ref_sphere_locations_from_img_y = [333, 261, 228, 176, 93, 77]

    # Preprocess to match engine's level layout
    for i in range(len(ref_sphere_locations_from_img_x)):
        ref_sphere_locations_from_img_x[i] = (ref_sphere_locations_from_img_x[i] - 165) * 10
        ref_sphere_locations_from_img_y[i] = (ref_sphere_locations_from_img_y[i] - 255) * 10

    # Compute actual locations from desired
    ref_sphere_locations_x, ref_sphere_locations_y = closest_point_to_point(locations, ref_sphere_locations_from_img_x,
                                                                            ref_sphere_locations_from_img_y)
    # Add edge points
    ref_sphere_locations_x.append(x_coords[0])
    ref_sphere_locations_y.append(y_coords[0])

    ref_sphere_locations_x.append(x_coords[step - 1])
    ref_sphere_locations_y.append(y_coords[step - 1])

    ref_sphere_locations_x.append(x_coords[-step])
    ref_sphere_locations_y.append(y_coords[-step])

    ref_sphere_locations_x.append(x_coords[-1])
    ref_sphere_locations_y.append(y_coords[-1])

    # Plot the coordinates as dots on a 2D plot
    plt.scatter(x_coords, y_coords, c='b')
    plt.scatter(ref_sphere_locations_x, ref_sphere_locations_y, c='r')

    # # Set the axis limits to match the range of your coordinates
    plt.xlim(min(x_coords)-50, max(x_coords)+50)
    plt.ylim(min(y_coords)-50, max(y_coords)+50)

    plt.show()

    # Get coordinates ready for triangulation
    points = []
    for x, y in zip(ref_sphere_locations_x, ref_sphere_locations_y):
        points.append([x, y])

    points = np.array(points)

    # Compute Delaunay triangulation
    tri = Delaunay(points)

    plt.triplot(points[:, 0], points[:, 1], tri.simplices)
    plt.plot(points[:, 0], points[:, 1], 'o')
    plt.show()

    # Get coordinates of triangles
    tri_coords = tri.points[tri.simplices]

    img_indexes = get_corresponding_imgs_to_triangle(tri_coords, locations)

    return tri_coords, img_indexes


def get_corresponding_imgs_to_triangle(triangles, locations):

    locations = np.array(locations)
    img_indexes = []

    for triangle in triangles:
        temp = []
        for point in triangle:
            for index, location in enumerate(locations):
                if point[0] == location[0] and point[1] == location[1]:
                    temp.append(index)
                    break
        img_indexes.append(temp)

    return img_indexes


def closest_point_to_point(locations, desired_location_x, desired_location_y):
    x_locations = []
    y_locations = []

    for xx, yy in zip(desired_location_x, desired_location_y):
        current_min_x = 900000
        current_min_y = 900000
        current_min = 90000
        for x, y in locations:
            if math.sqrt((xx-x)**2 + (yy-y)**2) < current_min:
                current_min_x = x
                current_min_y = y
                current_min = math.sqrt((xx-x)**2 + (yy-y)**2)
        x_locations.append(current_min_x)
        y_locations.append(current_min_y)

    return x_locations, y_locations


def create_bi_linear_img(img_location, triangles, img_indexes, images):
    # img_location is location of bi linear interpolation
    # triangles is a list of triangles

    img_location = np.array(img_location)

    corresponding_index = None
    image_locations = None

    for i, triangle in enumerate(triangles):
        if point_in_triangle(img_location, triangle[0], triangle[1], triangle[2]):
            # print("Found triangle!")
            corresponding_index = img_indexes[i]
            image_locations = triangle
            break

    if corresponding_index is None or image_locations is None:
        print("Error should not happen!")
        exit(2)

    # Reference images used for interpolation
    img1 = images[corresponding_index[0]]
    img2 = images[corresponding_index[1]]
    img3 = images[corresponding_index[2]]

    return bi_linear_interpolate(img1, img2, img3, image_locations, img_location)


def point_in_triangle(pt, v1, v2, v3):
    d1 = sign(pt, v1, v2)
    d2 = sign(pt, v2, v3)
    d3 = sign(pt, v3, v1)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not(has_neg and has_pos)


def sign(p1, p2, p3):
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])


def bi_linear_interpolate(img1, img2, img3, image_locations, desired_location):

    p1 = image_locations[0]
    p2 = image_locations[1]
    p3 = image_locations[2]

    # Compute the distances between the input image locations and the desired location
    d1 = np.linalg.norm(desired_location - p1)
    d2 = np.linalg.norm(desired_location - p2)
    d3 = np.linalg.norm(desired_location - p3)

    # Compute the weights for each input image based on their distances
    w1 = 1.0 / (d1 + 1e-9)
    w2 = 1.0 / (d2 + 1e-9)
    w3 = 1.0 / (d3 + 1e-9)

    # Normalize the weights between [0, 1]
    w_total = w1 + w2 + w3
    w1 /= w_total
    w2 /= w_total
    w3 /= w_total

    # Compute the size of the output image
    out_shape = img1.shape[:-1]

    # Create an empty array to hold the output image
    out_img = np.zeros(out_shape + (3,))

    for i in range(out_img.shape[0]):
        for j in range(out_img.shape[1]):

            pixel1 = img1[i, j]
            pixel2 = img2[i, j]
            pixel3 = img3[i, j]

            new_pixel = w1 * pixel1 + w2 * pixel2 + w3 * pixel3
            out_img[i, j] = new_pixel

    return out_img


def convert_array_to_img_and_display(array, display=False):
    # [0, 1] back to [0, 255] but clip away decimal values
    img_array = array * 255.
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    if display:
        plt.imshow(img_array)
        plt.show()
    return img_array


def save_to_disk(data, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(data, f)


def load_from_disk(name):
    with open(name + '.pkl', 'rb') as f:
        data = pickle.load(f)
    return data


def load_train_data():
    GT_images = load_from_disk(r'GT_images')
    bi_linear_images = load_from_disk(r'bi_linear_images')
    return GT_images, bi_linear_images


def convert_to_256x256(array):
    # Create a new array of zeros with shape (256, 256, 3)
    new_array = np.zeros((256, 256, 3))

    # Compute the indices to copy the original array into the new array
    start_row = (256 - 128) // 2
    end_row = start_row + 128
    start_col = 0
    end_col = 256

    # Copy the original array into the new array
    new_array[start_row:end_row, start_col:end_col, :] = array

    return new_array


def get_train_data():
    GT_images_raw, bi_linear_images_raw = load_train_data()

    GT_images = []
    bi_linear_images = []

    for img in GT_images_raw:
        GT_images.append(convert_to_256x256(img))
    for img in bi_linear_images_raw:
        bi_linear_images.append(convert_to_256x256(img))

    return np.array(GT_images), np.array(bi_linear_images)


def get_half_train_data():
    GT_images_raw, bi_linear_images_raw = load_train_data()

    GT_images = []
    bi_linear_images = []

    for i in range(0, len(GT_images_raw), 2):
        GT_images.append(convert_to_256x256(GT_images_raw[i + 1]))
        bi_linear_images.append(convert_to_256x256(bi_linear_images_raw[i + 1]))

    return np.array(GT_images), np.array(bi_linear_images)


def get_least_data():
    GT_images_raw, bi_linear_images_raw = load_train_data()

    GT_images = []
    bi_linear_images = []

    for i in range(0, len(GT_images_raw), 49):
        GT_images.append(convert_to_256x256(GT_images_raw[i + 1]))
        bi_linear_images.append(convert_to_256x256(bi_linear_images_raw[i + 1]))

    return np.array(GT_images), np.array(bi_linear_images)


def get_center_data():
    GT_images_raw, bi_linear_images_raw = load_train_data()
    GT_images = []
    bi_linear_images = []

    for i in range(2300-70, 2300):
        GT_images.append(convert_to_256x256(GT_images_raw[i]))
        bi_linear_images.append(convert_to_256x256(bi_linear_images_raw[i]))

    return np.array(GT_images), np.array(bi_linear_images)


if __name__ == '__main__':

    GT_images, bi_linear_images = get_data()

    save_to_disk(GT_images, r'GT_images')
    save_to_disk(bi_linear_images, r'bi_linear_images')
