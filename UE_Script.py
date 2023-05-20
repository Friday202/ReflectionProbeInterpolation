import unreal
import os
import glob
import csv
import time


def set_actor_location(actor, x, y):
    actor.set_actor_location(unreal.Vector(x, y, 170), False, True)


def get_actor_location(actor):
    return actor.get_actor_location()


def get_capture_cube():
    # Will always be one actor!
    actor_class = unreal.SceneCaptureCube
    actors = unreal.GameplayStatics.get_all_actors_of_class(unreal.EditorLevelLibrary.get_editor_world(), actor_class)
    return actors[0]


def export_hdr(filepath):
    # Update task filepath
    task.filename = filepath
    # Set exporter task and export HDR
    exporter.export_task = task
    exporter.run_asset_export_task(task)


def clear_files(should_do=False):
    if should_do:
        dir_path = r"C:\Data\HDR"

        hdr_files = glob.glob(os.path.join(dir_path, "*.hdr"))

        for file in hdr_files:
            os.remove(file)

        dir_path = r"C:\Data\Locations"

        csv_files = glob.glob(os.path.join(dir_path, "*.csv"))

        for file in csv_files:
            os.remove(file)


def calculate_position(x_idx, y_idx):
    # Calculate the spacing between each position
    x_spacing = (LEVEL_RANGE_X[1] - LEVEL_RANGE_X[0]) / float(X_MAX)
    y_spacing = (LEVEL_RANGE_Y[1] - LEVEL_RANGE_Y[0]) / float(Y_MAX)

    # Calculate the new position based on the given indices
    x_pos = LEVEL_RANGE_X[0] + (x_idx * x_spacing)
    y_pos = LEVEL_RANGE_Y[0] + (y_idx * y_spacing)

    return x_pos, y_pos


def export_location(number, location, file_path):

    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow((number, location.x, location.y))


if __name__ == "__main__":

    X_MAX = 70
    Y_MAX = 70

    LEVEL_RANGE_X = (-1650, 2350)
    LEVEL_RANGE_Y = (-2550, 1450)

    start_position = (-1650, 1450)

    # Get scene capture cube
    SceneCaptureCube = get_capture_cube()

    # Reset its location
    set_actor_location(SceneCaptureCube, start_position[0], start_position[1])

    # Remove HDR files and Location file
    clear_files(True)

    # Setup exporter and task
    asset_path = "/Game/CubeRenderTarget"

    # Create an instance of the exporter
    exporter = unreal.TextureCubeExporterHDR()

    # Set the export task properties
    task = unreal.AssetExportTask()
    task.object = unreal.load_asset(asset_path)

    currentHDR = 0

    for i in range(0, X_MAX):
        for j in range(0, Y_MAX):
            # Calculate position
            new_position = calculate_position(i, j)

            # Set actor location
            set_actor_location(SceneCaptureCube, new_position[0], new_position[1])

            # Capture scene must be triggered manually!
            SceneCaptureCube.capture_component_cube.capture_scene()

            # Export HDR
            export_hdr(r'C:\Data\HDR\Level_0_' + str(currentHDR) + '.hdr')

            # Get actor location
            location = get_actor_location(SceneCaptureCube)

            # Export location to CSV file
            file_path = r"C:\Data\Locations\Level_0_locations.csv"
            export_location(currentHDR, location, file_path)

            currentHDR += 1
