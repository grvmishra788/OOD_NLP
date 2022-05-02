import os

import Utils

ENERGY_TEMP = 5
SOFTMAX_TEMP = 2.5

DATASETS = ["Reuters8", "Reuters52", "20NG", "IMDB_CR", "IMDB_MR"]

RE_TRAIN = False
DEBUG = False
DEBUG_ERROR = True
SAVE = True
PLOT_FIG = False

# init names
OUTPUT_FOLDER_NAME = "outputs"
DATA_NAME = "IMDB"
OOD_DATA_NAME = "MR"

# folders

# init features folders
FEATURES_FOLDER = os.path.join(OUTPUT_FOLDER_NAME, "features")
FEATURES_DATA_FOLDER = os.path.join(FEATURES_FOLDER, DATA_NAME)
OOD_FEATURES_DATA_FOLDER = os.path.join(FEATURES_FOLDER, OOD_DATA_NAME)

# init distances folders
DISTS_FOLDER = os.path.join(OUTPUT_FOLDER_NAME, "dists")
DISTS_DATA_FOLDER = os.path.join(DISTS_FOLDER, DATA_NAME)
OOD_DISTS_DATA_FOLDER = os.path.join(DISTS_FOLDER, OOD_DATA_NAME)

# init closest class folders
CLOSEST_CLASS_FOLDER = os.path.join(OUTPUT_FOLDER_NAME, "closest_classes")
CLOSEST_CLASS_DATA_FOLDER = os.path.join(CLOSEST_CLASS_FOLDER, DATA_NAME)
OOD_CLOSEST_CLASS_DATA_FOLDER = os.path.join(CLOSEST_CLASS_FOLDER, OOD_DATA_NAME)


# init labels folders
LABELS_FOLDER = os.path.join(OUTPUT_FOLDER_NAME, "labels")
LABELS_DATA_FOLDER = os.path.join(LABELS_FOLDER, DATA_NAME)


# init means folders
MEANS_FOLDER = os.path.join(OUTPUT_FOLDER_NAME, "means")
MEANS_DATA_FOLDER = os.path.join(MEANS_FOLDER, DATA_NAME)

# init radius folders
RADIUS_FOLDER = os.path.join(OUTPUT_FOLDER_NAME, "radius")
RADIUS_DATA_FOLDER = os.path.join(RADIUS_FOLDER, DATA_NAME)


def set_data_names(dataset, ood_dataset):
    global DATA_NAME, OOD_DATA_NAME
    DATA_NAME = dataset
    OOD_DATA_NAME = ood_dataset

    # init features folders
    global FEATURES_FOLDER, FEATURES_DATA_FOLDER, OOD_FEATURES_DATA_FOLDER
    FEATURES_FOLDER = os.path.join(OUTPUT_FOLDER_NAME, "features")
    FEATURES_DATA_FOLDER = os.path.join(FEATURES_FOLDER, DATA_NAME)
    OOD_FEATURES_DATA_FOLDER = os.path.join(FEATURES_FOLDER, OOD_DATA_NAME)

    # init distances folders
    global DISTS_FOLDER, DISTS_DATA_FOLDER, OOD_DISTS_DATA_FOLDER
    DISTS_FOLDER = os.path.join(OUTPUT_FOLDER_NAME, "dists")
    DISTS_DATA_FOLDER = os.path.join(DISTS_FOLDER, DATA_NAME)
    OOD_DISTS_DATA_FOLDER = os.path.join(DISTS_FOLDER, OOD_DATA_NAME)

    # init closest class folders
    global CLOSEST_CLASS_FOLDER, CLOSEST_CLASS_DATA_FOLDER, OOD_CLOSEST_CLASS_DATA_FOLDER
    CLOSEST_CLASS_FOLDER = os.path.join(OUTPUT_FOLDER_NAME, "closest_classes")
    CLOSEST_CLASS_DATA_FOLDER = os.path.join(CLOSEST_CLASS_FOLDER, DATA_NAME)
    OOD_CLOSEST_CLASS_DATA_FOLDER = os.path.join(CLOSEST_CLASS_FOLDER, OOD_DATA_NAME)

    # init labels folders
    global LABELS_FOLDER, LABELS_DATA_FOLDER
    LABELS_FOLDER = os.path.join(OUTPUT_FOLDER_NAME, "labels")
    LABELS_DATA_FOLDER = os.path.join(LABELS_FOLDER, DATA_NAME)

    # init means folders
    global MEANS_FOLDER, MEANS_DATA_FOLDER
    MEANS_FOLDER = os.path.join(OUTPUT_FOLDER_NAME, "means")
    MEANS_DATA_FOLDER = os.path.join(MEANS_FOLDER, DATA_NAME)

    # init radius folders
    global RADIUS_FOLDER, RADIUS_DATA_FOLDER
    RADIUS_FOLDER = os.path.join(OUTPUT_FOLDER_NAME, "radius")
    RADIUS_DATA_FOLDER = os.path.join(RADIUS_FOLDER, DATA_NAME)

    Utils.init_folders()
