import os

DEBUG = False
SAVE = True

# init names
OUTPUT_FOLDER_NAME = "outputs"
DATA_NAME = "Reuters40"
OOD_DATA_NAME = "Reuters12"

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
