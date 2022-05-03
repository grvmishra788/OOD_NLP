import os
import constants
import Utils
from Utils import printD
import numpy as np


def load_train_features(NUM_CLASSES):
    train_features = []
    for classID in range(NUM_CLASSES):
        save_name = "{0}_train_features.npy".format(classID)
        save_location = os.path.join(constants.FEATURES_DATA_FOLDER, save_name)
        Utils.check_path(save_location)
        x = np.load(save_location)
        train_features.append(x)
    train_features = np.array(train_features, dtype='object')
    printD(f"train_features.shape = {train_features.shape}")
    return train_features


def load_test_features():
    save_name = "test_features.npy"
    save_location = os.path.join(constants.FEATURES_DATA_FOLDER, save_name)
    test_features = np.load(save_location)
    printD(f"test_features.shape = {test_features.shape}")
    return test_features


def load_ood_features():
    save_name = "ood_set_features.npy"
    save_location = os.path.join(constants.OOD_FEATURES_DATA_FOLDER, save_name)
    ood_set_features = np.load(save_location)
    printD(f"ood_set_features.shape = {ood_set_features.shape}")
    return ood_set_features

def load_test_logits():
    save_name = "test_logits.npy"
    save_location = os.path.join(constants.FEATURES_DATA_FOLDER, save_name)
    test_logits = np.load(save_location)
    printD(f"test_logits.shape = {test_logits.shape}")
    return test_logits


def load_ood_logits():
    save_name = "ood_set_logits.npy"
    save_location = os.path.join(constants.OOD_FEATURES_DATA_FOLDER, save_name)
    ood_set_logits = np.load(save_location)
    printD(f"ood_set_logits.shape = {ood_set_logits.shape}")
    return ood_set_logits

def load_class_means():
    save_name = "train_class_means.npy"
    save_location = os.path.join(constants.MEANS_DATA_FOLDER, save_name)
    class_means = np.load(save_location)
    printD(f"class_means.shape = {class_means.shape}")


def load_class_radii():
    save_name = "train_class_radii.npy"
    save_location = os.path.join(constants.RADIUS_DATA_FOLDER, save_name)
    class_radii = np.load(save_location)
    printD(f"class_radii.shape = {class_radii.shape}")
    return class_radii


def load_train_dists_and_closest_classes(NUM_CLASSES):
    train_distances = []
    train_closest_classes = []
    for classID in range(NUM_CLASSES):
        # load distances
        save_name = "{0}_train_distances.npy".format(classID)
        save_location = os.path.join(constants.DISTS_DATA_FOLDER, save_name)
        Utils.check_path(save_location)
        x = np.load(save_location)
        train_distances.append(x)
        # load closest classes
        save_name = "{0}_train_closest_classes.npy".format(classID)
        save_location = os.path.join(constants.CLOSEST_CLASS_DATA_FOLDER, save_name)
        Utils.check_path(save_location)
        x = np.load(save_location)
        train_closest_classes.append(x)
    train_distances = np.asarray(train_distances)
    train_closest_classes = np.asarray(train_closest_classes)
    printD(f"train_distances.shape = {train_distances.shape}")
    printD(f"train_closest_classes.shape = {train_closest_classes.shape}")

    return train_distances, train_closest_classes


def load_test_dists_and_closest_classes():
    save_name = "test_distances.npy"
    save_location = os.path.join(constants.DISTS_DATA_FOLDER, save_name)
    Utils.check_path(save_location)
    test_distances = np.load(save_location)
    printD(f"test_distances.shape = {test_distances.shape}")
    # load test closest classes
    save_name = "test_closest_classes.npy"
    save_location = os.path.join(constants.CLOSEST_CLASS_DATA_FOLDER, save_name)
    test_closest_classes = np.load(save_location)
    printD(f"test_closest_classes.shape = {test_closest_classes.shape}")
    return test_distances, test_closest_classes


def load_OOD_dists_and_closest_classes():
    save_name = "ood_set_distances.npy"
    save_location = os.path.join(constants.OOD_DISTS_DATA_FOLDER, save_name)
    ood_set_distances = np.load(save_location)
    printD(f"ood_set_distances.shape = {ood_set_distances.shape}")
    # load ood set closest classes
    save_name = "ood_set_closest_classes.npy"
    save_location = os.path.join(constants.OOD_CLOSEST_CLASS_DATA_FOLDER, save_name)
    ood_set_closest_classes = np.load(save_location)
    printD(f"ood_set_closest_classes.shape = {ood_set_closest_classes.shape}")
    return ood_set_distances, ood_set_closest_classes


