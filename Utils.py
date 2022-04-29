import numpy as np

import Utils
from constants import *


def printD(x):
    if DEBUG:
        print(x)


def check_path(path):
    if not os.path.exists(path):
        print(f"Err: {path} doesn't exist!")
    else:
        print(f"{path} exists!")


def partition_data_in_two(dataset, dataset_labels, in_sample_labels, oos_labels):
    '''
    :param dataset: the text from text_to_rank
    :param dataset_labels: dataset labels
    :param in_sample_labels: a list of newsgroups which the network will/did train on
    :param oos_labels: the complement of in_sample_labels; these newsgroups the network has never seen
    :return: the dataset partitioned into in_sample_examples, in_sample_labels,
    oos_examples, and oos_labels in that order
    '''
    _dataset = dataset[:]  # aliasing safeguard
    _dataset_labels = dataset_labels

    in_sample_idxs = np.zeros(np.shape(_dataset_labels), dtype=bool)
    ones_vec = np.ones(np.shape(_dataset_labels), dtype=int)
    for label in in_sample_labels:
        in_sample_idxs = np.logical_or(in_sample_idxs, _dataset_labels == label * ones_vec)

    return _dataset[in_sample_idxs], _dataset_labels[in_sample_idxs], \
           _dataset[np.logical_not(in_sample_idxs)], _dataset_labels[np.logical_not(in_sample_idxs)]


def relabel_in_sample_labels(labels):
    labels_as_list = labels.tolist()

    set_of_labels = []
    for label in labels_as_list:
        set_of_labels.append(label)
    labels_ordered = sorted(list(set(set_of_labels)))

    relabeled = np.zeros(labels.shape, dtype=int)
    for i in range(len(labels_as_list)):
        relabeled[i] = labels_ordered.index(labels_as_list[i])

    return relabeled


def init_folders():
    if not os.path.exists(OUTPUT_FOLDER_NAME):
        os.makedirs(OUTPUT_FOLDER_NAME)

    if not os.path.exists(FEATURES_DATA_FOLDER):
        os.makedirs(FEATURES_DATA_FOLDER)

    if not os.path.exists(OOD_FEATURES_DATA_FOLDER):
        os.makedirs(OOD_FEATURES_DATA_FOLDER)

    if not os.path.exists(DISTS_DATA_FOLDER):
        os.makedirs(DISTS_DATA_FOLDER)

    if not os.path.exists(OOD_DISTS_DATA_FOLDER):
        os.makedirs(OOD_DISTS_DATA_FOLDER)

    if not os.path.exists(CLOSEST_CLASS_DATA_FOLDER):
        os.makedirs(CLOSEST_CLASS_DATA_FOLDER)

    if not os.path.exists(OOD_CLOSEST_CLASS_DATA_FOLDER):
        os.makedirs(OOD_CLOSEST_CLASS_DATA_FOLDER)

    if not os.path.exists(LABELS_DATA_FOLDER):
        os.makedirs(LABELS_DATA_FOLDER)

    if not os.path.exists(MEANS_DATA_FOLDER):
        os.makedirs(MEANS_DATA_FOLDER)

    if not os.path.exists(RADIUS_DATA_FOLDER):
        os.makedirs(RADIUS_DATA_FOLDER)

    printD(f"Initialized folders at {OUTPUT_FOLDER_NAME}")


def check_all_paths():
    check_path(OUTPUT_FOLDER_NAME)
    check_path(FEATURES_DATA_FOLDER)
    check_path(OOD_FEATURES_DATA_FOLDER)
    check_path(DISTS_DATA_FOLDER)
    check_path(OOD_DISTS_DATA_FOLDER)
    check_path(CLOSEST_CLASS_DATA_FOLDER)
    check_path(OOD_CLOSEST_CLASS_DATA_FOLDER)
    check_path(LABELS_DATA_FOLDER)
    check_path(MEANS_DATA_FOLDER)
    check_path(RADIUS_DATA_FOLDER)


def get_class_info(in_sample_labels):
    classes = list(set(in_sample_labels))
    NUM_CLASSES = len(classes)
    printD(f"classes = {classes}")
    printD(f"NUM_CLASSES = {NUM_CLASSES}")
    return classes, NUM_CLASSES


def get_per_class_info(in_sample_examples, in_sample_labels, classes):
    total = 0
    per_class_examples = []
    for classID in classes:
        in_samples_class = in_sample_examples[in_sample_labels == classID]
        ss = len(in_samples_class)
        total += ss
        per_class_examples.append(ss)
    printD(f"per_class_examples = {per_class_examples}")
    printD(f"total == len(in_sample_examples)  == ({total == len(in_sample_examples)})")
    return per_class_examples


