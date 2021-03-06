import os
import numpy as np
from tqdm import tqdm
from Utils import printD
import constants

from scipy.spatial.distance import cdist


def gen_train_features(classes, in_sample_examples, in_sample_labels, sess, fel, x, is_training):
    printD("gen_train_features():")
    for classID in tqdm(classes):
        in_samples_class = in_sample_examples[in_sample_labels == classID]
        train_set_features = sess.run([fel], feed_dict={x: in_samples_class, is_training: False})
        train_set_features = np.squeeze(np.asarray(train_set_features))
        if train_set_features.ndim == 1:
            train_set_features = np.expand_dims(train_set_features, 0)
        if constants.SAVE:
            save_name = "{0}_train_features.npy".format(classID, train_set_features.shape[0])
            save_location = os.path.join(constants.FEATURES_DATA_FOLDER, save_name)
            np.save(save_location, train_set_features)
            printD("\tSaved generated features for " + str(classID) + " at " + save_location + " having shape - " + str(
                train_set_features.shape))

    if not constants.SAVE:
        printD("\tCalculated but didn't save train_set_features")


def gen_test_features(test_in_sample_examples, sess, fel, x, is_training):
    printD("gen_test_features():")
    test_set_features = sess.run([fel], feed_dict={x: test_in_sample_examples, is_training: False})
    test_set_features = np.squeeze(np.asarray(test_set_features))
    if constants.SAVE:
        save_name = "test_features.npy"
        save_location = os.path.join(constants.FEATURES_DATA_FOLDER, save_name)
        np.save(save_location, test_set_features)
        printD("\tSaved generated test set features" + " at " + save_location + " having len - " + str(
            test_set_features.shape))
    else:
        printD("\tCalculated but didn't save test_set_features")

def gen_test_logits(test_in_sample_examples, sess, logits, x, is_training):
    printD("gen_test_logits():")
    test_set_logits = sess.run([logits], feed_dict={x: test_in_sample_examples, is_training: False})
    test_set_logits = np.squeeze(np.asarray(test_set_logits))
    if constants.SAVE:
        save_name = "test_logits.npy"
        save_location = os.path.join(constants.FEATURES_DATA_FOLDER, save_name)
        np.save(save_location, test_set_logits)
        printD("\tSaved generated test_set_logits" + " at " + save_location + " having len - " + str(
            test_set_logits.shape))
    else:
        printD("\tCalculated but didn't save test_set_logits")

def gen_ood_features(test_oos_examples, sess, fel, x, is_training):
    printD("gen_ood_features():")
    ood_set_features = sess.run([fel], feed_dict={x: test_oos_examples, is_training: False})
    ood_set_features = np.squeeze(np.asarray(ood_set_features))
    if constants.SAVE:
        save_name = "ood_set_features.npy"
        save_location = os.path.join(constants.OOD_FEATURES_DATA_FOLDER, save_name)
        np.save(save_location, ood_set_features)
        printD("\tSaved generated ood set features" + " at " + save_location + " having len - " + str(
            ood_set_features.shape))
    else:
        printD("\tCalculated but didn't save ood_set_features")

def gen_ood_logits(test_oos_examples, sess, logits, x, is_training):
    printD("gen_ood_logits():")
    ood_set_logits = sess.run([logits], feed_dict={x: test_oos_examples, is_training: False})
    ood_set_logits = np.squeeze(np.asarray(ood_set_logits))
    if constants.SAVE:
        save_name = "ood_set_logits.npy"
        save_location = os.path.join(constants.OOD_FEATURES_DATA_FOLDER, save_name)
        np.save(save_location, ood_set_logits)
        printD("\tSaved generated ood set features" + " at " + save_location + " having len - " + str(
            ood_set_logits.shape))
    else:
        printD("\tCalculated but didn't save ood_set_logits")

def gen_class_means(classes):
    printD("gen_class_means():")
    class_means = []
    for classID in tqdm(classes):
        save_name = "{0}_train_features.npy".format(classID)
        class_features_path = os.path.join(constants.FEATURES_DATA_FOLDER, save_name)
        class_features = np.load(class_features_path)
        cMean = np.mean(class_features, axis=0)
        class_means.append(cMean)
    class_means = np.asarray(class_means)
    if constants.SAVE:
        save_name = "train_class_means.npy"
        save_location = os.path.join(constants.MEANS_DATA_FOLDER, save_name)
        np.save(save_location, class_means)
        printD("\tSaved generated means at " + save_location + " having shape - " + str(class_means.shape))
    else:
        printD("\tCalculated but didn't save class_means")


def gen_train_dists_and_closest_classes(classes):
    printD("gen_train_dists_and_closest_classes():")
    for classID in tqdm(classes):
        train_dists = []
        train_closest_classes = []

        # load features
        save_name = "{0}_train_features.npy".format(classID)
        save_location = os.path.join(constants.FEATURES_DATA_FOLDER, save_name)
        train_set_features = np.load(save_location)

        # load means
        save_name = "train_class_means.npy"
        save_location = os.path.join(constants.MEANS_DATA_FOLDER, save_name)
        class_means = np.load(save_location)

        for feature in train_set_features:
            d = cdist(class_means, np.expand_dims(feature, axis=0), metric='cosine')
            idx = np.argmin(d)
            d = d[idx][0]
            train_closest_classes.append(idx)
            train_dists.append(d)

        train_dists = np.asarray(train_dists)
        train_closest_classes = np.asarray(train_closest_classes)

        if constants.SAVE:
            # distances
            save_name = "{0}_train_distances.npy".format(classID)
            save_location = os.path.join(constants.DISTS_DATA_FOLDER, save_name)
            np.save(save_location, train_dists)
            printD("\tSaved distances for " + str(classID) + " at " + save_location + " having shape - " + str(
                train_dists.shape))
            # closest classes
            save_name = "{0}_train_closest_classes.npy".format(classID)
            save_location = os.path.join(constants.CLOSEST_CLASS_DATA_FOLDER, save_name)
            np.save(save_location, train_closest_classes)
            printD("\tSaved closest classes for " + str(classID) + " at " + save_location + " having shape - " + str(
                train_closest_classes.shape))

    if not constants.SAVE:
        printD("\tCalculated but didn't save train_distances and train_closest_classes")


def gen_test_dists_and_closest_classes():
    printD("gen_test_dists_and_closest_classes():")
    test_dists = []
    test_closest_classes = []
    save_name = "test_features.npy"
    save_location = os.path.join(constants.FEATURES_DATA_FOLDER, save_name)
    test_set_features = np.load(save_location)

    # load means
    save_name = "train_class_means.npy"
    save_location = os.path.join(constants.MEANS_DATA_FOLDER, save_name)
    class_means = np.load(save_location)

    for feature in test_set_features:
        d = cdist(class_means, np.expand_dims(feature, axis=0), metric='cosine')
        idx = np.argmin(d)
        d = d[idx][0]
        test_closest_classes.append(idx)
        test_dists.append(d)
    test_dists = np.asarray(test_dists)
    test_closest_classes = np.asarray(test_closest_classes)

    if constants.SAVE:
        # dists
        save_name = "test_distances.npy"
        save_location = os.path.join(constants.DISTS_DATA_FOLDER, save_name)
        np.save(save_location, test_dists)
        printD("Saved distances for test data at " + save_location + " having shape - " + str(test_dists.shape))
        # closest classes
        save_name = "test_closest_classes.npy"
        save_location = os.path.join(constants.CLOSEST_CLASS_DATA_FOLDER, save_name)
        np.save(save_location, test_closest_classes)
        printD("\tSaved closest classes for test data at " + save_location + " having shape - " + str(
            test_closest_classes.shape))
    else:
        printD("\tCalculated but didn't save test_dists and test_closest_classes")


def gen_ood_dists_and_closest_classes():
    printD("gen_ood_dists_and_closest_classes():")
    ood_set_dists = []
    ood_set_closest_classes = []
    save_name = "ood_set_features.npy"
    save_location = os.path.join(constants.OOD_FEATURES_DATA_FOLDER, save_name)
    ood_set_features = np.load(save_location)

    # load means
    save_name = "train_class_means.npy"
    save_location = os.path.join(constants.MEANS_DATA_FOLDER, save_name)
    class_means = np.load(save_location)

    for feature in ood_set_features:
        d = cdist(class_means, np.expand_dims(feature, axis=0), metric='cosine')
        idx = np.argmin(d)
        d = d[idx][0]
        ood_set_closest_classes.append(idx)
        ood_set_dists.append(d)
    ood_set_dists = np.asarray(ood_set_dists)
    ood_set_closest_classes = np.asarray(ood_set_closest_classes)

    if constants.SAVE:
        # dists
        save_name = "ood_set_distances.npy"
        save_location = os.path.join(constants.OOD_DISTS_DATA_FOLDER, save_name)
        np.save(save_location, ood_set_dists)
        printD("Saved distances for ood set data at " + save_location + " having shape - " + str(ood_set_dists.shape))
        # closest classes
        save_name = "ood_set_closest_classes.npy"
        save_location = os.path.join(constants.OOD_CLOSEST_CLASS_DATA_FOLDER, save_name)
        np.save(save_location, ood_set_closest_classes)
        printD("\tSaved closest classes for ood set data at " + save_location + " having shape - " + str(
            ood_set_closest_classes.shape))
    else:
        printD("\tCalculated but didn't save ood_set_dists and ood_set_closest_classes")


def gen_radii(classes, per_class_examples):
    printD("gen_radii():")
    FRACTION_TO_COVER = 0.95

    class_radii = []
    for classID in tqdm(classes):
        class_size = per_class_examples[classID]
        modified_class_size = int(class_size * FRACTION_TO_COVER)
        # load distances
        save_name = "{0}_train_distances.npy".format(classID, class_size)
        save_location = os.path.join(constants.DISTS_DATA_FOLDER, save_name)
        class_distances = np.load(save_location)
        # set radius
        class_radii.append(class_distances[modified_class_size])

    class_radii = np.asarray(class_radii)
    if constants.SAVE:
        save_name = "train_class_radii.npy"
        save_location = os.path.join(constants.RADIUS_DATA_FOLDER, save_name)
        np.save(save_location, class_radii)
        printD("\tSaved generated radii at " + save_location + " having shape - " + str(class_radii.shape))
    else:
        printD("\tCalculated but didn't save class_radii")


def gen_all(classes, per_class_examples, in_sample_examples, in_sample_labels, test_in_sample_examples, test_oos_examples, sess, fel, x, y, is_training, logits, noisy_test_in_sample_examples, noisy_oos_examples):
    gen_train_features(classes, in_sample_examples, in_sample_labels, sess, fel, x, is_training)
    gen_test_features(test_in_sample_examples, sess, fel, x, is_training)
    gen_ood_features(test_oos_examples, sess, fel, x,is_training)
    gen_class_means(classes)
    gen_train_dists_and_closest_classes(classes)
    gen_test_dists_and_closest_classes()
    gen_ood_dists_and_closest_classes()
    gen_radii(classes, per_class_examples)
    gen_test_logits(test_in_sample_examples, sess, logits, x, is_training)
    gen_ood_logits(test_oos_examples, sess, logits, x, is_training)
