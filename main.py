import numpy as np

import Utils
import analysis
import constants
from Utils import printD
from Distance import generate_arrays
from Distance.distance_reuters8 import Reuters8
from Distance.distance_reuters52 import Reuters52
from Distance.distance_newsgroup20 import Newsgroup20
from Distance.Sentiment.distance_IMDB import IMDB
import sklearn.metrics as sk


def main():
    # Utils.init_folders()

    # constants.set_data_names("Reuters6", "Reuters2")
    # Utils.init_folders()
    # reuters8 = Reuters8()
    # in_sample_examples, in_sample_labels, dev_in_sample_examples, \
    # dev_in_sample_labels, test_in_sample_examples, test_in_sample_labels, oos_examples, oos_labels = reuters8.get_data()
    # sess, saver, graph, fel, logits, x, y, is_training, safe, risky = reuters8.train_model()

    # constants.set_data_names("Reuters40", "Reuters12")
    # Utils.init_folders()
    # reuters8 = Reuters52()
    # in_sample_examples, in_sample_labels, oos_examples, oos_labels, \
    # dev_oos_labels, test_in_sample_examples, test_in_sample_labels, test_oos_examples, dev_oos_labels = reuters8.get_data()
    # sess, saver, graph, fel, logits, x, y, is_training, safe, risky = reuters8.train_model()

    # constants.set_data_names("15Newsgroup", "5Newsgroup")
    # Utils.init_folders()
    # print(constants.DATA_NAME, constants.OOD_DATA_NAME)
    # reuters8 = Newsgroup20()
    # in_sample_examples, in_sample_labels, oos_examples, oos_labels, \
    # dev_in_sample_examples, dev_in_sample_labels, dev_oos_examples, \
    # dev_oos_labels, test_in_sample_examples, test_in_sample_labels, test_oos_examples, dev_oos_labels = reuters8.get_data()
    # sess, saver, graph, fel, logits, x, y, is_training, safe, risky = reuters8.train_model()

    constants.set_data_names("IMDB", "CR")
    # constants.set_data_names("IMDB", "MR")
    reuters8 = IMDB()
    X_dev, Y_dev, in_sample_examples, in_sample_labels, test_in_sample_examples, test_in_sample_labels, oos_examples, oos_labels, oos_examples1, oos_labels1 = reuters8.get_data()
    sess, saver, graph, fel, logits, x, y, is_training, safe, risky, risky1 = reuters8.train_model()

    total_in_sample_classes = len(list(set(in_sample_labels)))
    total_out_sample_classes = len(list(set(oos_labels)))
    print(f"Total in-sample test examples = { len(list(test_in_sample_labels)) }")
    print(f"Total out-sample test examples = { len(list(oos_examples)) }")
    print(f"Total in-sample classes = { total_in_sample_classes }")
    print(f"Total out-sample classes = { total_out_sample_classes }")

    classes, NUM_CLASSES = Utils.get_class_info(in_sample_labels)

    FEATURE_LAYER_SIZE = fel.shape[-1]
    print(FEATURE_LAYER_SIZE)

    # per_class_examples = Utils.get_per_class_info(in_sample_examples, in_sample_labels, classes)
    # generate_arrays.gen_all(classes, per_class_examples, in_sample_examples, in_sample_labels, test_in_sample_examples,
    #                         oos_examples, sess, fel, x, y, is_training, logits)

    labels = np.zeros((safe.shape[0] + risky.shape[0]), dtype=np.int32)
    labels[:safe.shape[0]] += 1
    examples = np.squeeze(np.vstack((safe, risky)))
    print('AUPR (%):', round(100 * sk.average_precision_score(labels, examples), 2))
    print('AUROC (%):', round(100 * sk.roc_auc_score(labels, examples), 2))

    safe = np.squeeze(safe).tolist()
    risky = np.squeeze(risky).tolist()
    #
    # analysis.calculate_all(NUM_CLASSES)
    analysis.calculate_all2(safe, risky, True)


if __name__ == "__main__":
    main()
