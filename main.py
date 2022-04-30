import Utils
import analysis
from Utils import printD
from Distance import generate_arrays
from Distance.distance_reuters8 import Reuters8
from Distance.distance_reuters52 import Reuters52
from Distance.distance_newsgroup20 import Newsgroup20
from Distance.Sentiment.distance_IMDB import IMDB


def main():
    Utils.init_folders()
    # reuters8 = Reuters8()
    # in_sample_examples, in_sample_labels, dev_in_sample_examples, \
    # dev_in_sample_labels, test_in_sample_examples, test_in_sample_labels, oos_examples, oos_labels = reuters8.get_data()

    # reuters8 = Reuters52()
    # in_sample_examples, in_sample_labels, oos_examples, oos_labels, \
    # dev_oos_labels, test_in_sample_examples, test_in_sample_labels, test_oos_examples, dev_oos_labels = reuters8.get_data()

    # reuters8 = Newsgroup20()
    # in_sample_examples, in_sample_labels, oos_examples, oos_labels, \
    # dev_in_sample_examples, dev_in_sample_labels, dev_oos_examples, \
    # dev_oos_labels, test_in_sample_examples, test_in_sample_labels, test_oos_examples, dev_oos_labels = reuters8.get_data()

    reuters8 = IMDB()
    in_sample_examples, in_sample_labels, X_dev, Y_dev, test_in_sample_examples, test_in_sample_labels, test_oos_examples1, test_oos_labels1, test_oos_examples, test_oos_labels = reuters8.get_data()

    print(f"Total in-sample test examples = { len(list(test_in_sample_labels)) }")
    print(f"Total out-sample test examples = { len(list(test_oos_examples)) }")
    print(f"Total in-sample classes = { len(list(set(in_sample_labels))) }")
    print(f"Total out-sample classes = { len(list(set(test_oos_labels))) }")

    sess, saver, graph, fel, x, y, is_training = reuters8.train_model()
    classes, NUM_CLASSES = Utils.get_class_info(in_sample_labels)

    FEATURE_LAYER_SIZE = fel.shape[-1]
    print(FEATURE_LAYER_SIZE)

    per_class_examples = Utils.get_per_class_info(in_sample_examples, in_sample_labels, classes)
    generate_arrays.gen_all(classes, per_class_examples, in_sample_examples, in_sample_labels, test_in_sample_examples,
                            test_oos_examples, sess, fel, x, y, is_training)

    analysis.calculate_all(NUM_CLASSES)


if __name__ == "__main__":
    main()
