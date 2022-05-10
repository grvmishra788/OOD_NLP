import os

import numpy as np
import pandas as pd
# from torch import softmax
import Utils
import analysis
import constants
import noise
from Utils import printD
from Distance import generate_arrays
from Distance.distance_reuters8 import Reuters8
from Distance.distance_reuters52 import Reuters52
from Distance.distance_newsgroup20 import Newsgroup20
from Distance.Sentiment.distance_IMDB import IMDB
import sklearn.metrics as sk
from pprint import pprint
import argparse


def main():
    parser = argparse.ArgumentParser(description='Arguements for the OOD detection')
    parser.add_argument('--energy_temp', default=5,
                        help='Temperature value for enerrgy based OOD')
    parser.add_argument('--softmax_temp', default=10,
                        help='Temperature value for temperature scaling OOD')
    parser.add_argument('--retrain', default=False,
                        help='retrain')
    parser.add_argument('--debug', default=False,
                        help='retrain')                        
    parser.add_argument('--output_folder', default="outputs",
                        help='debug')                                                                        

    args = parser.parse_args()
    constants.set_energy_temp(args.energy_temp)
    constants.set_energy_temp(args.softmax_temp)
    constants.set_retrain(args.retrain)
    constants.set_debug(args.debug)
    constants.set_output_folder(args.output_folder)
    # with open("Results_presentation.log", "w") as log_file:
    for data in constants.DATASETS:
        results = pd.DataFrame()
        if data == "Reuters8":
            constants.set_data_names("Reuters6", "Reuters2")
            reuters8 = Reuters8()
            in_sample_examples, in_sample_labels, dev_in_sample_examples, \
            dev_in_sample_labels, test_in_sample_examples, test_in_sample_labels, oos_examples, oos_labels = reuters8.get_data()
            sess, saver, graph, fel, logits, x, y, is_training, safe, risky = reuters8.train_model()
        elif data == "Reuters52":
            constants.set_data_names("Reuters40", "Reuters12")
            reuters8 = Reuters52()
            in_sample_examples, in_sample_labels, dev_in_sample_examples, \
            dev_in_sample_labels, test_in_sample_examples, test_in_sample_labels, oos_examples, oos_labels = reuters8.get_data()
            sess, saver, graph, fel, logits, x, y, is_training, safe, risky = reuters8.train_model()
        elif data == "20NG":
            constants.set_data_names("15Newsgroup", "5Newsgroup")
            reuters8 = Newsgroup20()
            in_sample_examples, in_sample_labels, dev_in_sample_examples, \
            dev_in_sample_labels, test_in_sample_examples, test_in_sample_labels, oos_examples, oos_labels = reuters8.get_data()
            sess, saver, graph, fel, logits, x, y, is_training, safe, risky = reuters8.train_model()
        elif data == "IMDB_CR":
            constants.set_data_names("IMDB", "CR")
            reuters8 = IMDB()
            X_dev, Y_dev, in_sample_examples, in_sample_labels, test_in_sample_examples, test_in_sample_labels, oos_examples, oos_labels, oos_examples1, oos_labels1 = reuters8.get_data()
            sess, saver, graph, fel, logits, x, y, is_training, safe, risky, risky1 = reuters8.train_model()
        else:
            constants.set_data_names("IMDB", "MR")
            reuters8 = IMDB()
            X_dev, Y_dev, in_sample_examples, in_sample_labels, test_in_sample_examples, test_in_sample_labels, oos_examples1, oos_labels1, oos_examples, oos_labels = reuters8.get_data()
            sess, saver, graph, fel, logits, x, y, is_training, safe, risky1, risky = reuters8.train_model()

        print("---------------In- and out- domain Class Statistics------------------")
        total_in_sample_classes = len(list(set(in_sample_labels)))
        total_out_sample_classes = len(list(set(oos_labels)))
        print(f"Total in-sample test examples = {len(list(test_in_sample_labels))}")
        print(f"Total out-sample test examples = {len(list(oos_examples))}")
        print(f"Total in-sample classes = {total_in_sample_classes}")
        print(f"Total out-sample classes = {total_out_sample_classes}")
        print("---------------------------------------------------------------------")

        # get noisy data
        noisy_test_in_sample_examples = noise.add_noise(test_in_sample_examples)
        noisy_oos_examples = noise.add_noise(oos_examples)

        classes, NUM_CLASSES = Utils.get_class_info(in_sample_labels)

        per_class_examples = Utils.get_per_class_info(in_sample_examples, in_sample_labels, classes)
        generate_arrays.gen_all(classes, per_class_examples, in_sample_examples, in_sample_labels, test_in_sample_examples,
                                oos_examples, sess, fel, x, y, is_training, logits, noisy_test_in_sample_examples, noisy_oos_examples)

        sess.close()

        labels = np.zeros((safe.shape[0] + risky.shape[0]), dtype=np.int32)
        labels[:safe.shape[0]] += 1
        examples = np.squeeze(np.vstack((safe, risky)))
        print('AUPR (%):', round(100 * sk.average_precision_score(labels, examples), 2))
        print('AUROC (%):', round(100 * sk.roc_auc_score(labels, examples), 2))

        safe = np.squeeze(safe).tolist()
        risky = np.squeeze(risky).tolist()

        # RESULTS_TEMP, RESULTS_ENERGY, RESULTS_DIST
        a, b, c, d = analysis.calculate_all()
        results['Baseline'] = pd.Series(analysis.calculate_all2(safe, risky, True))
        results['Temp Scaling'], results['Energy'], results['Distance'], results['Ensemble'] = pd.Series(a), pd.Series(b), pd.Series(c), pd.Series(d)
        print(f"\n---------------{data}------------------")
        pprint(results)
        print("------------------------------------------\n")
        results.to_csv(os.path.join("outputs", "results", f"{data}.csv"))

        # pprint(f"\n---------------{data}------------------", log_file)
        # pprint(results, log_file)
        # pprint("------------------------------------------\n", log_file)


if __name__ == "__main__":
    main()
