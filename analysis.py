import os
import numpy as np
import matplotlib.pylab as plt
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from constants import *
from Distance import load_arrays
import Utils
from pprint import pprint


def calculate_all(NUM_CLASSES):
    """## ** Check output folders**"""
    Utils.check_all_paths()

    """## **Load data**"""
    # load train features
    # train_features = load_arrays.load_train_features(NUM_CLASSES)
    # load test features
    # test_features = load_arrays.load_test_features()
    # load ood features
    # ood_set_features = load_arrays.load_ood_features()
    # load class means
    # class_means = load_arrays.load_class_means()
    # load class radii
    class_radii = load_arrays.load_class_radii()
    # load train distances & closest classes
    # train_distances, train_closest_classes = load_arrays.load_train_dists_and_closest_classes(NUM_CLASSES)
    # load test distances
    test_distances, test_closest_classes = load_arrays.load_test_dists_and_closest_classes()
    # load OOD distances
    ood_set_distances, ood_set_closest_classes = load_arrays.load_OOD_dists_and_closest_classes()

    y_pred = []
    y_true = []
    for i in range(len(test_distances)):
        y_true.append(0)
        if test_distances[i] > class_radii[test_closest_classes[i]]:
            y_pred.append(1)
        else:
            y_pred.append(0)

    for i in range(len(ood_set_distances)):
        y_true.append(1)
        if ood_set_distances[i] > class_radii[ood_set_closest_classes[i]]:
            y_pred.append(1)
        else:
            y_pred.append(0)

    RESULTS = {"TNR at 95% TPR": calculate_TNR_at_95_TPR(test_distances, ood_set_distances),
               "Detection Accuracy": calculate_detection_accuracy(y_true, y_pred),
               "AUPR_out": calculate_AUPR_out(y_true, test_distances, ood_set_distances),
               "AUPR_in": calculate_AUPR_in(y_true, test_distances, ood_set_distances),
               "AUROC": calculate_AUROC(y_true, test_distances, ood_set_distances)}

    """## **Overall results**"""
    pprint(RESULTS)


def calculate_TNR_at_95_TPR(test_distances, ood_set_distances):
    """## **Calculate TNR at 95% TPR**"""

    # sort scores
    test_dists_sorted = sorted(test_distances)
    ood_set_dists_sorted = sorted(ood_set_distances)

    threshold = test_dists_sorted[int(0.95 * len(test_dists_sorted))]
    print(threshold)

    out_of_bound_scores = [a for a in ood_set_dists_sorted if a > threshold]
    TOTAL_OUT_OF_BOUND_SCORES = len(out_of_bound_scores)
    TOTAL_IN_BOUND_SCORES = len(ood_set_dists_sorted) - TOTAL_OUT_OF_BOUND_SCORES
    print("Total out-of-bound scores - " + str(TOTAL_OUT_OF_BOUND_SCORES))
    print("Total in-bound scores - " + str(TOTAL_IN_BOUND_SCORES))
    res = TOTAL_OUT_OF_BOUND_SCORES * 100 / len(ood_set_dists_sorted)
    print("TNR at 95% TPR - {0}%".format(res))

    plt.plot(test_dists_sorted)
    plt.plot(ood_set_dists_sorted)

    x_coordinates = [int(0.95 * len(test_dists_sorted)), int(0.95 * len(test_dists_sorted))]
    y_coordinates = [0, threshold]
    plt.plot(x_coordinates, y_coordinates, 'k--')

    x_coordinates = [TOTAL_IN_BOUND_SCORES, int(0.95 * len(test_dists_sorted))]
    y_coordinates = [threshold, threshold]
    plt.plot(x_coordinates, y_coordinates, 'k--')

    x_coordinates = [TOTAL_IN_BOUND_SCORES, TOTAL_IN_BOUND_SCORES]
    y_coordinates = [0, threshold]
    plt.plot(x_coordinates, y_coordinates, 'k--')

    plt.title("Sorted scores of in- and out- distribution test samples")
    plt.xlabel("Sample No.")
    plt.ylabel("Scores with cosine distance")
    plt.legend([DATA_NAME, OOD_DATA_NAME, "Calculating TNR at 95% TPR"])
    plt.savefig("TNR_at_95%_TPR.png")
    plt.show()
    plt.draw()

    return res


def calculate_detection_accuracy(y_true, y_pred):
    """## **Calculate Detection Accuracy**"""
    res = metrics.accuracy_score(y_true, y_pred)
    return res


def calculate_AUPR_out(y_true, test_distances, ood_set_distances):
    """## **Calculate AUPR_{out} score**"""
    precision, recall, thresholds = metrics.precision_recall_curve(y_true,
                                                                   np.concatenate((test_distances, ood_set_distances)),
                                                                   pos_label=1)

    from matplotlib import pyplot

    # plot model roc curve
    pyplot.plot(recall, precision)
    plt.title("Precision-Recall curve: Out distribution -> +ve")
    # axis labels
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    # show the legend
    pyplot.legend()
    # show the plot
    plt.savefig("PR_Curve_out.png")
    pyplot.show()

    return metrics.auc(recall, precision)


def calculate_AUPR_in(y_true, test_distances, ood_set_distances):
    """## **Calculate AUPR_{in} score**"""
    precision, recall, thresholds = metrics.precision_recall_curve(y_true,
                                                                   1 - np.concatenate((test_distances, ood_set_distances)),
                                                                   pos_label=0)

    from matplotlib import pyplot

    # plot model roc curve
    pyplot.plot(recall, precision)
    plt.title("Precision-Recall curve: In distribution -> +ve")
    # axis labels
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    # show the legend
    pyplot.legend()
    # show the plot
    plt.savefig("PR_Curve_in.png")
    pyplot.show()

    return metrics.auc(recall, precision)


def calculate_AUROC(y_true, test_distances, ood_set_distances):
    """## **Calculate AUROC**"""
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(y_true, np.concatenate((test_distances, ood_set_distances)))
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.4f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    plt.savefig("ROC_Curve.png")
    plt.show()
    return roc_auc

