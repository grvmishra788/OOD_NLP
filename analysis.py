import os
import numpy as np
import matplotlib.pylab as plt
from matplotlib import pyplot
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from Distance import load_arrays
import Utils
from Utils import printD
import constants
from scipy.special import logsumexp


def calculate_all2(test_distances, ood_set_distances, reverse=False):
    if reverse:
        test_distances = -1 * np.array(test_distances)
        ood_set_distances = -1 * np.array(ood_set_distances)
    y_true = []
    for i in range(len(test_distances)):
        y_true.append(0)
    for i in range(len(ood_set_distances)):
        y_true.append(1)

    RESULTS = {"TNR at 95% TPR": calculate_TNR_at_95_TPR(test_distances, ood_set_distances, "Baseline"),
               "Detection Accuracy": calculate_detection_accuracy(y_true, test_distances, ood_set_distances),
               "AUPR_out": calculate_AUPR_out(y_true, test_distances, ood_set_distances),
               "AUPR_in": calculate_AUPR_in(y_true, test_distances, ood_set_distances),
               "AUROC": calculate_AUROC(y_true, test_distances, ood_set_distances)}

    """## **Overall results**"""
    # pprint(RESULTS)
    return RESULTS

def calculate_all(data="Reuters8"):
    """## ** Check output folders**"""
    Utils.check_all_paths()

    """## **Load data**"""
    class_radii = load_arrays.load_class_radii()
    test_distances, test_closest_classes = load_arrays.load_test_dists_and_closest_classes()
    ood_set_distances, ood_set_closest_classes = load_arrays.load_OOD_dists_and_closest_classes()

    test_logits, ood_logits = load_arrays.load_test_logits(), load_arrays.load_ood_logits()
    test_energy_score, ood_energy_score = calculate_energy_score(test_logits), calculate_energy_score(ood_logits)
    test_soft_score, ood_soft_score = softmax_temp_score(test_logits), softmax_temp_score(ood_logits)

    test_ensemble_score, ood_ensemble_score = ensemble(test_distances, ood_set_distances, test_energy_score,
                                                       ood_energy_score, test_soft_score, ood_soft_score, data)
    y_true = []
    for i in range(len(test_distances)):
        y_true.append(0)
    for i in range(len(ood_set_distances)):
        y_true.append(1)

    printD("-----------------------------------------------------------------------------------")
    printD("Results of energy based OOD")
    RESULTS_ENERGY = {"TNR at 95% TPR": calculate_TNR_at_95_TPR(test_energy_score, ood_energy_score, "Energy"),
                      "Detection Accuracy": calculate_detection_accuracy(y_true, test_energy_score, ood_energy_score),
                      "AUPR_out": calculate_AUPR_out(y_true, test_energy_score, ood_energy_score),
                      "AUPR_in": calculate_AUPR_in(y_true, test_energy_score, ood_energy_score),
                      "AUROC": calculate_AUROC(y_true, test_energy_score, ood_energy_score)}
    # pprint(RESULTS_ENERGY)

    printD("-----------------------------------------------------------------------------------")
    printD("Results of temperature scaling based OOD")
    RESULTS_TEMP = {"TNR at 95% TPR": calculate_TNR_at_95_TPR(test_soft_score, ood_soft_score, "Temperature Scaling"),
                    "Detection Accuracy": calculate_detection_accuracy(y_true, test_soft_score, ood_soft_score),
                    "AUPR_out": calculate_AUPR_out(y_true, test_soft_score, ood_soft_score),
                    "AUPR_in": calculate_AUPR_in(y_true, test_soft_score, ood_soft_score),
                    "AUROC": calculate_AUROC(y_true, test_soft_score, ood_soft_score)}
    # pprint(RESULTS_TEMP)

    printD("-----------------------------------------------------------------------------------")
    printD("Results of Distance based OOD")
    RESULTS_DIST = {"TNR at 95% TPR": calculate_TNR_at_95_TPR(test_distances, ood_set_distances, "Cosine distance"),
                    "Detection Accuracy": calculate_detection_accuracy(y_true, test_distances, ood_set_distances),
                    "AUPR_out": calculate_AUPR_out(y_true, test_distances, ood_set_distances, "Cosine distance" ),
                    "AUPR_in": calculate_AUPR_in(y_true, test_distances, ood_set_distances, "Cosine distance"),
                    "AUROC": calculate_AUROC(y_true, test_distances, ood_set_distances, "Cosine distance")}

    printD("-----------------------------------------------------------------------------------")
    printD("Results of Ensemble based OOD")
    RESULTS_ENSEMBLE = {"TNR at 95% TPR": calculate_TNR_at_95_TPR(test_ensemble_score, ood_ensemble_score, "Ensemble"),
                        "Detection Accuracy": calculate_detection_accuracy(y_true, test_ensemble_score,
                                                                           ood_ensemble_score),
                        "AUPR_out": calculate_AUPR_out(y_true, test_ensemble_score, ood_ensemble_score),
                        "AUPR_in": calculate_AUPR_in(y_true, test_ensemble_score, ood_ensemble_score),
                        "AUROC": calculate_AUROC(y_true, test_ensemble_score, ood_ensemble_score)}

    if data == "IMDB_CR" or data == "IMDB_MR":
        RESULTS_ENSEMBLE = Utils.calculate_ensemble(RESULTS_ENSEMBLE)

    """## **Overall results**"""
    # pprint(RESULTS_DIST)
    return RESULTS_TEMP, RESULTS_ENERGY, RESULTS_DIST, RESULTS_ENSEMBLE


def calculate_TNR_at_95_TPR(test_distances, ood_set_distances, method=""):
    """## **Calculate TNR at 95% TPR**"""

    # sort scores
    test_dists_sorted = sorted(test_distances)
    ood_set_dists_sorted = sorted(ood_set_distances)

    threshold = test_dists_sorted[int(0.95 * len(test_dists_sorted))]

    out_of_bound_scores = [a for a in ood_set_dists_sorted if a > threshold]
    TOTAL_OUT_OF_BOUND_SCORES = len(out_of_bound_scores)
    TOTAL_IN_BOUND_SCORES = len(ood_set_dists_sorted) - TOTAL_OUT_OF_BOUND_SCORES
    printD("Total out-of-bound scores - " + str(TOTAL_OUT_OF_BOUND_SCORES))
    printD("Total in-bound scores - " + str(TOTAL_IN_BOUND_SCORES))
    res = TOTAL_OUT_OF_BOUND_SCORES / len(ood_set_dists_sorted)
    printD("TNR at 95% TPR - {0}%".format(res))

    if constants.PLOT_FIG and method == "Cosine distance":
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
        plt.ylabel(f"{method} scores")
        plt.legend([constants.DATA_NAME, constants.OOD_DATA_NAME, "Calculating TNR at 95% TPR"])
        plt.savefig(os.path.join("outputs", "results", f"{method}_TNR_at_95%_TPR.png"))
        plt.show()
        plt.draw()

    return round(100 * res, 2)


def calculate_detection_accuracy(y_true, test_distances, ood_set_distances):
    """## **Calculate Detection Accuracy**"""
    # sort scores to get threshold
    test_dists_sorted = sorted(test_distances)
    threshold = test_dists_sorted[int(0.95 * len(test_dists_sorted))]

    y_pred = []
    for i in range(len(test_distances)):
        if test_distances[i] > threshold:
            y_pred.append(1)
        else:
            y_pred.append(0)

    for i in range(len(ood_set_distances)):
        if ood_set_distances[i] > threshold:
            y_pred.append(1)
        else:
            y_pred.append(0)

    res = round(100 * metrics.accuracy_score(y_true, y_pred), 2)
    return res


def calculate_AUPR_out(y_true, test_distances, ood_set_distances, method=""):
    """## **Calculate AUPR_{out} score**"""
    precision, recall, thresholds = metrics.precision_recall_curve(y_true,
                                                                   np.concatenate((test_distances, ood_set_distances)),
                                                                   pos_label=1)
    if constants.PLOT_FIG and method == "Cosine distance":
        # plot model roc curve
        pyplot.plot(recall, precision)
        plt.title("Precision-Recall curve: Out distribution -> +ve")
        # axis labels
        pyplot.xlabel('Recall')
        pyplot.ylabel('Precision')
        # show the legend
        pyplot.legend()
        # show the plot
        plt.savefig(os.path.join("outputs", "results", f"{method}_PR_Curve_out.png"))
        pyplot.show()

    return round(100 * metrics.auc(recall, precision), 2)


def calculate_AUPR_in(y_true, test_distances, ood_set_distances, method=""):
    """## **Calculate AUPR_{in} score**"""
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, - np.concatenate(
        (test_distances, ood_set_distances)), pos_label=0)
    if constants.PLOT_FIG and method == "Cosine distance":
        # plot model roc curve
        pyplot.plot(recall, precision)
        plt.title("Precision-Recall curve: In distribution -> +ve")
        # axis labels
        pyplot.xlabel('Recall')
        pyplot.ylabel('Precision')
        # show the legend
        pyplot.legend()
        # show the plot
        plt.savefig(os.path.join("outputs", "results", f"{method}_PR_Curve_in.png"))
        pyplot.show()

    return round(100 * metrics.auc(recall, precision), 2)


def calculate_AUROC(y_true, test_distances, ood_set_distances, method=""):
    """## **Calculate AUROC**"""
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(y_true, np.concatenate((test_distances, ood_set_distances)))
    roc_auc = auc(fpr, tpr)
    if constants.PLOT_FIG and method == "Cosine distance":
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
        plt.savefig(os.path.join("outputs", "results", f"{method}_ROC_Curve.png"))
        plt.show()
    return round(100 * roc_auc, 2)


def calculate_energy_score(logits):
    energy_score = []
    for l in logits:
        energy_score.append(-1 * constants.ENERGY_TEMP * logsumexp(l / constants.ENERGY_TEMP))
    return np.array(energy_score)


def calculate_prob_energy_score(logits):
    energy_score = []
    for l in logits:
        Ex = -constants.ENERGY_TEMP * logsumexp(l / constants.ENERGY_TEMP)
        energy_score.append(Ex)
    return np.array(energy_score)


def softmax_temp_score(logits):
    softmax_temp = []
    for l in logits:
        e_x = np.exp(l / constants.SOFTMAX_TEMP)
        soft = e_x / e_x.sum()
        softmax_temp.append(1 - np.max(soft))
    return np.array(softmax_temp)


def ensemble(test_distances, ood_set_distances, test_energy_score, ood_energy_score, test_soft_score, ood_soft_score, data="Reuters8"):

    W1 = 2
    W2 = 1
    W3 = 1

    if data == "IMDB_CR" or data == "IMDB_MR":
        W1 = 1
        W2 = 1
        W3 = 100*max(abs(test_energy_score))

    return W1 * test_distances + W2 * test_soft_score - W3 * (1 - test_energy_score/max(abs(test_energy_score))), \
           W1 * ood_set_distances + W2 * ood_soft_score - W3 *(1 - ood_energy_score/max(abs(test_energy_score)))
