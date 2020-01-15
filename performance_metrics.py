import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 

from sklearn.metrics import roc_curve, auc



def plot_roc(ground_truth, scores):
    """
    Ground truth: true anomaly labels of test data
    Scores: anomaly scores of test data
    """
    fpr, tpr, thresholds = roc_curve(ground_truth, scores)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(fpr, tpr, label="ROC Curve (AUC = %0.2f)" % auc(fpr, tpr))
    ax.plot([0, 1], [0, 1], color="black", linestyle="--")
    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.05))
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    plt.show()



def plot_normal_vs_anomalous(targets, scores, anomaly, quantile):
    """
    targets: true labels of test data (numpy array)
    scores: anomaly scores of test data (numpy array)
    anomaly: label of the anomalous class
    quantile: assumed percentage of anomalies
    """
    anomalous_mask = targets == anomaly
    normal_mask = targets != anomaly

    # subset the scores with boolean masks
    anomalous_scores = scores[anomalous_mask]
    normal_scores = scores[normal_mask]

    # classifier threshold
    limit = np.quantile(scores, quantile)

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.distplot(anomalous_scores, label="Anomalous")
    sns.distplot(normal_scores, label="Normal")
    ax.axvline(limit, color="red")
    ax.set_xlabel("Likelihood")
    ax.set_ylabel("Probability")
    ax.legend()
    plt.show()