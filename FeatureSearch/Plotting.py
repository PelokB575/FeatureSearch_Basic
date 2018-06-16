import numpy

import pandas
from matplotlib import pyplot
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn import metrics


def plot_auc(scorefilename):
    data_no = pandas.read_csv(scorefilename, names=['label', 'score'])
    labels_no = data_no['label']
    scores_no = data_no['score']
    labels_no = [int(e) for e in labels_no]
    scores_no = [float(e) for e in scores_no]
    auc_value_no = metrics.roc_auc_score(numpy.array(labels_no), numpy.array(scores_no))

    fpr_no, tpr_no, thresholds_no = metrics.roc_curve(labels_no, scores_no, pos_label=1)
    eer_no = brentq(lambda x: 1. - x - interp1d(fpr_no, tpr_no)(x), 0., 1.)
    print(eer_no)
    # thresh_no = interp1d(fpr_no, thresholds_no)(eer_no)

    pyplot.figure()
    lw = 2
    pyplot.plot(fpr_no, tpr_no, color='black', lw=lw, label='AUC = %0.4f' % auc_value_no)
    pyplot.plot([0, 1], [0, 1], color='darkorange', lw=lw, linestyle='--')
    pyplot.xlim([0.0, 1.0])
    pyplot.ylim([0.0, 1.05])
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.title('AUC')
    pyplot.legend(loc="lower right")
    pyplot.show()
    return


if __name__ == '__main__':
    plot_auc('9Files_largescale_onlyCPP_2018-06-12_13_46_LDA_resprob.csv')