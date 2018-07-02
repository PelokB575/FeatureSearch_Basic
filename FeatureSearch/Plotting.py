import numpy

import pandas
from matplotlib import pyplot
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn import metrics


def plot_auc(scorefiles):
    pyplot.figure()
    lw = 2
    for scorefilename, classifier_name in scorefiles:
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

        pyplot.plot(fpr_no, tpr_no, lw=lw, label=f'{classifier_name} -- AUC = %0.4f' % auc_value_no)
    pyplot.plot([0, 1], [0, 1], color='darkorange', lw=lw, linestyle='--')
    pyplot.xlim([-0.01, 1.01])
    pyplot.ylim([-0.01, 1.01])
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.title('AUC')
    pyplot.legend(loc="lower right")
    pyplot.show()
    return


if __name__ == '__main__':
    plot_auc([('../9Files_largescale_onlyCPP_2018-06-25_18_27_LR_resprob.csv', 'Logistic Regression'),
              ('../9Files_largescale_onlyCPP_2018-06-25_18_27_LDA_resprob.csv', 'Linear Discriminant Analysis'),
              ('../9Files_largescale_onlyCPP_2018-06-25_18_27_KNN_resprob.csv', 'K-Nearest Neighbor'),
              ('../9Files_largescale_onlyCPP_2018-06-25_18_27_CART_resprob.csv', 'Decision Tree'),
              ('../9Files_largescale_onlyCPP_2018-06-25_18_27_NB_resprob.csv', 'Naive Bayes'),
              ('../9Files_largescale_onlyCPP_2018-06-25_18_27_SVM_resprob.csv', 'Support Vector Machine'),
              ('../9Files_largescale_onlyCPP_2018-06-25_18_27_RF_resprob.csv', 'Random Forest')])
    # plot_auc([('../9Files_largescale_onlyCPP_2018-06-27_14_04_LR_resprob.csv', 'Logistic Regression'),
    #           ('../9Files_largescale_onlyCPP_2018-06-27_14_04_LDA_resprob.csv', 'Linear Discriminant Analysis'),
    #           ('../9Files_largescale_onlyCPP_2018-06-27_14_04_KNN_resprob.csv', 'K-Nearest Neighbor'),
    #           ('../9Files_largescale_onlyCPP_2018-06-27_14_04_CART_resprob.csv', 'Decision Tree'),
    #           ('../9Files_largescale_onlyCPP_2018-06-27_14_04_NB_resprob.csv', 'Naive Bayes'),
    #           ('../9Files_largescale_onlyCPP_2018-06-27_14_04_SVM_resprob.csv', 'Support Vector Machine'),
    #           ('../9Files_largescale_onlyCPP_2018-06-27_14_04_RF_resprob.csv', 'Random Forest')])
