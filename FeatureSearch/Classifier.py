import csv
import datetime
import numpy
import warnings
import pandas

from arff2pandas import a2p

from matplotlib import pyplot

from scipy.interpolate import interp1d
from scipy.optimize import brentq

from sklearn import model_selection, metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def create_dataset(file_path):
    with open(file_path, 'r') as f:
        dataset = a2p.load(f)

    return dataset


def two_step_classifier(dataset):
    if dataset.values.size < 18:
        print('Not enough values in dataset')
        return

    resprob = {'LR': [], 'LDA': [], 'KNN': [], 'CART': [], 'NB': [], 'SVM': [], 'RF': []}

    classifier_file = base.split('\\')[-1] + "_" + str(
        datetime.datetime.now().strftime('%Y-%m-%d_%H_%M')) + "_classifier_results.csv"
    with open(classifier_file, 'w', newline='') as f:
        names = []
        models = [('LR', LogisticRegression()), ('LDA', LinearDiscriminantAnalysis()), ('KNN', KNeighborsClassifier()),
                  ('CART', DecisionTreeClassifier()), ('NB', GaussianNB()), ('SVM', SVC(probability=True)),
                  ('RF', RandomForestClassifier())]

        all_results = [0 for _ in models]
        # print(all_results)
        csv_keys = ['Author'] + [n for n, _ in models]
        w = csv.writer(f, csv_keys)

        for user_no in range(0, int(dataset.shape[0] / 9)):
            dset_vals_pos = dataset[(user_no * 9):(user_no * 9) + 9].values

            dset_vals_neg = pandas.concat([dataset[0:(user_no * 9)], dataset[(user_no * 9) + 9:]]).sample(n=9).values

            dset_vals_neg[:, -1] = [0 for _ in dset_vals_neg]

            dset_vals = pandas.concat([pandas.DataFrame(dset_vals_pos), pandas.DataFrame(dset_vals_neg)]).values

            X = dset_vals[:, 0:-1]
            Y = dset_vals[:, -1]

            # print(X)

            val_size = 1 / 3

            seed = 65537

            X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=val_size,
                                                                                            random_state=seed)

            print(Y_train)
            print(Y_validation)
            # print(X_validation)

            scoring = 'accuracy'
            results = []

            for name, model in models:
                # print(f'Model name: {name}')
                kfold = model_selection.KFold(n_splits=12, random_state=seed)
                cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
                results.append(cv_results)
                names.append(name)

                model.fit(X_train, Y_train)
                resprob_temp = model.predict_proba(X_validation)
                # print(resprob_temp)

                for i in range(0, len(resprob_temp)):
                    resprob[name].append((int(Y_validation[i] != 'noname'), resprob_temp[i][0]))

                # msg = "%s: %f (%f)\n" % (name, cv_results.mean(), cv_results.std())
                # print(msg)
                # f.write(msg)
            rmeans = [r.mean() for r in results]
            # print(rmeans)
            # print(all_results)
            w.writerow([dset_vals_pos[0, -1]] + rmeans)
            all_results = [all_results[i] + rmeans[i] for i in range(0, len(models))]
            # print(all_results)
        # print([(sum(r)/len(all_results)).mean() for r in all_results])
        all_results = [i / int(dataset.shape[0] / 9) for i in all_results]
        # print(all_results)
        for name, _ in models:
            resprob_filename = base.split('\\')[-1] + "_" + str(datetime.datetime.now().strftime('%Y-%m-%d_%H_%M')) + f"_{name}_resprob.csv"
            with open(resprob_filename, 'w', newline='') as resprob_out:
                csv_out = csv.writer(resprob_out)
                for row in resprob[name]:
                    csv_out.writerow(row)
        w.writerow(['TOTAL'] + [str(i) for i in all_results])


def compare_to_user_files(user_no, target_features, classifier, dataset):
    used_classifier = [('LR', LogisticRegression()), ('LDA', LinearDiscriminantAnalysis()), ('KNN', KNeighborsClassifier()),
                       ('CART', DecisionTreeClassifier()), ('NB', GaussianNB()), ('SVM', SVC(probability=True)),
                       ('RF', RandomForestClassifier())][classifier]

    if dataset.shape[0] < 18:
        return -1

    dset_vals_pos = dataset[(user_no * 9):(user_no * 9) + 9].values

    dset_vals_neg = pandas.concat([dataset[0:(user_no * 9)], dataset[(user_no * 9) + 9:]]).sample(n=9).values

    dset_vals_neg[:, -1] = ['noname' for _ in dset_vals_neg]

    dset_vals = pandas.concat([pandas.DataFrame(dset_vals_pos), pandas.DataFrame(dset_vals_neg)]).values

    X = dset_vals[:, 0:-1]
    Y = dset_vals[:, -1]

    # print(X)

    val_size = 1 / 3

    seed = 65537

    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=val_size,
                                                                                    random_state=seed)

    print(X_train)
    scoring = 'accuracy'

    used_model = used_classifier[1]
    used_model.fit(X_train, Y_train)

    shaped_feats = numpy.array(list(target_features.values())[0: -1]).reshape(1, -1)

    return used_model.predict_proba(shaped_feats)[0][0]


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    base = "C:\\Users\\Bence\\Documents\\Lecke\\Diplomamunka\\CPP_Files\\9Files_largescale_onlyCPP"
    arff_dataset = create_dataset("9Files_largescale_onlyCPP_2018-05-28_23_57.arff")

    two_step_classifier(arff_dataset)

