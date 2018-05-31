import csv
import datetime
import warnings

import pandas
from arff2pandas import a2p
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
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

    classifier_file = base.split('\\')[-1] + "_" + str(
        datetime.datetime.now().strftime('%Y-%m-%d_%H_%M')) + "_classifier_results.csv"
    with open(classifier_file, 'w') as f:
        names = []
        models = [('LR', LogisticRegression()), ('LDA', LinearDiscriminantAnalysis()),
                  ('KNN', KNeighborsClassifier()),
                  ('CART', DecisionTreeClassifier()), ('NB', GaussianNB()), ('SVM', SVC())]

        all_results = [0 for _ in models]
        # print(all_results)
        csv_keys = ['Author'] + [n for n, _ in models]
        w = csv.writer(f, csv_keys)

        for user_no in range(0, int(dataset.shape[0] / 9)):

            # print('user' + str(user_no))
            dset_vals_pos = dataset[(user_no * 9):(user_no * 9) + 9].values
            # f.write('\n' + dset_vals_pos[0, 27] + ": \n")
            # print(dset_vals_pos[0, 27])

            dset_vals_neg = pandas.concat([dataset[0:(user_no * 9)], dataset[(user_no * 9) + 9:]]).sample(
                n=9).values

            # print(len(dset_vals_neg[0]))

            dset_vals_neg[:, 47] = ['noname' for _ in dset_vals_neg]

            # print("\n----------Positive values:----------\n")
            # print(dset_vals_pos)
            # print("\n----------Negative values:----------\n")
            # print(dset_vals_neg)

            dset_vals = pandas.concat([pandas.DataFrame(dset_vals_pos), pandas.DataFrame(dset_vals_neg)]).values

            X = dset_vals[:, 0:47]
            Y = dset_vals[:, 47]

            # print(X)

            val_size = 1 / 3

            seed = 65537

            X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=val_size,
                                                                                            random_state=seed)

            scoring = 'accuracy'
            results = []

            for name, model in models:
                kfold = model_selection.KFold(n_splits=12, random_state=seed)
                cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
                results.append(cv_results)
                names.append(name)
                # msg = "%s: %f (%f)\n" % (name, cv_results.mean(), cv_results.std())
                # print(msg)
                # f.write(msg)
            rmeans = [r.mean() for r in results]
            # print(rmeans)
            # print(all_results)
            w.writerow([dset_vals_pos[0, 47]] + rmeans)
            all_results = [all_results[i] + rmeans[i] for i in range(0, len(models))]
            # print(all_results)
        # print([(sum(r)/len(all_results)).mean() for r in all_results])
        all_results = [i / int(dataset.shape[0] / 9) for i in all_results]
        # print(all_results)
        w.writerow(['TOTAL'] + [str(i) for i in all_results])


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    base = "C:\\Users\\Bence\\Documents\\Lecke\\Diplomamunka\\CPP_Files\\9Files_largescale_onlyCPP"
    arff_dataset = create_dataset("9Files_largescale_onlyCPP_2018-05-28_23_57.arff")

    # print(len(arff_dataset[0][0]))
    two_step_classifier(arff_dataset)
