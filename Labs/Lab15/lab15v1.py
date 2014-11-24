import pandas as pd
import numpy as np
import datetime as dt
import pdb
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.cross_validation import train_test_split
#from matplotlib.pyplot import plt


def cross_validate(X, y, classifier, k_fold):
    "Scores classifier using kfold cross_validation"
    # derive a set of (random) training and testing indices
    k_fold_indices = KFold(len(X), n_folds=k_fold,
                           shuffle=True, random_state=0)

    k_score_total = 0
    test_target = []
    target_predicted_proba = []
    i = 1
    # train and score classifier for each slice
    for train_slice, test_slice in k_fold_indices :
        model = classifier(X[train_slice],y[train_slice])
        k_score = model.score(X[test_slice], y[test_slice])
        k_score_total += k_score
        test_target = y[test_slice]
        if i > 1:
            target_predicted_proba = model.predict_proba(diff_feats)
        else:
            diff_feats = X[test_slice]
            i = 2

    # for i, train_slice, test_slice in enumerate(k_fold_indices) :
    #     if i == 0: 
    #         model = classifier(X[train_slice], y[train_slice])
    #         test_target = y[test_slice]
    #         target_predicted_proba = model_lr.predict_proba(y[test_slice])
    #     else:
    #         continue

    # return the average accuracy
    return k_score_total/k_fold, test_target, target_predicted_proba



def plot_roc_curve(target_test, target_predicted_proba):
    fpr, tpr, thresholds = roc_curve(target_test, target_predicted_proba[:, 1
                                                                         ])
    roc_auc = auc(fpr, tpr)
    return roc_auc

    # # Plot ROC curve
    # plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.0])
    # plt.xlabel('False Positive Rate or (1 - Specifity)')
    # plt.ylabel('True Positive Rate or (Sensitivity)')
    # plt.title('Receiver Operating Characteristic')
    # plt.legend(loc="lower right")

# train_feat, test_feat, train_target, test_target = train_test_split(feat_val, target_val, train_size=0.5)

# model_lr = LogisticRegression(C=1).fit(train_feat, train_target)
# target_predicted_proba = model_lr.predict_proba(test_feat)
# plot_roc_curve(test_target, target_predicted_proba)


def cleanData():
    "Loads and cleans credit analysis data"
    data = pd.read_csv('credit_analysis.csv', header=None)
    data.replace({17 : { '-' : 0, '+' : 1}},inplace=True)
    data[16] = data[16].apply(lambda x: (dt.datetime.now() - dt.datetime.strptime(x, '%Y-%m-%d')).days)
    # categorical_features = [1]
    categorical = [1,4,5,6,7,9,10,12,13]
    for n in categorical:
        temp_df = pd.get_dummies(data[n])
        data.drop(n,axis=1,inplace=True)
        data = pd.merge(data, temp_df,left_index=True,right_index=True)  
    numeric_features = [2,14]
    for n in numeric_features:
        data[n] = data[n].convert_objects(convert_numeric=True,convert_dates=True)
    data.dropna(inplace=True)
    collist = data.columns.tolist()
    target = data[17].values
    collist.remove(17)
    collist.remove(0) 
    features = data[collist].values

    return features, target 


def scoreModels(features, target, folds=10):
    "Calcs cross-validation scores for multiple algorithms"
    #pdb.set_trace()
    models = []
    models.append(RandomForestClassifier(random_state=0).fit)
    models.append(LogisticRegression(C=1.0).fit)
    models.append(KNeighborsClassifier(3).fit)
    #models.append(SVC(C=1.0).fit)
    models.append(GaussianNB().fit)

    print "Hello"
    for alg in models:
        print alg
        x, y, z = cross_validate(features, target, alg, folds)
        print "HELLO"
        print x
        print y
        print z
#        print plot_roc_curve(features, target)


def main():
    features, target = cleanData()
    scoreModels(features, target)


#instead of importing functions, run the main() function above
if __name__ == '__main__':
    main()



