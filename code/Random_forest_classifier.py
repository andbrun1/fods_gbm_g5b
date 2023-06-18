from preprocessing import data_rad_c
from preprocessing import split_data, y_to_class

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, confusion_matrix, auc


import warnings
warnings.filterwarnings("ignore")




def get_confusion_matrix(y, y_pred):
    """
    compute the confusion matrix of a classifier yielding
    predictions y_pred for the true class labels y
    :param y: true class labels
    :type y: numpy array

    :param y_pred: predicted class labels
    :type y_pred: numpy array

    :return: comfusion matrix comprising the
             true positives (tp),
             true negatives  (tn),
             false positives (fp),
             and false negatives (fn)
    :rtype: four integers
    """

    # true/false pos/neg.
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(len(y)):
        if y[i] == 1 and y_pred[i] == 1:
            tp += 1
        elif y[i] == 0 and y_pred[i] == 1:
            fp += 1
        elif y[i] == 0 and y_pred[i] == 0:
            tn += 1
        elif y[i] == 1 and y_pred[i] == 0:
            fn += 1

    return tn, fp, fn, tp



def evaluation_metrics(RFC, y_test, X_test_sc):
    """
    compute multiple evaluation metrics for the provided classifier given the true labels
    and input features. Provides a plot of the roc curve on the given axis with the legend
    entry for this plot being specified, too.

    :param RFC: true class labels
    :type RFC: numpy array

    :param y: true class labels
    :type y: numpy array

    :param X: feature matrix
    :type X: numpy array

    :param ax: matplotlib axis to plot on
    :type legend_entry: matplotlib Axes

    :param legend_entry: the legend entry that should be displayed on the plot
    :type legend_entry: string

    :return: comfusion matrix comprising the
             true positives (tp),
             true negatives  (tn),
             false positives (fp),
             and false negatives (fn)
    :rtype: four integers
    """


    # Get the label predictions
    y_test_pred    = RFC.predict(X_test_sc)
    # X is X_test_sc in this function
    print(len(X_test_sc))
    print(y_test_pred)
    print(y_test)

    # Calculate the confusion matrix given the predicted and true labels with your function
    tn, fp, fn, tp = get_confusion_matrix(y_test, y_test_pred)
    print(tn, fp, fn, tp)


    tn_sk, fp_sk, fn_sk, tp_sk = confusion_matrix(y_test, y_test_pred).ravel()
    if np.sum([np.abs(tp-tp_sk), np.abs(tn-tn_sk), np.abs(fp-fp_sk), np.abs(fn-fn_sk)]) > 0:
        print('OWN confusion matrix failed!!! Reverting to sklearn.')
        tn = tn_sk
        tp = tp_sk
        fn = fn_sk
        fp = fp_sk
    else:
        print(':) Successfully implemented the confusion matrix!')


    # Calculate the evaluation metrics
    precision   = tp / (tp + fp)
    specificity = tn / (tn + fp)
    accuracy    = (tp + tn) / (tp + tn + fp + fn)
    recall      = tp / (tp + fn)
    f1          = 2 * (precision * recall) / (precision + recall)
    
    print("Precision: ", precision, "specificity: ", specificity, "accuracy: ", accuracy, "recall: " , recall, "f1: ", f1)
    

    # Get the roc curve using a sklearn function
    y_test_predict_proba  = RFC.predict_proba(X_test_sc)[:, 1]
    fp_rates, tp_rates, _ = roc_curve(y_test, y_test_predict_proba)

    # Calculate the area under the roc curve using a sklearn function
    roc_auc = auc(fp_rates, tp_rates)

    return accuracy, precision, recall, specificity, f1, roc_auc, fp_rates, tp_rates



X = data_rad_c[data_rad_c.columns.drop("Survival_from_surgery_days")]
y = data_rad_c["Survival_from_surgery_days"]
y_to_class(y, 456)
print(y.value_counts()) # to see the distribution

#Test split
X_train, X_val, X_test, y_train, y_val, y_test, X_split, y_split = split_data(X, y)


# Perform a 5-fold stratified crossvalidation - prepare the splitting
n_splits = 5
skf = StratifiedKFold(n_splits, shuffle=True, random_state=42)

number_trees = list(range(60, 130))
best_accuracy = 0
best_n = None
acc = []


# Prepare the performance overview data frame
df_performance = pd.DataFrame(columns = ['fold','RFC','accuracy','precision','recall', 'specificity','F1','roc_auc'])
df_LR_normcoef = pd.DataFrame(index = X.columns, columns = np.arange(n_splits))


# Standardize the data: --> 1 time per fold standardization
fold = 0
fig, axs = plt.subplots(1, 2, figsize=(9, 4))

for n in number_trees:
    accuracies = []
    # Loop over all splits
    for train_index, val_index in skf.split(X_split, y_split):

        # Get the relevant subsets for training and testing
        X_train_f, X_val_f = X_split.iloc[train_index], X_split.iloc[val_index]
        y_train_f, y_val_f = y_split.iloc[train_index], y_split.iloc[val_index]


        # Standardize the numerical features using training set statistics
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train_f)
        X_val_sc  = scaler.transform(X_val_f)


        # Creat prediction models and fit them to the training data
        # Random forest
        RFC = RandomForestClassifier(n_estimators=n, max_depth=5, random_state=42) 
        RFC.fit(X_train_f, y_train_f)
        y_pred = RFC.predict(X_val_f)

        accuracy = accuracy_score(y_val_f, y_pred)
        accuracies.append(accuracy)


        fold += 1

    avg_accuracy = np.mean(accuracies)
    acc.append(avg_accuracy)

    if avg_accuracy > best_accuracy:
        best_accuracy = avg_accuracy
        best_n = n

print("Best n:", best_n)


# elbow plot for accuracy
plt.clf()
plt.cla()
plt.plot(number_trees, acc, marker = 'o', color = 'grey')
plt.xlabel('numbers of decision trees')
plt.ylabel('Accuracy score')
plt.title('Elbow plot for the best number of trees')
plt.grid(color='gray', alpha=0.3, linestyle=':', linewidth=1)
plt.savefig('../output/best_n_of_trees_elbow_plot.png')
plt.show()




scaler = StandardScaler()
X_split_sc = scaler.fit_transform(X_split)
X_test_sc  = scaler.transform(X_test)

# Random forest
RFC = RandomForestClassifier(n_estimators=83, max_depth=5, random_state=42) 
RFC.fit(X_split_sc, y_split)

# Summarize the performance metrics over all folds
# split the data frame so you have the performance for LR and RF
df_performance_LR = df_performance.loc[df_performance["RFC"]=="LR"]
df_performance_RF = df_performance.loc[df_performance["RFC"]=="RF"]
print(df_performance_LR)
print(df_performance_RF)

# create an empty dataframe (called datatab) to be filled with mean and std of the performance metrics
accuracy, precision, recall, specificity, f1, roc_auc, fp_rates, tp_rates = evaluation_metrics(RFC, y_test, X_test_sc)

metric_names = ["accuracy", "precision", "recall", "specificity", "F1", "roc_auc"]
table_values = [[round(metric, 3)] for metric in (accuracy, precision, recall, specificity, f1, roc_auc)]

fig, ax = plt.subplots()

# hide plot
ax.axis("tight")
ax.axis("off")

tab1 = ax.table(cellText=table_values, colLabels=["Random Forest"], rowLabels=metric_names, loc='center')
tab1.set_fontsize(14)
tab1.scale(1, 1.5)
plt.tight_layout()
plt.savefig("../output/RandomForest_evaluation.pdf", bbox_inches='tight')
plt.show()

evaluation_rfc = {
    "accuracy":accuracy,
    "precision":precision,
    "recall":recall, 
    "specificity":specificity,
    "f1":f1, 
    "roc_auc":roc_auc, 
    "fp_rates":fp_rates.tolist(),
    "tp_rates":tp_rates.tolist()
}
with open("../output/evaluation_rfc.json", "w+") as f:
    f.write(json.dumps(evaluation_rfc))


print(table_values)


# plot
plt.plot(fp_rates, tp_rates, label = 'Random Forest classifier')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.plot([0, 1], [0, 1],'r--', label = 'random\nclassifier')
plt.title('ROC curve')
plt.legend()
plt.grid(color='gray', alpha=0.3, linestyle=':', linewidth=1)
plt.tight_layout()
plt.savefig('../output/ROC_RFC.png')
plt.show()

