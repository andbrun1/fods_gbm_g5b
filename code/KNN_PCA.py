import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, confusion_matrix, auc
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")

from preprocessing import data_clin_c, data_rad_c,split_data, y_to_class




def get_confusion_matrix(y, y_pred):
    # true/false pos/neg.
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    # define positive and negative classes.
    pos = 1
    neg = 0
    for i in range(0, len(y)):
        if y[i] == pos:
            # positive class.
            if y_pred[i] == pos:
                tp += 1
            else:
                fn += 1
        else:
            # negative class.
            if y_pred[i] == neg:
                tn += 1
            else:
                fp += 1
    return tn, fp, fn, tp

def evaluation_metrics(knn, y_test, X_test_sc):
    # prediction
    y_test_pred = knn.predict(X_test_sc)
    #print(len(X_test_sc))
    #print(y_test_pred)
    #print(y_test)

    # confusion matrix
    tn, fp, fn, tp = get_confusion_matrix(y_test, y_test_pred)
    #print(tn, fp, fn, tp)
    tn_sk, fp_sk, fn_sk, tp_sk = confusion_matrix(y_test, y_test_pred).ravel()
    if np.sum([np.abs(tp-tp_sk) + np.abs(tn-tn_sk) + np.abs(fp-fp_sk) + np.abs(fn-fn_sk)]) >0:
        print('OWN confusion matrix failed!!! Reverting to sklearn.')
        tn = tn_sk
        tp = tp_sk
        fn = fn_sk
        fp = fp_sk
    else:
        print(':) Successfully implemented the confusion matrix!')

    # calculate evaluation metrics
    precision   = tp / (tp + fp)
    specificity = tn / (tn + fp)
    accuracy    = (tp + tn) / (tp + fp + tn + fn)
    recall      = tp / (tp + fn)
    f1          = tp/(tp + 0.5*(fp+fn))

    print(precision)
    print(specificity)
    print(accuracy)
    print(recall)
    print(f1)

    # get roc curve
    y_test_predict_proba  = knn.predict_proba(X_test_sc)[:, 1]
    fp_rates, tp_rates, _ = roc_curve(y_test, y_test_predict_proba)

    # AUC
    roc_auc = auc(fp_rates, tp_rates)
    print(roc_auc)

    return [accuracy,precision,recall,specificity,f1, roc_auc,fp_rates, tp_rates]


X=data_rad_c[data_rad_c.columns.drop("Survival_from_surgery_days")]
y=data_rad_c["Survival_from_surgery_days"]
y_to_class(y, 456)
#print(y.value_counts()) #Um Aufteilung zu sehen

# PCA: unsupervised learning technique to reduce dimensionality of data (results in combination of original features)
# n_components = number of components you want to retain (additional: hyperparameter tuning would be an option)
pca = PCA(n_components = 10)
X_pca = pca.fit_transform(X)
#print(X_pca)
X_pca = pd.DataFrame(X_pca)
#print(X_pca)

#Test split
X_train, X_val, X_test, y_train, y_val, y_test, X_split, y_split = split_data(X_pca, y)
#print(X_train)


# k hyperparamter tuning
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state =2023)

k_values = [i for i in range(1,40)]
best_accuracy = 0
best_k = None
acc = []
plt.figure()


for k in k_values:
    accuracies = []
    for train_index, val_index in skf.split(X_split, y_split):
        #print(train_index)
        #print(val_index)
        X_train_f, X_val_f = X_split.iloc[train_index], X_split.iloc[val_index]
        y_train_f, y_val_f = y_split.iloc[train_index], y_split.iloc[val_index]

        # Standardize the numerical features using training set statistics
        sc = StandardScaler()
        X_train_sc = sc.fit_transform(X_train_f)
        X_val_sc  = sc.transform(X_val_f)


        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_f, y_train_f)
        y_pred = knn.predict(X_val_f)

        accuracy = accuracy_score(y_val_f, y_pred)
        accuracies.append(accuracy)
        #print(accuracy)
        #print(accuracies)

    avg_accuracy = np.mean(accuracies)
    acc.append(avg_accuracy)
    #print(acc)
    if avg_accuracy > best_accuracy:
        best_accuracy = avg_accuracy
        best_k = k

# best_k k with highest accuracy, use plot to see where stabilization is
print("Best k:", best_k)

# elbow plot for accuracy
plt.plot(k_values, acc, marker = 'o', color = 'grey')
plt.xlabel('K values')
plt.ylabel('Accuracy score')
plt.title('Elbow plot best k')
plt.grid(color='gray', alpha=0.3, linestyle=':', linewidth=1)
plt.minorticks_on()
plt.savefig('../output/best_k_elbow_plot.png')
plt.show()




# KNN
# KNN classifier
#knn = KNeighborsClassifier(n_neighbors = best_k)
knn = KNeighborsClassifier(n_neighbors = 9) #possible: additional hyperparameter tuning -> distance metric

# Standardize
scaler = StandardScaler()
X_split_sc = scaler.fit_transform(X_split)
X_test_sc = scaler.transform(X_test)

# fit
knn.fit(X_split_sc, y_split)




# evaluation
accuracy,precision,recall,specificity,f1, roc_auc,fp_rates, tp_rates = evaluation_metrics(knn, y_test, X_test_sc)
evaluation_knn = ['KNN',accuracy,precision,recall,specificity,f1, roc_auc,fp_rates, tp_rates]
#print(evaluation_knn)

# evaluation and comparison models
index = ['accuracy','precision', 'recall', 'specificity', 'F1','roc_auc']
data_tab = pd.DataFrame(index = index, columns = ['KNN']) #add other models??
data_tab.iloc[0,0] = round(accuracy,4)
data_tab.iloc[1,0] = round(precision,4)
data_tab.iloc[2,0] = round(recall, 4)
data_tab.iloc[3,0] = round(specificity,4)
data_tab.iloc[4,0] = round(f1,4)
data_tab.iloc[5,0] = round(roc_auc,4)

fig, ax = plt.subplots()
ax.axis('off')
ax.axis('tight')
tab1 = ax.table(cellText=data_tab.values,colLabels=data_tab.columns, rowLabels=data_tab.index, loc='center')
tab1.set_fontsize(14)
plt.tight_layout()
plt.savefig("../output/KNN_evaluation.pdf", bbox_inches='tight',pad_inches=0.2)
plt.show()

# plot
plt.plot(fp_rates, tp_rates, label = 'KNN classifier')
plt.xlabel('FPR')
plt.ylabel('TPR')
#add_identity(color="r", ls="--",label = 'random\nclassifier')
plt.plot([0, 1], [0, 1],'r--', label = 'random\nclassifier')
plt.title('ROC curve KNN')
plt.legend()
plt.grid(color='gray', alpha=0.3, linestyle=':', linewidth=1)
plt.tight_layout()
plt.minorticks_on()
plt.savefig('../output/ROC_knn.png')
plt.show()








