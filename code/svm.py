import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from preprocessing import data_rad_c
from preprocessing import split_data, y_to_class
import json
#General Variables
#Grid search parameters for SVM
grid_best_c = 1.438449888287663 #Set to reduce time cost according to grid search
grid_best_gamma = 0.004281332398719396 #Set to reduce time cost according to grid search

#Parameters to induce new computation
#grid_best_c = None
#grid_best_gamma = None 


X=data_rad_c[data_rad_c.columns.drop("Survival_from_surgery_days")]
y=data_rad_c["Survival_from_surgery_days"]
y_to_class(y, 456)
#print(y.value_counts()) #Um Aufteilung zu sehen

#Test split, Validation and training set are retrieved in merged form
_, _, X_test, _, _, y_test, X_train, y_train = split_data(X, y)
#Scale the data
sc=StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)


#Grid search
#Only done if no best parameters are available
#Done once and set in code at the beginning to reduce time cost
if not grid_best_c or not grid_best_gamma:
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    #search for Hyperparameters
    c_range = np.logspace(-3, 3, 20, base=10)
    gamma_range = np.logspace(0, 3, 20, base=10)
    grid_params = dict(gamma=gamma_range, C=c_range)
    cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state=2023)
    grid = GridSearchCV(SVC(), param_grid=grid_params, cv=cv, n_jobs=-1)
    grid.fit(X_train_sc, y_train)
    grid_best_c = grid.best_params_["C"]
    grid_best_gamma = grid.best_params_["gamma"]
    
#print("The best parameters are %s"% (f"C: {grid_best_c}, gamma: {grid_best_gamma}"))

#Implement support vector machine with found parameters
clf_SVM = svm.SVC(probability=True, kernel = "rbf", C=grid_best_c, gamma=grid_best_gamma)
#Hyperparameters are C and gamma
#high c -> close fitting to achieve best fit of training data, low c -> smoother, less overfitting, squared l2 penalty
#gamma defines how much influence one example has, low reaches far, high reaches close

#Fit model
clf_SVM.fit(X_train_sc, y_train)

    
#Evaluation
def eval_Performance(y_eval, X_eval, clf, clf_name = 'My Classifier'):

    y_pred = clf.predict(X_eval)
    y_pred_proba = clf.predict_proba(X_eval)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_eval, y_pred).ravel()

    # Evaluation
    specificity = tn/(tn+fp)
    accuracy  = accuracy_score(y_eval, y_pred)
    precision = precision_score(y_eval, y_pred)
    recall    = recall_score(y_eval, y_pred)
    f1        = f1_score(y_eval, y_pred)
    fp_rates, tp_rates, _ = roc_curve(y_eval, y_pred_proba)
    #if y_eval == y_test:
        #evaluation_svm = ['SVM',accuracy,precision,recall,specificity,f1, roc_auc,fp_rates, tp_rates]
    # Calculate the area under the roc curve using a sklearn function
    roc_auc = auc(fp_rates, tp_rates)

    return tp,fp,tn,fn,accuracy, precision, recall, specificity, f1, roc_auc, fp_rates, tp_rates

df_performance = pd.DataFrame(columns = ['tp','fp','tn','fn','accuracy', 'precision', 'recall', 'specificity', 'f1', 'roc_auc', 'fp_rates', 'tp_rates'] )
df_performance.loc['SVM (train)',:] = eval_Performance(y_train, X_train_sc, clf_SVM, clf_name = 'SVM')
df_performance.loc['SVM (test)',:] = eval_Performance(y_test, X_test_sc, clf_SVM, clf_name = 'SVM (test')
print(df_performance)
#evaluation_svm=list(eval_Performance(y_test, X_test_sc, clf_SVM, clf_name = 'SVM (test'))
_,_,_,_,accuracy, precision, recall, specificity, f1, roc_auc, fp_rates, tp_rates = eval_Performance(y_test, X_test_sc, clf_SVM, clf_name = 'SVM (test')
evaluation_svm = {
    "accuracy":accuracy,
    "precision":precision,
    "recall":recall, 
    "specificity":specificity,
    "f1":f1, 
    "roc_auc":roc_auc, 
    "fp_rates":fp_rates.tolist(),
    "tp_rates":tp_rates.tolist()
}
with open("../output/evaluation_svm.json", "w+") as f:
    f.write(json.dumps(evaluation_svm))
#evaluation_svm=evaluation_svm[4:12]
#evaluation_svm.insert(0,'SVM')
#print(evaluation_svm)

#Visualisation

# Plot of Roc curve
fig, ax = plt.subplots(figsize=(9, 6))
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
plt.plot([0, 1], [0, 1], color="r", ls="--", label='random\nclassifier')
ax.set_title("Roc curve")
ax.grid(color='gray', alpha=0.3, linestyle=':', linewidth=1)
ax.minorticks_on()
plt.legend
ax.plot(fp_rates, tp_rates, color="grey")
plt.show()

