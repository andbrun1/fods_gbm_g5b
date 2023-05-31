import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,recall_score, precision_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from preprocessing import data_clin_c, data_rad_c
from preprocessing import split_data, y_to_class

def add_identity(axes, *line_args, **line_kwargs): #copied from HW5
    identity, = axes.plot([], [], *line_args, **line_kwargs)
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes

def evaluation_metrics(clf, y, X): #copied from HW5 and adapted
    """
    compute multiple evaluation metrics for the provided classifier given the true labels
    and input features. Provides a plot of the roc curve on the given axis with the legend
    entry for this plot being specified, too.
    """

    # Get the label predictions
    y_test_pred = log_reg.predict(X) # X corresponds to X_test_sc

    tn, fp, fn, tp = confusion_matrix(y, y_test_pred).ravel()

    # Calculate the evaluation metrics
    precision   = tp / (tp + fp)
    specificity = tn / (tn+fp)
    accuracy    = (tp +tn)/(tp +fp+tn+fn)
    recall      = tp / (tp + fn)  #same as sensitivity
    f1          = 2*((precision*recall)/(precision+recall)) #takes into account precision and recall

    # Get the roc curve using a sklearn function
    y_test_predict_proba  = clf.predict_proba(X)[:,1]
    fp_rates, tp_rates, _ = roc_curve(y, y_test_predict_proba)

    # Calculate the area under the roc curve using a sklearn function
    roc_auc = auc(fp_rates, tp_rates)

    # Plot the roc curve
    plt.plot(fp_rates, tp_rates)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.grid(color= "grey", alpha = 0.3, linestyle = ":", linewidth =1)
    plt.minorticks_on()
    plt.title('ROC curve for logistic regression')
    add_identity(axes=ax,linestyle='dashed', color="r", label='random classifier')
    plt.savefig('../output/roc_curve_log_reg.png')
    # plt.show()


    return [accuracy,precision,recall,specificity,f1, roc_auc, fp_rates, tp_rates]

survival_threshold = 456
#456 is median survival rate for this cancer (see paper)
X = data_rad_c[data_rad_c.columns.drop("Survival_from_surgery_days")]
y = data_rad_c["Survival_from_surgery_days"]
y_to_class(y, survival_threshold) #ignore warning
# print(y.value_counts()) #to see distribution

#Test split
X_train_final, X_val_final, X_test_final, y_train_final, y_val_final, y_test_final, X_split, y_split = split_data(X, y)
#I only need split (and test at the end); split is later divided into train and val several times
#the final is added so one can differentiate from the data in the fold
#print(y_test_final.shape)

# Initialize a 5 fold cross-validator and model
n_splits = 5
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2023)
# Initialize the model
log_reg = LogisticRegression(penalty="none", random_state=2023)

# Metrics to store the results
metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
all_coefficients = pd.DataFrame(index=range(1, 6), columns=X_train_final.columns)

# Evaluate the model using 5-fold cross-validation
fold = 1
for train_index, val_index in cv.split(X_split, y_split): # divide split into train & evaluate
    # above n_splits = 5 --> 5 loops
    X_train = X_split.iloc[train_index]
    X_val = X_split.iloc[val_index]
    y_train = y_split.iloc[train_index]
    y_val = y_split.iloc[val_index]   # use y or y_split? does it matter at all?

    # Standardize the data  within the loop!
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_val_std = scaler.transform(X_val)

    # Train and predict
    # print("y_train in loop", y_train.shape)
    log_reg.fit(X_train_std, y_train)
    y_pred = log_reg.predict(X_val_std)

    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    # Store the metrics
    metrics['accuracy'].append(accuracy)
    metrics['precision'].append(precision)
    metrics['recall'].append(recall)
    metrics['f1'].append(f1)

    all_coefficients.loc[fold] = np.abs(log_reg.coef_) #save the coefficients of each model, take absolute numbers
    fold += 1
"""
    if fold == 5:
        TN, FP, FN, TP = confusion_matrix(y_val, y_pred).ravel()
        print('Confusion matrix:\n\t\t  |y_true = 0\t|y_true = 1')
        print('----------|-------------|------------')
        print('y_pred = 0|  ' + str(TP) + '\t\t\t|' + str(FP))
        print('y_pred = 1|  ' + str(FN) + '\t\t\t|' + str(TN))
"""



#print(metrics)
#print(all_coefficients)

#calculate the importance, then mean of that
normcoef = pd.DataFrame(index = np.arange(n_splits), columns = X_train.columns)
#print(normcoef)
#print(len(X_train), len(X_split))
for l in range(0, X_split.shape[1]): #samples are different in train and split, number of feat is the same
    for i in range(0, n_splits):
        normcoef.iloc[i,l] = np.asarray(all_coefficients.iloc[i,l]/np.sum(all_coefficients.iloc[i])).flatten() #understand i&l!
#print(normcoef)

#get the mean and std of the normalised coefficients and sort them
mean_values = normcoef.mean()
std_values = normcoef.std()
datatab = pd.DataFrame({"mean": mean_values, "std": std_values})
#print(datatab)
#print(datatab.loc["T1GD_ED_Intensity_Energy"])
datatab = datatab.sort_values("mean", ascending=False)
#print(datatab)
datatab_100 = datatab.iloc[:100]
#print(datatab_100)

#print the importance of the 100 best features
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(np.arange(100), datatab_100["mean"], yerr=datatab_100["std"])
#ax.set_xticks(np.arange(100))
#ax.set_xticklabels(datatab_100.index.tolist(), rotation=90)
ax.set_title("Normalized importance of best 100 features", fontsize=20)
ax.set_xlabel("Feature", fontsize=16)
ax.set_ylabel("Coefficient (absolute)", fontsize=16)
#plt.tight_layout()
plt.savefig('../output/importance_100.png')
#plt.show()

#there is no clear ellbow. possible cut offs are after 14, 28 or 60...
#print the importance of the 60 best features
datatab_60 = datatab.iloc[:60]
fig, ax = plt.subplots(figsize=(15, 6))
ax.bar(np.arange(60), datatab_60["mean"], yerr=datatab_60["std"])
#ax.set_xticks(np.arange(60))
#ax.set_xticklabels(datatab_60.index.tolist(), rotation=50, fontsize=12)
ax.set_title("Normalized importance of best 60 features", fontsize=20)
ax.set_xlabel("Feature", fontsize=16)
ax.set_ylabel("Coefficient (absolute)", fontsize=16)
plt.tight_layout()
#plt.subplots_adjust(bottom=0.8)
plt.savefig('../output/importance_60.png')
#plt.show()

# create table with the x labels
x_labels = np.arange(1,61)
x_labels = pd.Series(x_labels)
features_60 = datatab_60.index.tolist()
features_60 = pd.Series(features_60)
table_data = pd.concat([x_labels, features_60], axis=1)
table_data.columns = ["Importance ranking", "Feature name"]
fig, ax = plt.subplots(figsize=(8, 14))
ax.axis('off')
ax.axis('tight')
table = ax.table(cellText=table_data.values, colLabels=table_data.columns, loc='center')
row_height = 1
table.scale(1, row_height)
column_widths = [0.15, 0.7]
for i, width in enumerate(column_widths):
    for l in range(0, 61):
        table[l, i].set_width(width)

plt.title("The 60 most relevant features")
plt.tight_layout()
plt.savefig('../output/best_60_features.pdf')
#plt.show()


# select the best feature from datatab
cut_off = 60 # decided to just do that; could also analyze the 3 and compare
best_feature_indexes = datatab.index[:cut_off]

# extract the best features from the bigger df
X_best_train = X_split.loc[:, best_feature_indexes]
X_best_test = X_test_final.loc[:, best_feature_indexes]


# Standardize the data
X_train_std = scaler.fit_transform(X_best_train)
X_test_std = scaler.transform(X_best_test)

# fit the new model
log_reg.fit(X_train_std, y_split)


# roc curve from homework 5
fig,ax = plt.subplots(1,1,figsize=(6, 4))
df_performance_log_reg = pd.DataFrame(columns = ['accuracy','precision','recall',
                                         'specificity','F1','roc_auc',"fp_rates", "tp_rates"])

eval_metrics = evaluation_metrics(log_reg, y_test_final, X_test_std)
df_performance_log_reg.loc[len(df_performance_log_reg),:] = eval_metrics
# print(metrics)
# print(df_performance_log_reg)
# comparison by eye: model with 60 features performs better than with all features :)

evaluation_lr = ["LR"]
evaluation_lr.extend(eval_metrics)
# print(evaluation_lr)
# evaluation_lr --> export for comparison with other models

# calculate importance (absolute values), order them, plot them, elbow plot style --> take most relevant
# extract most important features, make new model only with those (with train & test), evaluate performance

