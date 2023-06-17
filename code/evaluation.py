import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
# import evaluation list from model files, format: ['model name',accuracy,precision,recall,specificity,f1, roc_auc,fp_rates, tp_rates]

# hard coding solution
"""
evaluation_knn = ['KNN', 0.5735294117647058, 0.3888888888888889, 0.28, 0.7441860465116279, 0.32558139534883723, 0.5576744186046512, np.array([0.        , 0.        , 0.02325581, 0.06976744, 0.25581395,
       0.44186047, 0.58139535, 0.8372093 , 0.95348837, 1.        ]), np.array([0.  , 0.04, 0.08, 0.2 , 0.28, 0.44, 0.72, 0.88, 0.96, 1.  ])]
evaluation_lr = ['LR', 0.6470588235294118, 0.5185185185185185, 0.56, 0.6976744186046512, 0.5384615384615384, 0.666046511627907, np.array([0.        , 0.        , 0.        , 0.09302326, 0.09302326,
       0.11627907, 0.11627907, 0.18604651, 0.18604651, 0.27906977,
       0.27906977, 0.30232558, 0.30232558, 0.39534884, 0.39534884,
       0.62790698, 0.62790698, 0.6744186 , 0.6744186 , 0.72093023,
       0.72093023, 0.90697674, 0.90697674, 0.93023256, 0.93023256,
       1.        ]), np.array([0.  , 0.04, 0.12, 0.12, 0.32, 0.32, 0.48, 0.48, 0.52, 0.52, 0.56,
       0.56, 0.6 , 0.6 , 0.72, 0.72, 0.76, 0.76, 0.84, 0.84, 0.88, 0.88,
       0.92, 0.92, 1.  , 1.  ])]
evaluation_RFC = ['RFC', 0.6470588235294118, 0.6, 0.12, 0.9534883720930233, 0.19999999999999998, 0.5581395348837208, np.array([0.        , 0.        , 0.        , 0.02325581, 0.02325581,
       0.09302326, 0.09302326, 0.11627907, 0.11627907, 0.18604651,
       0.18604651, 0.25581395, 0.25581395, 0.27906977, 0.27906977,
       0.44186047, 0.44186047, 0.46511628, 0.46511628, 0.48837209,
       0.48837209, 0.60465116, 0.60465116, 0.6744186 , 0.6744186 ,
       0.72093023, 0.72093023, 0.76744186, 0.76744186, 0.8372093 ,
       0.8372093 , 1.        , 1.        ]), np.array([0.  , 0.04, 0.08, 0.08, 0.12, 0.12, 0.2 , 0.2 , 0.28, 0.28, 0.32,
       0.32, 0.36, 0.36, 0.44, 0.44, 0.52, 0.52, 0.56, 0.56, 0.64, 0.64,
       0.68, 0.68, 0.76, 0.76, 0.8 , 0.8 , 0.84, 0.84, 0.88, 0.88, 1.  ])]
evaluation_svm = ['SVM', 0.6323529411764706, 0.5, 0.2, 0.8837209302325582, 0.28571428571428575, 0.5823255813953488, np.array([0.        , 0.        , 0.02325581, 0.02325581, 0.09302326,
       0.09302326, 0.11627907, 0.11627907, 0.1627907 , 0.1627907 ,
       0.18604651, 0.18604651, 0.20930233, 0.20930233, 0.27906977,
       0.27906977, 0.39534884, 0.39534884, 0.41860465, 0.41860465,
       0.44186047, 0.44186047, 0.55813953, 0.55813953, 0.58139535,
       0.58139535, 0.60465116, 0.60465116, 0.81395349, 0.81395349,
       0.88372093, 0.88372093, 0.93023256, 0.93023256, 0.97674419,
       0.97674419, 1.        , 1.        ]), np.array([0.  , 0.04, 0.04, 0.16, 0.16, 0.2 , 0.2 , 0.28, 0.28, 0.32, 0.32,
       0.36, 0.36, 0.4 , 0.4 , 0.48, 0.48, 0.52, 0.52, 0.56, 0.56, 0.64,
       0.64, 0.68, 0.68, 0.72, 0.72, 0.76, 0.76, 0.8 , 0.8 , 0.88, 0.88,
       0.92, 0.92, 0.96, 0.96, 1.  ])]


"""
evaluation_knn = None
evaluation_lr = None
evaluation_rfc = None
evaluation_svm = None

with open("../output/evaluation_knn.json", "r") as f:
    evaluation_knn = json.load(f)
with open("../output/evaluation_lr.json", "r") as f:
    evaluation_lr = json.load(f)
with open("../output/evaluation_rfc.json", "r") as f:
    evaluation_rfc = json.load(f)
with open("../output/evaluation_svm.json", "r") as f:
    evaluation_svm = json.load(f)


"""
evaluation_svm_array = [
    evaluation_svm['accuracy'],
    evaluation_svm['precision'],
    evaluation_svm['recall'],
    evaluation_svm['specificity'],
    evaluation_svm['f1'],
    evaluation_svm['roc_auc'],
    evaluation_svm['fp_rates'],
    evaluation_svm['tp_rates']]
    """
"""
# Plot of Roc curve
fig, ax = plt.subplots(figsize=(9, 6))
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
plt.plot([0, 1], [0, 1], color="r", ls="--", label='random\nclassifier')
ax.set_title("Roc curve")
ax.grid(color='gray', alpha=0.3, linestyle=':', linewidth=1)
ax.minorticks_on()
plt.legend
ax.plot(evaluation_rfc['fp_rates'], evaluation_rfc['tp_rates'], color="grey")
plt.show()
exit()

# read csv with saved lists (important files for models have to run first!!
evaluation_knn = pd.read_csv("../data/evaluation_knn.csv")
evaluation_lr = pd.read_csv("../data/evaluation_lr.csv")
evaluation_RFC = pd.read_csv("../data/evaluation_rfc.csv")
evaluation_svm = pd.read_csv("../data/evaluation_svm.csv")

# transorm data frame back into a list
evaluation_knn = np.array(evaluation_knn.values.tolist())
evaluation_lr = np.array(evaluation_lr.values.tolist())
evaluation_RFC = np.array(evaluation_RFC.values.tolist())
evaluation_svm = np.array(evaluation_svm.values.tolist())

# flatten lists
evaluation_knn = evaluation_knn.flatten()
evaluation_lr = evaluation_lr.flatten()
evaluation_RFC = evaluation_RFC.flatten()
evaluation_svm = evaluation_svm.flatten()



print(evaluation_knn)
print(evaluation_lr)
print(evaluation_RFC)
print(evaluation_svm)
"""
# roc curve (all models in one plot for comparison)
plt.plot(evaluation_knn["fp_rates"],evaluation_knn["tp_rates"],label = "KNN")
plt.plot(evaluation_lr["fp_rates"],evaluation_lr["tp_rates"],label = "LR")
plt.plot(evaluation_rfc["fp_rates"],evaluation_rfc["tp_rates"],label = "RFC")
plt.plot(evaluation_svm["fp_rates"],evaluation_svm["tp_rates"],label = "SVM")
"""
plt.plot(evaluation_knn["fp_rates"],evaluation_knn[tp_rates],label = evaluation_knn[0])
plt.plot(evaluation_lr[7],evaluation_lr[8],label = evaluation_lr[0])
plt.plot(evaluation_rfc[7],evaluation_rfc[8],label = evaluation_rfc[0])
plt.plot(evaluation_svm[7],evaluation_svm[8],label = evaluation_svm[0])
"""
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.plot([0, 1], [0, 1],'r--', label = 'random\nclassifier')
plt.title('ROC curve')
plt.legend()
plt.grid(color='gray', alpha=0.3, linestyle=':', linewidth=1)
plt.tight_layout()
plt.minorticks_on()
plt.savefig('../output/ROC_models.png')
plt.show()

# table with evaluation metrics for every model
col_names = ["KNN","LR","RFC","SVM"]
keys = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc"]
data_tab = pd.DataFrame(index=keys, columns=col_names)
for key in keys:
    data_tab.loc[key,"KNN"] = round(evaluation_knn[key],2)
    data_tab.loc[key,"LR"] = round(evaluation_lr[key],2)
    data_tab.loc[key,"RFC"] = round(evaluation_rfc[key],2)
    data_tab.loc[key,"SVM"] = round(evaluation_svm[key],2)
print(data_tab)

fig, ax = plt.subplots()
ax.axis("tight")
ax.axis("off")
tab1 = ax.table(cellText=data_tab.values, colLabels=data_tab.columns, rowLabels=data_tab.index, loc='center')
tab1.set_fontsize(14)
tab1.scale(1, 1.5)
plt.tight_layout()
plt.savefig("../output/model_evaluation.pdf", bbox_inches='tight')
plt.show()

