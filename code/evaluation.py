import pandas as pd
import matplotlib.pyplot as plt
import json

#Initiate names
evaluation_knn = None
evaluation_lr = None
evaluation_rfc = None
evaluation_svm = None

#Load data
with open("../output/evaluation_knn.json", "r") as f:
    evaluation_knn = json.load(f)
with open("../output/evaluation_lr.json", "r") as f:
    evaluation_lr = json.load(f)
with open("../output/evaluation_rfc.json", "r") as f:
    evaluation_rfc = json.load(f)
with open("../output/evaluation_svm.json", "r") as f:
    evaluation_svm = json.load(f)


# roc curve (all models in one plot for comparison)
plt.plot(evaluation_knn["fp_rates"],evaluation_knn["tp_rates"],label = "KNN")
plt.plot(evaluation_lr["fp_rates"],evaluation_lr["tp_rates"],label = "LR")
plt.plot(evaluation_rfc["fp_rates"],evaluation_rfc["tp_rates"],label = "RFC")
plt.plot(evaluation_svm["fp_rates"],evaluation_svm["tp_rates"],label = "SVM")

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

