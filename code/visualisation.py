import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from preprocessing import data_clin_c, data_rad_c,d1, d2, data_clinical_days, data_clinical, data_radiomic
from KNN_PCA import X,y, y_train, y_val, y_test

# data before preprocessing
#print(data_clinical.shape)
#print(data_radiomic.shape)

# data after preprocessing
# data_clin_c
#print('data_clin_c')
#print(data_clin_c.shape)
#print(data_clin_c.dtypes)
#print(data_clin_c.head())
#print(data_clin_c.describe())

# outcome data_rad_c['Survival_from_surgery_days']
#Plot for the distribution of survival days
fig, axs = plt.subplots(2,1,figsize=(12,8), gridspec_kw={'height_ratios': [1, 4]})
fig1 = sns.boxplot(data=data_clinical_days, x=data_clinical_days['Survival_from_surgery_days'],ax=axs[0],color='gray')
fig1.minorticks_on()
fig1.set_xlabel(" ")
fig2 = sns.histplot(data_clinical_days['Survival_from_surgery_days'], binwidth=10, binrange=(0, 2200), kde=True, ax=axs[1],color='gray')
#axs = sns.histplot(data_clinical_days['Survival_from_surgery_days'])
#fig2.set_title("Count of patients by survival days")
fig2.set_xlabel("Survival days after surgery", size=15)
fig2.set_ylabel("Count of patients", size=15)
fig2.grid(color='gray', alpha=0.3, linestyle=':', linewidth=1)
fig2.minorticks_on()
plt.suptitle("Survival from surgery", size=20)
plt.tight_layout()
plt.savefig('../output/dist_surviaval_days.png')
plt.show()


#print('outcome')
#print(data_clin_c['Survival_from_surgery_days'])
days = data_clin_c['Survival_from_surgery_days'].values
days = [int(x) for x in days]
#print(days)
#mean_days = days.sum()/len(days)
#print(np.mean(days)) # mean is 422.1659292035398

# data_rad_c
#print('data_rad_c')
#print(data_rad_c.shape)
#print(data_rad_c.dtypes)
#print(data_rad_c.head())

# X
#print('X')
#print(X.shape)
#print(X.head())
#print(X.info)

# summary label
#print(data_clin_c['Survival_from_surgery_days'].describe())
#print(data_clin_c['Survival_from_surgery_days'].max)

# visualisation NaNs
# 1) full data_rad (Subject_Id = index) -> d2
# total number of missing values in d2
data_rad_with = d2.replace(to_replace='Not Available', value = np.nan)
num_NaN = data_rad_with.isna().sum().sum()
num_en = data_rad_with.shape[0] * data_rad_with.shape[1]
#print(num_en)
# % NaN in data rad
#print(num_NaN/num_en)

#print()
missing_values = data_rad_with.isna().sum()
missing_values = missing_values[missing_values.values != 0]
#print(missing_values)
NaN_count = pd.DataFrame(missing_values.value_counts())
#print(NaN_count)

# distribution of number missing values
fig3 = sns.histplot(missing_values,binwidth=10, color ='grey')
fig3.set_xlabel('Number of missing values')
fig3.set_ylabel('Count features')
fig3.set_title('Count of features by number of missing values')
plt.tight_layout()
fig3.minorticks_on()
plt.savefig('../output/missing_values.png')
plt.show()

gig, ax = plt.subplots()
ax.axis('off')
ax.axis('tight')
tab1 = ax.table(cellText=NaN_count.values,colLabels=['Number of Features'],rowLabels=NaN_count.index, loc='center')
tab1.set_fontsize(14)
plt.savefig("../output/missing_values.pdf", bbox_inches='tight',pad_inches=0.2)

# 2) full data_clin (Subject_Id = index) -> d1
num_NaN = d1.isna().sum().sum()
#print(num_NaN)
num_en = d1.shape[0] * d1.shape[1]
#print(num_en)
# % NaN in data rad
#print(num_NaN/num_en)
#print(d1.isna().sum()) #only in one column NaNs

# look at distribution in test splits
index = ['Percentage 0','Percentage 1']
data_tab1 = pd.DataFrame(index = index, columns = ['original','train','val', 'test'])
data_tab1.iloc[0,0] = y.value_counts()[0]/y.value_counts().sum().round(decimals = 4)
data_tab1.iloc[1,0] = y.value_counts()[1]/y.value_counts().sum().round(decimals = 4)
data_tab1.iloc[0,1] = y_train.value_counts()[0]/y_train.value_counts().sum().round(decimals = 4)
data_tab1.iloc[1,1] = y_train.value_counts()[1]/y_train.value_counts().sum().round(decimals = 4)
data_tab1.iloc[0,2] = y_val.value_counts()[0]/y_val.value_counts().sum().round(decimals = 4)
data_tab1.iloc[1,2] = y_val.value_counts()[1]/y_val.value_counts().sum().round(decimals = 4)
data_tab1.iloc[0,3] = y_test.value_counts()[0]/y_test.value_counts().sum().round(decimals = 4)
data_tab1.iloc[1,3] = y_test.value_counts()[1]/y_test.value_counts().sum().round(decimals = 4)

fig, ax = plt.subplots()
ax.axis('off')
ax.axis('tight')
tab1 = ax.table(cellText=data_tab1.values,colLabels=data_tab1.columns, rowLabels=data_tab1.index, loc='center')
tab1.set_fontsize(14)
plt.savefig("../output/distribution_outcome_sets.pdf", bbox_inches='tight',pad_inches=0.2)