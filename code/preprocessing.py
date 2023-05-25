import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel


# load the data
data_clinical = pd.read_csv("../data/clinFeatures_UPENN.csv")
data_radiation = pd.read_csv("../data/radFeatures_UPENN.csv")

# look at data: -> !!!!!
#print(data_radiation.shape)

# data types
#print(data_clinical.dtypes)
#print(data_radiation.dtypes)
#print(data_radiation.dtypes.value_counts()) #only float and int (ID object)

#check for duplicates -> !!!!
#print(data_clinical.shape)
#print(data_radiation.shape)
data_clinical = data_clinical.drop_duplicates()
data_radiation = data_radiation.drop_duplicates()
#print(data_clinical.shape)
#print(data_radiation.shape)
# shape stays the same -> no dublicates

#print(data_radiation['FLAIR_ED_Intensity_Energy'])

# set index SubjectID
d1 = data_clinical.set_index('SubjectID', drop= True)
d2 = data_radiation.set_index('SubjectID', drop= True)
#print((d2.isna()).sum().sum())
#print(d2.isna().sum())


# combine d2 with column 'Survival_from_surgery_days' from d1
#data_clinical_combine = pd.concat([d2, data_clinical_red],axis=1, ignore_index = True)
data_clinical_combine = d2.join(d1['Survival_from_surgery_days'])
#print(data_clinical_combine.isna().sum())
#print(data_clinical_combine)
#print((data_clinical_combine.isna()).sum().sum())
#print(data_clinical_combine.shape)

# remove all NaN/'Not Available' in 'Survival_from_surgery_days'
# transform all 'Not Available' to NaN
data_clinical_days = data_clinical_combine.replace(to_replace='Not Available', value = np.nan)
data_clinical_days = data_clinical_days[data_clinical_days['Survival_from_surgery_days'].notna()].apply(pd.to_numeric)

#print((data_clinical_days.isna()).sum().sum())
# new df: only with patients (rows) with an entry in 'Survival_from_surgery_days'
#print(data_clinical_days['Survival_from_surgery_days'])
#print(data_clinical_days.shape) #wie viele Patienten noch?

# overview features -> number of missing values for each feature
missing_count = data_clinical_days.isna().sum()
#print(missing_count)
# number of features with 0 missing values
#print(missing_count.value_counts())
#print((missing_count == 0).sum()) #result 577-1 = 576 (-1 because label (cleaned before) is a column in data_clinical_days)

# overview patients
#missing_pat = data_clinical_days.dropna()
#print(missing_pat.shape) #(312, 4753) 312 patients are complete
#print(missing_pat)
# -> don't drop patiente!!

# data_clinical
#print((data_clinical.isna()).sum()) #no missing values appart from PsP_TP_score (611 missing values for PsP_TP_score)


# clean data set goal: complete data set
# for data_clinical_days (idea: drop all incomplete features)
data_rad_c = data_clinical_days.dropna(axis = 1)
#print(data_rad_c.isna().sum().sum()) # =0
#print(data_rad_c.shape)

# correlation between features -> drop features which correlate
# plot difficult -> many possible combinations!

#data_rad_only = data_rad_c.drop('Survival_from_surgery_days', axis = 1)

corr_matrix = data_rad_c.corr().abs()
#print(corr_matrix)

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(bool))

#find index of columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

data_rad_c = data_rad_c.drop(data_rad_c[to_drop],axis =1)
#print(data_rad_c.shape)
#print(data_rad_c['Survival_from_surgery_days'])


# for data_clinical drop columns with PsP_TP_score and drop all patients droped in data_clinical_days
data_clin_c = d1.drop('PsP_TP_score', axis=1)
#print((data_clin_c.isna()).sum())
data_clin_c = data_clin_c[data_clin_c.index.isin(data_rad_c.index)]
#print(data_clin_c)
#print(data_clin_c.shape)
#print((data_clin_c.isna()).sum())



"""
Result of preprocessing:
data_rad_c: shape (452,367), no NaN, label (Survival_from_surgery_days) column added, all incomplete features droped, no correlated features
data_clin_c: shape (452,8) same indices as in data_rad_c, no NaN
-> nur noch diese data sets verwenden!

flow chart data preprocessing
correlation features?
"""

def split_data(X, y):
    #Split of the data in three sets, train 70%, test 15%, validation 15%
    X_split, X_test, y_split, y_test = train_test_split(X,y,test_size=0.15, stratify=y, train_size=0.85, random_state=2023)
    X_train, X_validate, y_train, y_validate = train_test_split(X_split,y_split,test_size=0.1765, stratify=y_split, train_size=0.8235, random_state=2023)
    
    return X_train, X_validate, X_test, y_train, y_validate, y_test, X_split, y_split

def y_to_class(y, limit):
    #y is numerical, Survival from surgery days
    #y above or equal to limit is 1, long-term survivors
    #y under limit is 0, short-term survivors
    for i in range(len(y)):
        if y[i] >= limit:
            y[i] = 1
        else:
            y[i] = 0
    return y
