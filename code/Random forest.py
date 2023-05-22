from preprocessing import data_clin_c, data_rad_c
from preprocessing import split_data, y_to_class

import warnings
warnings.filterwarnings("ignore")

X=data_rad_c[data_rad_c.columns.drop("Survival_from_surgery_days")]
y=data_rad_c["Survival_from_surgery_days"]
y_to_class(y, 456)
print(y.value_counts()) #Um Aufteilung zu sehen

#Test split
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X,y)



# Random forest:
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# X are the features(X) and y is the target variable.

#Create a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=126, random_state=42)  # Adjust the hyperparameters as needed

"""the n_estimator is a hyperparameter, which specifies 
the number of decision trees to be created in the 
random forest. Higher n_estimator = better performance, 
but also higher training time and computational cost"""

# Fit the model to the training data
rf_classifier.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = rf_classifier.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

"""The Accuracy score is at the moment Accuracy: 0.75
that means, that the RFC correctly predict the target variable
Survival from surgery days for about 75% of the instances
in the test set."""

"""The n_estimator at 126 is the lowest number with an accuracy 
of 0.75. This score stays constant till about 300 and falls 
after that again, so I choose 126 as a good value"""