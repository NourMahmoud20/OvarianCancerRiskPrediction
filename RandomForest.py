
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_validate
from statistics import mean
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from sklearn import metrics
import graphviz
from collections import Counter
from subprocess import call
import pickle
import seaborn as sns
from google.colab import drive
drive.mount('/content/drive')

#read PLCO data March 2022 and NHIS 2019,2020 and 2021
dataPLCO = pd.read_csv('/content/drive/My Drive/PLCO.csv')

data19= pd.read_csv('/content/drive/My Drive/adult19.csv')
data20=pd.read_csv('/content/drive/My Drive/adult20.csv')
data21=pd.read_csv('/content/drive/My Drive/adult21.csv')
dataNHIS = pd.concat([data19, data20, data21])
#Select all rows from all NHIS dataset where sex != 1 (male)
dataNHIS = dataNHIS[dataNHIS.SEX_A != 1]

#Select relevant input features from datasets and merge them

columns_to_drop =[' '] 

dataPLCO = dataPLCO.drop(columns=columns_to_drop)
my_list = list(dataPLCO)
print (my_list)

dataNH_subset = dataNHIS[['']]

# Rename common columns in dataNHIS to match name in dataPLCO
dataNH_subset = dataNH_subset.rename(columns={''})
#print(dataNH_subset.describe())

# Combine dataPLCO and dataNHIS
merged_data = pd.concat([dataPLCO, dataNH_subset])


merged_data = merged_data.dropna(subset=['ovar_cancer'])


merged_datax=merged_data.drop(columns=['ovar_cancer'])

print(merged_datax.head())
print(merged_datax.describe())
print(merged_datax.info())

# Imputing missing categorical values with mode
imputer = SimpleImputer(missing_values=np.NaN, strategy='most_frequent')
x_merged_df = imputer.fit_transform(merged_datax)

#Output column
ydata_subset = merged_data['ovar_cancer']
print(ydata_subset.describe())
print(ydata_subset.shape)


#count cancer vs no cancer cases 
num_classes = len(np.unique(ydata_subset))
print("Number of classes:", num_classes)
class_counts = Counter(ydata_subset)
print("Class counts:", class_counts)

#visualize the count
sns.countplot(ydata_subset, label="count classes")
plt.show()


#splitting dataset
X_train, X_test, y_train, y_test = train_test_split(x_merged_df, ydata_subset, test_size=0.2, random_state=123, stratify=ydata_subset)


#RandomTreeClassifier hyperparameter tuning by GridSearchCV
'''
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [5,7,10],
    'max_features': [2,3,6,7],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [3,4,5],
    'n_estimators': [500]
}
'''
# Create a based model (best model found so far)
rf = RandomForestClassifier(class_weight='balanced', random_state=0,bootstrap= True, max_depth= 7, max_features= 7, min_samples_leaf= 3, min_samples_split= 3, n_estimators= 500)

'''
rf = RandomForestClassifier(class_weight='balanced', random_state=0)

# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                         cv = 3, n_jobs = -1, verbose = 2, scoring='balanced_accuracy')

'''
def evaluate(model, test_features, test_labels):
    y_pred = model.predict(test_features)
    clf_report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True))
    print("Test Result:\n================================================")        
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("_______________________________________________")
    print(f"CLASSIFICATION REPORT:\n{clf_report}")
    print("_______________________________________________")
    print(f"Confusion Matrix: \n {confusion_matrix(y_test, y_pred)}\n")
    print(f"Balanced Accuracy: \n {balanced_accuracy_score(y_test,y_pred)}\n")

rf.fit(X_train,y_train)


# Saving model to current directory
# Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.
pickle.dump(rf, open('model.pkl','wb'))


evaluate(rf,X_test,y_test)
'''
# Fit the grid search to the data
grid_search.fit(X_train,y_train)
# Get the best grid parameters
best_params = grid_search.best_params_
print("Best grid parameters:", best_params)
#Evaluate best estimator on testing data
best_grid = grid_search.best_estimator_
evaluate(best_grid,X_test,y_test)
'''
# Extract single tree
estimator = rf.estimators_[0]
# Export as dot file
export_graphviz(estimator, out_file='tree.dot', 
                feature_names = merged_datax.columns.values,
                class_names=['No Cancer', 'Cancer'],
                rounded = True, proportion = False, 
                precision = 2, filled = True)

# Convert to png using system command (requires Graphviz)

call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

# Display in jupyter notebook
from IPython.display import Image
Image(filename = 'tree.png')
