from imblearn.under_sampling._prototype_selection import RandomUnderSampler
import torch
import numpy as np
import pandas as pd
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import TomekLinks, CondensedNearestNeighbour, NeighbourhoodCleaningRule
from sklearn.metrics import balanced_accuracy_score
from sklearn.decomposition import PCA
import graphviz
from collections import Counter
import seaborn as sns
import torch.nn as nn
from google.colab import drive
drive.mount('/content/drive')


#read PLCO data March 2022
dataPLCO = pd.read_csv('/content/drive/My Drive/PLCO.csv')


#Inputs: Family History (Ovarian and Breast cancer), Body Type (BMI, Weight,Height), NSAIDS (Aspirin and Ibuprofen intake), Diseases (Arthritis, Hypertension,Diabetes, Osteoporosis, and Colorectal Polyps), and Smoking
columns_to_drop =['ovar_reasfoll', 'ovar_reassurv', 'ovar_reassymp', 'ovar_reasoth', 'ovar_intstat_cat', 'ovar_annyr', 
'ovar_cancer_site', 'ovar_stage', 'ovar_stage_7e', 'ovar_stage_t', 'ovar_stage_n', 'ovar_stage_m', 'ovar_clinstage', 
'ovar_clinstage_7e', 'ovar_clinstage_t', 'ovar_clinstage_n', 'ovar_clinstage_m', 'ovar_pathstage', 'ovar_pathstage_7e', 
'ovar_pathstage_t', 'ovar_pathstage_n', 'ovar_pathstage_m', 'ovar_grade', 'ovar_topography', 
'ovar_morphology', 'ovar_behavior', 'ovar_histtype', 'ovar_exitstat', 'ovar_exitage', 'ovar_seer',
'ovar_cancer_first', 'ovar_curative_surg', 'ovar_curative_chemo', 'ovar_primary_trt', 'ovar_num_heslide_imgs',
'ovar_has_deliv_heslide_img', 'plco_id', 'build', 'build_cancers', 'build_incidence_cutoff', 'ovar_exitdays',
'ovar_primary_trt_days', 'ovar_cancer_diagdays', 'biopolink0', 'biopolink1', 'biopolink2', 'biopolink3', 'biopolink4',
'biopolink5', 'ovar_mra_stat0', 'ovar_mra_stat1', 'ovar_mra_stat2', 'ovar_mra_stat3', 'ovar_mra_stat4', 
'ovar_mra_stat5', 'tvu_result0', 'tvu_result1', 'tvu_result2','ca125_result0', 'ca125_result1', 
'ca125_result2', 'ca125_result3', 'ca125_result4', 'ca125_level0', 'ca125_level1', 'ca125_level2',
'ca125_level3', 'ca125_level4', 'ca125_level5', 'ca125_src0', 'ca125_src1', 'ca125_src2', 'ca125_src3', 'ca125_src4',
'ca125_src5', 'ca125ii_level0', 'ca125ii_level1', 'ca125ii_level2', 'ca125ii_level3', 'ca125ii_level4', 
'ca125ii_level5', 'orem_fyro', 'ca125_prot', 'ca125_days0', 'ca125_days1', 'ca125_days2', 'ca125_days3', 
'ca125_days4', 'ca125_days5', 'tvu_days0', 'tvu_days1', 'tvu_days2', 'tvu_days3','bq_adminm','bq_returned', 
'bq_age','bq_compdays','d_dthovar','f_dthovar','d_codeath_cat', 'f_codeath_cat','d_cancersite', 
'f_cancersite', 'd_seer_death', 'f_seer_death', 'is_dead_with_cod', 'is_dead', 'mortality_exitage', 'mortality_exitstat',
'build_death_cutoff', 'dth_days', 'mortality_exitdays','entryage_bq', 'entryage_dqx', 'entryage_dhq', 'entryage_sqx', 
'entryage_muq','ph_any_bq', 'ph_any_dqx', 'ph_any_dhq', 'ph_any_sqx', 'ph_any_muq', 'ph_ovar_bq', 'ph_ovar_dqx', 'ph_ovar_dhq',
'ph_ovar_sqx', 'ph_ovar_muq','ovar_eligible_bq', 'ovar_eligible_sqx', 'ovar_eligible_dhq', 'ovar_eligible_dqx', 'entrydays_bq',
'entrydays_dqx', 'entrydays_dhq', 'entrydays_sqx', 'entrydays_muq','fstcan_exitstat', 'fstcan_exitage', 'fstcan_exitdays',
'in_TGWAS_population', 'reconsent_outcome', 'reconsent_outcome_days','dual','center','arm','rndyear'] 

dataPLCO = dataPLCO.drop(columns=columns_to_drop)

dataPLCO = dataPLCO.dropna(subset=['ovar_cancer'])


dataPLCOx=dataPLCO.drop(columns=['ovar_cancer'])

#imputing numeric values with mean value
dataPLCOx[['weight_f', 'height_f', 'breast_fh_age', 'breast_fh_cnt', 'ovarsumm_fh_age', 'ovarsumm_fh_cnt']] = dataPLCOx[['weight_f', 'height_f', 'breast_fh_age', 'breast_fh_cnt', 'ovarsumm_fh_age', 'ovarsumm_fh_cnt']].fillna(dataPLCOx[['weight_f', 'height_f', 'breast_fh_age', 'breast_fh_cnt', 'ovarsumm_fh_age', 'ovarsumm_fh_cnt']].mean())

# Imputing missing categorical values with mode
imputer = SimpleImputer(missing_values=np.NaN, strategy='most_frequent')
dataPLCOx = imputer.fit_transform(dataPLCOx)

print(dataPLCOx.shape)
print(dataPLCOx)
'''
#scaling input features
dataPLCOx = StandardScaler().fit_transform(dataPLCOx)

#Applying PCA
pca = PCA(n_components=10)

principalComponents = pca.fit_transform(dataPLCOx)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['p1', 'p2','p3','p4','p5','p6','p7','p8','p9','p10'])
'''
#Output column
ydata_subset = dataPLCO['ovar_cancer']
print(ydata_subset.describe())
print(ydata_subset.shape)


#count cancer vs no cancer cases 
num_classes = len(np.unique(ydata_subset))
print("Number of classes:", num_classes)
class_counts = Counter(ydata_subset)
print("Class counts:", class_counts)

#visualize the count
sns.countplot(ydata_subset,label="count before oversampling")
plt.show()

#splitting dataset
X_train, X_test, y_train, y_test = train_test_split(dataPLCOx, ydata_subset, test_size=0.2, random_state=42, shuffle=True, stratify=ydata_subset)

#rus = RandomUnderSampler()
#X_train, y_train= rus.fit_resample(X_train, y_train)
#ros=RandomOverSampler(sampling_strategy=0.5, random_state=42)
#X_train, y_train= ros.fit_resample(X_train, y_train)

def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
        
    elif train==False:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")
        print(f"Balanced Accuracy: \n {balanced_accuracy_score(y_test,pred)}\n")


params = {
    "criterion":("gini", "entropy"), 
    "splitter":("best", "random"), 
    "max_depth":(list(range(1, 20))), 
    "min_samples_split":[2, 3, 4], 
    "min_samples_leaf":list(range(1, 20)), 
}


tree_clf = DecisionTreeClassifier(random_state=42, class_weight='balanced')
tree_cv = GridSearchCV(
    tree_clf, 
    params, 
    scoring="balanced_accuracy", 
    n_jobs=-1, 
    verbose=1, 
    cv=5
)

tree_cv.fit(X_train, y_train)
best_params = tree_cv.best_params_
print(f"Best paramters: {best_params})")

tree_clf = DecisionTreeClassifier(**best_params)
tree_clf.fit(X_train, y_train)
print_score(tree_clf, X_train, y_train, X_test, y_test, train=True)
print_score(tree_clf, X_train, y_train, X_test, y_test, train=False)
'''
tree_clf = DecisionTreeClassifier(class_weight='balanced',random_state=123,criterion= 'gini', max_depth= 10, min_samples_leaf= 2, min_samples_split= 6, splitter= 'best')
tree_clf.fit(X_train, y_train)
print_score(tree_clf, X_train, y_train, X_test, y_test, train=True)
print_score(tree_clf, X_train, y_train, X_test, y_test, train=False)
'''
# Export the trained decision tree as a GraphViz format file
dot_data = export_graphviz(tree_clf, out_file=None, feature_names=dataPLCOx.columns.values, class_names=['No Cancer', 'Cancer'], filled=True, rounded=True, special_characters=True)

# Render the decision tree
graph = graphviz.Source(dot_data)
graph.render('decision_tree')

# Display the decision tree
graph


