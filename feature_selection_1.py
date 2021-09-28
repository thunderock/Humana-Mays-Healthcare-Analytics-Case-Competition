from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import warnings
import gc
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
# import swifter
gc.enable()
warnings.filterwarnings('ignore')
sns.set(rc={'figure.figsize': (20, 20)})
student_id = 2000728661

target = "covid_vaccination"
df = pd.read_csv("dataset/transformed_dataset.csv", low_memory=False)
reg_cols, cat_cols = [], []
id = "ID"

for i in df.columns:
	if i not in [id, target]:
		if df[i].nunique() > 100:
			reg_cols.append(i)
		else:
			cat_cols.append(i)
print("size of reg :{}, size of cat: {}".format(len(reg_cols), len(cat_cols)))
print("reg")
print(reg_cols)
print("cat")
print(cat_cols)

training_cols = reg_cols + cat_cols
X, y = df[training_cols], df[target]

pca = PCA(n_components=50, random_state=student_id)
fs = SelectKBest(score_func=f_classif, k=100)


combined_features = FeatureUnion([('univ_select', fs), ('pca', pca)], n_jobs=4)

X_features = combined_features.fit(X, y).transform(X)
print("features in: {}".format(combined_features.n_features_in_))
print("Combined space has", X_features.shape[1], "features")

tree = RandomForestClassifier(n_jobs=1, random_state=student_id, max_depth=15)

selection_pipeline = Pipeline([('features', combined_features), ('tree', tree)])

params = dict(
    features__pca__n_components=[50],
    features__univ_select__k=[100],
    tree__n_estimators=[700, 600])

search = GridSearchCV(selection_pipeline, param_grid=params, verbose=2, n_jobs=2, cv=3, scoring='roc_auc')
search.fit(X, y)

print("-------")
print(search.best_estimator_)
print("-------")
print(search.best_params_)
print("-------")
print(search.cv_results_)
print("-------")
print(search.best_score_)
print("-------")
print(search.best_estimator_.named_steps['features'])
print("-------")

joblib.dump(search, 'grid_search_best_tree.pkl')
# joblib.load("model_file_name.pkl")

