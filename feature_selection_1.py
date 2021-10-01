from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest, mutual_info_classif
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
from functools import partial
import lightgbm as lgb

gc.enable()
warnings.filterwarnings('ignore')
sns.set(rc={'figure.figsize': (20, 20)})
student_id = 2000728661

target = "covid_vaccination"
df = pd.read_csv("dataset/transformed_dataset.csv", low_memory=False).sample(10000)
reg_cols, cat_cols = [], []
id = "ID"

for idx, i in enumerate(df.columns):
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
cat_index = [i for i in range(len(reg_cols), len(training_cols))]
score_func = partial(mutual_info_classif, discrete_features=cat_index)


print(len(reg_cols), len(cat_cols))
print(cat_index)

fs = SelectKBest(score_func=score_func, k=150)


combined_features = FeatureUnion([('univ_select', fs)], n_jobs=4)

cat_features = combined_features.fit(X, y).transform(X)
print("features in: {}".format(combined_features.n_features_in_))
print("Combined space has", X_features.shape[1], "features")

tree = lgb.LGBMClassier(n_jobs=2, random_seed=student_id, deterministic=True, #device_type='GPU',
                        max_depth=15, learning_rate=.1)


selection_pipeline = Pipeline(
	[('features', combined_features), ('tree', tree)])

params = dict(
    # features__pca__n_components=[50],
    features__univ_select__k=[150],
    tree__num_leaves=[700, 600])

search = GridSearchCV(selection_pipeline, param_grid=params,
                      verbose=2, n_jobs=2, cv=3, scoring='roc_auc')
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
