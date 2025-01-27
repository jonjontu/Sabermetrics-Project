#Import Libraries
import pandas as pd
import numpy as np
import warnings
import joblib
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
from catboost import CatBoostClassifier
from lightgbm.sklearn import LGBMClassifier

warnings.filterwarnings("ignore", category=FutureWarning, message="'force_all_finite' was renamed to 'ensure_all_finite'")


# Read Data
dict_dat = pd.read_csv("/Users/gamerboyj/data-dictionary.csv")
train_dat = pd.read_csv("/Users/gamerboyj/data-train.csv")
test_dat = pd.read_csv("/Users/gamerboyj/data-test.csv")


# Split Data
X_train = train_dat.drop(columns=["pitch_id", "gamedate", "venue_id", "lf_id", "cf_id", "rf_id", "first_fielder", "is_airout"])
y_train = train_dat["is_airout"]
X_test = test_dat.drop(columns=["pitch_id","gamedate","venue_id","p_airout"])


# Preprocess Data
categorical_feats = ["level", "bat_side", "pitch_side", "inning",
"top", "pre_balls", "pre_strikes", "pre_outs"]
numeric_feats = ["temperature", "exit_speed", "hit_spin_rate", "vert_exit_angle", "horz_exit_angle"]

ct = make_column_transformer(
    (make_pipeline(SimpleImputer(), StandardScaler()), numeric_feats),
    (OneHotEncoder(handle_unknown='ignore'), categorical_feats))
kf = KFold(n_splits=5, shuffle=True, random_state=1)


# Function that performs hyperparameter tuning, fits the model, and prints the best score and best parameters
models = []

def fit_to_result(name, model, param_grid, X_train, y_train):
    pipe_model = make_pipeline(ct, model)
    model_gs = GridSearchCV(pipe_model, param_grid=param_grid, n_jobs=-1, return_train_score=True, cv=kf)
    model_gs.fit(X_train, y_train)
    model_results_dict = {"Model Name": name, "Best Score": model_gs.best_score_, "Best Params": model_gs.best_params_}
    print(model_results_dict)
    models.append(model_results_dict)


# Hyperparameter optimization for Logistic Regression
lr_name = "Logistic Regression"
lr_param_grid = {
    "logisticregression__C": [0.001, 0.01, 0.1, 1, 10, 100]
    }
lr = LogisticRegression(random_state=1, max_iter=1000)
fit_to_result(lr_name, lr, lr_param_grid, X_train, y_train)


#Hyperparameter optimization for Random Forest
rf_name = "Random Forest"
rf_param_grid = {
    "randomforestclassifier__n_estimators": [100], 
    "randomforestclassifier__max_depth": [10, 20, None]
    }
rf = RandomForestClassifier(random_state=1)
rf.class_weight = "balanced"
fit_to_result(rf_name, rf, rf_param_grid, X_train, y_train)


#Hyperparameter optimization for CatBoost
cb_name = "CatBoost" 
cb_param_grid = {
    "catboostclassifier__iterations": [1000], 
    "catboostclassifier__learning_rate": [0.1, 0.3],
    "catboostclassifier__depth": [4, 6, 8]
    }
cb = CatBoostClassifier(random_state=1, verbose=False)
fit_to_result(cb_name, cb, cb_param_grid, X_train, y_train)


#Hyperparameter optimization for LightGBM
lgbm_name = "LightGBM"
lgbm_param_grid = {
    "lgbmclassifier__n_estimators": [100], 
    "lgbmclassifier__learning_rate": [0.1, 0.3],
    "lgbmclassifier__max_depth": [4, 6, 8]
    }
lgbm = LGBMClassifier(random_state=1, verbose=-1)
fit_to_result(lgbm_name, lgbm, lgbm_param_grid, X_train, y_train)


# Show best model with optimal parameters
highest_score_model = max(models, key=lambda x: x["Best Score"])
print(highest_score_model)


# Fit the best model
unique_model_params = {
    "Logistic Regression": {
        "random_state": 1,
        "max_iter": 1000
        },
    "Random Forest": {
        "random_state": 1,
        "class_weight": "balanced"
        },
    "CatBoost": {
        "random_state": 1,
        "verbose": False
        },
    "LightGBM": {
        "random_state": 1,
        "verbose": -1
        }
    }

best_model_name = highest_score_model["Model Name"]
best_params = highest_score_model["Best Params"].copy()

if best_model_name == "Logistic Regression":
    best_model = LogisticRegression(**best_params)
elif best_model_name == "Random Forest":
    best_model = RandomForestClassifier(**best_params)
elif best_model_name == "CatBoost":
    best_params = {
        key.replace('catboostclassifier__', ''): value
        for key, value in highest_score_model['Best Params'].items()
        }
    best_model = CatBoostClassifier(**best_params)
elif best_model_name == "LightGBM":
    best_model = LGBMClassifier(**best_params)
else:
    raise ValueError(f"Unknown model name: {best_model_name}")

pipe_best_model = make_pipeline(ct, best_model)
pipe_best_model.fit(X_train, y_train)
prob = pipe_best_model.predict_proba(X_test)


# Fill in predicted probability values for p_airout
p_airout_df = pd.DataFrame(prob, columns=["prob_0", "p_airout"])
test_probs = pd.concat([test_dat.drop(columns=["p_airout"]), p_airout_df], axis=1)
print(test_probs.head())


# Feature Importances
importance_values = pipe_best_model.named_steps["catboostclassifier"].get_feature_importance()
feature_names = pipe_best_model.named_steps["columntransformer"].get_feature_names_out()
feature_pairs = list(zip(feature_names, importance_values))
feature_pairs.sort(key=lambda x: x[1], reverse=True)
for f, i in feature_pairs:
    print(f"{f}: {i}")


# Save to Flask app
test_probs.to_csv("/Users/gamerboyj/test_probs.csv", index=False)
joblib.dump(highest_score_model, "/Users/gamerboyj/model_info.pkl")
joblib.dump(feature_pairs, "/Users/gamerboyj/feature_pairs.pkl")