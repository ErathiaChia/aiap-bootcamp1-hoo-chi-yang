
#model_mgr.py
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

class Model_Mgr:
    def __init__(self):
        # hyperparams
        # include __ in front of params because of pipeline in build_models.py
        self.ridge_params = {'model__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
        self.lasso_params = {'model__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
        self.elastic_params = {'model__alpha': [0.001, 0.01, 0.1, 1, 10, 100], 'model__l1_ratio': [0.2, 0.5, 0.8]}
        self.rf_params = {'model__n_estimators': [50, 100, 200], 'model__max_depth': [None, 10, 20, 30], 'model__min_samples_split': [2, 5, 10]}
        self.xgb_params = {'model__n_estimators': [50, 100, 200], 'model__max_depth': [3,5,7], 'model__learning_rate': [0.01, 0.5, 1], 'model__subsample': [0.5, 0.7, 1], 'model__colsample_bytree': [0.5, 0.7, 1]}
        self.svm_linear_params = {'model__C': [0.1, 1, 10], 'model__kernel': ['linear']}
        self.svm_radial_params = {'model__C': [0.1, 1, 10], 'model__kernel': ['rbf']}
        self.svm_poly_params = {'model__C': [0.1, 1, 10], 'model__kernel': ['poly'], 'model__degree': [2, 3]}
        self.knn_params = {'model__n_neighbors': [3, 5, 7], 'model__weights': ['uniform', 'distance']}
    
    # models
        self.models = {
            'Linear Regression': (LinearRegression(), {}),
            'Ridge': (Ridge(), self.ridge_params),
            'Lasso': (Lasso(), self.lasso_params),
            'ElasticNet': (ElasticNet(), self.elastic_params),
            'Random Forest': (RandomForestRegressor(), self.rf_params),
            'XGBoost': (XGBRegressor(), self.xgb_params),
            'SVM Linear': (SVR(), self.svm_linear_params),
            'SVM Radial': (SVR(), self.svm_radial_params),
            'SVM Poly': (SVR(), self.svm_poly_params),
            'KNN': (KNeighborsRegressor(), self.knn_params)
        }
        self.cv = 5
        self.seed = 2023

    def get_params(self, model_name):
        return self.models[model_name][1]

    def get_cv_folds(self):
        return self.cv
    
    def get_seed(self):
        return self.seed

    def get_models(self):
        return self.models
