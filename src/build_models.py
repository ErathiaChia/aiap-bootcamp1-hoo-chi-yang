
# implement age 
# add back attendance rate


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model_mgr import Model_Mgr

class ModelBuilder:
    def __init__(self, X_train, X_test, y_train, y_test,seed=2023):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test  = X_test
        self.y_test  = y_test
        self.seed    = seed

    def train_and_evaluate(self, model_name="all"):
        X_train = self.X_train
        X_test  = self.X_test
        y_train = self.y_train
        y_test  = self.y_test

        model_manager = Model_Mgr()
        
        # Retrieve hyper params
        models = model_manager.get_models()
        
        # Retrieve cv folds
        cv = model_manager.get_cv_folds()
        print(f"Using {cv} folds for cross-validation")

        # If model_name is not "all", filter the models dictionary
        if model_name != "all":
            models = {model_name: models[model_name]}

        # Print params for each model
        for model_name, (model, params) in models.items():
            print(f"\n{model_name} Testing the following Parameters:")
            print(params)

        best_models = {}
        for model_name, (model, params) in models.items():
            # Use pipeline to ensure that scaling is done within each CV fold
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
            
            grid = GridSearchCV(pipeline, params, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
            grid.fit(X_train, y_train)
            
            best_models[model_name] = grid.best_estimator_
            print(f"{model_name} - Best Parameters: {grid.best_params_}")
            print(f"{model_name} - Best Cross-Validation Score: {-grid.best_score_}")

        from sklearn.metrics import mean_squared_error

        test_scores = {}
        for model_name, model in best_models.items():
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            test_scores[model_name] = mse
            print(f"{model_name} - Test MSE: {mse}")

        best_model_name = min(test_scores, key=test_scores.get)
        best_model = best_models[best_model_name]
        print(f"\nThe best model is {best_model_name} with a Test MSE of {test_scores[best_model_name]}")


        if best_model_name in ["Linear Regression", "Ridge", "Lasso", "ElasticNet"]:
            # If the best model is linear, we can display its coefficients
            if best_model_name == "Linear Regression":
                coefficients = best_model.named_steps['model'].coef_
            else:
                coefficients = best_model.named_steps['model'].coef_
            features = X_train.columns
            feature_importances = pd.DataFrame({'Feature': features, 'Coefficient': coefficients})
            print("\nFeature Coefficients:")
            print(feature_importances.sort_values(by='Coefficient', ascending=False))

        elif best_model_name == "Random Forest":
            # If the best model is Random Forest, we can display feature importances
            importances = best_model.named_steps['model'].feature_importances_
            features = X_train.columns
            feature_importances = pd.DataFrame({'Feature': features, 'Importance': importances})
            print("\nFeature Importances:")
            print(feature_importances.sort_values(by='Importance', ascending=False))
        
        return best_models, test_scores