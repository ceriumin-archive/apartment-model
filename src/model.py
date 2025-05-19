from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

# the import is glitched and comes up as an error
from tensorflow import keras
from tensorflow.keras import layers

from xgboost import XGBRegressor

'''This Class handles all the model processing and training'''
class Model:
    # Returns the model based off the input provided for simplicity and any parameters
    def __init__(self, model_type='linear', params=None):
        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'random_forest':
            if params is None:
                params = {}
            self.model = RandomForestRegressor(random_state=42, n_jobs=-1, **params)
        elif model_type == 'xgboost':
            self.model = XGBRegressor(objective='reg:squarederror', random_state=42)
        elif model_type == 'stack':
            base_models = [
                ('linear', LinearRegression()),
                ('rf', RandomForestRegressor(n_estimators=100)),
                ('xgb', XGBRegressor(objective='reg:squarederror'))
            ]

            meta_model = LinearRegression()
            stacked_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)
            self.model = stacked_model

        # Neural Network with preset parameters, 64 seems to be the most optimized value for density
        elif model_type == 'keras':
            self.model = keras.Sequential([
                layers.Input(shape=(6,)),
                layers.Dense(64, activation='relu'),
                layers.Dense(1)
            ])
            self.model.compile(optimizer='adam', loss='mean_squared_error')

        else:
            raise ValueError('Invalid model type')

    def train(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    # Evaluation in a Human Readable Format 
    def evaluate(self, X_test, Y_test, model):
        Y_pred = self.predict(X_test)
        mse = mean_squared_error(Y_test, Y_pred)
        mae = mean_absolute_error(Y_test, Y_pred)
        r2 = r2_score(Y_test, Y_pred)

        # Fancy formatting just to make it look nicer on the eyes
        print(f"\n\033[1;4m{model} Evaluation\033[0m")
        print(f"\nR2 Score: {r2} \nMean Squared Error: {mse}\nMean Absolute Error: {mae}")

    # Hyperparameter tuning function is stored here
    def tune_hyperparameters(self, X_train, Y_train, param_grid):
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='r2')
        grid_search.fit(X_train, Y_train)
        self.model = grid_search.best_estimator_
        return grid_search.best_params_

    # Function for running cross validation
    def cross_validate(self, model, X, Y):
        scores = cross_val_score(model, X, Y, cv=10, scoring='r2')
        mean = scores.mean()
        std = scores.std()

        # This is usually ran after the evaluation anyways
        print(f"Cross Validation: {mean}\nStandard Deviation: {std}")
        return scores













