from data import DataLoader
from model import Model
from evaluate import Visualize

# Everything here runs all the methods together in sequence

'''Linear Regression Model'''
def linear(X_train, Y_train, X_test, Y_test, visualize):
    name = "Linear Regression"
    model = Model(model_type='linear')

    model.train(X_train, Y_train)
    model.evaluate(X_test, Y_test, name)
    score = model.cross_validate(model.model, X_train, Y_train)

    visualize.scatter(name, Y_test, model.predict(X_test))
    visualize.validation(score, name)
    visualize.residual(name, Y_test, model.predict(X_test))

'''Random Forest Regressor Ensemble'''
def forest(X_train, Y_train, X_test, Y_test, visualize):
    name = "Random Forest"
    # Stock model is initialized here
    model = Model(model_type='random_forest')
    # Tuned parameter grid for hyperparameters
    param_grid = {
        'n_estimators': [100,200, 300],
        'max_depth': [None, 10, 20, 30],
        'max_features': ['sqrt', 'log2', None]
    }
    best_params = model.tune_hyperparameters(X_train, Y_train, param_grid)
    tuned = Model(model_type='random_forest', params=best_params)
    tuned.train(X_train, Y_train)

    model.evaluate(X_test, Y_test, name)
    score = model.cross_validate(model.model, X_train, Y_train)

    visualize.scatter(name, Y_test, model.predict(X_test))
    visualize.validation(score, name)
    visualize.residual(name, Y_test, model.predict(X_test))
    visualize.importance(model.model, name, X_train.columns.tolist())

'''XGBoost Gradient Regressor Ensemble'''
def xg(X_train, Y_train, X_test, Y_test, visualize):
    name = "XGBoost"
    model = Model(model_type='xgboost')
    model.train(X_train, Y_train)

    model.evaluate(X_test, Y_test, name)
    score = model.cross_validate(model.model, X_train, Y_train)

    visualize.scatter(name, Y_test, model.predict(X_test))
    visualize.validation(score, name)
    visualize.residual(name, Y_test, model.predict(X_test))
    visualize.importance(model.model, name, X_train.columns.tolist())

'''Regression Stacking Ensemble'''
def stack(X_train, Y_train, X_test, Y_test, visualize):
    name = "Stacking Ensemble"
    model = Model(model_type='stack')
    model.train(X_train, Y_train)

    model.evaluate(X_test, Y_test, name)
    score = model.cross_validate(model.model, X_train, Y_train)

    visualize.scatter(name, Y_test, model.predict(X_test))
    visualize.validation(score, name)
    visualize.residual(name, Y_test, model.predict(X_test))

'''Keras Neural Network'''  # Experiemntal Model to see what would happen
def neural(X_train, Y_train, X_test, Y_test, visualize):
    name = "Neural Network"
    model = Model(model_type='keras')
    model.model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_test, Y_test), verbose=0)

    model.evaluate(X_test, Y_test, name)
    # Cross Validation Unavailable :(
    # model.cross_validate(model.model, X_train, Y_train)

    visualize.scatter(name, Y_test, model.predict(X_test))

def main():
    filepath = '../data/data.csv'
    target = 'price'

    ld = DataLoader(filepath, target)
    ld.load_data()
    ld.preprocess_data()
    ld.split_data()

    X_train, X_test, Y_train, Y_test = ld.get_data()

    viz = Visualize()

    # Runs all the models, both trains them and runs predictions + evaluations for convenience
    linear(X_train, Y_train, X_test, Y_test, viz)
    forest(X_train, Y_train, X_test, Y_test, viz)
    xg(X_train, Y_train, X_test, Y_test, viz)
    neural(X_train, Y_train, X_test, Y_test, viz)
    stack(X_train, Y_train, X_test, Y_test, viz)

# Initializes the main method which is the entrypoint
if __name__ == "__main__":
    main()

