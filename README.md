## Introduction 

Accurate price predictions for houses and apartments are essential for stakeholders in the
real estate market, for buyers, sellers and investors as prices tend to be unpredictable and
hard to estimate based on different circumstances. This project leverages machine
learning techniques to analyze apartment price data in Poland based off several key
factors. By using multiple techniques and technologies, we can aim to uncover trends and
identify key factors influencing prices. In addition, visualization techniques are
implemented to present insights and model predictions clearly. The following report
sections the detail of the processes of machine learning, such as data exploration, pre-
processing, training, prediction etc.

### Objectives 

The primary objectives of the project include:
- Identify and explore data to ensure it is suitable for machine learning
- Train and optimize the machine learning models to ensure they are the most accurate
they can possibly be
- Evaluate the models’ performance using appropriate and understandable metrics
- Visualize key insights and trends for understanding model prediction patterns

### Dataset Exploration 

The dataset in particular is a dataset found online which lists thousands of Polish
Apartments in three major cities, Kraków, Poznań and Warszawa (Warsaw). There are
many other features which will be listed, but the main features include square feet of the
apartment, floor it is on, coordinates, year of the transaction etc. The data in the project is
stored as a CSV file in the data directory. We can inspect the data using the pandas library
inside the code, but since the data was manually inspected and for the purposes of the
report, here is the first 5 elements of the CSV file. This can be replicated using the head()
function, but there are countless other functions we could use to inspect the data. Please
note, the price is in Polish Złote. 5zł is equivalent to £1.

| Address                     | City        | Floor | ID       | Latitude    | Longitude   | Price     | Rooms | Sq.m | Year |
|-----------------------------|-------------|-------|----------|-------------|-------------|-----------|-------|------|------|
| Podgórze Zabłocie Stanisława| Kraków      | 2.0   | 23918.0  | 50.0492242  | 19.9703793  | 749000.0  | 3.0   | 74.05| 2021 |
| Praga-Południe Grochowska   | Warszawa    | 3.0   | 17828.0  | 52.2497745  | 21.1068857  | 240548.0  | 1.0   | 24.38| 2021 |
| Krowodrza Czarnowiejska     | Kraków      | 2.0   | 22784.0  | 50.0669642  | 19.9200249  | 427000.0  | 2.0   | 37.0 | 1970 |
| Grunwald                    | Poznań      | 2.0   | 4315.0   | 52.404212   | 16.882542   | 1290000.0 | 5.0   | 166.0| 1935 |
| Ochota Gotowy budynek.      | Warszawa    | 1.0   | 11770.0  | 52.212225   | 20.9726299  | 996000.0  | 5.0   | 105.0| 2020 |

**PLEASE NOTE THE TABLE HAS BEEN CONDENSED AS AN EXAMPLE**

### Dataset Insights

There were several machine learning models which were considered for the project, the
ones which fit perfectly are:
- Linear Regression: A simple lightweight interpretation model for price prediction.
- Random Forest: An ensemble method good for capturing non-linear relationships.
- Gradient Boosting: A powerful model which provides high predictive accuracy.
- Neural Networks: For modeling highly intricate dataset patterns.

## Dataset Pre-Processing

Data Pre-Processing is the most important part of managing data before submitting it for
the machine learning model to train. There has been some form of data cleaning by the
author of the dataset however it has been described that only the null values have been
kindly removed, but we will still demonstrate this for purposes of the project. There are still
a lot of features and flaws in the dataset that need to be resolved within the program, and
preparing the data before submission to training the machine learning models. As the
saying goes, garbage in garbage out, so we need to make sure the data is clean and
perfect. Depending on the dataset, we may need to change any missing data such as
replacing it with the mean, median etc values, however this wouldn’t have worked for our
dataset as it would significantly change results (should there have been any missing data),
and data would be very inaccurate so the best practice was to just simply remove the data
using the dropna() function. Here is the data pre-processing function:

```python

def preprocess_data(self):

  self.data.drop(columns=['Unnamed: 0', 'id', 'address'], inplace=True)
  self.data = self.data.dropna()
  self.data = pd.get_dummies(self.data)

  for column in self.data.select_dtypes(include=[np.number]).columns:
    if abs(self.data[column].skew()) > 0.5:
      self.data[column] = np.log1p(self.data[column])
```

### Feature Removal

```python self.data.drop(columns=['Unnamed: 0', 'id', 'address'], inplace=True) ```

Any features (columns) which contribute minimally towards the accuracy are removed from
the dataset. Typically all the functions use the in-place parameter for convenience as
copying the dataset etc would make it a lot more confusing especially in a OOP format,
and the parameter simply edits the existing dataset, but for other purposes it may be
useful to create a copy. The columns removed include the following:
- Unnamed: 0: Placeholder variable created by the author, possibly an error and has no
relationship with anything.
- ID: Unique ID of the apartment transaction, this also has no relationship and is a unique
identifier for the apartments and nothing to do with the data itself.
- Address: This isn’t numerical so it may be difficult for the computer to interpret, the
coordinates replace this feature regardless so it can be removed.

### Feature Encoding 

```python self.data = pd.get_dummies(self.data) ```

Machine learning algorithms are only ever good with numerical data, so it would be hard
for the machine learning to interpret what the columns mean. We have used a pandas
function which uses one-hot encoding and replaces the data with the encoded data for
easier interpretation.

### Handling Outliers

```python

for column in self.data.select_dtypes(include=[np.number]).columns:
  if abs(self.data[column].skew()) > 0.5:
    self.data[column] = np.log1p(self.data[column])
```

As there may be outliers in the dataset, we need to make sure to edit them as the results
may be very different due to data imbalances which can drastically affect the accuracy of
the models. This skews the data logarithmically, and it has been edited and tested, the
values above are the best possible out of all combinations, the skew number does not
make a drastic difference but 0.5 seems to be the best value.

## Model Training

This section provides information on training the models. As outlined in Section 1.4, we
have used four models for predicting data. Model training ensures preparation and
optimization of algorithms to make sure they are accurate and reliable. The four models, to
remind you, are:

- Linear Regression: A simple lightweight interpretation model for price prediction.
- Random Forest: An ensemble method good for capturing non-linear relationships.
- Gradient Boosting: A powerful model which provides high predictive accuracy.
- Neural Networks: For modeling highly intricate dataset patterns.
  
We can test outcomes of all the models to compare them, and these stood out as they are
the best for performance and accuracy. There is no general function for training, as there is
only one function to fit the data, it is rather split across multiple models to ensure
compatibility. The function for training is:

```python

def train(self, X_train, Y_train):
  self.model.fit(X_train, Y_train)
```

### Data Splitting

To evaluate generalizability, we have went for a 80/20 split in the data. 80% is used for
training and 20% for prediction. The results are a lot more accurate and clear at this split.

```python

def split_data(self):
  Y = self.data[self.target]
  X = self.data.drop(columns=['price'], axis=1)
  X = (X - X.mean()) / (X.std())

  self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
```
**X IN THIS FUNCTION IS DIVIDED TO ENSURE Z-SCORE STANDARDIZATION.**

To evaluate generalizability, we have went for a 80/20 split in the data. 80% is used for
training and 20% for prediction. The results are a lot more accurate and clear at this split.

```python

def split_data(self):
  Y = self.data[self.target]
  X = self.data.drop(columns=['price'], axis=1)
  X = (X - X.mean()) / (X.std())

    self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
```

**X IN THIS FUNCTION IS DIVIDED TO ENSURE Z-SCORE STANDARDIZATION.**


### Algorithm Selection

This section describes the machine learning algorithms used for predictive modeling. A selection function dynamically chooses the model based on a specified `model_type`.

#### Linear Regression

Linear Regression was chosen for its simplicity and interpretability. It provides insights into linear relationships between the target variable and features, serving as a benchmark model.

```python
def linearRegression(X_train, Y_train, X_test, Y_test, visualize):
    lr_model = Model(model_type='linear')
    lr_model.train(X_train, Y_train)
```

---

#### Random Forest Regressor

Random Forest is an ensemble learning method that handles non-linear relationships well. It is robust to overfitting and captures complex patterns effectively. Hyperparameter tuning via grid search enhances its accuracy.

```python
def randomForest(X_train, Y_train, X_test, Y_test, visualize):
    rf_model = Model(model_type='random_forest')

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'max_features': ['sqrt', 'log2', None]
    }

    best_params = rf_model.tune_hyperparameters(X_train, Y_train, param_grid)
    rf_model = Model(model_type='random_forest', params=best_params)
    rf_model.train(X_train, Y_train)
```

---

#### Gradient Boosting (XGBoost)

XGBoost is a powerful gradient boosting model with high predictive accuracy. It builds models sequentially to minimize errors and includes regularization to prevent overfitting.

```python
def xgBoost(X_train, Y_train, X_test, Y_test, visualize):
    xgb_model = Model(model_type='xgboost')
    xgb_model.train(X_train, Y_train)
```

---

#### Neural Network

Neural Networks are effective for capturing high-dimensional, non-linear relationships. Implemented using TensorFlow and Keras, this model was configured with two dense layers.

```python
def keras(X_train, Y_train, X_test, Y_test, visualize):
    keras_model = Model(model_type='keras')
    keras_model.model.fit(X_train, Y_train, epochs=100, 
                          batch_size=32, validation_data=(X_test, Y_test), 
                          verbose=0)
    mean, r2 = keras_model.evaluate(X_test, Y_test)

# Model structure
elif model_type == 'keras':
    self.model = keras.Sequential([
        layers.Input(shape=(9,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(1)
    ])
    self.model.compile(optimizer='adam', loss='mean_squared_error')
```

---

#### Stacking

Stacking combines Linear Regression, Random Forest, and XGBoost to leverage the strengths of each model and improve accuracy.

```python
elif model_type == 'all':
    base_models = [
        ('linear', LinearRegression()),
        ('rf', RandomForestRegressor(n_estimators=100)),
        ('xgb', XGBRegressor(objective='reg:squarederror'))
    ]

    meta_model = LinearRegression()
    stacked_model = StackingRegressor(estimators=base_models, 
                                      final_estimator=meta_model)

    self.model = stacked_model
```

### Hyperparameter Tuning & Optimization

Hyperparameter tuning is crucial to improve model performance. Values for each parameter were extensively tested to find optimal settings. The primary techniques used include **Grid Search** and **Cross-Validation**, which help in identifying the best performing model configurations while mitigating overfitting.

#### Grid Search

Grid search tests all combinations of parameters from a predefined set to find the best performing model configuration.

```python
def tune_hyperparameters(self, X_train, Y_train, param_grid):
    grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='r2')
    grid_search.fit(X_train, Y_train)
    self.model = grid_search.best_estimator_
    return grid_search.best_params_
```

#### Cross-Validation

Cross-validation ensures that the model generalizes well by evaluating it across multiple folds of the training data.

```python
def cross_validate(self, model, X, Y):
    scores = cross_val_score(model, X, Y, cv=5, scoring='r2')
    return scores.mean(), scores.std()
```

> ⚠️ Note: While tuning significantly improved model performance, it increased computation time—especially for Random Forest. Despite the long runtime, the improvement in accuracy justified the extra processing time.

### Model Evaluation

Model evaluation is essential to assess how well a model performs. We use several metrics to measure the prediction accuracy and generalization of each model.

#### Evaluation Metrics

The following three metrics were used:

- **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual values.
- **Mean Absolute Error (MAE)**: Measures the average absolute difference between predicted and actual values.
- **R² Score (R²)**: Indicates the proportion of variance in the dependent variable predictable from the independent variables.

```python
def evaluate(self, X_test, Y_test):
    Y_pred = self.predict(X_test)
    mse = mean_squared_error(Y_test, Y_pred)
    mae = mean_absolute_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)
```

---

#### Linear Regression Scores

- **R² Score**: 0.7282  
- **MSE**: 0.06099  
- **MAE**: 0.1820  
- **Cross-Validation Score**: 0.6793  
- **Standard Deviation**: 0.0775  

> 🔎 Verdict: Too simplistic for non-linear housing price data. Fragile to outliers and underperforms on complex datasets.

---

#### Random Forest Scores

- **R² Score**: 0.9184  
- **MSE**: 0.0183  
- **MAE**: 0.0798  
- **Cross-Validation Score**: 0.8998  
- **Standard Deviation**: 0.0088  

> 🔎 Verdict: Very accurate, handles non-linear relationships well. Computationally expensive, but robust and effective.

---

#### XGBoost Scores

- **R² Score**: 0.9109  
- **MSE**: 0.0200  
- **MAE**: 0.0937  
- **Cross-Validation Score**: 0.8970  
- **Standard Deviation**: 0.0090  

> 🔎 Verdict: Nearly matches Random Forest in performance but significantly faster. Highly effective on structured, high-dimensional data.

---

#### Neural Network (Keras) Scores

- **R² Score**: 0.7807  
- **MSE**: 0.0492  
- **MAE**: 0.1606  

> ⚠️ Note: Keras does not support cross-validation natively. The model may overfit; considered a “black-box” and is harder to interpret.

---

#### Stacking Ensemble Scores

- **R² Score**: 0.9212  
- **MSE**: 0.0177  
- **MAE**: 0.0837  
- **Cross-Validation Score**: 0.9038  
- **Standard Deviation**: 0.0094  

> 🔎 Verdict: Slightly better than individual models but computationally expensive and more complex to implement and tune.

---

#### Final Verdict

XGBoost stands out as the optimal model, balancing **accuracy**, **efficiency**, and **interpretability**. Linear Regression performed the worst due to the dataset’s complexity and non-linear nature. Stacking showed marginal improvements but at a higher computational cost.

## Visualization

Visualizing results is essential for understanding model performance and identifying areas for improvement. This section includes visual tools to assess accuracy and diagnose potential issues.

### Key Visualizations

#### Model Evaluation

- **Predicted vs Actual**: Compares predicted values against true values using a regression plot.
- **Feature Importance**: Displays the impact of each feature in tree-based models.

#### Error Analysis

- **Cross Validation**: Assesses model stability across different folds.
- **Residual Plot**: Reveals patterns in prediction errors to detect systematic issues.

```python
def importance(self, model, name, features):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title(f"{name}: Feature Importance")
    plt.bar(range(len(features)), importances[indices])
    plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()
```

---

### Linear Regression

**Regression Plot**  
**[Placeholder for Linear Regression Regression Plot]**

**K-Fold Cross Validation**  
**[Placeholder for Linear Regression Cross Validation Plot]**

**Residual Plot**  
**[Placeholder for Linear Regression Residual Plot]**

**Verdict**  
Linear Regression is unsuitable for this dataset. It underfits and struggles with complex, non-linear relationships, showing clear systematic errors.

---

### Random Forest

**Regression Plot**  
**[Placeholder for Random Forest Regression Plot]**

**K-Fold Cross Validation**  
**[Placeholder for Random Forest Cross Validation Plot]**

**Residual Plot**  
**[Placeholder for Random Forest Residual Plot]**

**Feature Importance**  
**[Placeholder for Random Forest Feature Importance Plot]**

**Verdict**  
Performs accurately and robustly, though training time is high. Best suited for capturing non-linear patterns in housing data.

---

### XGBoost

**Regression Plot**  
**[Placeholder for XGBoost Regression Plot]**

**K-Fold Cross Validation**  
**[Placeholder for XGBoost Cross Validation Plot]**

**Residual Plot**  
**[Placeholder for XGBoost Residual Plot]**

**Feature Importance**  
**[Placeholder for XGBoost Feature Importance Plot]**

**Verdict**  
Almost matches Random Forest in performance, but with better speed and efficiency. Excels in structured, high-dimensional data.

---

### Neural Network

**Regression Plot**  
**[Placeholder for Neural Network Regression Plot]**

**Verdict**  
Too complex and opaque for this task. Likely overfitting and difficult to interpret. Not well-suited without extensive tuning and more data.

---

### Stacking Ensemble

**Regression Plot**  
**[Placeholder for Stacking Regression Plot]**

**K-Fold Cross Validation**  
**[Placeholder for Stacking Cross Validation Plot]**

**Residual Plot**  
**[Placeholder for Stacking Residual Plot]**

**Verdict**  
Slightly better performance than individual models but demands far more computational resources, making it inefficient for regular use.

---

### Summary

- **XGBoost** was the most effective model overall, balancing speed, accuracy, and reliability.
- **Linear Regression** performed the worst due to its inability to handle complex patterns.
- **Stacking** provided marginal improvements but was computationally heavy.
- **Neural Networks** showed promise but were too opaque and inefficient for the dataset size.

This pipeline could be adapted to other real estate markets with similar data structures. While perfect predictions are unrealistic, the models provide valuable price estimates to support buyers, sellers, and investors.

---

## References

- [Kaggle Dataset - Polish Apartment Prices](https://www.kaggle.com/datasets/dawidcegielski/house-prices-in-poland)








