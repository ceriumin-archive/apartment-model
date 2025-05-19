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


