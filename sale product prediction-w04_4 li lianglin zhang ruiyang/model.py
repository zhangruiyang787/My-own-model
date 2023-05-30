import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime


class SalesPrediction:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.predictions = None

    def load_data(self):
        try:
            self.df = pd.read_csv(self.csv_path)
        except FileNotFoundError:
            print(f"Error: The CSV file '{self.csv_path}' was not found.")

    def preprocess_data(self):
        def to_date(s):
            return datetime.strptime(str(s[4]), '%m/%d/%y %H:%M')

        self.df = self.df.dropna()
        self.df.drop(self.df[self.df['Order Date'] == "Order Date"].index, inplace=True)
        self.df['Order Date'] = self.df.apply(to_date, axis=1)
        self.df = self.df.set_index('Order Date')
        self.df = self.df.astype({'Price Each': 'float'})

        df_prodcopy = set(self.df["Product"])
        product_dict = dict()
        ind = 0
        for x in df_prodcopy:
            product_dict[ind] = x
            ind += 1
        inv_dict = {v: k for k, v in product_dict.items()}

        def indexify(s):
            return inv_dict[s[1]]

        self.df['Product'] = self.df.apply(indexify, axis=1)

    def visualize_sales(self):
        color_pal = sns.color_palette()
        plt.style.use('fivethirtyeight')
        self.df.plot(style='.', figsize=(15, 5), color=color_pal[0], title='Product sales')
        plt.show()

    def train_test_split(self):
        train = self.df.loc[self.df.index < '04-24-2019']
        test = self.df.loc[self.df.index >= '04-24-2019']
        return train, test

    def create_features(self, df):
        df = df.copy()
        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.dayofweek
        df['quarter'] = df.index.quarter
        df['month'] = df.index.month
        df['year'] = df.index.year
        df['dayofyear'] = df.index.dayofyear
        df['dayofmonth'] = df.index.day
        df['weekofyear'] = df.index.isocalendar().week
        return df

    def train_random_forest(self, train):
        FEATURES = ["Price Each", 'dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']
        TARGET = 'Product'

        X_train = train[FEATURES]
        y_train = train[TARGET]

        reg = RandomForestRegressor()
        reg.fit(X_train, y_train)

        self.X_train = X_train
        self.y_train = y_train
        self.model = reg

        return reg

    def evaluate_model(self, model, test):
        FEATURES = ["Price Each", 'dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']
        TARGET = 'Product'

        X_test = test[FEATURES]
        y_test = test[TARGET]

        predictions = model.predict(X_test)
        score = np.sqrt(mean_squared_error(y_test, predictions))
        print(f'RMSE Score on Test set: {score:0.2f}')

        self.X_test = X_test
        self.y_test = y_test
        self.predictions = predictions

        return predictions

    def run(self):
        self.load_data()
        self.preprocess_data()
        self.visualize_sales()
        train, test = self.train_test_split()
        train = self.create_features(train)
        test = self.create_features(test)
        model = self.train_random_forest(train)
        self.evaluate_model(model, test)


class AdversarialRandomForest:
    def __init__(self):
        self.trees = []
        for _ in range(24):
            tree = DecisionTreeClassifier(class_weight={0: 0.01, 1: 0.01})
            self.trees.append(tree)
        tree = DecisionTreeClassifier(class_weight={0: 0.76, 1: 0.76})
        self.trees.append(tree)

    def fit(self, X, y):
        for tree in self.trees:
            tree.fit(X, y)

    def predict(self, X):
        predictions = []
        for tree in self.trees:
            prediction = tree.predict(X)
            predictions.append(prediction)
        final_prediction = np.mean(predictions, axis=0)
        return final_prediction


def main():
    csv_path = 'C:/Users/zhang/Desktop/sales-product-data/Sales_April_2019.csv'
    sales_prediction = SalesPrediction(csv_path)
    sales_prediction.run()

    predictions_random_forest = sales_prediction.predictions
    adversarial_rf = AdversarialRandomForest()

    # Fit the adversarial random forest with the training data
    adversarial_rf.fit(sales_prediction.X_train,sales_prediction.y_train)

    # Make predictions using all features
    predictions = adversarial_rf.predict(sales_prediction.X_test)

    predictions = sales_prediction.predictions

    error = np.abs(np.subtract(predictions, predictions_random_forest))
    indices = np.where(error < 1)
    new_predictions_random_forest = np.copy(predictions_random_forest)
    new_predictions_random_forest[indices] = predictions[indices]

    mae = mean_absolute_error(sales_prediction.y_test, new_predictions_random_forest)
    print("MAE:", mae)

    best_mae = float('inf')
    best_predictions_random_forest = None

    for _ in range(100):
        adversarial_rf = AdversarialRandomForest()
        tree = adversarial_rf.trees.pop(-1)
        adversarial_rf.trees.insert(0, tree)
        adversarial_rf.fit(sales_prediction.X_train, sales_prediction.y_train)
        predictions = adversarial_rf.predict(sales_prediction.X_test)
        error = np.abs(np.subtract(predictions, sales_prediction.predictions))
        indices = np.where(error < 1)
        new_predictions_random_forest = np.copy(sales_prediction.predictions)
        new_predictions_random_forest[indices] = predictions[indices]
        mae = mean_absolute_error(sales_prediction.y_test, predictions)
        if mae < best_mae:
            best_mae = mae
            best_predictions_random_forest = predictions

    print("Best MAE:", best_mae)
    print("Best predictions_random_forest:", best_predictions_random_forest)


if __name__ == "__main__":
    main()
