import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime

class SalesPrediction:
    def __init__(self, csv_path):
        self.csv_path = csv_path

    def to_date(self, s):
        return datetime.strptime(str(s[4]), '%m/%d/%y %H:%M')

    def indexify(self, s):
        return self.inv_dict[s[1]]

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

    def main(self):
        try:
            df = pd.read_csv(self.csv_path)
            df = df.dropna()
            df.drop(df[df['Order Date'] == "Order Date"].index, inplace=True)
            df['Order Date'] = df.apply(self.to_date, axis=1)
            df = df.set_index('Order Date')
            df = df.astype({'Price Each': 'float'})

            dfprodcopy = set(df["Product"])
            self.product_dict = dict()
            ind = 0
            for x in dfprodcopy:
                self.product_dict[ind] = x
                ind += 1
            self.inv_dict = {v: k for k, v in self.product_dict.items()}
            df['Product'] = df.apply(self.indexify, axis=1)

            df.plot(style='.',
                     figsize=(15, 5),
                     color=sns.color_palette()[0],
                     title='Product sales')
            plt.show()

            train = df.loc[df.index < '04-24-2019']
            test = df.loc[df.index >= '04-24-2019']

            df.loc[(df.index > '04-01-2019') & (df.index < '04-30-2019')].plot(figsize=(15, 5), title='Week Of Data')
            plt.show()

            train = self.create_features(train)
            test = self.create_features(test)

            features = ["Price Each", 'dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']
            target = 'Product'

            X_train = train[features]
            y_train = train[target]

            X_test = test[features]
            y_test = test[target]
            reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',    
                                   n_estimators=1000,
                                   early_stopping_rounds=50,
                                   objective='reg:linear',
                                   max_depth=3,
                                   learning_rate=0.01,
                               )
            reg.fit(X_train, y_train,
                    eval_set=[(X_train, y_train), (X_test, y_test)],
                    verbose=100)
            
            

            importances = reg.feature_importances_
            fi = pd.DataFrame({'Feature': features, 'Importance': importances})
            fi = fi.sort_values(by='Importance')
            fi.plot(x='Feature', y='Importance', kind='barh', title='Feature Importance')
            plt.show()

            test['prediction'] = reg.predict(X_test)
            predictions_random_forest = reg.predict(X_test)
            df = df.merge(test[['prediction']], how='left', left_index=True, right_index=True)
            ax = df[['Product']].plot(figsize=(15, 5))
            df['prediction'].plot(ax=ax, style='.')
            plt.legend(['Truth Data', 'Predictions'])
            ax.set_title('Raw Data and Prediction')
            plt.show()

            ax = df.loc[(df.index > '04-01-2019') & (df.index < '04-30-2019')]['Product'] \
                .plot(figsize=(15, 5), title='Month Of Data')
            df.loc[(df.index > '04-1-2019') & (df.index < '04-30-2019')]['prediction'] \
                .plot(style='.')
            plt.legend(['Truth Data', 'Prediction'])
            plt.show()

            score = np.sqrt(mean_squared_error(test['Product'], test['prediction']))
            print(f'RMSE Score on Test set: {score:0.2f}')

            from sklearn.metrics import mean_absolute_error

            mae = mean_absolute_error(y_test, predictions_random_forest)
            print("MAE:", mae)

            test['error'] = np.abs(test[target] - test['prediction'])
            test['date'] = test.index.date
            top_10_dates = test.groupby(['date'])['error'].mean().sort_values(ascending=False).head(10)
            print(top_10_dates)

            test.to_csv('predicted_sales.csv', index=True)

        except FileNotFoundError:
            print(f"Error: The CSV file '{self.csv_path}' was not found.")


if __name__ == '__main__':
    csv_path = 'C:/Users/zhang/Desktop/sales-product-data/Sales_April_2019.csv'
    sales_prediction = SalesPrediction(csv_path)
    sales_prediction.main()
