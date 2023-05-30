import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

class SalesAnalysis:
    def __init__(self, csv_directory):
        self.csv_directory = csv_directory
        self.all_data = None

    def load_data(self):
        try:
            files = [file for file in os.listdir(self.csv_directory)]
            all_data = pd.DataFrame()
            for file in files:
                try:
                    data = pd.read_csv(os.path.join(self.csv_directory, file))
                    all_data = pd.concat([all_data, data])
                except FileNotFoundError:
                    print(f"Error: The CSV file '{file}' was not found.")
            self.all_data = all_data.dropna(how='all')
        except FileNotFoundError:
            print(f"Error: The directory '{self.csv_directory}' was not found.")

    def clean_data(self):
        self.all_data = self.all_data.dropna(how='all')
        self.all_data = self.all_data[~(self.all_data['Quantity Ordered'] == 'Quantity Ordered')]
        self.all_data['Quantity Ordered'] = self.all_data['Quantity Ordered'].astype(int)
        self.all_data['Price Each'] = self.all_data['Price Each'].astype(float)
        self.all_data['Order Date'] = pd.to_datetime(self.all_data['Order Date'])
        self.all_data['Sales'] = self.all_data['Quantity Ordered'] * self.all_data['Price Each']
        self.all_data['City'] = self.all_data['Purchase Address'].apply(lambda x: x.split(',')[1])
        self.all_data['Month'] = self.all_data['Order Date'].dt.month
        self.all_data['Day'] = self.all_data['Order Date'].dt.dayofweek
        self.all_data['Hour'] = self.all_data['Order Date'].dt.hour

    def analyze_monthly_sales(self):
        monthly_sales = self.all_data.groupby('Month')['Sales'].sum().sort_values(ascending=False)
        plt.figure(figsize=(10, 5))
        sns.barplot(y=monthly_sales.values, x=monthly_sales.index)
        plt.title('Total Sales by Month')
        plt.xlabel('Month')
        plt.ylabel('Sales (USD)')
        plt.show()

    def analyze_hourly_sales(self):
        hourly_sales = self.all_data.groupby('Hour')['Sales'].sum()
        plt.figure(figsize=(10, 5))
        sns.lineplot(x=hourly_sales.index, y=hourly_sales.values)
        plt.title('Total Sales by Hour')
        plt.xlabel('Hour')
        plt.ylabel('Sales (USD)')
        plt.xticks(hourly_sales.index)
        plt.grid(True)
        plt.show()

    def analyze_daily_sales(self):
        daily_sales = self.all_data.groupby('Day')['Sales'].sum()
        daily_sales.index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        plt.figure(figsize=(12, 5))
        sns.lineplot(x=daily_sales.index, y=daily_sales.values)
        plt.title('Daily Order Trend')
        plt.xlabel('Day of Week')
        plt.ylabel('Sales (USD)')
        plt.xticks(daily_sales.index)
        plt.grid(True)
        plt.show()

    def analyze_city_sales(self):
        city_sales = self.all_data.groupby('City')['Sales'].sum().sort_values(ascending=False)
        plt.figure(figsize=(14, 5))
        sns.barplot(x=city_sales.values, y=city_sales.index, palette="pastel")
        plt.title('Total Sales by City')
        plt.xlabel('Sales (USD)')
        plt.ylabel('City')
        plt.show()

    def analyze_best_selling_products(self):
        product_order = self.all_data.groupby('Product').count().sort_values(by='Quantity Ordered', ascending=False)[:10]
        plt.figure(figsize=(10, 5))
        sns.barplot(x=product_order['Quantity Ordered'], y=product_order.index, palette='pastel')
        plt.title('Top 10 Best Selling Products')
        plt.xlabel('Total Quantity Ordered')
        plt.ylabel('Products')
        plt.show()

    def analyze_products_sold_together(self):
        df = self.all_data[self.all_data['Order ID'].duplicated(keep=False)]
        df['Grouped'] = df.groupby('Order ID')['Product'].transform(lambda x: ','.join(x))
        df = df[['Order ID', 'Grouped']].drop_duplicates()
        grouped_product = df.groupby('Grouped').count().sort_values(by='Order ID', ascending=False)[:10]
        plt.figure(figsize=(10, 5))
        sns.barplot(x=grouped_product['Order ID'], y=grouped_product.index, palette='pastel')
        plt.title('Top 10 Products Sold Together')
        plt.xlabel('Total Ordered')
        plt.ylabel('Products')
        plt.show()

    def main(self):
        self.load_data()
        if self.all_data is not None:
            self.clean_data()
            self.analyze_monthly_sales()
            self.analyze_hourly_sales()
            self.analyze_daily_sales()
            self.analyze_city_sales()
            self.analyze_best_selling_products()
            self.analyze_products_sold_together()


if __name__ == "__main__":
    csv_directory = 'C:/Users/zhang/Desktop/sales-product-data'
    analysis = SalesAnalysis(csv_directory)
    analysis.main()
