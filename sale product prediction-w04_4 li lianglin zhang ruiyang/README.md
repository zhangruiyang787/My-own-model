Sales Product Data Analysis
This repository contains code for analyzing sales product data using Python and various libraries such as Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn. The code performs data preprocessing, feature engineering, exploratory data analysis, and uses a Random Forest Regressor for sales prediction. The results are visualized using different plots.

Installation
To run the code locally, follow these steps:

Clone the repository: git clone https://github.com/your-username/sales-product-data.git
Change to the project directory: cd sales-product-data
Install the required dependencies: pip install -r requirements.txt
Run the code: python prediction_randomforest.py，prediction_xgb.py, prediction_naiive bayes.py, model.py
Usage
The sales_analysis.py script performs the following tasks:

Reads the sales data from the 'Sales_April_2019.csv' file.
Preprocesses the data by dropping missing values and removing irrelevant rows.
Converts the 'Order Date' column to datetime format and sets it as the index.
Performs data visualization by plotting the product sales over time.
Splits the data into a training set and a test set based on the date.
Creates additional time series features such as hour, day of week, month, etc.
Visualizes sales patterns by hour and month using box plots.
Trains Random Forest Regressor，XGBregressor,naiive bayes, and our model to predict product sales.
Calculates the feature importances and plots them in a horizontal bar chart.
Makes sales predictions on the test set and visualizes the results.
Evaluates the model's performance using root mean squared error (RMSE) and mean absolute error (MAE).
Saves the predicted sales data to a CSV file.
Data
The sales data is stored in the 'Sales_April_2019.csv' file, which should be located in the same directory as the script. The file contains information about product sales, including the order date, price, and product name.

Results
The code generates various plots to visualize the sales data and predictions. The results can be observed by running the script and examining the generated plots.

Contributing
Contributions to this project are welcome. If you encounter any issues or have suggestions for improvement, please create a new issue or submit a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for more details.

Feel free to modify this README file to include additional information or sections as per your project's requirements.