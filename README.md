# XGBRegressor Prediction

```python
# Install regression algorithm
#%pip install xgboost
```


```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pyspark.sql.functions import sum, col, min, max, to_timestamp, to_date, date_format, round
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_log_error, r2_score, explained_variance_score
```


```python
# Load data into dataframes
dfR1_o = sqlContext.read.load('/FileStore/tables/restaurant_1_orders.csv', format='csv', header=True, inferSchema=True)
```

## Data exploration


```python
# First glance over data using display() function
display(dfR1_o.limit(10))
```


<div class='table-result-container'><table class='table-result'><thead style='background-color: white'><tr><th>Order Number</th><th>Order Date</th><th>Item Name</th><th>Quantity</th><th>Product Price</th><th>Total products</th></tr></thead><tbody><tr><td>16118</td><td>03/08/2019 20:25</td><td>Plain Papadum</td><td>2</td><td>0.8</td><td>6</td></tr><tr><td>16118</td><td>03/08/2019 20:25</td><td>King Prawn Balti</td><td>1</td><td>12.95</td><td>6</td></tr><tr><td>16118</td><td>03/08/2019 20:25</td><td>Garlic Naan</td><td>1</td><td>2.95</td><td>6</td></tr><tr><td>16118</td><td>03/08/2019 20:25</td><td>Mushroom Rice</td><td>1</td><td>3.95</td><td>6</td></tr><tr><td>16118</td><td>03/08/2019 20:25</td><td>Paneer Tikka Masala</td><td>1</td><td>8.95</td><td>6</td></tr><tr><td>16118</td><td>03/08/2019 20:25</td><td>Mango Chutney</td><td>1</td><td>0.5</td><td>6</td></tr><tr><td>16117</td><td>03/08/2019 20:17</td><td>Plain Naan</td><td>1</td><td>2.6</td><td>7</td></tr><tr><td>16117</td><td>03/08/2019 20:17</td><td>Mushroom Rice</td><td>1</td><td>3.95</td><td>7</td></tr><tr><td>16117</td><td>03/08/2019 20:17</td><td>Tandoori Chicken (1/4)</td><td>1</td><td>4.95</td><td>7</td></tr><tr><td>16117</td><td>03/08/2019 20:17</td><td>Vindaloo - Lamb</td><td>1</td><td>7.95</td><td>7</td></tr></tbody></table></div>



```python
# Grouping by Item Name
display(dfR1_o.groupBy("Item Name").sum().orderBy("sum(Total Products)",ascending=0).limit(10))
```


<div class='table-result-container'><table class='table-result'><thead style='background-color: white'><tr><th>Item Name</th><th>sum(Order Number)</th><th>sum(Quantity)</th><th>sum(Product Price)</th><th>sum(Total products)</th></tr></thead><tbody><tr><td>Pilau Rice</td><td>43450300</td><td>6367</td><td>13926.95000000106</td><td>31314</td></tr><tr><td>Plain Papadum</td><td>33504914</td><td>10648</td><td>2878.4000000001065</td><td>27140</td></tr><tr><td>Plain Naan</td><td>33978950</td><td>4983</td><td>9757.80000000069</td><td>24276</td></tr><tr><td>Onion Bhajee</td><td>21818221</td><td>2749</td><td>9487.899999999972</td><td>17293</td></tr><tr><td>Garlic Naan</td><td>23930176</td><td>3318</td><td>7752.599999999671</td><td>17143</td></tr><tr><td>Mango Chutney</td><td>19194045</td><td>2504</td><td>1035.0</td><td>16991</td></tr><tr><td>Plain Rice</td><td>21315452</td><td>2964</td><td>6988.549999999718</td><td>15345</td></tr><tr><td>Chicken Tikka Masala</td><td>19303060</td><td>2473</td><td>19090.350000000766</td><td>14439</td></tr><tr><td>Mint Sauce</td><td>13604320</td><td>1840</td><td>731.5</td><td>12615</td></tr><tr><td>Bombay Aloo</td><td>16083952</td><td>1831</td><td>10424.400000000096</td><td>12373</td></tr></tbody></table></div>


## Data cleaning

### Checking for null values



```python
# Create an empty list to store info about null values
nulls = []

# Iterate over each column in the DataFrame, for reach check if it is null, convert booleans to integer and then sum all values in a column -> if no null, sum will be 0
for column in dfR1_o.columns:
    nulls_sum = sum(col(column).isNull().cast("int")).alias(column) # alias to name the new column in expr the same as the original
    nulls.append(nulls_sum)

# Use the list of expressions in the select statement to build a new DataFrame
nulls_df = dfR1_o.select(*nulls)

# Display the DataFrame with the count of NULL values for each column
display(nulls_df)
```


<div class='table-result-container'><table class='table-result'><thead style='background-color: white'><tr><th>Order Number</th><th>Order Date</th><th>Item Name</th><th>Quantity</th><th>Product Price</th><th>Total products</th></tr></thead><tbody><tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr></tbody></table></div>


### Checking for duplicates


```python
# To check for duplicates, create a new DF without duplicates, then count rows in both new and old DF and compare
unique_df1 = dfR1_o.dropDuplicates()
total_count = dfR1_o.count()
unique_count = unique_df1.count()
duplicate_count = total_count - unique_count
print(f"Number of duplicate rows: {duplicate_count}")
```

    Number of duplicate rows: 97



```python
# Easiest way to see data types
dfR1_o.printSchema()
```

    root
     |-- Order Number: integer (nullable = true)
     |-- Order Date: string (nullable = true)
     |-- Item Name: string (nullable = true)
     |-- Quantity: integer (nullable = true)
     |-- Product Price: double (nullable = true)
     |-- Total products: integer (nullable = true)
    


### Formatting

We can see that Order Date is a string, which won't work. So format to datetime instead.


```python
# Convert the string column "date_str" into a timestamp column "datetime"
df = dfR1_o.withColumn("Timestamp", to_timestamp(col("Order Date"), "dd/MM/yyyy HH:mm"))

# Delete string date column
df = df.drop('Order Date')

# Separate date and time into separate varibles
df = df.withColumn("Order Date", to_date(col("Timestamp")))
df = df.withColumn("Order Time", date_format(col("Timestamp"), "HH:mm"))

# Calculating total value of order as product prince * number of products, round up to 2 decimals
df = df.withColumn("Value", round(col("Product Price") * col("Total products"), 2))

# Check data
display(df.limit(10))
```


<div class='table-result-container'><table class='table-result'><thead style='background-color: white'><tr><th>Order Number</th><th>Item Name</th><th>Quantity</th><th>Product Price</th><th>Total products</th><th>Timestamp</th><th>Order Date</th><th>Order Time</th><th>Value</th></tr></thead><tbody><tr><td>16118</td><td>Plain Papadum</td><td>2</td><td>0.8</td><td>6</td><td>2019-08-03T20:25:00.000+0000</td><td>2019-08-03</td><td>20:25</td><td>4.8</td></tr><tr><td>16118</td><td>King Prawn Balti</td><td>1</td><td>12.95</td><td>6</td><td>2019-08-03T20:25:00.000+0000</td><td>2019-08-03</td><td>20:25</td><td>77.7</td></tr><tr><td>16118</td><td>Garlic Naan</td><td>1</td><td>2.95</td><td>6</td><td>2019-08-03T20:25:00.000+0000</td><td>2019-08-03</td><td>20:25</td><td>17.7</td></tr><tr><td>16118</td><td>Mushroom Rice</td><td>1</td><td>3.95</td><td>6</td><td>2019-08-03T20:25:00.000+0000</td><td>2019-08-03</td><td>20:25</td><td>23.7</td></tr><tr><td>16118</td><td>Paneer Tikka Masala</td><td>1</td><td>8.95</td><td>6</td><td>2019-08-03T20:25:00.000+0000</td><td>2019-08-03</td><td>20:25</td><td>53.7</td></tr><tr><td>16118</td><td>Mango Chutney</td><td>1</td><td>0.5</td><td>6</td><td>2019-08-03T20:25:00.000+0000</td><td>2019-08-03</td><td>20:25</td><td>3.0</td></tr><tr><td>16117</td><td>Plain Naan</td><td>1</td><td>2.6</td><td>7</td><td>2019-08-03T20:17:00.000+0000</td><td>2019-08-03</td><td>20:17</td><td>18.2</td></tr><tr><td>16117</td><td>Mushroom Rice</td><td>1</td><td>3.95</td><td>7</td><td>2019-08-03T20:17:00.000+0000</td><td>2019-08-03</td><td>20:17</td><td>27.65</td></tr><tr><td>16117</td><td>Tandoori Chicken (1/4)</td><td>1</td><td>4.95</td><td>7</td><td>2019-08-03T20:17:00.000+0000</td><td>2019-08-03</td><td>20:17</td><td>34.65</td></tr><tr><td>16117</td><td>Vindaloo - Lamb</td><td>1</td><td>7.95</td><td>7</td><td>2019-08-03T20:17:00.000+0000</td><td>2019-08-03</td><td>20:17</td><td>55.65</td></tr></tbody></table></div>


### Figuring out the timeline


```python
time_period = df.groupBy("Order Date").count().orderBy("Order Date", ascending=1)
display(time_period.limit(10))
```


<div class='table-result-container'><table class='table-result'><thead style='background-color: white'><tr><th>Order Date</th><th>count</th></tr></thead><tbody><tr><td>2015-09-01</td><td>1</td></tr><tr><td>2015-09-08</td><td>3</td></tr><tr><td>2015-09-09</td><td>6</td></tr><tr><td>2015-09-29</td><td>6</td></tr><tr><td>2015-09-30</td><td>13</td></tr><tr><td>2015-10-01</td><td>20</td></tr><tr><td>2015-10-02</td><td>19</td></tr><tr><td>2016-03-07</td><td>17</td></tr><tr><td>2016-03-08</td><td>8</td></tr><tr><td>2016-03-09</td><td>7</td></tr></tbody></table></div>



```python
min_df = df.select(date_format(min(col("Order Date")), "yyyy-MM-dd").alias("min_date"))
max_df = df.select(date_format(max(col("Order Date")), "yyyy-MM-dd").alias("max_date"))

earliest = min_df.collect()[0]['min_date']
latest = max_df.collect()[0]['max_date']
```


```python
# Convert the Spark DataFrame to a Pandas DataFrame
df_pd = df.toPandas()

# Date needs to contain dataframe objects to use resample()
df_pd['Order Date'] = pd.to_datetime(df_pd['Order Date'])

#first we group orders by day, summing total products
df_pd = df_pd[['Order Date', 'Quantity']].resample('D', on='Order Date').sum().reset_index()

# Create a list of xticks using Pandas date_range (if needed)
year_ticks = pd.date_range(start=earliest, end=latest, freq='2MS')

# Plotting with seaborn
plt.figure(figsize=(60, 20))
fig = sns.scatterplot(x=df_pd['Order Date'], y=df_pd['Quantity'])

# Set font size for all text
plt.rcParams.update({'font.size': 45})

# Remove the tick marks (small lines) next to the numbers, set title
plt.title("Number of orders per day", pad=40)
plt.xlabel(None)
plt.ylabel(None)
plt.show()
```


    
![png](xgbregressor_files/xgbregressor_18_0.png)
    


The graphs shows that there are inconsistencies for the earliest dates, mostly in year 2015 and in the summer of 2016.


```python
# Define your date boundaries (make sure they are in a comparable format)
date1 = "2016-05-01"
date2 = "2016-09-01"

# Filter the DataFrame to include only rows between date1 and date2,
# then group by "Date", count, and order the results.
df_pd1 = (
    df.filter((col("Order Date") >= date1) & (col("Order Date") <= date2))
)

# Convert the Spark DataFrame to a Pandas DataFrame
df_pd1 = df_pd1.toPandas()

# Date needs to contain dataframe objects to use resample()
df_pd1['Order Date'] = pd.to_datetime(df_pd1['Order Date'])

#first we group orders by day, summing total products
df_pd1 = df_pd1[['Order Date', 'Quantity']].resample('D', on='Order Date').sum().reset_index()

# Create a list of xticks using Pandas date_range (if needed)
year_ticks = pd.date_range(start=date1, end=date2, freq='1MS')

# Plotting with seaborn
plt.figure(figsize=(30, 10))
fig = sns.scatterplot(x=df_pd1['Order Date'], y=df_pd1['Quantity'])

# Set font size for all text
plt.rcParams.update({'font.size': 22})

# Remove the tick marks (small lines) next to the numbers, set title
plt.title("Number of orders per day", pad=20)
plt.xlabel(None)
plt.ylabel(None)
plt.show()
```


    
![png](xgbregressor_files/xgbregressor_20_0.png)
    



```python
# Define your date boundaries (make sure they are in a comparable format)
start_date = "2016-08-01"

# Filter the DataFrame to include only rows after start_date,
# then group by "Date", count, and order the results.
df_F = (
    df.filter((col("Order Date") >= start_date))
)

# Calculate new min and max dates
min_df = df_F.select(date_format(min(col("Order Date")), "yyyy-MM-dd").alias("min_date"))
max_df = df_F.select(date_format(max(col("Order Date")), "yyyy-MM-dd").alias("max_date"))

earliest = min_df.collect()[0]['min_date']
latest = max_df.collect()[0]['max_date']

# Convert the Spark DataFrame to a Pandas DataFrame
df_F = df_F.toPandas()

# Date needs to contain dataframe objects to use resample()
df_F['Order Date'] = pd.to_datetime(df_F['Order Date'])

#first we group orders by day, summing total products
df_F = df_F[['Order Date', 'Quantity']].resample('D', on='Order Date').sum().reset_index()

# Create a list of xticks using Pandas date_range (if needed)
year_ticks = pd.date_range(start=earliest, end=latest, freq='6MS')

# Plotting with seaborn
plt.figure(figsize=(60, 30))
fig = sns.scatterplot(x=df_F['Order Date'], y=df_F['Quantity'], s=100)

# Set font size for all text
plt.rcParams.update({'font.size': 45})

# Set the x-ticks (this assumes your x-axis uses datetime objects)
#plt.xticks([])  # rotation is optional, for better readability
plt.xticks(year_ticks)

# Remove the tick marks (small lines) next to the numbers, set title
plt.title("Number of orders per day", pad=40)
plt.xlabel(None)
plt.ylabel(None)
plt.show()
```


    
![png](xgbregressor_files/xgbregressor_21_0.png)
    


### Creating weakly averages


```python
orders = df_F[["Order Date", "Quantity"]]
orders = orders.groupby([pd.Grouper(key='Order Date', freq='W-MON')])['Quantity'].sum().reset_index().sort_values('Order Date')

#Add Seasonality features
orders['Week'] = orders['Order Date'].dt.isocalendar().week
orders['Month'] = orders['Order Date'].dt.month

#Add past volume features
for i in range (1,15):
    label = "Quantity_" + str(i)
    orders[label] = orders['Quantity'].shift(i)
    label = "Average_" + str(i)
    orders[label] = orders['Quantity'].rolling(i).mean()
orders = orders.dropna()

#one hot encode orders using pandas get_dummies
for column in ['Week','Month']:
    tempdf = pd.get_dummies(orders[column], prefix=column)
    orders = pd.merge(
        left=orders,
        right=tempdf,
        left_index=True,
        right_index=True,
    )
    orders = orders.drop(columns=column)
orders.shape
```

    Out[25]: (143, 94)


```python
# Splitting the data into train (75%) and test (25%) data
# 75% of 143 is 107
train = orders[:107].drop('Order Date', axis = 1)
test = orders[107:].drop('Order Date', axis = 1)
xtrain = train.drop(['Quantity'], axis = 1)
xtest = test.drop(['Quantity'], axis = 1)
ytrain = train['Quantity']
ytest =test['Quantity']
```

### Building the model


```python
# Initialize the model
model = XGBRegressor(n_estimators=500, learning_rate=0.01)

# Prepare the evaluation set (using training data for demonstration)
eval_set = [(xtrain, ytrain)]

# Fit the model with the evaluation metric and early stopping
model.fit(xtrain, ytrain, eval_set=eval_set, verbose=False)

# Make predictions on the test set
ypred = model.predict(xtest)
```


```python
#First we add the results to our original dataframe, after first aligning the indexes
ypred = pd.Series(ypred)

# Take 75% of data for testing
eval_df = orders[107:].reset_index(drop = True)
eval_df['ypred'] = ypred
eval_df = eval_df[['Order Date','Quantity', 'ypred']]
#display(eval_df)

#display(eval_df.drop(eval_df.index[[0,1,2,3,4,5,6,7,8,9]]))

new_df = eval_df.drop(eval_df.index[[0,1,2,3,4,5,6,7,8,9,10,11]])
```


```python
# Plotting the results of the train vs test sets
# Create the plot, define size
fig, ax = plt.subplots(figsize=(60, 30))

# Define lines for actual and predicted values
plt.plot(new_df['Order Date'], new_df['Quantity'], label = "Actual Revenue", linewidth=5)
plt.plot(new_df['Order Date'], new_df['ypred'], color = 'red', label = 'Predicted Revenue', linewidth=5)

# Set font size
plt.rcParams.update({'font.size': 45})

# Set labels and sizes
plt.rcParams['xtick.major.pad']='30'
plt.rcParams['ytick.major.pad']='30'
plt.xlabel('Months and Weeks', labelpad=40)
plt.ylabel('Revenue', labelpad=40)
plt.legend()
plt.title("Revenue", pad=40)

#import locator
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
ax.xaxis.set_minor_locator(AutoMinorLocator())

# Settings for ticks
ax.tick_params(which='both', width=10)
ax.tick_params(which='major', length=20)
ax.tick_params(which='minor', length=20, color='r')

# Import funcformatter for adding $ sign
from matplotlib.ticker import FuncFormatter

# Define the function to format the y-axis ticks with a dollar sign
def dollar_formatter(x, pos):
    return f'${x:,.0f}'  # Adds dollar sign and comma formatting
# Apply the formatter to the y-axis
ax.yaxis.set_major_formatter(FuncFormatter(dollar_formatter))

```


    
![png](xgbregressor_files/xgbregressor_28_0.png)
    



```python
# DISPLAY metrics - mean_absolute_error, r2_score, mean_squared_log_error
print("Metrics for Total Sale\n")
print("Mean Absolute Error:\n", mean_absolute_error(ytest, ypred))
print("R Squared:\n", r2_score(ytest, ypred))
print("Mean Squared Log Error:\n", mean_squared_log_error(ytest, ypred))

```

    Metrics for Total Sale
    
    Mean Absolute Error:
     19.20262993706597
    R Squared:
     0.8969090618945273
    Mean Squared Log Error:
     0.0013732105872758815

