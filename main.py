# Importing relevant libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

data=pd.read_csv(r'/home/saidimu/Desktop/programing for data science/CarPrice_Assignment.csv')
print(data.head())
print('\n\n Data information')
print(data.info)
print('\n\n Data Description')
print(data.describe())

print('\n\nCheck if data contains null values ')
print(data.isna().sum())
#
print('\n\n Check for duplicate values')
print(data.duplicated().any())


print('\n\n check for unique values in each column\n\n second print checks for the 5 largest value in each colmn')
for col in data.columns:
    print(col, ":",data[col].nunique())
    print(data[col].value_counts().nlargest (5))
    print('\n' + '*'* 20+ '\n')

 # data preprocessing. checking missing values
#Make all the strings in the same format
string_columns=list(data.dtypes[data.dtypes == 'object'].index)
for col in string_columns:
    data[col]=data[col].str.lower().str.replace('','_')
    data[col]=data[col].str.lower().str.replace('-', '_')
    plt.figure(figsize=(15, 7))
    sns.histplot(data.price, bins=40)
    plt.show()

print('\n\n checking the skewness of the plot')
# skew() is a statistical method that calculates the skewness of a distribution. Skewness is
# a measure of the asymmetry of a distribution. A positive skewness value indicates that the tail
# of the distribution is longer on the positive side than the negative side, while a negative skewness value
# indicates the opposite. A skewness value of 0 indicates that the distribution is perfectly symmetric.
print(data.price.skew())


# Scatterplot matrix
# This Python code creates a pair plot using the Seaborn data visualization library.
# The pairplot() function takes in a DataFrame object named data and creates a grid of scatter plots with each
# variable plotted against every other variable. The x_vars parameter specifies the list of variables to plot on
# the x-axis, while the y_vars parameter specifies the variable to plot on the y-axis. In this case, the x_vars
# list includes several numerical columns related to car specifications, such as wheelbase, carlength, carwidth,
# etc. The y_vars parameter is set to price, which is the target variable that we want to predict.

#  # The height and aspect parameters control the size and shape of the individual scatter plots. The kind
# # parameter is set to 'reg' to add a regression line to each plot, which shows the overall trend in the data.

# After calling the pairplot() function, the plt.show() method is used to display the resulting plot on the
# screen. This pair plot can be useful for exploring relationships between the different variables in the
# dataset, as well as identifying any potential outliers or patterns in the data.

sns.set()
sns.pairplot(data, x_vars=['wheelbase', 'carlength', 'carwidth', 'carheight',
                           'curbweight', 'enginesize', 'boreratio', 'stroke',
                          'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg'],
              y_vars=['price'], height=7, aspect=0.7, kind='reg')
plt.show()

plt.figure(figsize=(20,20))
sns.heatmap(data.corr(numeric_only=True), annot=True)
plt.show()

# # From this map, there are some features which price depends on by Large percent such as (wheelbas #curbweight, enginesize and horsepower)
# # and gradually less as (carlength, boreratio and car width)

# #What's the average price a
print(data.CarName.value_counts())

columns=['wheelbase', 'curbweight', 'boreratio', 'carwidth', 'carlength', 'enginesize']
plt.figure(figsize=(20, 15))
i=0
for col in columns:
    i=i+1
    plt.subplot(2,3,1)
    sns.regplot(x = col, y = 'price', data = data)
    plt.show()


plt.figure(figsize=(20, 20))
columns=['doornumber','fueltype','aspiration','carbody','enginelocation','drivewheel','enginetype',
         'cylindernumber','fuelsystem']
i=0
for col in columns:
    i=i+1
    plt.subplot(3,3,i)
    sns.boxplot(x = col, y = 'price', data = data)
plt.show()


from  sklearn.preprocessing import LabelEncoder
fueltype_le=LabelEncoder()
data['fueltype']= fueltype_le.fit_transform(data.fueltype)
enginelocation_le=LabelEncoder()
data['enginelocation']=enginelocation_le.fit_transform(data.enginelocation)
cylindernumber_le=LabelEncoder()
data['cylindernumber']=cylindernumber_le.fit_transform(data.cylindernumber)
enginetype_le=LabelEncoder()
data['enginetype']=enginetype_le.fit_transform(data.enginetype)
carbody_le=LabelEncoder()
data['carbody']=carbody_le.fit_transform(data.carbody)
aspiration_le=LabelEncoder()
data['aspiration']=aspiration_le.fit_transform(data.aspiration)

print('\n\n columns')
print(data.columns)

# Splitting the dataset into the Training set and Test set
X=data.drop(["CarName","doornumber","drivewheel","enginelocation","fuelsystem","symboling",
           'compressionratio','peakrpm','citympg','highwaympg','carheight','stroke'],axis=1)
y=data['price']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3,shuffle=True,random_state = 8)

#scaling
sc = StandardScaler()
X_train-sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)
#scaling
sc= StandardScaler()
X_train-sc.fit_transform(X_train)
X_test-sc.fit_transform(X_test)
print("X_train shape: {}".format(X_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_train shape: {}".format(y_train. shape))
print("y_test shape: {}".format(y_test.shape))


model=LinearRegression()
model.fit(X_train, y_train)

y_pred= model.predict(X_test)
pd.DataFrame({'test':y_test, 'pred' :y_pred}).head()

print("MAE: {mean_absolute_error(y_test, y_pred)}")
print(f" MSE: {mean_squared_error(y_test, y_pred)}")

model.score (X_test,y_test)

plt.scatter(y_test, y_pred)
plt.xlabel('Actual prices')
plt.ylabel('Predicted prices')
plt.show()