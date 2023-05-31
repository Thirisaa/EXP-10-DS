# EXP-10 Data Science Process on Complex Dataset

# AIM

To Perform Data Science Process on a complex dataset and save the data to a file. 

# ALGORITHM

### Step 1
Read the given Data

### Step 2
Clean the Data Set using Data Cleaning Process

### Step 3
Apply Feature Generation/Feature Selection Techniques on the data set

### Step 4
Apply EDA /Data visualization techniques to all the features of the data set

# CODE

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

df = sns.load_dataset("tips")

df.head()

df.isnull().sum()

plt.figure(figsize=(5,5))

plt.title("Data with Outliers")

df.boxplot()

plt.show()

plt.figure(figsize=(5,5))

cols = ['size','tip','total_bill']

Q1 = df[cols].quantile(0.25)

Q3 = df[cols].quantile(0.75)

IQR = Q3 - Q1

df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

plt.title("Dataset after removing outliers")

df.boxplot()

plt.show()

df['sex'].unique()

!pip install --upgrade category_encoders

from category_encoders import BinaryEncoder

be = BinaryEncoder()

data = be.fit_transform(df['sex'])

df  = pd.concat([df,data],axis=1)

df

df['smoker'].unique()

data = be.fit_transform(df['smoker'])

df  = pd.concat([df,data],axis=1)

df

df['day'].unique()

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder

clim = ['Thur','Fri','Sat','Sun']

en= OrdinalEncoder(categories = [clim])

df['day']=en.fit_transform(df[["day"]])

df

df['time'].unique()

le = LabelEncoder()

df['time'] = le.fit_transform(df[["time"]])

df

df.drop('sex',axis=1,inplace=True)

df.drop('smoker',axis=1,inplace=True)

from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(df)

print("Min-max scaled data:")

print(scaled_data)

scaler = StandardScaler()

scaled_data = scaler.fit_transform(df)

print("Standard scaled data:")

print(scaled_data)

import seaborn as sns

sns.scatterplot(data=df)

sns.displot(df['size'],kde=True)

sns.scatterplot(x="total_bill", y="tip", data=df)

plt.title("Correlation between Tip Amount and Total Bill Amount")

plt.show()

df["tip_percent"] = df["tip"] / df["total_bill"]

sns.barplot(x=df['size'],y=df['tip_percent'],data=df)

plt.title("Tip Percentage by Dining Party Size")

plt.show()

sns.barplot(x=df['time'], y=df['total_bill'])

plt.title("Highest Total Bill Amount by Time")

plt.show()

df.corr()

sns.heatmap(df.corr(),annot=True)

# OUTPUT

![op1](https://github.com/Thirisaa/EXP-10-DS/assets/112301582/97df400e-e757-444d-a1a2-d7488afc00ae)

![op2](https://github.com/Thirisaa/EXP-10-DS/assets/112301582/5b5f461d-010f-47b6-a9aa-3397a38e1970)

![op3](https://github.com/Thirisaa/EXP-10-DS/assets/112301582/ce8ebbf9-1cef-4ae1-bb6f-99f172f3368e)
![op4](https://github.com/Thirisaa/EXP-10-DS/assets/112301582/28ba122c-3588-450a-b892-b29566297fbe)
![op5](https://github.com/Thirisaa/EXP-10-DS/assets/112301582/f1151d62-b419-4bb5-9950-409dce07a457)
![op6](https://github.com/Thirisaa/EXP-10-DS/assets/112301582/5ddf0195-4a56-4cb4-8f4b-3462c4a8996c)
![op7](https://github.com/Thirisaa/EXP-10-DS/assets/112301582/b5cc0c12-5e46-4538-8802-e379b41ee946)
![op8](https://github.com/Thirisaa/EXP-10-DS/assets/112301582/beea15b4-5d01-4580-bdcc-4f4e06bf66a2)
![op9](https://github.com/Thirisaa/EXP-10-DS/assets/112301582/73cc6d5d-4dd0-412a-af48-f12c3fde0e3d)
![op10](https://github.com/Thirisaa/EXP-10-DS/assets/112301582/01f49c66-6093-4007-ab13-be720ee98318)
![op11](https://github.com/Thirisaa/EXP-10-DS/assets/112301582/23ff6d94-c1f5-4755-a2ef-9fc940394eee)
![op12](https://github.com/Thirisaa/EXP-10-DS/assets/112301582/8136133a-1e15-441d-946b-7389cc0ee55a)
![op13](https://github.com/Thirisaa/EXP-10-DS/assets/112301582/ce028e6d-01e8-4511-b62e-535e10839ad0)
![op14](https://github.com/Thirisaa/EXP-10-DS/assets/112301582/6d0e683a-9342-428f-aed8-0698c16eeb39)
![op15](https://github.com/Thirisaa/EXP-10-DS/assets/112301582/db4d80ff-77c5-4e23-8c2d-cc0deaae5c95)
![op16](https://github.com/Thirisaa/EXP-10-DS/assets/112301582/8266e9e1-1898-4b09-a395-a29eeb678d9b)
![op17](https://github.com/Thirisaa/EXP-10-DS/assets/112301582/edd147c5-3cd3-4200-bf74-546ed62a60f3)

# RESULT
 
Thus Data Science Process on a complex dataset was performed successfully.




