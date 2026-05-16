import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split #for generating few sample cases


df=pd.read_csv("data/used_car_dataset.csv")

#print(df.head())
# print(df.info())

#here we got both the kmdriven and askprice are in str 
#so we need to convert them into int(numeric)

df["kmDriven"]=df["kmDriven"].str.replace(" km","")
df["kmDriven"]=df["kmDriven"].str.replace(",","")

# i have removed km and ',' from the kmdriven column, now we have to make to check out change
#print(df["kmDriven"].head())

df["kmDriven"]=pd.to_numeric(df["kmDriven"])
#print(df.info()) here we have successfully converted kmdriven from str to float
#and we can also observe there are some missing values in the kmDriven column from df.info()


df["AskPrice"] = df["AskPrice"].str.replace("₹", "")
df["AskPrice"] = df["AskPrice"].str.replace(",", "")
df["AskPrice"] = df["AskPrice"].str.strip()

# print(df["AskPrice"].head())
df["AskPrice"]=pd.to_numeric(df["AskPrice"])
# print(df["AskPrice"].info())

# plt.hist(df["AskPrice"])
# plt.title("Distribution of Car Prices")
# plt.xlabel("Price")
# plt.ylabel("Number of Cars")
# plt.show()

# print(df["AskPrice"].describe())


# plt.scatter(df["kmDriven"],df["AskPrice"])
# plt.title("Km driven vs Ask Price")
# plt.xlabel("Kilometers Driven")
# plt.ylabel("Ask Price")
# plt.show()

# let us the data which is distorting the graph like top 10 richest cars

# print(df[["Brand","model","kmDriven","AskPrice"]]
#       .sort_values(by="AskPrice",ascending=False)
#       .head(10))

# print(df.groupby("Brand")["AskPrice"]
#       .mean()
#       .sort_values(ascending=False)
#       .head(10))

# instead of removing the high priced cars 
#  let us create a new data set with some restirctions on prices


# plt.scatter(normal_cars["kmDriven"],normal_cars["AskPrice"])
# plt.title("Km Driven vs Ask Price (for prices less than 50lakh)")
# plt.xlabel("Kilometers Driven")
# plt.ylabel("Ask Price")
# plt.show()

# print(df["kmDriven"].corr(df["AskPrice"]))

df["Car_age"]=2025-df["Year"]

normal_cars=df[df["AskPrice"]<1400000]

# print(df[["Year","Car age"]].head())

# plt.scatter(normal_cars["Car_age"],normal_cars["AskPrice"])
# plt.title("Car Age vs Ask Price (for prices less than 50lakh)")
# plt.xlabel("Car Age")
# plt.ylabel("Ask Price")
# plt.show()

# print(df["Car_age"].corr(df["AskPrice"]))
# print(df["FuelType"].unique())
# fuel_encoded = pd.get_dummies(df["FuelType"])

# print(fuel_encoded.head())



df = df.drop(columns=["AdditionInfo"]) #as encoding that requires advanced things NLP text processing

# print(df.select_dtypes(include="str").columns)
df = df.dropna()
# print(df.isnull().sum())

X = df.drop(columns=["AskPrice"])

y = df["AskPrice"]

# print(X.head())

# print(y.head())

X = pd.get_dummies(X)

print(X.head())

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42

)

print(X_train.shape)
print(X_test.shape)