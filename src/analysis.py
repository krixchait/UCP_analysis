import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split #for generating few sample cases
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error #average prediction error
from sklearn.metrics import r2_score #how well is our model explaning variance
from sklearn.ensemble import RandomForestRegressor



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

# print(X.head())

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42

)

# print(X_train.shape)
# print(X_test.shape)

model=LinearRegression() #this is the main line where are are training our data using linear regression
model.fit(X_train,y_train)
# print("Model trained sucessfully")

predictions=model.predict(X_test)
# print(predictions[:5])

comparision=pd.DataFrame({
    "Actual Price":y_test.values,
    "Predicted Price": predictions
})

# print(comparision.head(10))


mae=mean_absolute_error(y_test,predictions)
r2=r2_score(y_test,predictions)

# print("MAE :",mae)
# print("R2 score:",r2)


rf_model=RandomForestRegressor(
    n_estimators=100,
    random_state=42

)

rf_model.fit(X_train,y_train) #we are training it using random forest regressor
rf_predictions=rf_model.predict(X_test)
rf_mae=mean_absolute_error(y_test,rf_predictions)
rf_r2=r2_score(y_test,rf_predictions)

# print("Random Forest MAE :",rf_mae)
# print("Random Forest R2 Score :",rf_r2)

# print("Linear Regression R2:", r2)
# print("Random Forest R2:", rf_r2)

brands = sorted(df["Brand"].unique())

fuel_types = sorted(df["FuelType"].unique())

transmissions = sorted(df["Transmission"].unique())

owners = sorted(df["Owner"].unique())


print("\nAvailable Brands:\n")
for i, brand in enumerate(brands):
    print(i, "-", brand)
brand_choice=int(input("\nSelect Brand Number : "))
brand=brands[brand_choice]




models = sorted(
    df[df["Brand"] == brand]["model"].unique()
)
df["Brand"] == brand

print("\nAvailable Models:\n")

for i, model_name in enumerate(models):
    print(i, "-", model_name)

model_choice = int(input("\nSelect Model Number: "))

model_name = models[model_choice]

#transmission seleciton

print("\nAvailable Transmissions:\n")

for i, t in enumerate(transmissions):
    print(i, "-", t)

transmission_choice = int(input("\nSelect Transmission Number: "))

transmission = transmissions[transmission_choice]

#owner 
print("\nAvailable Owner Types:\n")

for i, o in enumerate(owners):
    print(i, "-", o)

owner_choice = int(input("\nSelect Owner Type Number: "))

owner = owners[owner_choice]

#fuel type
print("\nAvailable Fuel Types:\n")

for i, f in enumerate(fuel_types):
    print(i, "-", f)

fuel_choice = int(input("\nSelect Fuel Type Number: "))

fuel_type = fuel_types[fuel_choice]

year = int(input("\nEnter Manufacturing Year: "))

km_driven = float(input("Enter Kilometers Driven: "))

#we have taken inut form the user, now need to connect with our data
sample_car = {
    "Brand": brand,
    "model": model_name,
    "Year": year,
    "kmDriven": km_driven,
    "Transmission": transmission,
    "Owner": owner,
    "FuelType": fuel_type,
    "Car_age": 2025 - year
}
sample_df = pd.DataFrame([sample_car])
sample_df = pd.get_dummies(sample_df)
sample_df = sample_df.reindex(columns=X.columns, fill_value=0)
sample_prediction = rf_model.predict(sample_df)

print("\nPredicted Car Price: ₹", round(sample_prediction[0]))