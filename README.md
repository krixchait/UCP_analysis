<div align="center">

# UCP Analysis

### Used Car Price Analysis & Prediction with Linear Regression and Random Forest

<p>
  <strong>Pandas</strong> •
  <strong>Scikit-learn</strong> •
  <strong>Matplotlib</strong> •
  <strong>Regression Modeling</strong>
</p>

---

**UCP Analysis** cleans and explores a real-world used-car listings dataset, engineers features
from raw text fields, trains and compares two regression models, then lets you interactively
predict the price of a car by answering a series of prompts.

</div>

---

# Why UCP Analysis?

Raw scraped listing data is rarely usable as-is — prices come formatted with currency symbols,
mileage comes as a string with units baked in, and categorical fields need encoding before any
model can touch them. This project works through that pipeline end to end on a public
[used car dataset](data/used_car_dataset.csv): cleaning, exploratory analysis, feature
engineering, model training, model comparison, and a small interactive prediction tool built on
top of the final model.

This project emphasizes:

* Data cleaning of messy real-world (string-encoded) numeric fields
* Exploratory data analysis via price distribution and correlation checks
* Feature engineering (deriving car age from year)
* Categorical encoding with one-hot vectors
* Model comparison: Linear Regression vs. Random Forest
* An interactive CLI for predicting the price of a user-described car

---

# Dataset

`data/used_car_dataset.csv` — 9,581 used car listings with the following raw columns:

| Column          | Description                                          |
| ---------------- | ------------------------------------------------------ |
| `Brand`          | Manufacturer (39 unique brands)                       |
| `model`          | Model name                                            |
| `Year`           | Manufacturing year                                    |
| `kmDriven`       | Odometer reading, originally a string like `"98,000 km"` |
| `Transmission`   | `Manual` or `Automatic`                               |
| `Owner`          | `first` or `second` owner                             |
| `FuelType`       | `Petrol`, `Diesel`, or `Hybrid/CNG`                    |
| `AdditionInfo`   | Free-text listing description (dropped before modeling) |
| `AskPrice`       | Listed price, originally a string like `"₹ 1,95,000"`  |

---

# Pipeline

```text
Raw CSV
   │
   ▼
Clean kmDriven & AskPrice (strip units, commas, currency symbol → numeric)
   │
   ▼
Engineer Car_age = 2025 − Year
   │
   ▼
Drop AdditionInfo (free text, out of scope) + drop remaining NaNs
   │
   ▼
One-hot encode categoricals (Brand, model, Transmission, Owner, FuelType)
   │
   ▼
Train/test split (80/20)
   │
   ├──> Linear Regression ──> MAE, R²
   │
   └──> Random Forest Regressor (100 trees) ──> MAE, R²
                    │
                    ▼
        Interactive CLI prediction on the better-performing model
```

---

# Exploratory Analysis

The script includes (commented-out, toggleable) exploratory steps used to shape the modeling
decisions:

* **Price distribution** — histogram of `AskPrice` to check skew
* **Km driven vs. price** — scatter plot, then filtered to listings under ₹14,00,000 to remove
  outlier luxury listings that were distorting the trend
* **Car age vs. price** — scatter plot on the same filtered subset
* **Correlation checks** — `kmDriven` and `Car_age` against `AskPrice`
* **Top-10 highest-priced listings** and **average price by brand**, used to sanity-check which
  rows were driving outlier behavior before deciding on the price cutoff

---

# Model Comparison

Two models are trained on the same train/test split and evaluated with **MAE** (mean absolute
error) and **R²** (variance explained):

| Model                     | What it captures                                              |
| --------------------------- | ---------------------------------------------------------------- |
| **Linear Regression**      | Baseline linear relationship between features and price       |
| **Random Forest Regressor** | Non-linear interactions between brand, age, mileage, etc. (100 trees) |

Random Forest is the model used for the final interactive prediction, since tree ensembles
typically capture non-linear pricing effects (e.g. brand-specific depreciation curves) that a
plain linear model can't.

---

# Interactive Prediction

Running the script ends with a guided CLI: pick a brand, then a model (filtered to that brand),
transmission, owner type, and fuel type from numbered lists, then enter the manufacturing year
and kilometers driven. The inputs are assembled into a single-row DataFrame, one-hot encoded and
reindexed to match the training feature columns, and passed to the trained Random Forest model
for a price prediction.

```text
Available Brands:
0 - Audi
1 - BMW
2 - Chevrolet
...

Select Brand Number : 4

Available Models:
0 - City
1 - Civic
...

Select Model Number: 0
...

Predicted Car Price: ₹ 452000
```

---

# Project Structure

```text
UCP_analysis
│
├── data/
│   └── used_car_dataset.csv
│
├── src/
│   └── analysis.py          # Full pipeline: cleaning → EDA → training → CLI prediction
│
├── requirements.txt
└── README.md
```

---

# Running Locally

```bash
pip install -r requirements.txt
python src/analysis.py
```

Run from the repository root, since the script reads `data/used_car_dataset.csv` with a relative
path.

---
