"""
german_credit_from_csv.py

EDA of german-credit.csv, downloaded directly in CSV form from the University of Pennsylvania.
"""
import pandas as pd
import seaborn as sns

sns.set_style()

filename: str = "../../data/german-credit.csv"
df: pd.DataFrame = pd.read_csv(filename)

# Print some information about the dataframe and its columns
print(df.info())

for column in df.columns:
    vals = df[column].unique()
    print(f"Column: {column} has {len(vals)} unique values {vals}")

"""
This version of the CSV uses label encoding:
- Column: Occupation has 4 unique values [3 2 1 4]
- Column: Type of apartment has 3 unique values [1 2 3]

Neither of these seem like proper ordinal variables, which is why label encoding 
is not a suitable approach for these variables. We can conclude that this data is compromised.
"""