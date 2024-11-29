"""
pearson-statistical-analysis.py

Contains the code needed for the first exercise from the slides.
(Scatterplot can be found in pearson-linear-regression.py)
"""
#%% Imports
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind

#%% Configuration
fname: str = "../../data/pearson-father-son.csv"

# Read the Pearson Father son set as a DataFrame
df: pd.DataFrame = pd.read_csv(fname)

#%% Data exploration
# The dataframe contains two columns: father and son. Heights are specified in inches. There
# seems to be a positive correlation between the height of the father and the son
sns.pairplot(df)

#%% Calculate the correlation between the DF columns
print(df.corr(method='pearson', min_periods=1, numeric_only=False))

# Run a t-test on the father and son-columns to see how their means compare
print(ttest_ind(df['father'], df['son']))
