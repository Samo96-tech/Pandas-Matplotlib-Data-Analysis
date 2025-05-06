import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ## Task 1: Load & Explore
# Option A: load Iris via sklearn
from sklearn.datasets import load_iris
iris_data = load_iris(as_frame=True)
df = iris_data.frame

# Option B: load from CSV
# df = pd.read_csv('path/to/your.csv')
# Display first rows
df.head()
# Inspect data types and missing values
print(df.info())
print('\nMissing values per column:')
print(df.isnull().sum())

# If missing values exist, either drop or fill them.
# Example: drop any rows with missing values
df_clean = df.dropna().reset_index(drop=True)
# Or fill numeric NaNs with column mean:
# df_clean = df.fillna(df.mean())


# ## Task 2: Basic Analysis
# Summary statistics
df_clean.describe()

# Group by species and compute mean sepal length
grouped = df_clean.groupby('target')['sepal length (cm)'].mean().rename_axis('species_id')
grouped

# Map target IDs back to species names for readability
species_map = dict(zip(range(len(iris_data.target_names)), iris_data.target_names))
grouped.index = grouped.index.map(species_map)
grouped        

# ## Task 3: Visualizations
plt.figure()
plt.plot(df_clean.index, df_clean['sepal length (cm)'], label='Sepal Length')
plt.title('Sepal Length over Sample Index')
plt.xlabel('Sample Index')
plt.ylabel('Sepal Length (cm)')
plt.legend()
plt.show()     

# Bar Chart (Average by Category)
plt.figure()
grouped.plot(kind='bar')
plt.title('Average Sepal Length per Iris Species')
plt.xlabel('Species')
plt.ylabel('Mean Sepal Length (cm)')
plt.xticks(rotation=45)
plt.show()

# Histogram (Distribution of Petal Length)
plt.figure()
plt.hist(df_clean['petal length (cm)'], bins=15)
plt.title('Distribution of Petal Length')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')
plt.show()

# Scatter Plot (Relationship between Two Numericals)
plt.figure()
plt.scatter(df_clean['sepal length (cm)'], df_clean['petal length (cm)'], alpha=0.7)
plt.title('Sepal Length vs. Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.show()

# Error Handling on File Read
try:
    df2 = pd.read_csv('nonexistent.csv')
except FileNotFoundError as e:
    print("Error: File not found. Please check the path and try again.")
    df2 = pd.DataFrame()  # fallback

