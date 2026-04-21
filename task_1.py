# ===============================
# 1. Import Libraries
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Settings
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10,6)

# ===============================
# 2. Load Dataset
# ===============================
df = pd.read_csv(r"C:\Users\abiji\OneDrive\Desktop\Internship_\archive (4)\netflix_titles.csv")

# Preview
print(df.head())
print(df.info())

# ===============================
# 3. Data Cleaning
# ===============================

# Remove duplicates
df.drop_duplicates(inplace=True)

# Handle missing values
df['director'].fillna('Unknown', inplace=True)
df['cast'].fillna('Not Available', inplace=True)
df['country'].fillna('Unknown', inplace=True)
df['rating'].fillna(df['rating'].mode()[0], inplace=True)

# Convert date column
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')

# Extract year added
df['year_added'] = df['date_added'].dt.year

# ===============================
# 4. Exploratory Data Analysis
# ===============================

# Movies vs TV Shows
type_counts = df['type'].value_counts()

plt.figure()
sns.countplot(data=df, x='type')
plt.title("Movies vs TV Shows")
plt.show()

# Top 10 countries
top_countries = df['country'].value_counts().head(10)

plt.figure()
sns.barplot(x=top_countries.values, y=top_countries.index)
plt.title("Top 10 Content Producing Countries")
plt.show()

# Content added over years
year_data = df['year_added'].value_counts().sort_index()

plt.figure()
sns.lineplot(x=year_data.index, y=year_data.values)
plt.title("Content Added Over Years")
plt.xlabel("Year")
plt.ylabel("Count")
plt.show()

# ===============================
# 5. Genre Analysis
# ===============================

# Split genres
df['listed_in'] = df['listed_in'].str.split(',')

# Explode genres
genre_df = df.explode('listed_in')

# Clean spaces
genre_df['listed_in'] = genre_df['listed_in'].str.strip()

top_genres = genre_df['listed_in'].value_counts().head(10)

plt.figure()
sns.barplot(x=top_genres.values, y=top_genres.index)
plt.title("Top 10 Genres on Netflix")
plt.show()

# ===============================
# 6. Correlation Heatmap (if applicable)
# ===============================

# Only numeric columns
numeric_df = df.select_dtypes(include=[np.number])

if not numeric_df.empty:
    plt.figure()
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

# ===============================
# 7. Save Cleaned Dataset
# ===============================
df.to_csv("data/cleaned_netflix_data.csv", index=False)

print("Project Completed Successfully!")