import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("titanic")
print("Dataset loaded. Shape:", df.shape)

print("\nFirst 5 rows:\n", df.head())

print("\nMissing values:\n", df.isnull().sum())

print("\nBasic statistics:\n", df.describe(include="all"))

plt.figure(figsize=(6,4))
sns.countplot(x="survived", data=df)
plt.title("Survival Count")
plt.savefig("eda_survival_count.png")
plt.close()

plt.figure(figsize=(6,4))
sns.countplot(x="sex", hue="survived", data=df)
plt.title("Survival by Gender")
plt.savefig("eda_survival_by_gender.png")
plt.close()

plt.figure(figsize=(6,4))
sns.histplot(df["age"].dropna(), kde=True, bins=30)
plt.title("Age Distribution")
plt.savefig("eda_age_distribution.png")
plt.close()

plt.figure(figsize=(6,4))
sns.boxplot(x="pclass", y="age", data=df)
plt.title("Age vs Passenger Class")
plt.savefig("eda_age_vs_class.png")
plt.close()

print("\nEDA complete. Plots saved as PNG files.")
