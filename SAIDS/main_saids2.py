import pandas as pd

# load raw data
df = pd.read_csv("student_preprocessing_dataset_Lab2.csv")

print("Original Dataset:")
print(df.head())

# fix missing values (mean for numbers, mode for city)
print("\nMissing Values:")
print(df.isnull().sum())

df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Marks"] = df["Marks"].fillna(df["Marks"].mean())
df["Attendance"] = df["Attendance"].fillna(df["Attendance"].mean())
df["City"] = df["City"].fillna(df["City"].mode()[0])

# drop any duplicate rows
print("\nDuplicate Rows:", df.duplicated().sum())
df = df.drop_duplicates()
print("Duplicate Rows After Removal:", df.duplicated().sum())

# encode categoricals to numbers
df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1})
df["Department"] = df["Department"].astype("category").cat.codes
df["City"] = df["City"].astype("category").cat.codes

# save output
print("\nCleaned Dataset:")
print(df.head())

df.to_csv("Cleaned_Student_Dataset.csv", index=False)
print("\nPreprocessing done! Cleaned CSV file created.")
