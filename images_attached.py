import pandas as pd

# Load your cleaned Excel file
df = pd.read_excel("PatientImages_PLOS2017.xlsx")

# Add filename column
df["filename"] = df.index.astype(str) + ".png"

# Save final matched Excel
df.to_excel("PatientImages_MATCHED.xlsx", index=False)

print("Done! Saved as PatientImages_MATCHED.xlsx")
