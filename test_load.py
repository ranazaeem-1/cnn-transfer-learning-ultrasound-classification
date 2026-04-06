import os, cv2, pandas as pd

IMAGES_DIR='processed_images'
df = pd.read_excel('PatientImages_MATCHED.xlsx')
diag_col = [c for c in df.columns if 'diagn' in c.lower()][0]
df['Diagnosis_Clean'] = df[diag_col].astype(str).str.strip()
c_exist = 0
c_img = 0
c_diag = 0
for _, row in df.iterrows():
    fn = str(row['filename'])
    diag = row['Diagnosis_Clean']
    img_path = os.path.join(IMAGES_DIR, fn)
    if os.path.exists(img_path):
        c_exist += 1
        img = cv2.imread(img_path)
        if img is not None:
            c_img += 1
        if diag in ['N', 'D', 'P', 'I']:
            c_diag += 1
print(f'Exists: {c_exist}, ImgLoaded: {c_img}, ValidDiag: {c_diag}')

