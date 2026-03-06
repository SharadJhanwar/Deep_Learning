import json

notebook_path = r'c:\Users\yx084\OneDrive\Python-Modules\DL\2.Number\6.churn.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if not source:
            continue
        
        # Join lines to check content easily, or check line by line
        # Fix 1: Loop inplace replace
        new_source = []
        modified = False
        for line in source:
            if "df1[col].replace({'Yes': 1,'No': 0},inplace=True)" in line:
                new_source.append("    df1[col] = df1[col].replace({'Yes': 1,'No': 0})\n")
                modified = True
            elif "df1['gender'].replace({'Female':1,'Male':0},inplace=True)" in line:
                new_source.append("df1['gender'] = df1['gender'].replace({'Female':1,'Male':0})\n")
                modified = True
            elif "df2 = pd.get_dummies(data=df1, columns=['InternetService','Contract','PaymentMethod'])" in line:
                new_source.append("df2 = pd.get_dummies(data=df1, columns=['InternetService','Contract','PaymentMethod'], dtype=int)\n")
                modified = True
            else:
                new_source.append(line)
        
        if modified:
            cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
