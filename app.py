# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, Normalizer
import seaborn as sns
import matplotlib.pyplot as plt

# Title of the app
st.title("pIC50 Prediction App")

# Link to the dataset on Google Drive
data_link_id = "1WweCeesg7nFwzBfCHGL28Prf_VdUaqZL"
data_link = f'https://drive.google.com/uc?id={data_link_id}'
df = pd.read_csv(data_link)

# Display data preview
st.write("Data Preview:")
st.dataframe(df.head())

# Data preprocessing
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(axis=0, inplace=True)

X = df.drop('pIC50', axis=1)
y = df['pIC50']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

# Scale the input data using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Building a Regression Model using Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test_scaled)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

st.write(f'Mean Absolute Error: {mae}')
st.write(f'Mean Squared Error: {mse}')
st.write(f'Root Mean Squared Error: {rmse}')
st.write(f'R-squared (R2) Score: {r2}')

# Plotting
fig, ax = plt.subplots()
sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha': 0.3, 'color': 'blue'}, line_kws={'color':'red'}, ax=ax)
ax.set_xlabel('Experimental pIC50', fontsize=14, fontweight='bold')
ax.set_ylabel('Predicted pIC50', fontsize=14, fontweight='bold')
ax.set_xlim(2, 11.5)
ax.set_ylim(2, 11.5)
ax.figure.set_size_inches(8, 6)
plt.title('Actual vs Predicted pIC50', fontsize=16, fontweight='bold')
st.pyplot(fig)

# Uploading SMILES data
smiles_file = st.file_uploader("Upload your sample.csv", type="csv")
if smiles_file is not None:
    sample = pd.read_csv(smiles_file)
    st.write("Sample Data Preview:")
    st.dataframe(sample.head())

    # Calculate Lipinski descriptors for the ligands
    def lipinski(SMILES):
        moldata = []
        for elem in SMILES:
            if isinstance(elem, str):
                mol = Chem.MolFromSmiles(elem)
                moldata.append(mol)
        baseData = np.zeros((len(moldata), 4))
        for i, mol in enumerate(moldata):
            desc_MolWt = Descriptors.MolWt(mol)
            desc_MolLogP = Descriptors.MolLogP(mol)
            desc_NumHDonors = Lipinski.NumHDonors(mol)
            desc_NumHAcceptors = Lipinski.NumHAcceptors(mol)
            baseData[i] = [desc_MolWt, desc_MolLogP, desc_NumHDonors, desc_NumHAcceptors]
        columnNames = ["MW", "LogP", "NumHDonors", "NumHAcceptors"]
        return pd.DataFrame(data=baseData, columns=columnNames)

    df_ligands_lipinski = lipinski(sample['SMILES'])
    st.write("Lipinski Descriptors:")
    st.dataframe(df_ligands_lipinski.head())

    # Getting RDKit molecular descriptors
    def RDKit_descriptors(SMILES):
        mols = [Chem.MolFromSmiles(i) for i in SMILES]
        calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
        desc_names = calc.GetDescriptorNames()

        Mol_descriptors = []
        for mol in mols:
            mol = Chem.AddHs(mol)
            descriptors = calc.CalcDescriptors(mol)
            Mol_descriptors.append(descriptors)
        return Mol_descriptors, desc_names

    MoleculeDescriptors_list, desc_names = RDKit_descriptors(sample['SMILES'])
    df_ligands_descriptors = pd.DataFrame(MoleculeDescriptors_list, columns=desc_names)

    # Combine Lipinski and molecular descriptors
    fp_ligands = pd.concat([df_ligands_descriptors, df_ligands_lipinski], axis=1)
    fp_ligands = fp_ligands.drop('MW', axis=1)
    st.write("Data Preview:")
    st.dataframe(fp.ligands.head())

    # Predictions
    sample['predicted_pIC50'] = model.predict(scaler.transform(fp_ligands))
    st.write("Predicted pIC50 Values:")
    st.dataframe(sample[['SMILES', 'predicted_pIC50']])
# -


