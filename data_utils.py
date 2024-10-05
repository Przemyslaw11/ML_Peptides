import pandas as pd
from rdkit import Chem


def sample_target_rows(df, n=10):
    positive_samples = df[df['target'] == 'Positive'].sample(n=n, random_state=42)
    negative_samples = df[df['target'] == 'Negative'].sample(n=n, random_state=42)

    result = pd.concat([positive_samples, negative_samples])
    result = result.reset_index(drop=True)

    return result


def string_to_smiles(input_string, smiles_dict):
    smiles_chars = [smiles_dict.get(char, char) for char in input_string]
    smiles_string = ''.join(smiles_chars)
    mol = Chem.MolFromSmiles(smiles_string)
    return Chem.MolToSmiles(mol) if mol is not None else smiles_string
