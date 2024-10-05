import pandas as pd
import os

# Binary Classification (positive, negative peptides) based on amino sequence
DATA_DIR = 'data/HemoPI'


def extract_peptides(file_content, type_, dataset):
    lines = file_content.strip().split('\n')
    peptides = []
    current_peptide = None

    for line in lines:
        if line.startswith('>'):
            if current_peptide:
                peptides.append(current_peptide)
            current_peptide = {'name': line[1:], 'sequence': '', 'type': type_, 'dataset': dataset}
        else:
            current_peptide['sequence'] += line.strip()

    if current_peptide:
        peptides.append(current_peptide)

    return peptides


def create_dataframe(peptides):
    return pd.DataFrame(peptides)


def process_files(directory):
    all_peptides = []

    file_categories = {
        'pos_main': ('Positive', 'Main'),
        'pos_validation': ('Positive', 'Validation'),
        'neg_main': ('Negative', 'Main'),
        'neg_validation': ('Negative', 'Validation')
    }

    for filename in os.listdir(directory):
        if filename.endswith('.fa.txt'):
            type_, dataset = next((cat for prefix, cat in file_categories.items() if filename.startswith(prefix)),
                                  ('Unknown', 'Unknown'))

            with open(os.path.join(directory, filename), 'r') as file:
                file_content = file.read()
                peptides = extract_peptides(file_content, type_, dataset)
                all_peptides.extend(peptides)

    return create_dataframe(all_peptides)


def main():
    df = process_files(DATA_DIR)

    df = df[df['type'] != 'Unknown'].drop(columns=['name'])

    df = df.rename(columns={"type": "target"})
    df = df.drop(columns=['dataset'])
    print(df.head(2))
    print(df.shape)

    df.to_csv('hemo_pi.csv', index=False)


if __name__ == '__main__':
    main()
