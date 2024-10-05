from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import os

# Binary Classification (positive, negative peptides) based on amino sequence
DATA_PATH = 'data/HemoPI'


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


def vectorize_sequence_column(df, sequence_column='sequence'):
    vectorizer = CountVectorizer(analyzer='char', lowercase=False)

    # Fit and transform the sequences
    vector_matrix = vectorizer.fit_transform(df[sequence_column])

    # Get the feature names (amino acids)
    feature_names = vectorizer.get_feature_names_out()

    vector_df = pd.DataFrame(vector_matrix.toarray(), columns=feature_names)
    result_df = pd.concat([df, vector_df], axis=1)

    return result_df


def main():
    DATA_PATH = 'data/HemoPI'
    df = process_files(DATA_PATH)

    df = df[df['type'] != 'Unknown'].drop(columns=['name'])

    df_val = df[df['dataset'] == 'Validation']
    df_main = df[df['dataset'] == 'Main']

    print(df_val.shape)
    print(df_main.shape)

    # Vectorized?
    df_results = vectorize_sequence_column(df_main)
    df_results = df_results.rename(columns={"type": "target"})
    df_results = df_results.drop(columns=['sequence', 'dataset'])
    print(df_results.head(2))


if __name__ == '__main__':
    main()
