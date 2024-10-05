from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import os


DATA_DIR = 'data/PestycydoweAMPs'


def process_amp_files(directory):
    data = []
    categories = ['amp', 'namp_faba', 'namp_viri']
    datasets = ['train', 'test']

    for category in categories:
        for dataset in datasets:
            filename = f"{category}_{dataset}.fasta" if category == 'amp' else f"{category}_{dataset}"
            file_path = os.path.join(directory, filename)

            if os.path.exists(file_path):
                with open(file_path, 'r') as file:
                    sequence = ""
                    for line in file:
                        if line.startswith('>'):
                            if sequence:
                                data.append({
                                    'category': category,
                                    'dataset': dataset,
                                    'sequence': sequence.strip()
                                })
                                sequence = ""
                        else:
                            sequence += line.strip()

                    # Add the last sequence
                    if sequence:
                        data.append({
                            'category': category,
                            'dataset': dataset,
                            'sequence': sequence.strip()
                        })

    return pd.DataFrame(data)


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
    df = process_amp_files(DATA_DIR)

    df = vectorize_sequence_column(df, sequence_column='sequence')

    df = (df.drop(columns=['sequence'])
            .rename(columns={'category': 'target'}))

    df_train = df[df['dataset'] == 'train']
    df_test = df[df['dataset'] == 'test']

    df_train = df_train.drop(columns='dataset')
    df_test = df_test.drop(columns='dataset')

    print(df_train.shape)
    print(df_test.shape)
    print(df_train.head(2))
    print('\n')
    print(df_test.head(2))


if __name__ == '__main__':
    main()
