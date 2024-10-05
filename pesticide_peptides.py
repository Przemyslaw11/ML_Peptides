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


def main():
    df = process_amp_files(DATA_DIR)

    df = (df.rename(columns={'category': 'target'}))
    df = df.drop(columns='dataset')
    print(df.head(2))

    df.to_csv('pesticides_peptides.csv', index=False)


if __name__ == '__main__':
    main()
