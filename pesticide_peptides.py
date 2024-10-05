from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import os


DATA_DIR = 'data/PestycydoweAMPs'


def process_fasta_files(directory):
    data = []
    processed_files = set()

    for filename in os.listdir(directory):
        if filename.endswith('.fasta'):
            base_filename = filename[:-6]  # Remove '.fasta'
            if base_filename in processed_files:
                continue
            processed_files.add(base_filename)

            file_path = os.path.join(directory, filename)

            parts = base_filename.split('_')
            if parts[0] == 'amp':
                category = 'amp'
            elif len(parts) >= 2:
                category = f"{parts[0]}_{parts[1]}"
            else:
                category = parts[0]

            with open(file_path, 'r') as file:
                current_id = None
                sequence = ""

                for line in file:
                    line = line.strip()
                    if line.startswith('>'):
                        if current_id and sequence:
                            data.append({
                                'id': current_id,
                                'category': category,
                                'sequence': sequence
                            })

                        current_id = line[1:]  # Remove the '>' character
                        sequence = ""
                    elif line:
                        sequence += line

                if current_id and sequence:
                    data.append({
                        'id': current_id,
                        'category': category,
                        'sequence': sequence
                    })

    df = pd.DataFrame(data)
    return df


def main():
    df = process_fasta_files(DATA_DIR)

    print(df.head())

    # Print category distribution
    category_counts = df['category'].value_counts()
    print("\nCategory distribution:")
    print(category_counts)
    df = df.drop(columns=['id']).rename(columns={'category': 'target'})

    df.to_csv('pesticides_peptides.csv', index=False)
    print(df.head(2))


if __name__ == '__main__':
    main()
