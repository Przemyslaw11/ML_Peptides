import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from tqdm import tqdm
import wandb
from dotenv import load_dotenv

load_dotenv()


class PeptideDataset(Dataset):
    def __init__(self, sequences, labels, max_length=128):
        self.sequences = sequences
        self.labels = labels
        self.max_length = max_length
        self.aa_to_int = {aa: i+1 for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        seq_ints = [self.aa_to_int.get(aa, 0) for aa in sequence[:self.max_length]]
        seq_ints = seq_ints + [0] * (self.max_length - len(seq_ints))
        
        return {
            'input_ids': torch.tensor(seq_ints, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class PeptideClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, max_length):
        super(PeptideClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        
    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(2)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    best_val_accuracy = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = correct_train / total_train

        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_accuracy = correct_val / total_val
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        scheduler.step(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_peptide_classifier.pth')
            # wandb.save('best_peptide_classifier.pth')
            print(f"New best model saved with validation accuracy: {best_val_accuracy:.4f}")

    return model


if __name__ == "__main__":
    wandb.init(project="peptide-classification", entity="ai-development")

    df = pd.read_csv('hemopi_data.csv')

    df = df[['sequence', 'target']]
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['target'].value_counts(normalize=True)}")

    le = LabelEncoder()
    df['target_encoded'] = le.fit_transform(df['target'])

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['target'], random_state=42)

    max_length = 128
    vocab_size = 22
    embed_dim = 32
    num_classes = len(le.classes_)

    wandb.config.update({
        "max_length": max_length,
        "vocab_size": vocab_size,
        "embed_dim": embed_dim,
        "num_classes": num_classes,
        "batch_size": 32,
    })

    train_dataset = PeptideDataset(train_df['sequence'].tolist(), train_df['target_encoded'].tolist(), max_length)
    val_dataset = PeptideDataset(val_df['sequence'].tolist(), val_df['target_encoded'].tolist(), max_length)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = (vocab_size, embed_dim, num_classes, max_length)

    wandb.watch(model)

    trained_model = train_model(model, train_loader, val_loader)

    torch.save(trained_model.state_dict(), 'final_peptide_classifier.pth')

    wandb.save('final_peptide_classifier.pth')
    wandb.finish()
