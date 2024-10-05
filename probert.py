from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Tuple
from torch.optim import AdamW
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch

class PeptideDataset(Dataset):
    def __init__(self, sequences: List[str], labels: List[int], max_length: int = 128, augment: bool = False):
        self.sequences = sequences
        self.labels = labels
        self.max_length = max_length
        self.augment = augment
        self.aa_to_int = {aa: i+1 for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
        
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        if self.augment and np.random.rand() < 0.5:
            sequence = self._augment_sequence(sequence)
        
        seq_ints = [self.aa_to_int.get(aa, 0) for aa in sequence[:self.max_length]]
        seq_ints = seq_ints + [0] * (self.max_length - len(seq_ints))
        
        return {
            'input_ids': torch.tensor(seq_ints, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
    def _augment_sequence(self, sequence: str) -> str:
        choice = np.random.choice(['substitute', 'insert', 'delete'])
        
        if choice == 'substitute':
            idx = np.random.randint(len(sequence))
            new_aa = np.random.choice(list(self.aa_to_int.keys()))
            sequence = sequence[:idx] + new_aa + sequence[idx+1:]
        elif choice == 'insert':
            idx = np.random.randint(len(sequence) + 1)
            new_aa = np.random.choice(list(self.aa_to_int.keys()))
            sequence = sequence[:idx] + new_aa + sequence[idx:]
        elif choice == 'delete':
            if len(sequence) > 1:
                idx = np.random.randint(len(sequence))
                sequence = sequence[:idx] + sequence[idx+1:]
        
        return sequence

class EnhancedPeptideClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int):
        super(EnhancedPeptideClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x).transpose(1, 2)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(2)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 100,
    lr: float = 0.001,
    patience: int = 10
) -> nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    best_val_accuracy = 0
    epochs_without_improvement = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
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
        correct_val, total_val, val_loss = 0, 0, 0.0
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
        
        scheduler.step(val_accuracy)
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_peptide_classifier.pth')
            print(f"New best model saved with validation accuracy: {best_val_accuracy:.4f}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after {patience} epochs without improvement.")
                break
    
    return model

def prepare_data(
    df: pd.DataFrame,
    test_size: float = 0.2
) -> Tuple[DataLoader, DataLoader, LabelEncoder]:
    le = LabelEncoder()
    df['target_encoded'] = le.fit_transform(df['target'])
    
    train_df, val_df = train_test_split(df, test_size=test_size, stratify=df['target'], random_state=42)
    
    max_length = 128
    
    train_dataset = PeptideDataset(train_df['sequence'].tolist(), train_df['target_encoded'].tolist(), max_length, augment=True)
    val_dataset = PeptideDataset(val_df['sequence'].tolist(), val_df['target_encoded'].tolist(), max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    return train_loader, val_loader, le

def main():
    df = pd.read_csv('hemopi_data.csv')
    
    df = df[['sequence', 'target']]
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['target'].value_counts(normalize=True)}")
    
    train_loader, val_loader, le = prepare_data(df)
    
    vocab_size = 22  # 20 amino acids + padding + unknown
    embed_dim = 64
    num_classes = len(le.classes_)
    
    model = EnhancedPeptideClassifier(vocab_size, embed_dim, num_classes)
    
    trained_model = train_model(model, train_loader, val_loader, num_epochs=100, lr=0.001, patience=10)
    
    torch.save(trained_model.state_dict(), 'final_peptide_classifier.pth')

if "__name__" == "main":
    main()