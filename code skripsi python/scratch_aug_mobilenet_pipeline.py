import torch
import numpy as np
import pandas as pd
import csv
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from tqdm import tqdm
from torchvision.models import mobilenet_v3_large
import random
import gc
from torch import nn

# Constants
SEED = 123
IMG_SIZE = 224
BATCH_SIZE = 64
LEARNING_RATE = 3e-5
EPOCHS = 1000
NUM_CLASSES = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FERDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

        # Ekstrak label dan piksel
        self.labels = self.dataframe['emotion'].values
        self.pixels = self.dataframe['pixels'].apply(self.string_to_image).values

    def string_to_image(self, pixels_string):
        # Konversi string piksel menjadi numpy array dan reshape ke 48x48
        pixels = np.array(pixels_string.split(), dtype='float32')
        image = pixels.reshape(48, 48)
        image = np.expand_dims(image, axis=-1)  # Tambahkan channel dimensi
        return image

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image = self.pixels[idx]
        label = self.labels[idx]
        
        image = Image.fromarray(image.squeeze().astype('uint8'), mode='L')

        # Jika ada transformasi, terapkan ke image
        if self.transform:
            image = self.transform(image)

        return image, label
    
def create_transforms():
    train_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(size=IMG_SIZE, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    test_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return train_transforms, test_transforms

def load_and_split_data(data_path):
    data = pd.read_csv(data_path)
    data_train, data_test = train_test_split(data, test_size=0.1, stratify=data['emotion'], random_state=SEED)
    data_train, data_val = train_test_split(data_train, test_size=0.1, stratify=data_train['emotion'], random_state=SEED)
    return data_train, data_val, data_test

def create_datasets(data_train, data_val, data_test, train_transforms, test_transforms):
    train_dataset = FERDataset(data_train, transform=train_transforms)
    val_dataset = FERDataset(data_val, transform=test_transforms)
    test_dataset = FERDataset(data_test, transform=test_transforms)
    return train_dataset, val_dataset, test_dataset

def create_dataloaders(train_dataset, val_dataset, test_dataset):
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                           generator=torch.Generator().manual_seed(SEED))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            generator=torch.Generator().manual_seed(SEED))
    return train_loader, val_loader, test_loader

    
class MultiheadSelfAttentionBlock(nn.Module):
    def __init__(self,
                 embedding_dim:int=768,
                 num_heads:int=12,
                 attn_dropout:float=0.):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                    num_heads=num_heads,
                                                    dropout=attn_dropout,
                                                    batch_first=True)
    def forward(self, x):
        attn_output, _ = self.multihead_attn(query=x,
                                             key=x,
                                             value=x,
                                             need_weights=False)
        return attn_output
    
class MLPBlock(nn.Module):
    def __init__(self,
                 embedding_dim:int=768,
                 mlp_size:int=3072,
                 dropout:float=0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim,
                      out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size,
                      out_features=embedding_dim),
            nn.Dropout(p=dropout)
        )
    def forward(self, x):
        x = self.mlp(x)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 embedding_dim:int=768,
                 num_heads:int=12,
                 mlp_size:int=3072,
                 mlp_dropout:float=0.,
                 attn_dropout:float=0.):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(normalized_shape=embedding_dim, eps=1e-6)
        
        self.msa_block = MultiheadSelfAttentionBlock(embedding_dim=embedding_dim,
                                                     num_heads=num_heads,
                                                     attn_dropout=attn_dropout)
        
        self.layer_norm2 = nn.LayerNorm(normalized_shape=embedding_dim, eps=1e-6)
        
        self.mlp_block =  MLPBlock(embedding_dim=embedding_dim,
                                   mlp_size=mlp_size,
                                   dropout=mlp_dropout)
    def forward(self, x):
        x = self.msa_block(self.layer_norm1(x)) + x 
        
        x = self.mlp_block(self.layer_norm2(x)) + x 
        
        return x

class ViT(nn.Module):
    def __init__(self,
                 img_size:int=224,
                 in_channels:int=3,
                 patch_size:int=16,
                 num_transformer_layers:int=12,
                 embedding_dim:int=768,
                 mlp_size:int=3072,
                 num_heads:int=12,
                 attn_dropout:float=0.,
                 mlp_dropout:float=0.,
                 embedding_dropout:float=0.,
                 num_classes:int=1000):
        super().__init__()
        assert img_size % 32 == 0, f"Image size must be divisible by 32, image size: {img_size}"
        
        self.backbone = mobilenet_v3_large(pretrained=True).features

        # Projection layer to match ViT's embedding dimension
        self.projection = nn.Conv2d(960, embedding_dim, kernel_size=1)
                 
        self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim),
                                            requires_grad=True)
        
        self.num_patches = (img_size // 32) ** 2  # MobileNet reduces spatial size by 32x
        
        self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches+1, embedding_dim),
                                               requires_grad=True)
                
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        
        self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim,
                                                                            num_heads=num_heads,
                                                                            mlp_size=mlp_size,
                                                                            mlp_dropout=mlp_dropout) for _ in range(num_transformer_layers)])
       
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim, eps=1e-6),
            nn.Linear(in_features=embedding_dim, 
                      out_features=num_classes)
        )
    
    def forward(self, pixel_values, labels=None):
        batch_size = pixel_values.shape[0]

        # Extract features using MobileNetV3
        features = self.backbone(pixel_values)  # Output shape: (batch_size, backbone_out_channels, H', W')
        features = self.projection(features)  # Project to embedding_dim: (batch_size, embedding_dim, H', W')
        
        # Flatten the feature maps into a sequence of tokens
        features = features.flatten(2).transpose(1, 2)  # Shape: (batch_size, num_patches, embedding_dim)
        
        class_token = self.class_embedding.expand(batch_size, -1, -1) # (n_samples, 1, embed_dim)

        x = torch.cat((class_token, features), dim=1) # (n_samples, 1 + n_patches, embed_dim)

        x = self.position_embedding + x # add position embed

        x = self.embedding_dropout(x)

        x = self.transformer_encoder(x)

        logits = self.classifier(x[:, 0])

        # Jika labels diberikan, hitung loss
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
    
        return (loss, logits) if loss is not None else logits

class BestModelSaver:
    def __init__(self, save_path, model_name):
        self.save_path = save_path
        self.model_name = model_name
        self.best_accuracy = -float('inf')
        os.makedirs(self.save_path, exist_ok=True)

    def save(self, model, current_accuracy, epoch):
        if current_accuracy > self.best_accuracy:
            self.best_accuracy = current_accuracy
            model_path = os.path.join(self.save_path, f"{self.model_name}_best.pt")
            torch.save(model.state_dict(), model_path)
            print(f"Best model saved at {model_path} with accuracy: {self.best_accuracy:.4f}")

class MetricsPlotter:
    def __init__(self, save_path, model_name):
        self.save_path = save_path
        self.model_name = model_name
        os.makedirs(self.save_path, exist_ok=True)

    def plot_and_save(self, train_metrics, val_metrics, metric_name, epoch):
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(train_metrics) + 1), train_metrics, label=f"Training {metric_name}", marker='o')
        plt.plot(range(1, len(val_metrics) + 1), val_metrics, label=f"Validation {metric_name}", marker='o')
        plt.title(f"{self.model_name} {metric_name} per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel(metric_name)
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(self.save_path, f"{self.model_name}_{metric_name.lower()}.png")
        plt.savefig(plot_path)
        plt.close()

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.epochs_no_improve = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1

        if self.epochs_no_improve >= self.patience:
            self.early_stop = True

        return self.early_stop
    
class Validator:
    def __init__(self, model, criterion, device):
        self.model = model
        self.criterion = criterion
        self.device = device

    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_targets = []
        all_predicted = []

        with torch.no_grad():  # Disable gradient computation
            pbar = tqdm(val_loader, desc="Validation")
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # Update statistics
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

                # Collect all targets and predictions for F1-score
                all_targets.extend(targets.cpu().numpy())
                all_predicted.extend(predicted.cpu().numpy())

                # Update progress bar
                pbar.set_postfix({
                    "Loss": f"{val_loss / (batch_idx + 1):.4f}",
                    "Acc": f"{val_correct / val_total:.4f}"
                })

        # Calculate validation accuracy, loss, and F1-score
        val_accuracy = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        val_f1 = f1_score(all_targets, all_predicted, average="weighted")

        return avg_val_loss, val_accuracy, val_f1
    
def train_model(model, train_loader, val_loader, config_idx, results_dir, epoch_csv_path):
    # Initialize training utilities
    model_saver = BestModelSaver(save_path=results_dir, model_name=f"model{config_idx}_augment_mobilenet")
    metrics_plotter = MetricsPlotter(save_path=results_dir, model_name=f"model{config_idx}_augment_mobilenet")
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    early_stopping = EarlyStopping(patience=15, min_delta=0)
    validator = Validator(model=model, criterion=criterion, device=DEVICE)

    # Initialize lists to store metrics
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    
    with open(epoch_csv_path, mode='a', newline='') as epoch_file:
        epoch_writer = csv.writer(epoch_file)
        
        for epoch in range(EPOCHS):
            # Training phase
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                pbar.set_postfix({
                    "Loss": f"{train_loss / (batch_idx + 1):.4f}",
                    "Acc": f"{correct / total:.4f}"
                })

            # Calculate training metrics
            train_accuracy = correct / total
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_accuracy)

            # Validation phase
            avg_val_loss, val_accuracy, val_f1 = validator.validate(val_loader)
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)

            # Update learning rate
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']

            # Save best model
            model_saver.save(model, val_accuracy, epoch)

            # Save plots
            metrics_plotter.plot_and_save(train_losses, val_losses, "Loss", epoch)
            metrics_plotter.plot_and_save(train_accuracies, val_accuracies, "Accuracy", epoch)
            
            # Save epoch results
            epoch_writer.writerow([
                config_idx, epoch + 1, avg_train_loss, train_accuracy,
                avg_val_loss, val_accuracy, val_f1, current_lr
            ])

            # Early stopping check
            if early_stopping(avg_val_loss):
                print(f"Early stopping triggered at epoch {epoch + 1}!")
                break

    return model

def evaluate_model(model, test_loader, config_idx, results_dir, test_csv_path):
    criterion = nn.CrossEntropyLoss()
    model.eval()

    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_test_targets = []
    all_test_predicted = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()

            all_test_targets.extend(targets.cpu().numpy())
            all_test_predicted.extend(predicted.cpu().numpy())

            pbar.set_postfix({
                "Loss": f"{test_loss / (batch_idx + 1):.4f}",
                "Acc": f"{test_correct / test_total:.4f}"
            })

    # Calculate test metrics
    test_accuracy = test_correct / test_total
    avg_test_loss = test_loss / len(test_loader)
    test_f1 = f1_score(all_test_targets, all_test_predicted, average="weighted")

    # Save test results
    with open(test_csv_path, mode='a', newline='') as test_file:
        test_writer = csv.writer(test_file)
        test_writer.writerow([config_idx, avg_test_loss, test_accuracy, test_f1])

    print(f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.4f}, Test F1: {test_f1:.4f}")

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    print(f"Using device: {DEVICE}")
    print(f"PyTorch version: {torch.__version__}")

    # Create directories
    results_dir = "../Hasil Eksperimen"
    os.makedirs(results_dir, exist_ok=True)

    # Initialize CSV files
    config_csv_path = os.path.join(results_dir, "augment_mobilenet_model_configurations.csv")
    epoch_csv_path = os.path.join(results_dir, "augment_mobilenet_epoch_results.csv")
    test_csv_path = os.path.join(results_dir, "augment_mobilenet_test_results.csv")
    if not os.path.exists(config_csv_path):
        with open(config_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Config", "Patch Size", "Num Heads", "Embedding Dim", "Num Transformer Layers"])
    if not os.path.exists(epoch_csv_path):
        with open(epoch_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Config", "Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc", "F1 Score", "Learning Rate"])
    if not os.path.exists(test_csv_path):
        with open(test_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Config", "Test Loss", "Test Acc", "F1 Score"])

    # Data preparation
    train_transforms, test_transforms = create_transforms()
    data_train, data_val, data_test = load_and_split_data("../data/fer2013_clean.csv")
    train_dataset, val_dataset, test_dataset = create_datasets(data_train, data_val, data_test, train_transforms, test_transforms)
    train_loader, val_loader, test_loader = create_dataloaders(train_dataset, val_dataset, test_dataset)

    # Model configurations to test
    patch_sizes = [16, 32]
    num_heads = [8]
    embedding_dims = [256, 384, 512, 768]
    num_transformer_layers = [6, 12]
    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    config_idx = 1  # Starting configuration index

    with open(config_csv_path, mode='a', newline='') as config_file:
        config_writer = csv.writer(config_file)
        
        for patch_size in patch_sizes:
            for num_head in num_heads:
                for embedding_dim in embedding_dims:
                    for num_transformer_layer in num_transformer_layers:
                        print(f"\nStarting model{config_idx} with configuration:")
                        print(f"Patch size: {patch_size}, Num heads: {num_head}, "
                              f"Embedding dim: {embedding_dim}, Num layers: {num_transformer_layer}")

                        # Model initialization
                        model = ViT(
                            num_classes=len(class_names),
                            in_channels=3,
                            patch_size=patch_size,
                            num_heads=num_head,
                            embedding_dim=embedding_dim,
                            num_transformer_layers=num_transformer_layer
                        ).to(DEVICE)

                        # Train the model
                        trained_model = train_model(model, train_loader, val_loader, config_idx, results_dir, epoch_csv_path)

                        # Evaluate on test set
                        evaluate_model(trained_model, test_loader, config_idx, results_dir, test_csv_path)

                        # Save configuration
                        config_writer.writerow([
                            config_idx, patch_size, num_head, 
                            embedding_dim, num_transformer_layer
                        ])

                        # Clean up
                        del trained_model
                        torch.cuda.empty_cache()
                        gc.collect()

                        config_idx += 1

if __name__ == "__main__":
    main()