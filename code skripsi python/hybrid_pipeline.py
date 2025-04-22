import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW
import os
from tqdm import tqdm
from sklearn.metrics import f1_score
from torchvision.models import mobilenet_v3_large
import random
from torch.utils.data import WeightedRandomSampler
from torch import nn
import csv
import gc

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
    # Create transform pipeline manually
    train_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),  # Randomly flip horizontally
        transforms.RandomRotation(10),     # Randomly rotate by 10 degrees
        transforms.RandomResizedCrop(
            size=IMG_SIZE,  # Output size
            scale=(0.8, 1.0)  # Range of the random crop size relative to the input size
        ),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Adjust brightness
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]) 

    # Create transform pipeline manually
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

def create_dataloaders(train_dataset, val_dataset, test_dataset, sampler):
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             generator=torch.Generator().manual_seed(SEED), sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
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
                 img_size:int=224, # Training resolution from Table 3 in ViT paper
                 in_channels:int=3, # Number of channels in input image
                 patch_size:int=16, # Patch size
                 num_transformer_layers:int=12, # Layers from Table 1 for ViT-Base
                 embedding_dim:int=768, # Hidden size D from Table 1 for ViT-Base
                 mlp_size:int=3072, # MLP size from Table 1 for ViT-Base
                 num_heads:int=12, # Heads from Table 1 for ViT-Base
                 attn_dropout:float=0., # Dropout for attention projection
                 mlp_dropout:float=0., # Dropout for dense/MLP layers 
                 embedding_dropout:float=0., # Dropout for patch and position embeddings
                 num_classes:int=1000): # Default for ImageNet but can customize this
        super().__init__()
         
        assert img_size % 32 == 0, f"Image size must be divisible by 32, image size: {img_size}"
        
        self.mobilenet = mobilenet_v3_large(pretrained=True).features
        
        self.projection = nn.Conv2d(in_channels=960, 
                                    out_channels=embedding_dim,
                                    kernel_size=1)
                 
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
       
        self.norm = nn.LayerNorm(normalized_shape=embedding_dim, eps=1e-6)
        self.head = nn.Linear(in_features=embedding_dim, out_features=num_classes)
    
    def forward(self, pixel_values, labels=None):
        
        batch_size = pixel_values.shape[0]

        # Extract features using MobileNet
        features = self.mobilenet(pixel_values)  # Output shape: (batch_size, 1280, H', W')
        features = self.projection(features)  # Project to embedding_dim: (batch_size, embedding_dim, H', W')

        # Flatten the feature maps into a sequence of tokens
        features = features.flatten(2).transpose(1, 2)  # Shape: (batch_size, num_patches, embedding_dim)
        
        class_token = self.class_embedding.expand(batch_size, -1, -1)

        x = torch.cat((class_token, features), dim=1)  # Shape: (batch_size, num_patches + 1, embedding_dim)

        x = x + self.position_embedding

        x = self.embedding_dropout(x)

        x = self.transformer_encoder(x)

        x = self.norm(x)
        
        cls_token_final = x[:, 0]

        logits = self.head(cls_token_final)

        return logits

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
    
class ASAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, eta=0.01, adaptive=True, **kwargs):
        """
        Adaptive SAM (ASAM) optimizer.
        
        Args:
            params: Model parameters.
            base_optimizer: Base optimizer (e.g., SGD, Adam).
            rho: Maximum perturbation radius (default: 0.05).
            eta: Learning rate for rho adaptation (default: 0.01).
            adaptive: Enable layer-wise adaptive rho (default: True).
        """
        defaults = dict(rho=rho, eta=eta, adaptive=adaptive, **kwargs)
        super(ASAM, self).__init__(params, defaults)
        
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.state['step'] = 0
        
        for group in self.param_groups:
            group.setdefault('rho', rho)
            group.setdefault('eta', eta)
            group.setdefault('adaptive', adaptive)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        Perturb parameters adaptively based on layer-wise gradients.
        """
        grad_norm = self._grad_norm()
        
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)
            
            # Layer-wise adaptive rho
            if group['adaptive']:
                layer_grad_norm = torch.norm(
                    torch.stack([torch.norm(p.grad) for p in group['params'] if p.grad is not None]),
                    p=2
                )
                adaptive_rho = group['rho'] * (1 + group['eta'] * layer_grad_norm)
                scale = adaptive_rho / (grad_norm + 1e-12)
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                # Save original parameters
                self.state[p]["old_p"] = p.data.clone()
                
                # Apply adaptive perturbation
                e_w = scale * p.grad
                p.add_(e_w)
        
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        Update parameters using gradients at perturbed point.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                # Restore original parameters
                p.data = self.state[p]["old_p"]
        
        # Base optimizer update
        self.base_optimizer.step()
        
        if zero_grad:
            self.zero_grad()
        
        self.state['step'] += 1

    def _grad_norm(self):
        """
        Compute L2 norm of gradients across all parameters.
        """
        norm = torch.norm(
            torch.stack([
                torch.norm(p.grad) if p.grad is not None else torch.tensor(0.)
                for group in self.param_groups for p in group["params"]
            ]),
            p=2
        )
        return norm

    def step(self, closure=None):
        raise NotImplementedError("ASAM requires first_step() and second_step().")
        
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
    model_saver = BestModelSaver(save_path=results_dir, model_name=f"model{config_idx}_hybrid")
    base_optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    optimizer = ASAM(model.parameters(), base_optimizer, rho=0.05, eta=0.1, adaptive=True)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    early_stopping = EarlyStopping(patience=10, min_delta=0)
    validator = Validator(model=model, criterion=criterion, device=DEVICE)

    # Initialize lists to store training and validation metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    with open(epoch_csv_path, mode='a', newline='') as epoch_file:
        epoch_writer = csv.writer(epoch_file)

        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            # Training
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Backward pass and optimize
                loss.backward()
                optimizer.first_step(zero_grad=True)

                # Second forward-backward pass
                criterion(model(inputs), targets).backward()
                optimizer.second_step(zero_grad=True)  # Update weights

                # Update statistics
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # Update progress bar
                pbar.set_postfix({
                    "Loss": f"{train_loss / (batch_idx + 1):.4f}",
                    "Acc": f"{correct / total:.4f}"
                })

            # Calculate training accuracy and loss
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

def evaluate_model(best_model, test_loader, config_idx, results_dir, test_csv_path):
    criterion = nn.CrossEntropyLoss()
    best_model.eval()

    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_test_targets = []
    all_test_predicted = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = best_model(inputs)
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
    config_csv_path = os.path.join(results_dir, "hybrid_model_configurations.csv")
    epoch_csv_path = os.path.join(results_dir, "hybrid_epoch_results.csv")
    test_csv_path = os.path.join(results_dir, "hybrid_test_results.csv")
    if not os.path.exists(config_csv_path):
        with open(config_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Config", "Num Heads", "Embedding Dim", "Num Transformer Layers"])
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
    data_train, data_val, data_test = load_and_split_data("../data/fer2013v2_clean.csv")
    train_dataset, val_dataset, test_dataset = create_datasets(data_train, data_val, data_test, train_transforms, test_transforms)
    
    train_labels = data_train["emotion"]
    class_counts = np.bincount(train_labels)
    class_weights = 1. / class_counts
    sample_weights = class_weights[train_labels]
    # Create a weighted sampler
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    train_loader, val_loader, test_loader = create_dataloaders(train_dataset, val_dataset, test_dataset, sampler)

    num_heads = [8]
    embedding_dims = [256, 512, 768]
    num_transformer_layers = [6, 12]
    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    config_idx = 1  # Starting configuration index
    
    with open(config_csv_path, mode='a', newline='') as config_file:
        config_writer = csv.writer(config_file)
        
        for num_head in num_heads:
            for embedding_dim in embedding_dims:
                for num_transformer_layer in num_transformer_layers:
                    print(f"\nStarting model{config_idx} with configuration:")
                    print(f"Num heads: {num_head}, "
                          f"Embedding dim: {embedding_dim}, Num layers: {num_transformer_layer}")

                    # Model initialization
                    model = ViT(
                        num_classes=len(class_names),
                        in_channels=3,
                        patch_size=8,
                        num_heads=num_head,
                        embedding_dim=embedding_dim,
                        num_transformer_layers=num_transformer_layer
                    ).to(DEVICE)

                    # Train the model
                    train_model(model, train_loader, val_loader, config_idx, results_dir, epoch_csv_path)
                    
                    # Load best model
                    best_model = ViT(
                        num_classes=len(class_names),
                        in_channels=3,
                        patch_size=8,
                        num_heads=num_head,
                        embedding_dim=embedding_dim,
                        num_transformer_layers=num_transformer_layer
                    ).to(DEVICE)
                    best_model.load_state_dict(torch.load(f"../Hasil Eksperimen/model{config_idx}_hybrid_best.pt", weights_only=False))

                    # Evaluate on test set
                    evaluate_model(best_model, test_loader, config_idx, results_dir, test_csv_path)

                    # Save configuration
                    config_writer.writerow([
                        config_idx, num_head, 
                        embedding_dim, num_transformer_layer
                    ])

                    # Clean up
                    del model
                    del best_model
                    torch.cuda.empty_cache()
                    gc.collect()

                    config_idx += 1


if __name__ == "__main__":
    main()