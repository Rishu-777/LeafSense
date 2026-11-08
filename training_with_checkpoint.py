import os
import shutil
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

print("\n" + "="*70)
print("PLANT DISEASE DETECTION - TRAINING WITH CHECKPOINT SUPPORT")
print("="*70 + "\n")

# ============ CONFIGURATION ============
SOURCE_DIR = "Dataset/raw/color"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
MODEL_SAVE_PATH = "Flask Deployed App/plant_disease_model_1.pt"
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth")

# Training parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001
TRAIN_RATIO = 0.8

print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print()

# ============ STEP 1: VERIFY DATASET PATH ============
print("STEP 1: VERIFYING DATASET PATH")
print("-" * 70)

print(f"Looking for dataset at: {SOURCE_DIR}")

if not os.path.exists(SOURCE_DIR):
    print(f"❌ ERROR: Path not found!")
    print(f"\nYour folder structure should be:")
    print(f"  Plant-Disease-Detection-main/")
    print(f"  ├── Dataset/raw/color/ (with disease folders)")
    print(f"  ├── Flask Deployed App/")
    print(f"  └── complete_training_final.py")
    exit()

# Get disease folders
disease_folders = [d for d in os.listdir(SOURCE_DIR)
                  if os.path.isdir(os.path.join(SOURCE_DIR, d))]
disease_folders.sort()

print(f"✓ Found dataset!")
print(f"✓ Disease categories: {len(disease_folders)}")
print()

# ============ STEP 2: DATASET ORGANIZATION ============
print("STEP 2: ORGANIZING DATASET")
print("-" * 70)

if os.path.exists(os.path.join(TRAIN_DIR, "Apple_Scab")):
    print("✓ Dataset already organized!")
    print(f"  Training data: {TRAIN_DIR}/")
    print(f"  Validation data: {VAL_DIR}/")
else:
    print("Organizing dataset into train/val split...\n")

    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)

    total_images = 0

    for idx, disease in enumerate(disease_folders, 1):
        disease_path = os.path.join(SOURCE_DIR, disease)
        images = [f for f in os.listdir(disease_path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if len(images) == 0:
            continue

        print(f"  [{idx}/{len(disease_folders)}] {disease}: {len(images)} images", end=" -> ")

        random.shuffle(images)
        split_point = int(len(images) * TRAIN_RATIO)
        train_images = images[:split_point]
        val_images = images[split_point:]

        train_disease_dir = os.path.join(TRAIN_DIR, disease)
        val_disease_dir = os.path.join(VAL_DIR, disease)
        os.makedirs(train_disease_dir, exist_ok=True)
        os.makedirs(val_disease_dir, exist_ok=True)

        for img in train_images:
            src = os.path.join(disease_path, img)
            dst = os.path.join(train_disease_dir, img)
            try:
                shutil.copy2(src, dst)
            except:
                pass

        for img in val_images:
            src = os.path.join(disease_path, img)
            dst = os.path.join(val_disease_dir, img)
            try:
                shutil.copy2(src, dst)
            except:
                pass

        print(f"Train: {len(train_images)}, Val: {len(val_images)} ✓")
        total_images += len(images)

    print(f"\n✓ Organization complete! Total images: {total_images}")

print()

# ============ STEP 3: DATA LOADING ============
print("STEP 3: LOADING DATA")
print("-" * 70)

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

try:
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transforms)
except Exception as e:
    print(f"❌ ERROR loading dataset: {e}")
    exit()

NUM_CLASSES = len(train_dataset.classes)

print(f"✓ Data loaded successfully!")
print(f"  Classes: {NUM_CLASSES}")
print(f"  Training samples: {len(train_dataset)}")
print(f"  Validation samples: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                         shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                       shuffle=False, num_workers=0)

print()

# ============ STEP 4: BUILD MODEL ============
print("STEP 4: BUILDING MODEL")
print("-" * 70)

model = models.resnet50(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

for param in model.layer4.parameters():
    param.requires_grad = True

num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, NUM_CLASSES)
)

model = model.to(DEVICE)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"✓ Model built successfully!")
print(f"  Trainable parameters: {trainable_params:,}")

print()

# ============ STEP 5: SETUP TRAINING & CHECKPOINT ============
print("STEP 5: SETTING UP TRAINING")
print("-" * 70)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Create checkpoint directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ===== CHECKPOINT LOADING =====
start_epoch = 0
best_val_accuracy = 0
training_history = []

if os.path.exists(CHECKPOINT_PATH):
    print(f"⚠️  Found previous checkpoint!")
    print(f"   Do you want to:")
    print(f"   1. Resume from checkpoint")
    print(f"   2. Start fresh (delete checkpoint)")
    choice = input("   Enter 1 or 2: ").strip()

    if choice == "1":
        print(f"\n   Loading checkpoint...")
        checkpoint = torch.load(CHECKPOINT_PATH)

        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])

        start_epoch = checkpoint['epoch']
        best_val_accuracy = checkpoint['best_accuracy']
        training_history = checkpoint['history']

        print(f"   ✓ Checkpoint loaded!")
        print(f"   ✓ Resuming from epoch {start_epoch + 1}")
        print(f"   ✓ Previous best accuracy: {best_val_accuracy:.2f}%")
        print(f"   ✓ Training epochs completed: {len(training_history)}")

    elif choice == "2":
        print(f"\n   Deleting checkpoint and starting fresh...")
        os.remove(CHECKPOINT_PATH)
        start_epoch = 0
        best_val_accuracy = 0
        training_history = []
    else:
        print(f"   Invalid choice. Starting fresh...")
        start_epoch = 0
        best_val_accuracy = 0
        training_history = []

print(f"✓ Training setup complete!")
print()

# ============ STEP 6: TRAINING LOOP WITH CHECKPOINT ============
print("STEP 6: TRAINING MODEL")
print("="*70)
print(f"NOTE: You can stop training anytime with Ctrl+C")
print(f"      Progress will be saved and you can resume later!")
print("="*70)

patience_counter = 0
patience = 3

try:
    for epoch in range(start_epoch, EPOCHS):
        # --------- TRAINING PHASE ---------
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        print(f"\n[Epoch {epoch+1}/{EPOCHS}] Training...", flush=True)

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            if (batch_idx + 1) % 100 == 0:
                current_acc = 100 * train_correct / train_total
                print(f"  Batch [{batch_idx + 1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}, Acc: {current_acc:.2f}%")

        train_accuracy = 100 * train_correct / train_total
        train_loss /= len(train_loader)

        # --------- VALIDATION PHASE ---------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        print(f"[Epoch {epoch+1}/{EPOCHS}] Validating...", flush=True)

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                if (batch_idx + 1) % 100 == 0:
                    print(f"  Batch [{batch_idx + 1}/{len(val_loader)}]")

        val_accuracy = 100 * val_correct / val_total
        val_loss /= len(val_loader)

        # Store history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_accuracy,
            'val_loss': val_loss,
            'val_acc': val_accuracy
        })

        # --------- EPOCH RESULTS ---------
        print("\n" + "-"*70)
        print(f"Epoch [{epoch+1}/{EPOCHS}] RESULTS:")
        print(f"  Training   → Loss: {train_loss:.4f} | Acc: {train_accuracy:.2f}%")
        print(f"  Validation → Loss: {val_loss:.4f} | Acc: {val_accuracy:.2f}%")

        scheduler.step()

        # --------- SAVE BEST MODEL ---------
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  ✓ NEW BEST MODEL SAVED! (Accuracy: {best_val_accuracy:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement (Patience: {patience_counter}/{patience})")

            if patience_counter >= patience:
                print(f"\n  Early stopping triggered!")
                break

        # ===== SAVE CHECKPOINT =====
        checkpoint = {
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_accuracy': best_val_accuracy,
            'history': training_history
        }
        torch.save(checkpoint, CHECKPOINT_PATH)
        print(f"  ✓ Checkpoint saved (can resume training later)")

        print("-"*70)

except KeyboardInterrupt:
    print("\n\n" + "="*70)
    print("⚠️  TRAINING INTERRUPTED BY USER")
    print("="*70)
    print(f"\n✓ Progress saved!")
    print(f"✓ Checkpoint saved to: {CHECKPOINT_PATH}")
    print(f"\nTo resume training, run this script again and choose option 1")
    print("="*70 + "\n")
    exit()

print()

# ============ TRAINING COMPLETE ============
print("="*70)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("="*70)

print(f"\n✓ Best Validation Accuracy: {best_val_accuracy:.2f}%")
print(f"✓ Model saved to: {MODEL_SAVE_PATH}")
print(f"✓ Total epochs completed: {len(training_history)}")

# Clean up checkpoint after successful completion
if os.path.exists(CHECKPOINT_PATH):
    os.remove(CHECKPOINT_PATH)
    print(f"✓ Checkpoint cleaned up (training complete)")

print(f"\n" + "="*70)
print("NEXT STEPS - TEST YOUR MODEL")
print("="*70)

print(f"\n1. Run Flask Application:")
print(f"   cd 'Flask Deployed App'")
print(f"   python app.py")

print(f"\n2. Open Web Browser:")
print(f"   http://127.0.0.1:5000")

print(f"\n3. Upload leaf images and get predictions!")

if os.path.exists(MODEL_SAVE_PATH):
    model_size_mb = os.path.getsize(MODEL_SAVE_PATH) / (1024 * 1024)
    print(f"\n✓ Model ready for deployment!")
    print(f"✓ Model file: {MODEL_SAVE_PATH} ({model_size_mb:.2f} MB)")

print("="*70 + "\n")
