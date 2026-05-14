# Classification Models

Hey there! 👋 This is where we keep our three powerful CNN models that help classify spinal X-ray images. The goal? Figure out whether an image shows pathology or is all clear (no findings).

## 📋 Overview

We've built an ensemble framework that brings together three different deep learning architectures. Think of it as getting three expert opinions instead of just one – it really helps improve our results!

### Models Included

| Model | Parameters | AUROC | Sensitivity | Specificity | F1-Score |
|-------|-----------|-------|-------------|-------------|----------|
| **DenseNet-121** | 8M | 90.25% | 83.32% | 82.34% | 82.46% |
| **EfficientNetV2-S** | 21M | 89.44% | 70.80% | 91.12% | 79.85% |
| **ResNet-50** | 25.6M | 88.88% | 82.72% | 78.13% | 80.42% |
| **Ensemble** | — | **90.25%** | **83.32%** | **82.34%** | **82.46%** |

## 🗂️ Files

```
classification/
├── README.md                    # This file
├── train_densenet121.py        # DenseNet-121 training script
├── train_efficientnet.py       # EfficientNetV2-S training script
├── train_resnet50.py           # ResNet-50 training script
└── ensemble_submission.py       # Ensemble inference and submission
```

## 🚀 Quick Start

### What You'll Need First

Before diving in, let's get all the necessary libraries installed:

```bash
pip install torch torchvision timm pandas numpy scikit-learn pillow tqdm
```

### Training Individual Models

Ready to train some models? Let's go through each one!

#### 1. DenseNet-121

```bash
python train_densenet121.py
```

**Configuration**:
- Architecture: 4 dense blocks (6, 12, 24, 16 layers)
- Growth rate: k=32
- Compression factor: θ=0.5
- Input size: 384×384
- Batch size: 32
- Epochs: 50
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)

**What to Expect**:
- Grab a coffee (or four) – this takes about 4 hours on an RTX 3050
- You'll get a shiny checkpoint file: `densenet121_best.pth`
- AUROC should land around 90.25% – pretty sweet!

#### 2. EfficientNetV2-S

```bash
python train_efficientnet.py
```

**Configuration**:
- Architecture: Fused-MBConv blocks with progressive training
- Input size: 384×384 (progressive: 128→256→384)
- Batch size: 24
- Epochs: 50
- Optimizer: AdamW (lr=1e-4, weight_decay=5e-5)
- Activation: SiLU (Swish)

**What to Expect**:
- Slightly longer training – about 5 hours on an RTX 3050
- Your checkpoint: `efficientnetv2_s_best.pth`
- AUROC around 89.44%
- This one's really good at avoiding false positives (91.12% specificity!)

#### 3. ResNet-50

```bash
python train_resnet50.py
```

**Configuration**:
- Architecture: 4 stages with bottleneck blocks
- Input size: 384×384
- Batch size: 32
- Epochs: 50
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
- Activation: ReLU

**What to Expect**:
- This one's a marathon – about 6 hours on an RTX 3050
- Your checkpoint: `resnet50_best.pth`
- AUROC around 88.88%

### Ensemble Prediction

Once you've trained all three models (congrats, by the way!), let's combine their powers:

```bash
python ensemble_submission.py
```

**Ensemble Strategy**:
- Weighted average of probabilities
- Test-Time Augmentation (TTA): horizontal flip
- Optimal threshold search for F1-score maximization
- Final predictions: `ensemble_submission.csv`

## 🔬 Training Details

### Data Augmentation

Let's talk about how we make our models more robust. All three models use the same augmentation tricks to learn better:

```python
train_transforms = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

**What We're Doing Here**:
- Resize: Everything gets resized to 384×384 pixels (keeps things consistent)
- Horizontal flip: 50/50 chance – because spine X-rays can be flipped
- Rotation: We wiggle them ±10 degrees (people aren't always perfectly straight!)
- Color jitter: Mess with brightness and contrast by ±20% (real-world images vary)
- Normalization: Using ImageNet statistics (industry standard)

### Loss Functions

**Binary Cross-Entropy with Logits** (sounds fancy, but it's just how we measure mistakes):
```
Loss = -[y·log(σ(ŷ)) + (1-y)·log(1-σ(ŷ))]
```

In plain English:
- y: the actual answer (0 for healthy, 1 for pathology)
- ŷ: what our model thinks (before converting to probability)
- σ: sigmoid function (turns model output into a probability)

**With Class Weights** (optional for balanced training):
```python
pos_weight = torch.tensor([n_negative / n_positive])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

### Optimization

**AdamW Optimizer**:
- Decoupled weight decay (fixes Adam's L2 regularization issue)
- β₁ = 0.9, β₂ = 0.999
- ε = 1e-8
- Learning rate schedule: Cosine annealing

**Learning Rate Schedule**:
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=50, eta_min=1e-6
)
```

### Training Procedure

```python
# Pseudo-code for training loop
for epoch in range(num_epochs):
    # Training phase
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Validation phase
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = torch.sigmoid(model(images))
            # Calculate metrics: AUROC, F1, Sensitivity, Specificity
    
    # Update learning rate
    scheduler.step()
    
    # Save best checkpoint based on AUROC
    if auroc > best_auroc:
        torch.save(model.state_dict(), 'best_checkpoint.pth')
```

## 📊 Evaluation Metrics

Let's break down how we measure success!

### 1. AUROC (Area Under ROC Curve)

```python
from sklearn.metrics import roc_auc_score
auroc = roc_auc_score(true_labels, predicted_probabilities)
```

**What This Means**: If you pick a random pathology case and a random healthy case, AUROC tells you the probability our model ranks the pathology case higher. The closer to 100%, the better!

### 2. Sensitivity (Recall/True Positive Rate)

```
Sensitivity = TP / (TP + FN)
```

**Why This Matters**: This tells us how good we are at catching actual pathology cases. High sensitivity means we're not missing sick patients – super important in medicine!

### 3. Specificity (True Negative Rate)

```
Specificity = TN / (TN + FP)
```

**Why This Matters**: This measures how well we identify healthy patients. High specificity means we're not freaking people out with false alarms.

### 4. F1-Score (Harmonic Mean of Precision and Recall)

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

where:
```
Precision = TP / (TP + FP)
Recall = Sensitivity = TP / (TP + FN)
```

### Optimal Threshold Selection

Binary classification requires threshold τ to convert probabilities to labels:

```python
# Search for optimal threshold
thresholds = np.arange(0.35, 0.60, 0.001)
best_f1 = 0
best_threshold = 0.5

for tau in thresholds:
    predictions = (probabilities >= tau).astype(int)
    f1 = f1_score(true_labels, predictions)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = tau
```

## 🧪 Model Architecture Details

### DenseNet-121

**Key Features**:
- Dense connectivity: Each layer receives all previous feature maps
- Alleviates vanishing gradient problem
- Encourages feature reuse
- Fewer parameters than traditional CNNs

**Composite Function H(·)**:
```
H(x) = Conv3×3(ReLU(BatchNorm(x)))
```

**Dense Block Transition**:
```
x_ℓ = [x₀, x₁, ..., x_{ℓ-1}] ⊕ H_ℓ([x₀, x₁, ..., x_{ℓ-1}])
```

### EfficientNetV2-S

**Key Features**:
- Compound scaling: Balances depth, width, resolution
- Fused-MBConv blocks: Faster than MBConv
- Progressive training: Gradually increases image size
- Squeeze-and-Excitation (SE) attention

**Scaling Factors**:
- Depth: α^φ
- Width: β^φ
- Resolution: γ^φ
- Constraint: α·β²·γ² ≈ 2

### ResNet-50

**Key Features**:
- Residual connections: Enables training very deep networks
- Bottleneck design: 1×1, 3×3, 1×1 convolutions
- Batch normalization after each convolution
- Identity shortcuts for gradient flow

**Residual Block**:
```
y = F(x, {W_i}) + x
```

where F(·) represents the residual mapping.

## 💾 Model Checkpoints

After training, you'll have the following checkpoints:

```
checkpoints/
├── densenet121_best.pth          # Best DenseNet-121 (AUROC-based)
├── efficientnetv2_s_best.pth     # Best EfficientNetV2-S
├── resnet50_best.pth             # Best ResNet-50
└── training_logs/
    ├── densenet121_log.csv       # Training history
    ├── efficientnetv2_s_log.csv
    └── resnet50_log.csv
```

**Checkpoint Contents**:
```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'best_auroc': best_auroc,
    'metrics': {
        'auroc': auroc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1_score': f1_score
    }
}
```

## 🔍 Troubleshooting

Run into issues? Don't worry, we've all been there! Here are some common problems and fixes:

### Out of Memory (OOM) Error

Your GPU is crying for help? Let's fix that!

**Solution 1**: Cut down the batch size
```python
BATCH_SIZE = 16  # Instead of 32 - your GPU will thank you
```

**Solution 2**: Use gradient accumulation (fancy way to process more data with less memory)
```python
accumulation_steps = 2
for i, (images, labels) in enumerate(train_loader):
    outputs = model(images)
    loss = criterion(outputs, labels) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Overfitting

**The Problem**: Your model is basically memorizing the training data instead of actually learning. You'll see great training scores but terrible validation scores.

**How to Fix It**:
1. Throw more augmentation at it (make it work harder)
2. Add dropout (p=0.5) – randomly ignore some neurons during training
3. Crank up weight decay (penalize complexity)
4. Use early stopping (quit while you're ahead!)

### Slow Convergence

**Training feels like watching paint dry? Try these**:
1. Bump up the learning rate (but be careful – too high and things explode)
2. Use learning rate warmup (ease into it)
3. Check your data loading pipeline (use `num_workers=4` for parallel loading)
4. Enable mixed precision training (faster and uses less memory)

```python
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    outputs = model(images)
    loss = criterion(outputs, labels)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 📈 Expected Results Timeline

Wondering if your training is on track? Here's what normal progress looks like:

### DenseNet-121
- Epoch 10: AUROC ~85% (off to a good start!)
- Epoch 25: AUROC ~88% (getting there...)
- Epoch 40-50: AUROC converges to ~90% (nailed it!)

### EfficientNetV2-S
- Epoch 15: AUROC ~86% (solid beginning)
- Epoch 30: AUROC ~88.5% (looking good)
- Epoch 45-50: AUROC converges to ~89.4% (that's our target!)

### ResNet-50
- Epoch 10: AUROC ~84% (right on schedule)
- Epoch 25: AUROC ~87% (making progress)
- Epoch 40-50: AUROC converges to ~88.9% (and we're done!)

## 🔗 References

1. Huang et al., "Densely Connected Convolutional Networks", CVPR 2017
2. Tan & Le, "EfficientNetV2: Smaller Models and Faster Training", ICML 2021
3. He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
4. Loshchilov & Hutter, "Decoupled Weight Decay Regularization", ICLR 2019

---

**Want to dive deeper?** Check out the detailed methodology and all the math in [`../docs/methodology.md`](../docs/methodology.md)

**Stuck or confused?** No worries! Check the main [README.md](../README.md) or open an issue on GitHub – we're here to help! 🚀
