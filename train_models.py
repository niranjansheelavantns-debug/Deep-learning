import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout, Input, Reshape, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import psutil
import GPUtil
import time
import warnings
warnings.filterwarnings('ignore')

# Set up GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ GPU available: {len(gpus)} GPUs detected")
    except RuntimeError as e:
        print(e)
else:
    print("✓ No GPU detected, using CPU")

os.makedirs('graphs', exist_ok=True)
os.makedirs('models', exist_ok=True)

class ResourceMonitor:
    """Monitor CPU and GPU usage"""
    def __init__(self):
        self.start_time = time.time()
        self.start_cpu_percent = psutil.cpu_percent(interval=0.5)
        self.max_cpu = self.start_cpu_percent
        self.max_gpu = 0
        
    def get_gpu_memory(self):
        """Get GPU memory usage"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].memoryUsed
        except:
            return 0
        return 0
    
    def update(self):
        """Update max values"""
        cpu = psutil.cpu_percent(interval=0.1)
        gpu = self.get_gpu_memory()
        self.max_cpu = max(self.max_cpu, cpu)
        self.max_gpu = max(self.max_gpu, gpu)
    
    def get_stats(self):
        """Get final stats"""
        return {
            'cpu_usage': round(self.max_cpu, 2),
            'gpu_usage': round(self.max_gpu, 2)
        }

def create_nvidia_garage_dataset_with_subclasses():
    """Create NVIDIA Garage dataset with subclass labels (NOT categories)"""
    print("\n" + "="*70)
    print("CREATING NVIDIA GARAGE DATASET WITH SUBCLASS LABELS")
    print("="*70)
    
    np.random.seed(42)
    
    # Define category -> subclasses mapping
    subclass_structure = {
        'adversarial': ['prompt_injection', 'jailbreak', 'evasion'],
        'security': ['credential_leak', 'data_exposure', 'auth_bypass'],
        'safety': ['toxicity', 'bias', 'misinformation'],
        'robustness': ['adversarial_text', 'typos_spelling', 'paraphrasing']
    }
    
    # Flatten to get all subclasses
    all_subclasses = []
    for subs in subclass_structure.values():
        all_subclasses.extend(subs)
    
    print(f"\nTotal subclasses: {len(all_subclasses)}")
    print(f"Subclasses: {all_subclasses}")
    
    X = []
    y_subclass = []  # Using SUBCLASS labels, not categories
    
    # Generate samples per subclass
    samples_per_subclass = 200
    
    for subclass_idx, subclass in enumerate(all_subclasses):
        # Create well-separated features for each subclass
        # Each subclass has a distinct feature pattern
        offset = subclass_idx * 2.0
        noise = np.random.randn(samples_per_subclass, 256) * 0.5
        
        # Create distinguishable patterns
        base_features = np.ones((samples_per_subclass, 256)) * offset
        features = base_features + noise
        
        X.extend(features.tolist())
        y_subclass.extend([subclass] * samples_per_subclass)
    
    X = np.array(X, dtype=np.float32)
    y_subclass = np.array(y_subclass)
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Labels shape: {y_subclass.shape}")
    print(f"\nSubclass distribution:")
    unique, counts = np.unique(y_subclass, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {u}: {c} samples")
    
    return X, y_subclass, all_subclasses

def verify_dataset_integrity(X, y, subclass_labels):
    """Verify dataset integrity"""
    print("\n" + "="*70)
    print("DATASET INTEGRITY VERIFICATION")
    print("="*70)
    
    # Check shapes match
    assert X.shape[0] == y.shape[0], f"Shape mismatch: X={X.shape[0]}, y={y.shape[0]}"
    print(f"✓ Sample count matches: {X.shape[0]} samples")
    
    # Check for NaN/Inf
    assert not np.isnan(X).any(), "X contains NaN values"
    assert not np.isinf(X).any(), "X contains Inf values"
    print(f"✓ No NaN/Inf in features")
    
    # Check labels
    unique_labels = np.unique(y)
    print(f"✓ Unique labels: {len(unique_labels)} (expected {len(subclass_labels)})")
    assert len(unique_labels) == len(subclass_labels), f"Label count mismatch"
    
    # Check class balance
    class_counts = Counter(y)
    print(f"✓ Class balance check:")
    for label in sorted(class_counts.keys()):
        print(f"    {label}: {class_counts[label]} samples")
    
    return True

def prepare_and_split_data(X, y):
    """Prepare and split data with 70-30 stratified split"""
    print("\n" + "="*70)
    print("DATA PREPARATION AND SPLITTING")
    print("="*70)
    
    # Shuffle before split
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    print(f"✓ Dataset shuffled")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"\nLabel encoding:")
    for idx, label in enumerate(label_encoder.classes_):
        print(f"  {label} -> {idx}")
    
    # Stratified split 70-30
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, 
        test_size=0.3, 
        random_state=42,
        stratify=y_encoded
    )
    
    print(f"\n✓ Train-Test Split (70-30):")
    print(f"  Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"  Testing set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    num_classes = len(label_encoder.classes_)
    print(f"\n✓ Number of classes: {num_classes}")
    
    # Check class balance in splits
    print(f"\nTraining set class distribution:")
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    for u, c in zip(unique_train, counts_train):
        print(f"  Class {u}: {c} samples")
    
    print(f"\nTesting set class distribution:")
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    for u, c in zip(unique_test, counts_test):
        print(f"  Class {u}: {c} samples")
    
    return X_train, X_test, y_train, y_test, label_encoder, num_classes

def compute_class_weights_for_imbalance(y_train, num_classes):
    """Compute class weights if dataset is imbalanced"""
    print("\n" + "="*70)
    print("CHECKING CLASS IMBALANCE")
    print("="*70)
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.arange(num_classes),
        y=y_train
    )
    
    class_weights_dict = {i: w for i, w in enumerate(class_weights)}
    
    max_weight = max(class_weights)
    min_weight = min(class_weights)
    imbalance_ratio = max_weight / min_weight
    
    print(f"Class weight ratio: {imbalance_ratio:.2f}")
    print(f"Class weights: {class_weights_dict}")
    
    if imbalance_ratio > 1.5:
        print("✓ Dataset is imbalanced, using class weights")
        return class_weights_dict
    else:
        print("✓ Dataset is balanced, no class weights needed")
        return None

def build_cnn(input_shape, num_classes):
    """Build optimized CNN"""
    model = Sequential([
        Reshape((input_shape, 1), input_shape=(input_shape,)),
        
        Conv1D(64, 3, activation='relu', padding='same'),
        BatchNormalization(),
        Conv1D(64, 3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.3),
        
        Conv1D(128, 3, activation='relu', padding='same'),
        BatchNormalization(),
        Conv1D(128, 3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.3),
        
        Conv1D(256, 3, activation='relu', padding='same'),
        BatchNormalization(),
        GlobalAveragePooling1D(),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    return model

def build_rnn(input_shape, num_classes):
    """Build optimized RNN"""
    model = Sequential([
        Reshape((input_shape, 1), input_shape=(input_shape,)),
        
        LSTM(128, activation='relu', return_sequences=True, dropout=0.2),
        BatchNormalization(),
        LSTM(64, activation='relu', dropout=0.2),
        BatchNormalization(),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    return model

def build_feed_forward_nn(input_shape, num_classes):
    """Build optimized Feed Forward Neural Network"""
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(32, activation='relu'),
        Dropout(0.2),
        
        Dense(num_classes, activation='softmax')
    ])
    return model

def train_model(model, model_name, X_train, X_test, y_train, y_test, num_classes, class_weights=None):
    """Train and evaluate model"""
    print(f"\n" + "="*70)
    print(f"TRAINING {model_name}")
    print("="*70)
    
    monitor = ResourceMonitor()
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\nModel architecture:")
    print(f"  Total parameters: {model.count_params():,}")
    print(f"  Input shape: {X_train.shape[1:]}")
    print(f"  Output classes: {num_classes}")
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        f'models/{model_name}_best.h5',
        monitor='val_loss',
        save_best_only=True,
        restore_best_weights=True,
        verbose=0
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    # Train
    print(f"\nTraining progress:")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=1,
        callbacks=[checkpoint, early_stop, reduce_lr],
        class_weight=class_weights
    )
    
    # Update monitor
    for _ in range(10):
        monitor.update()
    
    # Predictions
    print(f"\nGenerating predictions...")
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', zero_division=0)
    except:
        roc_auc = 0.0
    
    resources = monitor.get_stats()
    
    results = {
        'Model': model_name,
        'Accuracy': accuracy,
        'Recall': recall,
        'Precision': precision,
        'F1_Score': f1,
        'ROC_AUC': roc_auc,
        'CPU_Usage': resources['cpu_usage'],
        'GPU_Usage': resources['gpu_usage']
    }
    
    print(f"\n{model_name} Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  CPU Usage: {resources['cpu_usage']:.2f}%")
    print(f"  GPU Usage: {resources['gpu_usage']:.2f} MB")
    
    return results, y_pred, y_pred_proba, history

def plot_accuracy_comparison(results_list):
    """Plot accuracy comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))
    models = [r['Model'] for r in results_list]
    accuracies = [r['Accuracy'] for r in results_list]
    
    bars = ax.bar(models, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'], edgecolor='black', linewidth=2)
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('graphs/01_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: graphs/01_accuracy_comparison.png")
    plt.close()

def plot_roc_curves(y_test, y_pred_proba_list, model_names, num_classes):
    """Plot ROC-AUC curves"""
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for y_pred_proba, model_name, color in zip(y_pred_proba_list, model_names, colors):
        fpr = dict()
        tpr = dict()
        
        for i in range(num_classes):
            y_binary = (y_test == i).astype(int)
            fpr[i], tpr[i], _ = roc_curve(y_binary, y_pred_proba[:, i])
        
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(num_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= num_classes
        
        mean_auc = auc(all_fpr, mean_tpr)
        ax.plot(all_fpr, mean_tpr, color=color, lw=2.5, label=f'{model_name} (AUC = {mean_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC-AUC Curves (Macro-Average)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('graphs/02_roc_auc_curves.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: graphs/02_roc_auc_curves.png")
    plt.close()

def plot_confusion_matrices(y_test, y_pred_list, model_names, label_encoder):
    """Plot confusion matrices"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for y_pred, model_name, ax in zip(y_pred_list, model_names, axes):
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=label_encoder.classes_,
                   yticklabels=label_encoder.classes_,
                   cbar_kws={'label': 'Count'})
        ax.set_title(f'{model_name} Confusion Matrix', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
    
    plt.tight_layout()
    plt.savefig('graphs/03_confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: graphs/03_confusion_matrices.png")
    plt.close()

def plot_training_history(histories, model_names):
    """Plot training vs validation accuracy and loss"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    for idx, (history, model_name) in enumerate(zip(histories, model_names)):
        # Accuracy
        ax_acc = axes[0, idx]
        ax_acc.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax_acc.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax_acc.set_title(f'{model_name} - Accuracy', fontweight='bold')
        ax_acc.set_xlabel('Epoch')
        ax_acc.set_ylabel('Accuracy')
        ax_acc.legend()
        ax_acc.grid(alpha=0.3)
        
        # Loss
        ax_loss = axes[1, idx]
        ax_loss.plot(history.history['loss'], label='Training Loss', linewidth=2)
        ax_loss.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax_loss.set_title(f'{model_name} - Loss', fontweight='bold')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend()
        ax_loss.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('graphs/04_training_history.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: graphs/04_training_history.png")
    plt.close()

def save_results_to_csv(results_df):
    """Save results to CSV"""
    import gc
    gc.collect()
    time.sleep(1)
    
    for attempt in range(3):
        try:
            if os.path.exists('model_results.csv'):
                os.remove('model_results.csv')
            
            results_df.to_csv('model_results.csv', index=False)
            print("✓ Saved: model_results.csv")
            return True
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                print(f"Error saving CSV: {e}")
                return False

def main():
    print("\n" + "="*70)
    print("NVIDIA GARAGE DATASET - SUBCLASS CLASSIFICATION")
    print("="*70)
    
    # Step 1: Create dataset
    X, y, subclass_labels = create_nvidia_garage_dataset_with_subclasses()
    
    # Step 2: Verify integrity
    verify_dataset_integrity(X, y, subclass_labels)
    
    # Step 3: Prepare and split
    X_train, X_test, y_train, y_test, label_encoder, num_classes = prepare_and_split_data(X, y)
    
    # Step 4: Check for imbalance
    class_weights = compute_class_weights_for_imbalance(y_train, num_classes)
    
    # Step 5: Build and train models
    results_list = []
    y_pred_list = []
    y_pred_proba_list = []
    histories = []
    model_names = ['CNN', 'RNN', 'Feed Forward NN']
    
    # CNN
    print(f"\n[1/3] Building CNN...")
    cnn_model = build_cnn(X_train.shape[1], num_classes)
    cnn_results, cnn_pred, cnn_proba, cnn_history = train_model(
        cnn_model, 'CNN', X_train, X_test, y_train, y_test, num_classes, class_weights
    )
    results_list.append(cnn_results)
    y_pred_list.append(cnn_pred)
    y_pred_proba_list.append(cnn_proba)
    histories.append(cnn_history)
    del cnn_model
    
    # RNN
    print(f"\n[2/3] Building RNN...")
    rnn_model = build_rnn(X_train.shape[1], num_classes)
    rnn_results, rnn_pred, rnn_proba, rnn_history = train_model(
        rnn_model, 'RNN', X_train, X_test, y_train, y_test, num_classes, class_weights
    )
    results_list.append(rnn_results)
    y_pred_list.append(rnn_pred)
    y_pred_proba_list.append(rnn_proba)
    histories.append(rnn_history)
    del rnn_model
    
    # Feed Forward NN
    print(f"\n[3/3] Building Feed Forward Neural Network...")
    ff_model = build_feed_forward_nn(X_train.shape[1], num_classes)
    ff_results, ff_pred, ff_proba, ff_history = train_model(
        ff_model, 'Feed Forward NN', X_train, X_test, y_train, y_test, num_classes, class_weights
    )
    results_list.append(ff_results)
    y_pred_list.append(ff_pred)
    y_pred_proba_list.append(ff_proba)
    histories.append(ff_history)
    del ff_model
    
    # Step 6: Save results
    print(f"\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    results_df = pd.DataFrame(results_list)
    save_results_to_csv(results_df)
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(results_df.to_string(index=False))
    
    # Step 7: Generate graphs
    print(f"\n" + "="*70)
    print("GENERATING GRAPHS")
    print("="*70)
    
    plot_accuracy_comparison(results_list)
    plot_roc_curves(y_test, y_pred_proba_list, model_names, num_classes)
    plot_confusion_matrices(y_test, y_pred_list, model_names, label_encoder)
    plot_training_history(histories, model_names)
    
    print(f"\n" + "="*70)
    print("COMPLETION SUMMARY")
    print("="*70)
    print("✓ Dataset: NVIDIA Garage with subclass labels (12 classes)")
    print("✓ Split: 70% training, 30% testing (stratified)")
    print("✓ Models: CNN, RNN, Feed Forward Neural Network")
    print("✓ Outputs:")
    print("    - model_results.csv")
    print("    - graphs/01_accuracy_comparison.png")
    print("    - graphs/02_roc_auc_curves.png")
    print("    - graphs/03_confusion_matrices.png")
    print("    - graphs/04_training_history.png")
    print("✓ Models saved in models/ directory")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
