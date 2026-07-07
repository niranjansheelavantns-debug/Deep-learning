import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score, confusion_matrix, roc_curve, auc
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv2D, MaxPooling2D, Flatten, Dropout, Input, Reshape, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
        print(f"GPU available: {len(gpus)} GPUs detected")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected, using CPU")

# Create directories
os.makedirs('graphs', exist_ok=True)

class ResourceMonitor:
    """Monitor CPU and GPU usage"""
    def __init__(self):
        self.start_time = time.time()
        self.start_cpu_percent = psutil.cpu_percent(interval=1)
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
            'cpu_usage': self.max_cpu,
            'gpu_usage': self.max_gpu
        }

def load_real_dataset():
    """Load MNIST dataset for real high-quality results"""
    print("Loading MNIST dataset...")
    
    # Load MNIST from Keras
    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize to [0, 1]
    x_train_full = x_train_full.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Flatten images to 1D
    x_train_full = x_train_full.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)
    
    print(f"Training data shape: {x_train_full.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Number of classes: {len(np.unique(y_train_full))}")
    
    return x_train_full, x_test, y_train_full, y_test

def prepare_data(X_train, X_test, y_train, y_test):
    """Prepare data with 70-30 split"""
    # Combine and resplit to ensure 70-30 split
    X_combined = np.vstack([X_train, X_test])
    y_combined = np.hstack([y_train, y_test])
    
    # Shuffle and split 70-30
    indices = np.random.permutation(len(X_combined))
    X_combined = X_combined[indices]
    y_combined = y_combined[indices]
    
    split_idx = int(0.7 * len(X_combined))
    X_train_new = X_combined[:split_idx]
    X_test_new = X_combined[split_idx:]
    y_train_new = y_combined[:split_idx]
    y_test_new = y_combined[split_idx:]
    
    num_classes = len(np.unique(y_train_new))
    
    print(f"\nDataset prepared:")
    print(f"Training set size: {X_train_new.shape[0]}")
    print(f"Testing set size: {X_test_new.shape[0]}")
    print(f"Number of classes: {num_classes}")
    print(f"Feature dimension: {X_train_new.shape[1]}")
    
    return X_train_new, X_test_new, y_train_new, y_test_new, num_classes

# Model 1: Optimized CNN
def build_cnn(input_shape, num_classes):
    """Build optimized CNN for high accuracy"""
    model = Sequential([
        Reshape((28, 28, 1), input_shape=(input_shape,)),
        
        # Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Block 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        GlobalAveragePooling2D(),
        
        # Dense layers
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Model 2: Optimized RNN
def build_rnn(input_shape, num_classes):
    """Build optimized RNN for high accuracy"""
    model = Sequential([
        Reshape((28, 28), input_shape=(input_shape,)),
        
        # LSTM layers
        LSTM(128, activation='relu', return_sequences=True, dropout=0.2),
        BatchNormalization(),
        
        LSTM(64, activation='relu', dropout=0.2),
        BatchNormalization(),
        
        # Dense layers
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Model 3: Optimized Dense Neural Network
def build_neural_network(input_shape, num_classes):
    """Build optimized Dense Neural Network for high accuracy"""
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(num_classes, activation='softmax')
    ])
    return model

def train_model(model, model_name, X_train, X_test, y_train, y_test, num_classes):
    """Train and evaluate model"""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    monitor = ResourceMonitor()
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss', 
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=128,
        validation_split=0.2,
        verbose=1,
        callbacks=callbacks
    )
    
    # Update resource monitor
    for _ in range(10):
        monitor.update()
    
    # Predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # AUC (One-vs-Rest for multiclass)
    try:
        auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', zero_division=0)
    except:
        auc_score = 0.0
    
    resources = monitor.get_stats()
    
    results = {
        'Model': model_name,
        'Accuracy': accuracy,
        'Recall': recall,
        'F1_Score': f1,
        'Precision': precision,
        'AUC_Value': auc_score,
        'CPU_Usage': resources['cpu_usage'],
        'GPU_Usage': resources['gpu_usage']
    }
    
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f} ✓")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"AUC Value: {auc_score:.4f}")
    print(f"CPU Usage: {resources['cpu_usage']:.2f}%")
    print(f"GPU Usage: {resources['gpu_usage']:.2f} MB")
    
    return results, y_pred, y_pred_proba, history

def plot_accuracy_comparison(results_list):
    """Plot accuracy comparison"""
    models = [r['Model'] for r in results_list]
    accuracies = [r['Accuracy'] for r in results_list]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'], edgecolor='black', linewidth=2)
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.ylim([0.7, 1.0])
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('graphs/accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: graphs/accuracy_comparison.png")
    plt.close()

def plot_auc_curves(y_test, y_pred_proba_list, model_names, num_classes):
    """Plot AUC curves for all models"""
    plt.figure(figsize=(12, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, (y_pred_proba, model_name, color) in enumerate(zip(y_pred_proba_list, model_names, colors)):
        # One-vs-Rest approach for multiclass
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(num_classes):
            y_test_binary = (y_test == i).astype(int)
            fpr[i], tpr[i], _ = roc_curve(y_test_binary, y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot macro-average ROC curve
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(num_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= num_classes
        
        mean_auc = auc(all_fpr, mean_tpr)
        plt.plot(all_fpr, mean_tpr, color=color, lw=2.5,
                label=f'{model_name} (AUC = {mean_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('AUC Curves - Macro-Average (Multiclass)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('graphs/auc_curves.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: graphs/auc_curves.png")
    plt.close()

def plot_confusion_matrices(y_test, y_pred_list, model_names):
    """Plot confusion matrices for all models"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for idx, (y_pred, model_name, ax) in enumerate(zip(y_pred_list, model_names, axes)):
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=range(10),
                   yticklabels=range(10),
                   cbar_kws={'label': 'Count'},
                   cbar=True)
        ax.set_title(f'{model_name} Confusion Matrix', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('graphs/confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: graphs/confusion_matrices.png")
    plt.close()

def save_results_to_csv(results_df, filename='model_results.csv'):
    """Save results to CSV with error handling"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            import gc
            gc.collect()
            time.sleep(1)
            
            if os.path.exists(filename):
                try:
                    os.remove(filename)
                except:
                    pass
            
            results_df.to_csv(filename, index=False)
            print(f"✓ Successfully saved: {filename}")
            return True
            
        except PermissionError:
            print(f"Attempt {attempt+1}/{max_retries}: Permission denied. Retrying...")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                print(f"ERROR: Could not save {filename}")
                print("Make sure the file is not open in Excel")
                return False
        except Exception as e:
            print(f"Error saving CSV: {e}")
            return False

def main():
    print("="*70)
    print("MNIST DATASET - HIGH ACCURACY DEEP LEARNING MODEL TRAINING")
    print("="*70)
    
    # Load real dataset
    print("\n[1/5] Loading MNIST dataset (real image data)...")
    X_train_orig, X_test_orig, y_train_orig, y_test_orig = load_real_dataset()
    
    # Prepare data with 70-30 split
    print("\n[2/5] Preparing data (70% train, 30% test)...")
    X_train, X_test, y_train, y_test, num_classes = prepare_data(
        X_train_orig, X_test_orig, y_train_orig, y_test_orig
    )
    
    # Build and train models
    results_list = []
    y_pred_list = []
    y_pred_proba_list = []
    model_names = ['CNN', 'RNN', 'Neural Network']
    
    # Model 1: CNN
    print("\n[3/5] Training CNN...")
    cnn_model = build_cnn(X_train.shape[1], num_classes)
    print(f"CNN Parameters: {cnn_model.count_params():,}")
    cnn_results, cnn_pred, cnn_proba, _ = train_model(
        cnn_model, 'CNN', X_train, X_test, y_train, y_test, num_classes
    )
    results_list.append(cnn_results)
    y_pred_list.append(cnn_pred)
    y_pred_proba_list.append(cnn_proba)
    del cnn_model
    
    # Model 2: RNN
    print("\n[3/5] Training RNN...")
    rnn_model = build_rnn(X_train.shape[1], num_classes)
    print(f"RNN Parameters: {rnn_model.count_params():,}")
    rnn_results, rnn_pred, rnn_proba, _ = train_model(
        rnn_model, 'RNN', X_train, X_test, y_train, y_test, num_classes
    )
    results_list.append(rnn_results)
    y_pred_list.append(rnn_pred)
    y_pred_proba_list.append(rnn_proba)
    del rnn_model
    
    # Model 3: Neural Network
    print("\n[3/5] Training Neural Network...")
    nn_model = build_neural_network(X_train.shape[1], num_classes)
    print(f"Neural Network Parameters: {nn_model.count_params():,}")
    nn_results, nn_pred, nn_proba, _ = train_model(
        nn_model, 'Neural Network', X_train, X_test, y_train, y_test, num_classes
    )
    results_list.append(nn_results)
    y_pred_list.append(nn_pred)
    y_pred_proba_list.append(nn_proba)
    del nn_model
    
    # Save results to CSV
    print("\n[4/5] Saving results to CSV...")
    results_df = pd.DataFrame(results_list)
    
    if save_results_to_csv(results_df, 'model_results.csv'):
        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)
        print(results_df.to_string(index=False))
    
    # Generate graphs
    print("\n[5/5] Generating graphs...")
    plot_accuracy_comparison(results_list)
    plot_auc_curves(y_test, y_pred_proba_list, model_names, num_classes)
    plot_confusion_matrices(y_test, y_pred_list, model_names)
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nGenerated files:")
    print("  • model_results.csv")
    print("  • graphs/accuracy_comparison.png")
    print("  • graphs/auc_curves.png")
    print("  • graphs/confusion_matrices.png")
    print("\nKey Features:")
    print("  ✓ Real MNIST dataset (70,000 real handwritten digit images)")
    print("  ✓ 70-30 train-test split as requested")
    print("  ✓ 10 classes (digits 0-9)")
    print("  ✓ Expected accuracy: 0.95-0.99")
    print("  ✓ Batch Normalization for stable training")
    print("  ✓ Early stopping to prevent overfitting")
    print("  ✓ Learning rate scheduling for convergence")

if __name__ == "__main__":
    main()
