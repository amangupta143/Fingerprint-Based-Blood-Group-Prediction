# -*- coding: utf-8 -*-
"""
# Fingerprint Based Blood-Group Prediction
### Importing Libraries
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

# Prevent duplicate plugin registrations
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

import zipfile
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB3, ResNet50V2, Xception, ResNet50, VGG16
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import pandas as pd

print("\nGPU Availability:", tf.config.list_physical_devices('GPU'))
print("TensorFlow Version:", tf.__version__)
print("CUDA Version:", os.environ.get('CUDA_VERSION', 'Not found'))
print("cuDNN Version:", os.environ.get('CUDNN_VERSION', 'Not found'))

"""### Set random seeds for reproducibility"""

np.random.seed(42)
tf.random.set_seed(42)

"""### Define paths and parameters"""
# Updated paths for local environment
# Root directory of the project
root_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the dataset folder
data_dir = os.path.join(root_dir, "dataset")

# Other directories
test_dir = os.path.join(root_dir, "test_dataset")
results_dir = os.path.join(root_dir, "results")
models_dir = os.path.join(root_dir, "models")

# Match actual image dimensions
img_height, img_width = 103, 96
batch_size = 32  # Smaller batch size for better generalization
epochs = 150     # More epochs for better convergence

# Create necessary directories
os.makedirs(test_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

"""### Function to create test set"""

def create_test_set(data_dir, test_dir, test_split=0.15):
    blood_groups = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    for blood_group in blood_groups:
        os.makedirs(os.path.join(test_dir, blood_group), exist_ok=True)

        images = [f for f in os.listdir(os.path.join(data_dir, blood_group))
                 if f.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png'))]

        if len(images) < 3:
            print(f"Warning: {blood_group} has fewer than 3 images, skipping test split")
            continue

        test_size = max(1, int(len(images) * test_split))
        test_images = np.random.choice(images, size=test_size, replace=False)

        for img in test_images:
            src = os.path.join(data_dir, blood_group, img)
            dst = os.path.join(test_dir, blood_group, img)
            shutil.copy(src, dst)

"""### Create data generators with enhanced augmentation"""

def create_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.25,
        height_shift_range=0.25,
        shear_range=0.25,
        zoom_range=0.3,
        brightness_range=[0.9, 1.1],  # Brightness variation
        fill_mode='nearest',
        validation_split=0.15
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    return train_generator, validation_generator, test_generator

"""### Execute data preparation"""

print("Creating test dataset...")
create_test_set(data_dir, test_dir, test_split=0.15)

print("\nCreating data generators...")
train_generator, validation_generator, test_generator = create_data_generators()
num_classes = len(train_generator.class_indices)
input_shape = (img_height, img_width, 3)

"""### Define callbacks for training"""

def get_callbacks(model_name):
    checkpoint = ModelCheckpoint(
        os.path.join(models_dir, f'best_{model_name}.h5'),
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )

    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=30,  # More patience
        verbose=1,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,  # Stronger reduction
        patience=6,
        min_lr=1e-7,
        verbose=1
    )

    return [checkpoint, early_stopping, reduce_lr]

"""## Defining Models
### 1. Custom CNN
"""

def build_optimized_cnn(input_shape, num_classes):
    model = tf.keras.models.Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.3),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.4),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.4),

        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.5),

        Flatten(),
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

"""### 2. ResNet50 model"""

def build_resnet50_model(input_shape, num_classes):
    base_model = ResNet50(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )

    # Freeze base model initially
    base_model.trainable = False

    # Add layers for classification
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model, base_model

"""### 3. VGG16 model"""

def build_vgg16_model(input_shape, num_classes):
    base_model = VGG16(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )

    # Freeze base model initially
    base_model.trainable = False

    # Add layers for classification
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model, base_model

"""### 4. EfficientNetB3 with additional tuning"""

def build_efficient_net_model(input_shape, num_classes):
    base_model = EfficientNetB3(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )

    # Freeze base model initially
    base_model.trainable = False

    # Add layers for classification
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model, base_model

"""### Function to train and evaluate models"""

def train_and_evaluate_model(model, model_name, train_generator, validation_generator, test_generator, epochs):
    print(f"\nTraining {model_name}...")

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=get_callbacks(model_name.lower().replace(' ', '_'))
    )

    # Load the best weights
    model.load_weights(os.path.join(models_dir, f'best_{model_name.lower().replace(" ", "_")}.h5'))

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"{model_name} - Test accuracy: {test_acc:.4f}")

    return model, history, test_acc

"""### Function to fine-tune a pre-trained model"""

def fine_tune_model(model, base_model, model_name, train_generator, validation_generator, test_generator, unfreeze_layers=30):
    # Load best weights from transfer learning phase
    model.load_weights(os.path.join(models_dir, f'best_{model_name.lower().replace(" ", "_")}.h5'))

    # Unfreeze the top layers of the base model
    base_model.trainable = True
    for layer in base_model.layers[:-unfreeze_layers]:
        layer.trainable = False

    # Recompile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),  # Lower learning rate for fine-tuning
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"\nFine-tuning {model_name}...")

    # Fine-tune with fewer epochs
    fine_tune_epochs = epochs

    # Use consistent naming convention for fine-tuned models
    fine_tuned_name = f"{model_name.lower().replace(' ', '_')}_fine_tuned"

    history = model.fit(
        train_generator,
        epochs=fine_tune_epochs,
        validation_data=validation_generator,
        callbacks=get_callbacks(fine_tuned_name)
    )

    # Load the best weights after fine-tuning
    model.load_weights(os.path.join(models_dir, f'best_{fine_tuned_name}.h5'))

    # Evaluate after fine-tuning
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"{model_name} (Fine-tuned) - Test accuracy: {test_acc:.4f}")

    return model, history, test_acc

"""## Train and evaluate models"""

results = []

# Clear session and configure GPU memory growth
tf.keras.backend.clear_session()
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU config error: {e}")

"""### 1. Train CNN"""

print("\n==== Training Optimized CNN Model ====")
custom_model = build_optimized_cnn(input_shape, num_classes)
custom_model, custom_history, custom_acc = train_and_evaluate_model(
    custom_model, 'Optimized_CNN', train_generator, validation_generator, test_generator, epochs
)
results.append(('Optimized CNN', custom_acc))

"""### 2. Train and fine-tune ResNet50"""

print("\n==== Training ResNet50 Model ====")
resnet50_model, resnet50_base = build_resnet50_model(input_shape, num_classes)
resnet50_model, resnet50_history, resnet50_acc = train_and_evaluate_model(
    resnet50_model, 'ResNet50', train_generator, validation_generator, test_generator, epochs
)
results.append(('ResNet50', resnet50_acc))

# Fine-tune ResNet50
print("\n==== Fine-tuning ResNet50 Model ====")
resnet50_ft_model, resnet50_ft_history, resnet50_ft_acc = fine_tune_model(
    resnet50_model, resnet50_base, 'ResNet50', train_generator, validation_generator, test_generator
)
results.append(('ResNet50 (Fine-tuned)', resnet50_ft_acc))

"""### 3. Train and fine-tune VGG16"""

print("\n==== Training VGG16 Model ====")
vgg16_model, vgg16_base = build_vgg16_model(input_shape, num_classes)
vgg16_model, vgg16_history, vgg16_acc = train_and_evaluate_model(
    vgg16_model, 'VGG16', train_generator, validation_generator, test_generator, epochs
)
results.append(('VGG16', vgg16_acc))

# Fine-tune VGG16
print("\n==== Fine-tuning VGG16 Model ====")
vgg16_ft_model, vgg16_ft_history, vgg16_ft_acc = fine_tune_model(
    vgg16_model, vgg16_base, 'VGG16', train_generator, validation_generator, test_generator
)
results.append(('VGG16 (Fine-tuned)', vgg16_ft_acc))

"""### 4. Train and fine-tune EfficientNetB3"""

print("\n==== Training EfficientNetB3 Model ====")
efficientnet_model, efficientnet_base = build_efficient_net_model(input_shape, num_classes)
efficientnet_model, efficientnet_history, efficientnet_acc = train_and_evaluate_model(
    efficientnet_model, 'EfficientNetB3', train_generator, validation_generator, test_generator, epochs
)
results.append(('EfficientNetB3', efficientnet_acc))

# Fine-tune EfficientNetB3
print("\n==== Fine-tuning EfficientNetB3 Model ====")
efficientnet_ft_model, efficientnet_ft_history, efficientnet_ft_acc = fine_tune_model(
    efficientnet_model, efficientnet_base, 'EfficientNetB3', train_generator, validation_generator, test_generator
)
results.append(('EfficientNetB3 (Fine-tuned)', efficientnet_ft_acc))

"""### Create ensemble model"""

"""### Create and evaluate ensemble model"""

def create_ensemble_model(models, input_shape, num_classes):
    # Create a fresh input
    inputs = Input(shape=input_shape)

    # Process each model output separately, avoiding shared graph nodes
    outputs = []
    for i, model in enumerate(models):
        # Create a separate model that copies weights but has new graph nodes
        temp_model = tf.keras.models.clone_model(model)
        temp_model.set_weights(model.get_weights())

        # Create a separate functional model with unique naming
        temp_input = Input(shape=input_shape, name=f'input_model_{i}')
        temp_output = temp_model(temp_input)
        isolated_model = Model(inputs=temp_input, outputs=temp_output, name=f'isolated_model_{i}')

        # Use this isolated model in the ensemble
        outputs.append(isolated_model(inputs))

    # Average the outputs
    if len(outputs) > 1:
        ensemble_output = tf.keras.layers.Average()(outputs)
    else:
        ensemble_output = outputs[0]

    # Create the ensemble model
    ensemble_model = Model(inputs=inputs, outputs=ensemble_output)

    return ensemble_model

print("\n==== Creating and Evaluating Ensemble Model ====")
# Get top 3 models by accuracy for ensemble
sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
top_3_models = [m[0] for m in sorted_results[:3]]
print(f"Creating ensemble from top 3 models: {top_3_models}")

# Load the best models for ensemble
models_for_ensemble = []
for model_name in top_3_models:
    if "Fine-tuned" in model_name:
        model_filename = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
    else:
        model_filename = model_name.lower().replace(' ', '_')

    model_path = os.path.join(models_dir, f'best_{model_filename}.h5')
    print(f"Loading model from: {model_path}")

    try:
        model = tf.keras.models.load_model(model_path)
        models_for_ensemble.append(model)
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")

"""### Evaluate and save ensemble"""

# Create and compile the ensemble if we have models to ensemble
if models_for_ensemble:
    ensemble_model = create_ensemble_model(models_for_ensemble, input_shape, num_classes)

    ensemble_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Evaluate ensemble
    print("\nEvaluating Ensemble Model...")
    test_loss, test_acc = ensemble_model.evaluate(test_generator)
    print(f"Ensemble Model - Test accuracy: {test_acc:.4f}")

    # Save ensemble model
    ensemble_model_path = os.path.join(models_dir, 'ensemble_model.h5')
    ensemble_model.save(ensemble_model_path)
    print(f"Ensemble model saved to {ensemble_model_path}")

    # Add ensemble to results
    results.append(('Ensemble', test_acc))

    # If ensemble is the best, save it as final best model
    if test_acc >= sorted_results[0][1]:
        shutil.copy(
            ensemble_model_path,
            os.path.join(models_dir, 'final_best_model.h5')
        )
        print("Ensemble model copied as final_best_model.h5")
else:
    print("Could not create ensemble due to model loading errors")

"""### Final model evaluation and result comparison"""

print("\n==== Final Model Comparison ====")
results.sort(key=lambda x: x[1], reverse=True)
for model_name, accuracy in results:
    print(f"{model_name}: {accuracy:.4f}")

# Save the best model for deployment
best_model_name = results[0][0].lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
if best_model_name == 'ensemble':
    # Already saved above
    pass
else:
    if "fine_tuned" in best_model_name:
        # Copy the best model to a final best model file
        shutil.copy(
            os.path.join(models_dir, f'best_{best_model_name}.h5'),
            os.path.join(models_dir, 'final_best_model.h5')
        )
    else:
        # Copy the best model to a final best model file
        shutil.copy(
            os.path.join(models_dir, f'best_{best_model_name}.h5'),
            os.path.join(models_dir, 'final_best_model.h5')
        )

print(f"\nFinal best model: {results[0][0]} with accuracy: {results[0][1]:.4f}")

"""## Create results subdirectories"""

viz_dir = os.path.join(results_dir, 'visualizations')
os.makedirs(viz_dir, exist_ok=True)

"""### 1. Dataset Distribution Visualization"""

print("\n==== Visualizing Dataset Distribution ====")
def visualize_dataset_distribution():
    blood_groups = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    counts = []

    for blood_group in blood_groups:
        images = [f for f in os.listdir(os.path.join(data_dir, blood_group))
                 if f.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png'))]
        counts.append(len(images))

    # Create dataframe for easier plotting
    df = pd.DataFrame({'Blood Group': blood_groups, 'Count': counts})

    # Sort by count for better visualization
    df = df.sort_values('Count', ascending=False)

    # Plot bar chart
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Blood Group', y='Count', data=df, palette='viridis')
    plt.title('Distribution of Images Across Blood Groups', fontsize=16)
    plt.xlabel('Blood Group', fontsize=14)
    plt.ylabel('Number of Images', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'dataset_distribution.png'), dpi=300)
    plt.close()

    # Plot pie chart
    plt.figure(figsize=(10, 10))
    plt.pie(df['Count'], labels=df['Blood Group'], autopct='%1.1f%%', startangle=90,
            shadow=False, explode=[0.05]*len(df), colors=sns.color_palette('viridis', len(df)))
    plt.axis('equal')
    plt.title('Percentage Distribution of Blood Groups in Dataset', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'dataset_distribution_pie.png'), dpi=300)
    plt.close()

    return df

dataset_df = visualize_dataset_distribution()

"""### 2. Training History Visualization"""

print("\n==== Visualizing Training History ====")
def plot_training_history(history, model_name, metric='accuracy'):
    plt.figure(figsize=(12, 5))

    # Plot training & validation accuracy/loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history[metric], label=f'Training {metric}')
    plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
    plt.title(f'{model_name} - {metric.capitalize()} Over Epochs', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(metric.capitalize(), fontsize=12)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Loss Over Epochs', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f'{model_name.lower().replace(" ", "_")}_history.png'), dpi=300)
    plt.close()

# Plot training history for all models
histories = {
    'Optimized CNN': custom_history,
    'ResNet50': resnet50_history,
    'ResNet50 (Fine-tuned)': resnet50_ft_history,
    'VGG16': vgg16_history,
    'VGG16 (Fine-tuned)': vgg16_ft_history,
    'EfficientNetB3': efficientnet_history,
    'EfficientNetB3 (Fine-tuned)': efficientnet_ft_history
}

for model_name, history in histories.items():
    plot_training_history(history, model_name)

"""### 3. Model Comparison Visualization"""

print("\n==== Visualizing Model Comparison ====")
def visualize_model_comparison(results):
    df = pd.DataFrame(results, columns=['Model', 'Accuracy'])
    df = df.sort_values('Accuracy', ascending=False)

    # Bar chart
    plt.figure(figsize=(14, 8))
    bars = sns.barplot(x='Model', y='Accuracy', data=df, palette='viridis')
    plt.title('Accuracy Comparison Across Models', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Test Accuracy', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)

    # Add accuracy values on top of bars
    for i, v in enumerate(df['Accuracy']):
        bars.text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=10)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'model_comparison.png'), dpi=300)
    plt.close()

    # Save results as CSV
    df.to_csv(os.path.join(results_dir, 'model_comparison_results.csv'), index=False)

    return df

comparison_df = visualize_model_comparison(results)

"""### 4. Confusion Matrix for All Models"""

print("\n==== Generating Confusion Matrices ====")
def plot_confusion_matrix(model, test_generator, model_name):
    # Get predictions
    y_pred_probs = model.predict(test_generator)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_generator.classes

    # Get class labels
    class_indices = test_generator.class_indices
    class_labels = {v: k for k, v in class_indices.items()}
    ordered_labels = [class_labels[i] for i in range(len(class_labels))]

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot non-normalized confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=ordered_labels, yticklabels=ordered_labels)
    plt.title(f'{model_name} - Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png'), dpi=300)
    plt.close()

    # Plot normalized confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=ordered_labels, yticklabels=ordered_labels)
    plt.title(f'{model_name} - Normalized Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f'{model_name.lower().replace(" ", "_")}_confusion_matrix_norm.png'), dpi=300)
    plt.close()

    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=ordered_labels, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(results_dir, f'{model_name.lower().replace(" ", "_")}_classification_report.csv'))

    return cm, report_df

# Generate confusion matrices for all models
for model_name, _ in results:
    if model_name == 'Ensemble':
        model = ensemble_model
    else:
        model_filename = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        model_path = os.path.join(models_dir, f'best_{model_filename}.h5')
        model = tf.keras.models.load_model(model_path)

    cm, report_df = plot_confusion_matrix(model, test_generator, model_name)
    print(f"Generated confusion matrix and classification report for {model_name}")

"""### 5. Sample Predictions Visualization"""

print("\n==== Visualizing Sample Predictions ====")
def visualize_sample_predictions(model, test_generator, model_name, num_samples=20):
    # Get class mapping
    class_indices = test_generator.class_indices
    class_labels = {v: k for k, v in class_indices.items()}

    # Get a batch of images
    images, true_labels = next(test_generator)
    true_labels = np.argmax(true_labels, axis=1)

    # Make predictions
    predictions = model.predict(images)
    pred_labels = np.argmax(predictions, axis=1)

    # Limit to num_samples
    images = images[:num_samples]
    true_labels = true_labels[:num_samples]
    pred_labels = pred_labels[:num_samples]
    predictions = predictions[:num_samples]

    # Create grid plot
    n_cols = 4
    n_rows = (num_samples + n_cols - 1) // n_cols

    plt.figure(figsize=(15, n_rows * 4))
    for i, (image, true_label, pred_label, pred_probs) in enumerate(zip(images, true_labels, pred_labels, predictions)):
        plt.subplot(n_rows, n_cols, i + 1)

        # Convert from [0,1] to [0,255]
        image = image * 255
        image = image.astype(np.uint8)

        plt.imshow(image)

        # Color based on correctness
        color = 'green' if true_label == pred_label else 'red'

        # Add labels
        true_class = class_labels[true_label]
        pred_class = class_labels[pred_label]
        confidence = pred_probs[pred_label] * 100

        plt.title(f"True: {true_class}\nPred: {pred_class} ({confidence:.1f}%)",
                 color=color, fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f'{model_name.lower().replace(" ", "_")}_sample_predictions.png'), dpi=300)
    plt.close()

# Visualize sample predictions for best model
best_model_name = results[0][0]
if best_model_name == 'Ensemble':
    best_model = ensemble_model
else:
    best_model_filename = best_model_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
    best_model_path = os.path.join(models_dir, f'best_{best_model_filename}.h5')
    best_model = tf.keras.models.load_model(best_model_path)

visualize_sample_predictions(best_model, test_generator, best_model_name)

"""### 6. Learning Rate Analysis"""

print("\n==== Analyzing Learning Rate Impact ====")
def plot_lr_history(histories):
    plt.figure(figsize=(12, 6))

    for model_name, history in histories.items():
        if 'lr' in history.history:
            plt.plot(history.history['lr'], label=model_name)

    plt.title('Learning Rate Schedule Across Models', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Learning Rate', fontsize=14)
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(os.path.join(viz_dir, 'learning_rate_analysis.png'), dpi=300)
    plt.close()

plot_lr_history(histories)

"""### 7. Class-wise Performance Analysis"""

print("\n==== Class-wise Performance Analysis ====")
def visualize_class_performance(model, test_generator, model_name):
    # Get predictions
    y_pred_probs = model.predict(test_generator)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_generator.classes

    # Get class labels
    class_indices = test_generator.class_indices
    class_labels = {v: k for k, v in class_indices.items()}

    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=list(class_indices.keys()), output_dict=True)

    # Convert to DataFrame for easier plotting
    classes_df = pd.DataFrame({
        'Class': list(class_indices.keys()),
        'Precision': [report[cls]['precision'] for cls in class_indices.keys()],
        'Recall': [report[cls]['recall'] for cls in class_indices.keys()],
        'F1-Score': [report[cls]['f1-score'] for cls in class_indices.keys()],
        'Support': [report[cls]['support'] for cls in class_indices.keys()]
    })

    # Sort by F1-Score
    classes_df = classes_df.sort_values('F1-Score', ascending=False)

    # Plot metrics
    plt.figure(figsize=(14, 8))

    x = np.arange(len(classes_df))
    width = 0.2

    plt.bar(x - width, classes_df['Precision'], width, label='Precision')
    plt.bar(x, classes_df['Recall'], width, label='Recall')
    plt.bar(x + width, classes_df['F1-Score'], width, label='F1-Score')

    plt.xlabel('Blood Group', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.title(f'{model_name} - Class-wise Performance Metrics', fontsize=16)
    plt.xticks(x, classes_df['Class'], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(viz_dir, f'{model_name.lower().replace(" ", "_")}_class_performance.png'), dpi=300)
    plt.close()

    # Save to CSV
    classes_df.to_csv(os.path.join(results_dir, f'{model_name.lower().replace(" ", "_")}_class_performance.csv'))

    return classes_df

# Visualize class performance for all models
print("Generating class-wise performance analysis for all models...")
for model_name, _ in results:
    if model_name == 'Ensemble':
        model = ensemble_model
    else:
        model_filename = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        model_path = os.path.join(models_dir, f'best_{model_filename}.h5')
        model = tf.keras.models.load_model(model_path)

    classes_df = visualize_class_performance(model, test_generator, model_name)
    print(f"Generated class-wise performance visualization for {model_name}")

"""### 8. ROC Curve Analysis"""

print("\n==== ROC Curve Analysis ====")
def plot_roc_curves(model, test_generator, model_name):
    # Get predictions
    y_pred_probs = model.predict(test_generator)
    y_true = tf.keras.utils.to_categorical(test_generator.classes, num_classes=num_classes)

    # Get class labels
    class_indices = test_generator.class_indices
    class_labels = {v: k for k, v in class_indices.items()}

    # Plot ROC curves
    plt.figure(figsize=(12, 10))

    # Store AUC values
    auc_values = []

    # Plot ROC curve for each class
    for i in range(num_classes):
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        auc_values.append(roc_auc)

        plt.plot(fpr, tpr, lw=2, label=f'{class_labels[i]} (AUC = {roc_auc:.3f})')

    # Plot random guessing line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(f'{model_name} - ROC Curves for Each Blood Group', fontsize=16)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f'{model_name.lower().replace(" ", "_")}_roc_curves.png'), dpi=300)
    plt.close()

    # Create AUC summary dataframe
    auc_df = pd.DataFrame({
        'Blood Group': list(class_indices.keys()),
        'AUC': auc_values
    })
    auc_df = auc_df.sort_values('AUC', ascending=False)

    # Save AUC values
    auc_df.to_csv(os.path.join(results_dir, f'{model_name.lower().replace(" ", "_")}_auc_values.csv'), index=False)

    return auc_df

# Plot ROC curves for all models
print("Generating ROC curves for all models...")
for model_name, _ in results:
    if model_name == 'Ensemble':
        model = ensemble_model
    else:
        model_filename = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        model_path = os.path.join(models_dir, f'best_{model_filename}.h5')
        model = tf.keras.models.load_model(model_path)

    auc_df = plot_roc_curves(model, test_generator, model_name)
    print(f"Generated ROC curves for {model_name}")

"""# 9. Feature Importance Analysis (for interpretability)"""

print("\n==== Feature Importance Analysis ====")
def visualize_feature_importance(model, test_generator, model_name, num_images=5):
    try:
        # Import necessary libraries
        import matplotlib.cm as cm
        from tensorflow.keras.models import Model

        # Get a batch of correctly classified images
        images, labels = next(test_generator)
        true_labels = np.argmax(labels, axis=1)
        predictions = model.predict(images)
        pred_labels = np.argmax(predictions, axis=1)

        # Find correctly classified images
        correct_idx = np.where(true_labels == pred_labels)[0]
        if len(correct_idx) < num_images:
            num_images = len(correct_idx)

        # Sample images
        sample_idx = np.random.choice(correct_idx, num_images, replace=False)

        # Get class labels
        class_indices = test_generator.class_indices
        class_labels = {v: k for k, v in class_indices.items()}

        # Create a new model for visualization
        grad_model = Model(
            inputs=model.inputs,
            outputs=[model.output, model.layers[-3].output]  # Get the output of the last conv layer
        )

        plt.figure(figsize=(14, num_images * 4))

        for i, idx in enumerate(sample_idx):
            # Get the image
            img = images[idx]
            img_tensor = np.expand_dims(img, axis=0)

            # Get the class
            pred_class = pred_labels[idx]
            class_name = class_labels[pred_class]

            # Get the gradients
            with tf.GradientTape() as tape:
                preds, last_conv_layer_output = grad_model(img_tensor)
                class_output = preds[:, pred_class]

            # Get gradients for the class output with respect to the last conv layer
            grads = tape.gradient(class_output, last_conv_layer_output)

            # Get the channel-wise mean of the gradients
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            # Weight the last conv layer output by the gradients
            last_conv_layer_output = last_conv_layer_output[0]
            heatmap = tf.reduce_sum(tf.multiply(pooled_grads, last_conv_layer_output), axis=-1)
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            heatmap = heatmap.numpy()

            # Resize heatmap to match the original image size
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

            # Convert to RGB for visualization
            heatmap = np.uint8(255 * heatmap)
            heatmap = cm.jet(heatmap)[:, :, :3]
            heatmap = np.uint8(255 * heatmap)

            # Original image
            img_display = np.uint8(img * 255)

            # Superimpose heatmap on the original image
            superimposed_img = cv2.addWeighted(img_display, 0.6, heatmap, 0.4, 0)

            # Plot
            plt.subplot(num_images, 2, 2*i+1)
            plt.imshow(img_display)
            plt.title(f"Original: {class_name}", fontsize=14)
            plt.axis('off')

            plt.subplot(num_images, 2, 2*i+2)
            plt.imshow(superimposed_img)
            plt.title("Grad-CAM Heatmap", fontsize=14)
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f'{model_name.lower().replace(" ", "_")}_grad_cam.png'), dpi=300)
        plt.close()

        print(f"Generated Grad-CAM visualizations for {model_name}")

    except Exception as e:
        print(f"Could not generate Grad-CAM visualizations: {e}")

# Try to generate Grad-CAM visualizations
try:
    import cv2
    visualize_feature_importance(best_model, test_generator, best_model_name)
except ImportError:
    print("OpenCV (cv2) not available. Skipping Grad-CAM visualization.")

"""### 10. Training Time Analysis"""

print("\n==== Training Time Analysis ====")
def visualize_training_times(histories):
    # This assumes you've added timing to your history objects
    # If not, skip this visualization

    times = []
    for model_name, history in histories.items():
        if hasattr(history, 'epoch_times'):
            times.append((model_name, np.mean(history.epoch_times), np.sum(history.epoch_times)))

    if not times:
        print("No timing information available. Skipping training time analysis.")
        return

    # Create DataFrame
    times_df = pd.DataFrame(times, columns=['Model', 'Avg Epoch Time (s)', 'Total Training Time (s)'])

    # Convert to minutes for better readability
    times_df['Total Training Time (min)'] = times_df['Total Training Time (s)'] / 60

    # Sort by total training time
    times_df = times_df.sort_values('Total Training Time (s)')

    # Plot
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.barplot(x='Model', y='Avg Epoch Time (s)', data=times_df, palette='viridis')
    plt.title('Average Time per Epoch', fontsize=14)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.xticks(rotation=45, ha='right')

    plt.subplot(1, 2, 2)
    sns.barplot(x='Model', y='Total Training Time (min)', data=times_df, palette='viridis')
    plt.title('Total Training Time', fontsize=14)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Time (minutes)', fontsize=12)
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'training_time_analysis.png'), dpi=300)
    plt.close()

    # Save as CSV
    times_df.to_csv(os.path.join(results_dir, 'training_times.csv'), index=False)

# Try to visualize training times if available
try:
    visualize_training_times(histories)
except Exception as e:
    print(f"Could not generate training time analysis: {e}")

"""### 11. Model Parameter Count Analysis"""

print("\n==== Model Parameter Analysis ====")
def visualize_parameter_counts():
    # Get models and their parameter counts
    model_params = []

    # Custom CNN
    model_params.append(('Optimized CNN', custom_model.count_params()))

    # ResNet50
    model_params.append(('ResNet50', resnet50_model.count_params()))

    # VGG16
    model_params.append(('VGG16', vgg16_model.count_params()))

    # EfficientNetB3
    model_params.append(('EfficientNetB3', efficientnet_model.count_params()))

    # Ensemble
    model_params.append(('Ensemble', ensemble_model.count_params()))

    # Create DataFrame
    params_df = pd.DataFrame(model_params, columns=['Model', 'Parameter Count'])

    # Add a column for millions of parameters
    params_df['Parameters (M)'] = params_df['Parameter Count'] / 1e6

    # Sort by parameter count
    params_df = params_df.sort_values('Parameter Count')

    # Plot
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='Model', y='Parameters (M)', data=params_df, palette='viridis')

    # Add value labels on bars
    for i, v in enumerate(params_df['Parameters (M)']):
        ax.text(i, v + 0.1, f'{v:.2f}M', ha='center')

    plt.title('Model Parameter Count Comparison', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Parameters (Millions)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'parameter_count_analysis.png'), dpi=300)
    plt.close()

    # Save as CSV
    params_df.to_csv(os.path.join(results_dir, 'model_parameters.csv'), index=False)

    return params_df

params_df = visualize_parameter_counts()

"""### 12. Create Summary Report"""

print("\n==== Generating Summary Report ====")
def generate_summary_report():
    # Create a DataFrame to track model parameters
    params_data = []

    # Add parameter counts for all trained models
    params_data.append({
        'Model': 'Optimized CNN',
        'Parameter Count': custom_model.count_params()
    })

    params_data.append({
        'Model': 'ResNet50',
        'Parameter Count': resnet50_model.count_params()
    })

    params_data.append({
        'Model': 'ResNet50 (Fine-tuned)',
        'Parameter Count': resnet50_ft_model.count_params()
    })

    params_data.append({
        'Model': 'VGG16',
        'Parameter Count': vgg16_model.count_params()
    })

    params_data.append({
        'Model': 'VGG16 (Fine-tuned)',
        'Parameter Count': vgg16_ft_model.count_params()
    })

    params_data.append({
        'Model': 'EfficientNetB3',
        'Parameter Count': efficientnet_model.count_params()
    })

    params_data.append({
        'Model': 'EfficientNetB3 (Fine-tuned)',
        'Parameter Count': efficientnet_ft_model.count_params()
    })

    # Create DataFrame from the list
    params_df = pd.DataFrame(params_data)

    # Add ensemble model if it exists
    if 'Ensemble' in [r[0] for r in results]:
        # Find the ensemble model in the results
        ensemble_model_loaded = tf.keras.models.load_model(os.path.join(models_dir, 'ensemble_model.h5'))
        # Add to DataFrame
        params_df = pd.concat([params_df, pd.DataFrame([{
            'Model': 'Ensemble',
            'Parameter Count': ensemble_model_loaded.count_params()
        }])], ignore_index=True)

    # Create a summary of all results
    summary = {
        'Dataset Information': {
            'Total Classes': len(train_generator.class_indices),
            'Class Names': list(train_generator.class_indices.keys()),
            'Training Images': train_generator.samples,
            'Validation Images': validation_generator.samples,
            'Test Images': test_generator.samples
        },
        'Best Model': {
            'Name': results[0][0],
            'Accuracy': results[0][1],
            'Parameters': params_df[params_df['Model'] == results[0][0]]['Parameter Count'].values[0] if results[0][0] in params_df['Model'].values else 'Unknown'
        },
        'Model Comparison': {
            'Model': [m[0] for m in results],
            'Accuracy': [m[1] for m in results]
        }
    }

    # Ensure the visualizations directory exists
    os.makedirs(os.path.join(results_dir, 'visualizations'), exist_ok=True)

    # Create dataset distribution visualization
    plt.figure(figsize=(10, 6))
    plt.bar(summary['Dataset Information']['Class Names'],
            [len(os.listdir(os.path.join(data_dir, cls))) for cls in summary['Dataset Information']['Class Names']])
    plt.title('Dataset Distribution')
    plt.xlabel('Blood Group')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'visualizations', 'dataset_distribution.png'))
    plt.close()

    # Create model comparison visualization
    plt.figure(figsize=(12, 6))
    model_names = summary['Model Comparison']['Model']
    accuracies = summary['Model Comparison']['Accuracy']

    # Sort by accuracy for better visualization
    sorted_indices = np.argsort(accuracies)[::-1]
    model_names = [model_names[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]

    bars = plt.bar(model_names, accuracies, color='skyblue')
    bars[0].set_color('navy')  # Highlight best model
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)

    # Add accuracy values on top of bars
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'visualizations', 'model_comparison.png'))
    plt.close()

    # Generate model comparison table rows
    model_rows = ""
    for i, (model_name, accuracy) in enumerate(zip(summary['Model Comparison']['Model'],
                                                  summary['Model Comparison']['Accuracy'])):
        highlight = 'class="highlight"' if i == 0 else ''
        model_rows += f"<tr {highlight}><td>{model_name}</td><td>{accuracy:.4f}</td></tr>\n"

    # Get best model name for image paths
    best_model_name = summary['Best Model']['Name'].lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')

    # Create a nicely formatted HTML report
    html = f"""
    <html>
    <head>
        <title>Blood Group Classification Model Summary</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #3498db; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .highlight {{ background-color: #e8f4f8; font-weight: bold; }}
            img {{ max-width: 100%; height: auto; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <h1>Blood Group Classification Model Summary</h1>

        <h2>Dataset Information</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Classes</td><td>{summary['Dataset Information']['Total Classes']}</td></tr>
            <tr><td>Class Names</td><td>{', '.join(summary['Dataset Information']['Class Names'])}</td></tr>
            <tr><td>Training Images</td><td>{summary['Dataset Information']['Training Images']}</td></tr>
            <tr><td>Validation Images</td><td>{summary['Dataset Information']['Validation Images']}</td></tr>
            <tr><td>Test Images</td><td>{summary['Dataset Information']['Test Images']}</td></tr>
        </table>

        <h2>Best Model Performance</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr class="highlight"><td>Best Model</td><td>{summary['Best Model']['Name']}</td></tr>
            <tr class="highlight"><td>Accuracy</td><td>{summary['Best Model']['Accuracy']:.4f}</td></tr>
            <tr><td>Parameters</td><td>{summary['Best Model']['Parameters']:,}</td></tr>
        </table>

        <h2>Model Comparison</h2>
        <table>
            <tr><th>Model</th><th>Accuracy</th></tr>
            {model_rows}
        </table>

        <h2>Visualizations</h2>
        <h3>Dataset Distribution</h3>
        <img src="visualizations/dataset_distribution.png" alt="Dataset Distribution">

        <h3>Model Comparison</h3>
        <img src="visualizations/model_comparison.png" alt="Model Comparison">

        <p>Report generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </body>
    </html>
    """

    # Save the HTML report
    with open(os.path.join(results_dir, 'summary_report.html'), 'w') as f:
        f.write(html)

    print(f"Summary report saved to {os.path.join(results_dir, 'summary_report.html')}")

# Execute the report generation
generate_summary_report()

"""### Create a single comparison visualization of all model training histories"""

def plot_all_model_histories(histories):
    plt.figure(figsize=(18, 8))

    # Plot training accuracy
    plt.subplot(1, 2, 1)
    for model_name, history in histories.items():
        plt.plot(history.history['accuracy'], label=f'{model_name}')

    plt.title('Training Accuracy Comparison', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    for model_name, history in histories.items():
        plt.plot(history.history['val_accuracy'], label=f'{model_name}')

    plt.title('Validation Accuracy Comparison', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'all_model_histories.png'), dpi=300)
    plt.close()

# Plot all model histories
plot_all_model_histories(histories)

print("\n==== All visualizations have been saved to the results directory ====")