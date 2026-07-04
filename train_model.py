"""
Smart Skincare Recommender — Training Script
3 Skin Types: Oily, Dry, Normal
Architecture: MobileNetV2
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Input
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    print(f"TensorFlow {tf.__version__} ready")
except ImportError:
    print("Run: pip install tensorflow")
    sys.exit(1)

# ── CONFIG ──────────────────────────────────────────────────────
IMG_SIZE    = 224
BATCH_SIZE  = 16
EPOCHS_P1   = 30
EPOCHS_P2   = 20
LR_P1       = 1e-3
LR_P2       = 1e-5
DATASET_DIR = 'dataset'
MODEL_PATH  = 'skin_model.h5'

# Must match folder names EXACTLY
SKIN_TYPES = ['Dry', 'Normal', 'Oily']

# ── VALIDATE DATASET ────────────────────────────────────────────
def validate():
    print("\n=== Validating Dataset ===")
    ok = True
    for split in ['train', 'val']:
        for cls in SKIN_TYPES:
            path = os.path.join(DATASET_DIR, split, cls)
            if not os.path.exists(path):
                print(f"MISSING: {path}")
                ok = False
                continue
            imgs = [f for f in os.listdir(path)
                    if f.lower().endswith(('.jpg','.jpeg','.png','.webp'))]
            print(f"  {split}/{cls}: {len(imgs)} images")
            if len(imgs) == 0:
                print(f"  ERROR: {path} is empty!")
                ok = False
    if not ok:
        print("\nFix the above errors first.")
        sys.exit(1)
    print("Dataset OK!\n")

# ── GENERATORS ──────────────────────────────────────────────────
def get_generators():
    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.20,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode='nearest',
    )
    val_gen = ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_directory(
        os.path.join(DATASET_DIR, 'train'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=SKIN_TYPES,
        shuffle=True,
    )
    val_data = val_gen.flow_from_directory(
        os.path.join(DATASET_DIR, 'val'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=SKIN_TYPES,
        shuffle=False,
    )
    print(f"Class mapping: {train_data.class_indices}")
    print(f"Train: {train_data.samples} | Val: {val_data.samples}\n")
    return train_data, val_data

# ── BUILD MODEL ─────────────────────────────────────────────────
def build_model():
    print("=== Building MobileNetV2 Model ===")
    inp  = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    base = MobileNetV2(weights='imagenet', include_top=False, input_tensor=inp)
    base.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(len(SKIN_TYPES), activation='softmax')(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=Adam(LR_P1),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print(f"Trainable params: {sum(tf.size(w).numpy() for w in model.trainable_weights):,}\n")
    return model, base

# ── CALLBACKS ───────────────────────────────────────────────────
def callbacks():
    return [
        EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-8, verbose=1),
    ]

# ── TRAIN ───────────────────────────────────────────────────────
def train(model, base, train_data, val_data):
    print("=== Phase 1: Training Head ===")
    h1 = model.fit(train_data, epochs=EPOCHS_P1, validation_data=val_data, callbacks=callbacks(), verbose=1)
    print(f"Phase 1 best val accuracy: {max(h1.history['val_accuracy']):.2%}\n")

    print("=== Phase 2: Fine-tuning ===")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    model.compile(optimizer=Adam(LR_P2), loss='categorical_crossentropy', metrics=['accuracy'])
    h2 = model.fit(train_data, epochs=EPOCHS_P2, validation_data=val_data, callbacks=callbacks(), verbose=1)
    print(f"Phase 2 best val accuracy: {max(h2.history['val_accuracy']):.2%}\n")
    return h1, h2

# ── PLOT ────────────────────────────────────────────────────────
def plot(h1, h2):
    acc  = h1.history['accuracy']     + h2.history['accuracy']
    vacc = h1.history['val_accuracy'] + h2.history['val_accuracy']
    ep   = range(1, len(acc)+1)
    plt.figure(figsize=(10, 4))
    plt.subplot(1,2,1)
    plt.plot(ep, acc,  label='Train'); plt.plot(ep, vacc, label='Val')
    plt.title('Accuracy'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.subplot(1,2,2)
    loss  = h1.history['loss']     + h2.history['loss']
    vloss = h1.history['val_loss'] + h2.history['val_loss']
    plt.plot(ep, loss, label='Train'); plt.plot(ep, vloss, label='Val')
    plt.title('Loss'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    print("Saved: training_history.png")

# ── MAIN ────────────────────────────────────────────────────────
if __name__ == '__main__':
    validate()
    train_data, val_data = get_generators()
    model, base = build_model()
    h1, h2 = train(model, base, train_data, val_data)
    plot(h1, h2)
    print(f"\nDone! Model saved: {MODEL_PATH}")
    print("Now run: python app.py")
