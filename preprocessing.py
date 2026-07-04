"""
preprocessing.py — Image + Data cleaning for 3 skin types
Skin Types: Dry, Normal, Oily (alphabetical — matches training order)
"""
import os
import numpy as np
from PIL import Image, ImageOps

IMG_SIZE   = 224
SKIN_TYPES = ['Dry', 'Normal', 'Oily']

CONCERN_MAP = {
    'Oily'  : ['Acne', 'Pores', 'Whitehead/Blackhead', 'Hydration'],
    'Dry'   : ['Hydration', 'Barrier Damage', 'Dark Spots', 'Irritation'],
    'Normal': ['Dark Spots', 'Pigmentation', 'Pores', 'Hydration'],
}

ALL_CONCERNS = [
    'Acne', 'Dark Spots', 'Pigmentation', 'Pores',
    'Hydration', 'Irritation', 'Whitehead/Blackhead', 'Barrier Damage',
]

VALID_EXT = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def load_and_preprocess_image(image_path, img_size=IMG_SIZE):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = ImageOps.fit(img, (img_size, img_size), method=Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        return np.expand_dims(arr, axis=0)
    except Exception as e:
        raise RuntimeError(f"Failed to preprocess '{image_path}': {e}")


def validate_image_file(image_path):
    if not image_path:
        return False, "No image path"
    if not os.path.exists(image_path):
        return False, "File does not exist"
    if os.path.getsize(image_path) == 0:
        return False, "File is empty"
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in VALID_EXT:
        return False, f"Unsupported format: {ext}"
    try:
        img = Image.open(image_path)
        img.verify()
        return True, ""
    except Exception as e:
        return False, f"Corrupted image: {e}"


def clean_product_dataframe(df):
    import pandas as pd
    df = df.copy()
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip()
    df.replace('nan', pd.NA, inplace=True)
    df.dropna(subset=['Product', 'Category'], inplace=True)
    df['Category']    = df['Category'].str.strip().str.title()
    df['SkinType']    = df['SkinType'].str.strip()
    df['Concern']     = df['Concern'].fillna('')
    df['product_url'] = df['product_url'].fillna('#')
    df['product_pic'] = df['product_pic'].fillna('')
    df.reset_index(drop=True, inplace=True)
    return df


def product_matches_skin_type(skin_type_field, target):
    if not skin_type_field or not target:
        return False
    parts = [p.strip().title() for p in str(skin_type_field).split(',')]
    return target.strip().title() in parts


def product_concern_score(concern_field, selected_concerns):
    if not concern_field or not selected_concerns:
        return 0
    product_concerns = [c.strip() for c in str(concern_field).split('|')]
    return sum(1 for c in selected_concerns if c in product_concerns)
