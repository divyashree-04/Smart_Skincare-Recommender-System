"""
app.py — Smart Skincare Recommender
3 Skin Types: Dry, Normal, Oily
Real ML inference when skin_model.h5 exists.
"""
import os, base64, random
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from werkzeug.utils import secure_filename
from preprocessing import (
    load_and_preprocess_image, validate_image_file,
    clean_product_dataframe, product_matches_skin_type, product_concern_score,
    SKIN_TYPES, CONCERN_MAP, ALL_CONCERNS,
)

app = Flask(__name__)
app.secret_key = 'skincare_secret_2024'
app.config['UPLOAD_FOLDER']      = os.path.join('static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ── LOAD MODEL ONCE AT STARTUP ──────────────────────────────────
MODEL      = None
MODEL_PATH = 'skin_model.h5'

def load_model():
    global MODEL
    if os.path.exists(MODEL_PATH):
        try:
            import tensorflow as tf
            MODEL = tf.keras.models.load_model(MODEL_PATH)
            dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
            MODEL.predict(dummy, verbose=0)
            print(f"✅ Model loaded — {MODEL_PATH}")
        except Exception as e:
            print(f"⚠️  Model load failed: {e} — using demo mode")
            MODEL = None
    else:
        print("ℹ️  No skin_model.h5 found — running in demo mode")
        print("   To train: python train_model.py")

load_model()

# ── CONSTANTS ───────────────────────────────────────────────────
CATEGORIES = ['Face Wash', 'Serum', 'Moisturizer', 'Sunscreen', 'Toner', 'Other']
CAT_ICONS  = {
    'Face Wash':'🧴','Serum':'✨','Moisturizer':'💧',
    'Sunscreen':'☀️','Toner':'🌿','Other':'🫧',
}
SKIN_ICONS = {'Oily':'🫧','Dry':'🌵','Normal':'✨'}

# ── ML PREDICTION ───────────────────────────────────────────────
def predict_skin(image_path):
    """
    Returns: skin_type, concerns, confidence, all_probs, source
    Never crashes — falls back to demo if anything fails.
    """
    is_valid, err = validate_image_file(image_path)
    if not is_valid:
        print(f"[predict] Invalid image: {err}")
        return _demo_prediction()

    if MODEL is not None:
        try:
            arr        = load_and_preprocess_image(image_path)
            probs      = MODEL.predict(arr, verbose=0)[0]
            idx        = int(np.argmax(probs))
            skin_type  = SKIN_TYPES[idx]         # Dry/Normal/Oily (alphabetical)
            confidence = float(np.max(probs))
            all_probs  = {st: round(float(p), 4) for st, p in zip(SKIN_TYPES, probs)}
            concerns   = CONCERN_MAP.get(skin_type, ALL_CONCERNS[:3])
            print(f"[ML] {skin_type} — {confidence:.1%}")
            return skin_type, concerns, confidence, all_probs, 'model'
        except Exception as e:
            print(f"[predict] Error: {e} — demo fallback")
            return _demo_prediction()

    return _demo_prediction()


def _demo_prediction():
    skin_type  = random.choice(SKIN_TYPES)
    concerns   = CONCERN_MAP.get(skin_type, [])[:3]
    confidence = round(random.uniform(0.60, 0.85), 4)
    all_probs  = {st: round(random.uniform(0.05, 0.20), 4) for st in SKIN_TYPES}
    all_probs[skin_type] = confidence
    total = sum(all_probs.values())
    all_probs = {k: round(v/total, 4) for k, v in all_probs.items()}
    return skin_type, concerns, confidence, all_probs, 'demo'


# ── RECOMMENDATIONS ─────────────────────────────────────────────
def get_recommendations(skin_type, concerns):
    recs = {}
    try:
        df = clean_product_dataframe(pd.read_csv('skincare.csv'))
    except Exception as e:
        print(f"CSV error: {e}")
        return recs

    for cat in CATEGORIES:
        cat_df = df[df['Category'] == cat].copy()
        if cat_df.empty:
            continue
        matched = cat_df[cat_df['SkinType'].apply(
            lambda x: product_matches_skin_type(x, skin_type)
        )].copy()
        working = matched if not matched.empty else cat_df.copy()
        if concerns:
            working['_score'] = working['Concern'].apply(
                lambda x: product_concern_score(x, concerns)
            )
            working = working.sort_values('_score', ascending=False)
        top = working.head(2)
        if not top.empty:
            recs[cat] = top.drop(columns=['_score'], errors='ignore').to_dict('records')
    return recs


# ── ROUTES ──────────────────────────────────────────────────────
@app.route('/')
def index():
    session.clear()
    return render_template('index.html')

@app.route('/choice')
def choice():
    return render_template('choice.html')

@app.route('/upload', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        image_path = None
        f = request.files.get('skin_image')
        if f and f.filename:
            fname      = secure_filename(f.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
            f.save(image_path)
        elif request.form.get('webcam_data'):
            try:
                _, encoded = request.form['webcam_data'].split(',', 1)
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'webcam.jpg')
                with open(image_path, 'wb') as fh:
                    fh.write(base64.b64decode(encoded))
            except Exception as e:
                print(f"Webcam error: {e}")
        session['image_path'] = image_path
        return redirect(url_for('loading'))
    return render_template('upload.html')

@app.route('/loading')
def loading():
    return render_template('loading.html')

@app.route('/analyze')
def analyze():
    image_path = session.get('image_path')
    skin_type, concerns, confidence, all_probs, source = predict_skin(image_path)
    session.update({
        'detected_skin_type': skin_type,
        'detected_concerns' : concerns,
        'confidence'        : confidence,
        'all_probs'         : all_probs,
        'source'            : source,
    })
    return jsonify({
        'skin_type' : skin_type,
        'concerns'  : concerns,
        'confidence': round(confidence * 100, 1),
        'source'    : source,
    })

@app.route('/result')
def result():
    skin_type  = session.get('detected_skin_type', 'Normal')
    concerns   = session.get('detected_concerns',  [])
    confidence = session.get('confidence', 0.0)
    all_probs  = session.get('all_probs',  {})
    source     = session.get('source',     'demo')
    return render_template('result.html',
        skin_type  = skin_type,
        skin_icon  = SKIN_ICONS.get(skin_type, '✨'),
        concerns   = concerns,
        confidence = round(confidence * 100, 1),
        all_probs  = all_probs,
        source     = source,
    )

@app.route('/confirm-result', methods=['POST'])
def confirm_result():
    if request.form.get('accurate') == 'yes':
        session['skin_type'] = session.get('detected_skin_type', 'Normal')
        session['concerns']  = session.get('detected_concerns',  [])
        return redirect(url_for('recommendations'))
    return redirect(url_for('manual_select'))

@app.route('/manual-select', methods=['GET','POST'])
def manual_select():
    if request.method == 'POST':
        skin_type = request.form.get('skin_type','').strip()
        concerns  = request.form.getlist('concerns')
        if skin_type not in SKIN_TYPES:
            skin_type = 'Normal'
        concerns = [c for c in concerns if c in ALL_CONCERNS]
        session['skin_type'] = skin_type
        session['concerns']  = concerns
        return redirect(url_for('recommendations'))
    return render_template('manual.html',
        skin_types       = SKIN_TYPES,
        concerns         = ALL_CONCERNS,
        prefill_skin     = session.get('detected_skin_type', ''),
        prefill_concerns = session.get('detected_concerns',  []),
        skin_icons       = SKIN_ICONS,
    )

@app.route('/recommendations')
def recommendations():
    skin_type = session.get('skin_type', '')
    concerns  = session.get('concerns',  [])
    if not skin_type or skin_type not in SKIN_TYPES:
        return redirect(url_for('manual_select'))
    recs = get_recommendations(skin_type, concerns)
    return render_template('recommendations.html',
        skin_type      = skin_type,
        skin_icon      = SKIN_ICONS.get(skin_type, '✨'),
        concerns       = concerns,
        recommendations= recs,
        cat_icons      = CAT_ICONS,
    )

if __name__ == '__main__':
    app.run(debug=True, port=5000)
