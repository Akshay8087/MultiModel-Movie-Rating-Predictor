"""CinePredict Advanced Flask Platform"""
from flask import Flask, request, jsonify, render_template, send_file
import pickle, json, numpy as np, os

app = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))

def _load(p):
    with open(os.path.join(BASE, p), 'rb') as f: return pickle.load(f)

SCALER    = _load('scaler.pkl')
FEATURES  = _load('feature_names.pkl')
LE_TARGET = _load('label_encoder.pkl')
CLASSES   = ['Average', 'Good', 'Excellent']
with open(os.path.join(BASE, 'final_results.json')) as f: RESULTS = json.load(f)

SLUG_MAP = {
    'Logistic Regression':'baseline_logistic_regression.pkl',
    'Decision Tree':       'baseline_decision_tree.pkl',
    'Random Forest':       'baseline_random_forest.pkl',
    'Extra Trees':         'baseline_extra_trees.pkl',
    'Gradient Boosting':   'baseline_gradient_boosting.pkl',
    'AdaBoost':            'baseline_adaboost.pkl',
    'SVM (RBF)':           'baseline_svm_rbf.pkl',
    'Naive Bayes':         'baseline_naive_bayes.pkl',
    'KNN':                 'baseline_k-nearest_neighbors.pkl',
    'XGBoost':             'xgboost.pkl',
    'LightGBM':            'lightgbm.pkl',
    'RF (Tuned)':          'tuned_rf.pkl',
    'XGBoost (Tuned)':     'tuned_xgboost.pkl',
    'Voting Ensemble':     'ensemble.pkl',
}
MODELS = {}
for name, path in SLUG_MAP.items():
    full = os.path.join(BASE, path)
    if os.path.exists(full): MODELS[name] = _load(path)
print(f"Loaded {len(MODELS)} models")

def build_features(data):
    pop   = float(data.get('popularity', 10))
    vc    = float(data.get('vote_count', 5000))
    year  = float(data.get('release_year', 2010))
    lenc  = int(data.get('language_encoded', 0))
    isen  = int(data.get('is_english', 1))
    ovl   = float(data.get('overview_length', 250))
    wc    = float(data.get('word_count', 45))
    lp = np.log1p(pop); lv = np.log1p(vc); age = 2024 - year
    raw = np.array([[lp, lv, age, lenc, isen, ovl, wc, lp*lv, age*lv]])
    return SCALER.transform(raw)

def decode_pred(raw_pred, model_name):
    try:
        return LE_TARGET.inverse_transform([int(raw_pred)])[0]
    except Exception:
        return str(raw_pred)

def decode_classes(model_classes):
    try:
        return [LE_TARGET.inverse_transform([int(c)])[0] for c in model_classes]
    except Exception:
        return [str(c) for c in model_classes]

@app.route('/')
def index():
    return render_template('index.html', models=list(MODELS.keys()), results=RESULTS)

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    name = data.get('model', list(MODELS.keys())[0])
    if name not in MODELS: return jsonify({'error': f'Model not found: {name}'}), 400
    try:
        X = build_features(data); model = MODELS[name]
        raw = model.predict(X)[0]
        pred = decode_pred(raw, name)
        proba = model.predict_proba(X)[0].tolist()
        classes = decode_classes(model.classes_)
        pd2 = dict(zip(classes, proba))
        row = next((r for r in RESULTS if r['Model']==name), {})
        return jsonify({'model':name,'prediction':str(pred),'probabilities':{c:round(pd2.get(c,0),4) for c in CLASSES},'confidence':round(max(proba),4),'model_accuracy':row.get('Accuracy',0),'model_f1':row.get('F1',0),'model_type':row.get('Type','')})
    except Exception as e: return jsonify({'error':str(e)}), 400

@app.route('/api/compare', methods=['POST'])
def compare_all():
    data = request.get_json()
    try: X = build_features(data)
    except Exception as e: return jsonify({'error':str(e)}), 400
    out = []
    for name, model in MODELS.items():
        try:
            raw  = model.predict(X)[0]; pred = decode_pred(raw, name)
            proba = model.predict_proba(X)[0].tolist()
            classes = decode_classes(model.classes_)
            pd2 = dict(zip(classes, proba))
            row = next((r for r in RESULTS if r['Model']==name), {})
            out.append({'model':name,'prediction':str(pred),'confidence':round(max(proba),4),'probabilities':{c:round(pd2.get(c,0),4) for c in CLASSES},'accuracy':row.get('Accuracy',0),'f1_score':row.get('F1',0),'mcc':row.get('MCC',0),'type':row.get('Type','')})
        except Exception: continue
    out.sort(key=lambda x: x['accuracy'], reverse=True)
    return jsonify(out)

@app.route('/api/models')
def model_info(): return jsonify(RESULTS)

@app.route('/api/viz/<name>')
def serve_viz(name):
    p = os.path.join(BASE, f'viz_{name}.png')
    return send_file(p, mimetype='image/png') if os.path.exists(p) else (jsonify({'error':'not found'}),404)

if __name__ == '__main__': app.run(debug=True, port=5000, host='0.0.0.0')