"""WAICC PolERo: reproducible TF-IDF + LightGBM ensemble.

Run with train_data.csv, test_data.csv and sample_output.csv accessible
under /kaggle/input (or in the working directory).

Produces submission_ensemble_reproduced.csv. Uses the same algorithms,
folds and hyperparameters as Bianca's two notebooks.
"""

from pathlib import Path
import glob
import re

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

EVASION_CLASSES = [
    'Explicit', 'Implicit', 'General', 'Partial/half-answer', 'Dodging',
    'Deflection', 'Declining to answer', 'Claims ignorance', 'Clarification'
]
EVASION_TO_CLARITY = {
    'Explicit': 'Clear Reply', 'Implicit': 'Ambivalent',
    'General': 'Ambivalent', 'Partial/half-answer': 'Ambivalent',
    'Dodging': 'Ambivalent', 'Deflection': 'Ambivalent',
    'Declining to answer': 'Clear Non-Reply',
    'Claims ignorance': 'Clear Non-Reply',
    'Clarification': 'Clear Non-Reply'
}


def find_file(filename):
    matches = glob.glob(f'/kaggle/input/**/{filename}', recursive=True)
    if matches:
        return matches[0]
    local = Path(filename)
    if local.exists():
        return str(local)
    raise FileNotFoundError(f'Nu am gasit {filename}; verifica Input sau folderul curent.')


def make_context(train, test):
    # Keep the original row order in train/test; the sorted copy is only
    # used to look up preceding interview turns.
    train = train.copy()
    test = test.copy()
    train['is_test'] = 0
    test['is_test'] = 1
    full = pd.concat([train, test], ignore_index=True)
    full = full.sort_values(['url', 'question_order']).reset_index(drop=True)
    same_url = full['url'] == full['url'].shift(1)
    full['turn_gap'] = np.where(
        same_url, full['question_order'] - full['question_order'].shift(1), -1
    )
    full['prev_question'] = np.where(
        full['turn_gap'] == 1, full['interview_question'].shift(1), ''
    )
    full['prev_answer'] = np.where(
        full['turn_gap'] == 1, full['interview_answer'].shift(1), ''
    )
    cols = full[['datapointID', 'turn_gap', 'prev_question', 'prev_answer']]
    assert cols['datapointID'].is_unique
    train_ctx = train.drop(columns=['is_test']).merge(cols, on='datapointID', how='left')
    test_ctx = test.drop(columns=['is_test']).merge(cols, on='datapointID', how='left')
    return train_ctx, test_ctx


def build_text(df):
    return (
        'Presedinte: ' + df['president'].fillna('') +
        ' | Context: ' + df['context'].fillna('') +
        ' | Intrebare: ' + df['interview_question'].fillna('') +
        ' | Raspuns: ' + df['interview_answer'].fillna('')
    )


def extract_features(df):
    features = pd.DataFrame(index=df.index)
    question = df['interview_question'].fillna('').astype(str)
    answer = df['interview_answer'].fillna('').astype(str)
    q_lower = question.str.lower()
    a_lower = answer.str.lower()

    features['question_words'] = question.str.split().str.len()
    features['answer_words'] = answer.str.split().str.len()
    features['question_chars'] = question.str.len()
    features['answer_chars'] = answer.str.len()
    features['length_ratio'] = features['answer_words'] / (features['question_words'] + 1)
    features['answer_question_marks'] = answer.str.count(r'\?')
    features['answer_exclamations'] = answer.str.count('!')
    features['answer_commas'] = answer.str.count(',')
    features['answer_sentences'] = answer.str.count(r'[.!?]') + 1

    features['claims_ignorance'] = a_lower.str.contains(
        r'\bnu (?:știu|stiu|cunosc)\b|'
        r'\bnu (?:am|dețin|detin) (?:aceste )?(?:date|informații|informatii)\b',
        regex=True
    ).astype(int)
    features['refuses_answer'] = a_lower.str.contains(
        r'nu comentez|nu voi comenta|nu mă pronunț|nu ma pronunt|'
        r'nu pot răspunde|nu pot raspunde', regex=True
    ).astype(int)
    features['asks_clarification'] = a_lower.str.contains(
        r'vă referiți|va referiti|care dintre|ce anume|din partea cui',
        regex=True
    ).astype(int)

    def lexical_overlap(q, a):
        q_words = set(re.findall(r'\w+', q.lower()))
        a_words = set(re.findall(r'\w+', a.lower()))
        if not q_words:
            return 0.0
        return len(q_words & a_words) / len(q_words)

    features['lexical_overlap'] = [
        lexical_overlap(q, a) for q, a in zip(question, answer)
    ]
    features['turn_gap'] = df['turn_gap'].fillna(-1)
    features['has_previous_turn'] = (features['turn_gap'] == 1).astype(int)
    features['question_order'] = df['question_order'].fillna(0)
    features['president'] = df['president'].fillna('Unknown').astype('category')
    features['context'] = df['context'].fillna('Unknown').astype('category')
    return features


def main():
    train = pd.read_csv(find_file('train_data.csv'))
    test = pd.read_csv(find_file('test_data.csv'))
    sample = pd.read_csv(find_file('sample_output.csv'))
    assert train['datapointID'].is_unique and test['datapointID'].is_unique
    assert len(train) == 3278 and len(test) == 296

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train['fold'] = -1
    for fold, (_, val_idx) in enumerate(skf.split(train, train['evasion_label'])):
        train.loc[val_idx, 'fold'] = fold

    train_ctx, test_ctx = make_context(train, test)
    label2id = {label: i for i, label in enumerate(EVASION_CLASSES)}
    y_train = train_ctx['evasion_label'].map(label2id).values
    assert not pd.isna(y_train).any()
    X_train_text = build_text(train_ctx)
    X_test_text = build_text(test_ctx)
    baseline_oof = np.zeros((len(train_ctx), 9))
    baseline_test = np.zeros((len(test_ctx), 9))

    for fold in range(5):
        tr_mask = train_ctx['fold'] != fold
        va_mask = train_ctx['fold'] == fold
        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=25000, sublinear_tf=True)
        X_tr = vec.fit_transform(X_train_text[tr_mask])
        X_va = vec.transform(X_train_text[va_mask])
        X_te = vec.transform(X_test_text)
        clf = LogisticRegression(C=2.0, max_iter=1000, class_weight='balanced')
        clf.fit(X_tr, y_train[tr_mask])
        assert np.array_equal(clf.classes_, np.arange(9))
        baseline_oof[va_mask.to_numpy()] = clf.predict_proba(X_va)
        baseline_test += clf.predict_proba(X_te) / 5.0

    X_train = extract_features(train_ctx)
    X_test = extract_features(test_ctx)
    for col in ['president', 'context']:
        categories = pd.Index(pd.concat([
            X_train[col].astype(str), X_test[col].astype(str)
        ]).unique())
        X_train[col] = pd.Categorical(X_train[col], categories=categories)
        X_test[col] = pd.Categorical(X_test[col], categories=categories)

    oof_lgb = np.zeros((len(train_ctx), 9))
    test_lgb = np.zeros((len(test_ctx), 9))
    for fold in range(5):
        tr_mask = train_ctx['fold'] != fold
        va_mask = train_ctx['fold'] == fold
        model = lgb.LGBMClassifier(
            objective='multiclass', num_class=9,
            n_estimators=300, learning_rate=0.03,
            num_leaves=15, max_depth=5, min_child_samples=20,
            class_weight='balanced', reg_lambda=2.0,
            verbosity=-1, random_state=42, n_jobs=-1
        )
        model.fit(
            X_train.loc[tr_mask], y_train[tr_mask],
            categorical_feature=['president', 'context']
        )
        assert np.array_equal(model.classes_, np.arange(9))
        oof_lgb[va_mask.to_numpy()] = model.predict_proba(X_train.loc[va_mask])
        test_lgb += model.predict_proba(X_test) / 5.0

    ensemble_oof = 0.75 * baseline_oof + 0.25 * oof_lgb
    ensemble_test = 0.75 * baseline_test + 0.25 * test_lgb
    pred_9_oof = [EVASION_CLASSES[i] for i in ensemble_oof.argmax(axis=1)]
    pred_3_oof = [EVASION_TO_CLARITY[c] for c in pred_9_oof]
    f1_evasion = f1_score(train_ctx['evasion_label'], pred_9_oof, average='macro')
    f1_clarity = f1_score(train_ctx['clarity_label'], pred_3_oof, average='macro')
    print(f'OOF: Clarity={f1_clarity:.4f}, Evasion={f1_evasion:.4f}, '
          f'Combined={50*(f1_clarity+f1_evasion):.2f}/100')

    pred_9_test = [EVASION_CLASSES[i] for i in ensemble_test.argmax(axis=1)]
    pred_3_test = [EVASION_TO_CLARITY[c] for c in pred_9_test]
    map_3 = dict(zip(test_ctx['datapointID'], pred_3_test))
    map_9 = dict(zip(test_ctx['datapointID'], pred_9_test))
    submission = sample[['subtaskID', 'datapointID']].copy()
    submission['answer'] = np.where(
        submission['subtaskID'] == 1,
        submission['datapointID'].map(map_3),
        submission['datapointID'].map(map_9)
    )
    assert submission.shape == (592, 3)
    assert not submission['answer'].isna().any()
    assert not submission[['subtaskID', 'datapointID']].duplicated().any()
    assert (submission['subtaskID'].value_counts().sort_index().to_numpy() == [296,296]).all()
    submission.to_csv('submission_ensemble_reproduced.csv', index=False)
    print('Wrote submission_ensemble_reproduced.csv')


if __name__ == '__main__':
    main()
