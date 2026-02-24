import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.utils import resample
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier as rfc 
warnings.filterwarnings('ignore')

np.random.seed(42)

# =============================================================================
# LOAD DATA
# =============================================================================
df = pd.read_csv('results_output_2day_remove.csv')
df['label'] = df['finallabel'].apply(lambda x: 1 if x == 'improved' else 0)

# =============================================================================
# FEATURE COMBINATIONS (matching paper's 5 settings per group)
# =============================================================================
# DQ features by type
mood_dq       = ['moodmean', 'moodSTDV']
anxiety_dq    = ['anxietymean', 'anxietySTDV']
mood_dq_base  = ['moodmeanbaseline', 'moodSTDVbaseline']
anxiety_dq_base = ['anxietymeanbaseline', 'anxietySTDVbaseline']
qids_base     = ['Qbaseline']
qids_current  = ['Q']

def make_combos(dq_feats, dq_base_feats):
    return {
        'DQ':                      dq_feats,
        'DQ + DQ base':            dq_feats + dq_base_feats,
        'DQ + QIDS base':          dq_feats + qids_base,
        'DQ + DQ base\n+ QIDS base': dq_feats + dq_base_feats + qids_base,
        'All':                     dq_feats + dq_base_feats + qids_base + qids_current,
    }

feature_groups = {
    'Anxiety': make_combos(anxiety_dq, anxiety_dq_base),
    'Mood':    make_combos(mood_dq,    mood_dq_base),
    'Anxiety + Mood': make_combos(anxiety_dq + mood_dq, anxiety_dq_base + mood_dq_base),
}

# Comparison baseline (horizontal line in paper): QIDS + QIDS baseline
comparison_baseline_feats = qids_current + qids_base

# =============================================================================
# UPSAMPLING (match paper's class balancing approach)
# =============================================================================
maj = df[df['finallabel'] == 'nonImproved']
min_ = df[df['finallabel'] == 'improved']
min_up = resample(min_, replace=True, n_samples=len(maj), random_state=42)
df_bal = pd.concat([maj, min_up]).sample(frac=1, random_state=42).reset_index(drop=True)

# Stratify by gender + label (as in the LIBSVM script)
df_bal['gender_stratify'] = df_bal['Gender'].astype(str) + '_' + df_bal['finallabel']
stratify_labels = LabelEncoder().fit_transform(df_bal['gender_stratify'])
y_all = df_bal['label'].values

# =============================================================================
# SVM HYPERPARAMETER GRID (C and gamma as powers of 2, matching paper)
# =============================================================================
C_exps     = list(range(-7, 7))
gamma_exps = list(range(-7, 7))

# =============================================================================
# CROSS-VALIDATION + TRAINING FUNCTION
# =============================================================================
def run_cv(feature_cols, df_bal, stratify_labels):
    """Run 11-fold stratified CV with SVM RBF, grid search over C and gamma.
    Returns best (mean F1, precision, recall, specificity) across hyperparams."""

    X_all = df_bal[feature_cols].values
    y_all = df_bal['label'].values

    skf = StratifiedKFold(n_splits=11, shuffle=True, random_state=42)
    best_f1 = -1
    best_metrics = {}

    for C_exp in C_exps:
        for gamma_exp in gamma_exps:
            C_val     = 2 ** C_exp
            gamma_val = 2 ** gamma_exp

            fold_metrics = []
            for train_idx, val_idx in skf.split(X_all, stratify_labels):
                train_data = df_bal.iloc[train_idx].copy()
                val_data   = df_bal.iloc[val_idx].copy()

                # Re-balance training fold
                maj_tr  = train_data[train_data['finallabel'] == 'nonImproved']
                min_tr  = train_data[train_data['finallabel'] == 'improved']
                min_up  = resample(min_tr, replace=True, n_samples=len(maj_tr), random_state=42)
                train_bal = pd.concat([maj_tr, min_up]).sample(frac=1, random_state=42)

                scaler  = MinMaxScaler()
                X_train = scaler.fit_transform(train_bal[feature_cols])
                X_val   = scaler.transform(val_data[feature_cols])
                y_train = train_bal['label'].values
                y_val   = val_data['label'].values

                model = SVC(kernel='rbf', C=C_val, gamma=gamma_val) # SVM RBF
                #model = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False,
                                          #eval_metric='logloss', C=C_val, gamma=gamma_val)
                #model = rfc(n_estimators=100, max_depth=5, random_state=42)
                model.fit(X_train, y_train)
                preds = model.predict(X_val)

                f1   = f1_score(y_val, preds, zero_division=0)
                prec = precision_score(y_val, preds, zero_division=0)
                rec  = recall_score(y_val, preds, zero_division=0)
                tn, fp, fn, tp = confusion_matrix(y_val, preds, labels=[0,1]).ravel()
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

                fold_metrics.append((f1, prec, rec, spec))

            mean_f1   = np.mean([m[0] for m in fold_metrics])
            mean_prec = np.mean([m[1] for m in fold_metrics])
            mean_rec  = np.mean([m[2] for m in fold_metrics])
            mean_spec = np.mean([m[3] for m in fold_metrics])

            if mean_f1 > best_f1:
                best_f1 = mean_f1
                best_metrics = {
                    'f1':          mean_f1,
                    'precision':   mean_prec,
                    'recall':      mean_rec,
                    'specificity': mean_spec,
                }

    return best_metrics

# =============================================================================
# RUN ALL COMBINATIONS
# =============================================================================
print("Running CV for comparison baseline (QIDS + QIDS baseline)...")
baseline_metrics = run_cv(comparison_baseline_feats, df_bal, stratify_labels)
print(f"  Baseline → F1={baseline_metrics['f1']:.3f}, Spec={baseline_metrics['specificity']:.3f}")

results = {}  # results[group_name][combo_name] = metrics dict
for grp_name, combos in feature_groups.items():
    results[grp_name] = {}
    for combo_name, feats in combos.items():
        print(f"Running: {grp_name} | {combo_name.replace(chr(10),' ')} ...")
        metrics = run_cv(feats, df_bal, stratify_labels)
        results[grp_name][combo_name] = metrics
        print(f"  F1={metrics['f1']:.3f}  Prec={metrics['precision']:.3f}  "
              f"Rec={metrics['recall']:.3f}  Spec={metrics['specificity']:.3f}")

# =============================================================================
# PLOTTING — Figure 6 (F1 + Specificity) and Figure 7 (Precision + Recall)
# =============================================================================
group_names  = list(feature_groups.keys())       # Anxiety, Mood, Anxiety+Mood
combo_labels = list(list(feature_groups.values())[0].keys())  # 5 combos
n_combos     = len(combo_labels)
n_groups     = len(group_names)

bar_colors = ['#4878CF', '#6ACC65', '#D65F5F', '#B47CC7', '#C4AD66']
bar_width  = 0.14
group_gap  = 0.3
x_positions = []
for g in range(n_groups):
    base = g * (n_combos * bar_width + group_gap)
    x_positions.append([base + i * bar_width for i in range(n_combos)])

def make_figure(metric1_key, metric2_key, metric1_label, metric2_label, fig_title, filename):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    fig.suptitle(fig_title, fontsize=13, fontweight='bold', y=0.98)

    for ax_idx, (metric_key, metric_label) in enumerate(
            [(metric1_key, metric1_label), (metric2_key, metric2_label)]):
        ax = axes[ax_idx]

        for g, grp_name in enumerate(group_names):
            for c, combo_name in enumerate(combo_labels):
                val = results[grp_name][combo_name][metric_key]
                ax.bar(x_positions[g][c], val,
                       width=bar_width, color=bar_colors[c],
                       edgecolor='black', linewidth=0.5)

        # Baseline horizontal line per group
        bl_val = baseline_metrics[metric_key]
        for g in range(n_groups):
            x_left  = x_positions[g][0] - bar_width * 0.6
            x_right = x_positions[g][-1] + bar_width * 0.6
            ax.hlines(bl_val, x_left, x_right,
                      colors='black', linestyles='--', linewidth=1.5,
                      label='QIDS + QIDS base' if g == 0 else None)

        # Group labels on x-axis
        group_centers = [np.mean(x_positions[g]) for g in range(n_groups)]
        ax.set_xticks(group_centers)
        ax.set_xticklabels(group_names, fontsize=10)
        ax.set_ylabel(metric_label, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.yaxis.grid(True, linestyle=':', alpha=0.6)
        ax.set_axisbelow(True)

    # Legend
    patches = [mpatches.Patch(color=bar_colors[c],
                               label=combo_labels[c].replace('\n', ' '))
               for c in range(n_combos)]
    patches.append(plt.Line2D([0], [0], color='black', linestyle='--',
                               linewidth=1.5, label='QIDS + QIDS base (baseline)'))
    fig.legend(handles=patches, loc='lower center', ncol=3,
               fontsize=8.5, bbox_to_anchor=(0.5, 0.01), frameon=True)

    plt.tight_layout(rect=[0, 0.08, 1, 0.97])
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

make_figure('f1', 'specificity',
            'F1 Score', 'Specificity',
            'Figure 6 — Weekly Prediction: F1 Score and Specificity (SVM, 2 day removed)',
            #'Figure 6 — Weekly Prediction: F1 Score and Specificity (XGBoost, 2 day removed)',
            #'Figure 6 — Weekly Prediction: F1 Score and Specificity (Random Forest, 2 days removed)',
            '/Users/ryanseely/Desktop/figure6_f1_specificity_remove2_SVM.png')

make_figure('precision', 'recall',
            'Precision', 'Recall',
            'Figure 7 — Weekly Prediction: Precision and Recall (SVM, 2 days removed)',
            #'Figure 7 — Weekly Prediction: Precision and Recall (XGBoost, 2 day removed)',
             #'Figure 7 — Weekly Prediction: Precision and Recall (Random Forest, 2 days removed)',
            '/Users/ryanseely/Desktop/figure7_f1_P&R_remove2_SVM.png')

# =============================================================================
# SAVE NUMERIC RESULTS TABLE
# =============================================================================
rows = []
for grp_name, combos in results.items():
    for combo_name, m in combos.items():
        rows.append({
            'Feature Group': grp_name,
            'Input Combo': combo_name.replace('\n', ' '),
            'F1':          round(m['f1'], 4),
            'Precision':   round(m['precision'], 4),
            'Recall':      round(m['recall'], 4),
            'Specificity': round(m['specificity'], 4),
        })
rows.append({
    'Feature Group': '--- Comparison Baseline ---',
    'Input Combo': 'QIDS + QIDS baseline',
    'F1':          round(baseline_metrics['f1'], 4),
    'Precision':   round(baseline_metrics['precision'], 4),
    'Recall':      round(baseline_metrics['recall'], 4),
    'Specificity': round(baseline_metrics['specificity'], 4),
})
results_df = pd.DataFrame(rows)
results_df.to_csv('/Users/ryanseely/Desktop/SVMtable_remove2_csv', index=False)
#results_df.to_csv('/Users/ryanseely/Desktop/rfcTable_remove2.csv', index=False)
#results_df.to_csv('/Users/ryanseely/Desktop/xgbTable_remove1.csv', index=False)
print("\nAll results:")
print(results_df.to_string(index=False))
