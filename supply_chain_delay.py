
# PROJECT  : AI Supply Chain Delay Prediction
# Domain    : Logistics / Supply Chain
# AI Angle  : Random Forest classifier predicts shipment delay
#             BEFORE it departs — enabling proactive rerouting.
# CSVs Used : csv/shipments.csv | csv/suppliers.csv | csv/routes.csv

import os, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model     import LogisticRegression
from sklearn.model_selection  import train_test_split, cross_val_score
from sklearn.preprocessing    import StandardScaler, LabelEncoder
from sklearn.metrics          import (classification_report, confusion_matrix,
                                      roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
CSV  = os.path.join(BASE, '..', 'csv')
OUT  = BASE

print("=" * 58)
print("  PROJECT 3 — AI Supply Chain Delay Prediction")
print("=" * 58)

#  1. LOAD DATA 
shipments = pd.read_csv(os.path.join(CSV, 'shipments.csv'), parse_dates=['order_date'])
suppliers = pd.read_csv(os.path.join(CSV, 'suppliers.csv'))
routes    = pd.read_csv(os.path.join(CSV, 'routes.csv'))

# Merge all tables (mimicking SQL JOIN)
df = (shipments
      .merge(suppliers[['supplier_id','reliability_score','country',
                         'tier','category']], on='supplier_id', suffixes=('','_sup'))
      .merge(routes[['route_id','distance_km','avg_days']], on='route_id', suffixes=('','_rt')))

print(f"\n✅  Loaded  →  shipments: {len(shipments):,}  |  "
      f"suppliers: {len(suppliers)}  |  routes: {len(routes)}")
print(f"    Overall delay rate : {df['is_delayed'].mean():.1%}")
print(f"    Total order value  : ₹{df['order_value'].sum():,.0f}")

#  2. EXPLORATORY ANALYSIS 
print("\n--- Delay Rate by Transport Mode ---")
mode_delay = df.groupby('transport_mode')['is_delayed'].agg(['mean','count']).round(3)
mode_delay.columns = ['delay_rate','shipments']
mode_delay['delay_rate_pct'] = (mode_delay['delay_rate'] * 100).round(1)
print(mode_delay.to_string())

print("\n--- Delay Rate by Root Cause ---")
for col in ['weather_risk','port_congestion','supplier_issue','demand_spike']:
    with_factor  = df[df[col]==1]['is_delayed'].mean() * 100
    without      = df[df[col]==0]['is_delayed'].mean() * 100
    print(f"  {col:<20}: WITH={with_factor:.1f}%  WITHOUT={without:.1f}%  Delta=+{with_factor-without:.1f}pp")

print("\n--- Top 5 Worst Suppliers ---")
sup_delay = (df.groupby(['supplier_id','country','tier','reliability_score'])['is_delayed']
               .agg(['mean','count']).reset_index()
               .sort_values('mean', ascending=False).head(5))
sup_delay.columns = ['supplier_id','country','tier','reliability','delay_rate','shipments']
sup_delay['delay_rate'] = (sup_delay['delay_rate']*100).round(1)
print(sup_delay.to_string(index=False))

# 3. FEATURE ENGINEERING 
le = LabelEncoder()
df['transport_enc']  = le.fit_transform(df['transport_mode'])
df['tier_enc']       = le.fit_transform(df['tier'])
df['country_enc']    = le.fit_transform(df['country'])
df['category_enc']   = le.fit_transform(df['product_category'])

df['month']          = df['order_date'].dt.month
df['quarter']        = df['order_date'].dt.quarter
df['day_of_week']    = df['order_date'].dt.dayofweek
df['is_peak_season'] = df['month'].isin([10,11,12]).astype(int)
df['value_per_kg']   = df['order_value'] / df['weight_kg'].clip(lower=0.1)
df['days_buffer']    = df['expected_days'] - df['avg_days']   # schedule slack

feature_cols = [
    'reliability_score','country_enc','tier_enc',
    'distance_km','transport_enc','category_enc',
    'weight_kg','order_value','value_per_kg',
    'expected_days','days_buffer',
    'weather_risk','port_congestion','supplier_issue','demand_spike',
    'month','quarter','day_of_week','is_peak_season'
]

X = df[feature_cols].fillna(0)
y = df['is_delayed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                     stratify=y, random_state=42)
scaler   = StandardScaler()
X_tr_sc  = scaler.fit_transform(X_train)
X_te_sc  = scaler.transform(X_test)

#  4. TRAIN MULTIPLE AI MODELS
print("\n--- Training AI Models ---")
models = {
    'Random Forest':       RandomForestClassifier(n_estimators=200, max_depth=8,
                                                  class_weight='balanced', random_state=42),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=150, max_depth=4,
                                                      learning_rate=0.1, random_state=42),
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=500, random_state=42)
}

results = {}
for name, model in models.items():
    X_tr_use = X_tr_sc if name == 'Logistic Regression' else X_train
    X_te_use = X_te_sc if name == 'Logistic Regression' else X_test
    model.fit(X_tr_use, y_train)
    y_pred = model.predict(X_te_use)
    y_prob = model.predict_proba(X_te_use)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)
    cv_auc = cross_val_score(model, X_tr_use, y_train, cv=5, scoring='roc_auc').mean()
    results[name] = {'model':model, 'y_pred':y_pred, 'y_prob':y_prob,
                     'auc':auc, 'cv_auc':cv_auc}
    print(f"\n  {name}")
    print(f"    AUC  = {auc:.3f}  |  CV-AUC = {cv_auc:.3f}")
    print(classification_report(y_test, y_pred, target_names=['On-Time','Delayed'],
                                 zero_division=0, output_dict=False)
          .replace('\n', '\n    '))

# Best model
best_name = max(results, key=lambda k: results[k]['auc'])
best      = results[best_name]
best_model= best['model']
print(f"\n  🏆  Best Model: {best_name}  (AUC = {best['auc']:.3f})")

#  5. FEATURE IMPORTANCE 
if hasattr(best_model, 'feature_importances_'):
    feat_imp = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n--- Top 10 Delay Predictors ---")
    for feat, imp in feat_imp.head(10).items():
        bar = '█' * int(imp * 200)
        print(f"  {feat:<22} {bar:<20} {imp:.3f}")

#  6. RISK SCORING NEW SHIPMENTS 
# Score on test set
risk_df = X_test.copy()
risk_df['delay_probability'] = best['y_prob']
risk_df['actual_delay']      = y_test.values
risk_df['risk_level'] = pd.cut(risk_df['delay_probability'],
                                bins=[0, 0.3, 0.6, 1.0],
                                labels=['Low','Medium','High'])
risk_summary = risk_df.groupby('risk_level').agg(
    shipments   =('delay_probability','count'),
    avg_prob    =('delay_probability','mean'),
    actual_delay=('actual_delay','mean')
).round(3)
print("\n--- Risk Score Distribution ---")
print(risk_summary.to_string())

#  7. FINANCIAL IMPACT SIMULATION 
delay_cost_per_day = 50   # ₹ holding cost per day per shipment
df_test = df.iloc[X_test.index.tolist()].copy()
df_test['delay_prob'] = best['y_prob']
df_test['predicted_delay'] = best['y_pred']

caught_early  = df_test[(df_test['predicted_delay']==1) & (df_test['is_delayed']==1)]
missed_delays = df_test[(df_test['predicted_delay']==0) & (df_test['is_delayed']==1)]
savings = caught_early['delay_days'].sum() * delay_cost_per_day
print(f"\n--- Financial Impact of AI Model ---")
print(f"  Delays caught early (TP) : {len(caught_early):>4} shipments")
print(f"  Delays missed (FN)       : {len(missed_delays):>4} shipments")
print(f"  Estimated savings (₹{delay_cost_per_day}/day): ₹{savings:,.0f}")

#  8. VISUALISATIONS 
fig = plt.figure(figsize=(20, 13))
fig.suptitle("PROJECT 3 — AI Supply Chain Delay Prediction Dashboard", fontsize=15, fontweight='bold')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

# Plot 1: Delay rate by transport mode
ax1 = fig.add_subplot(gs[0, 0])
mode_sorted = mode_delay.sort_values('delay_rate_pct', ascending=True)
colors1 = ['#E74C3C' if v > 40 else '#F39C12' if v > 25 else '#2ECC71'
           for v in mode_sorted['delay_rate_pct']]
bars1 = ax1.barh(mode_sorted.index, mode_sorted['delay_rate_pct'], color=colors1)
ax1.bar_label(bars1, fmt='%.1f%%', padding=3)
ax1.set_title('Delay Rate by Transport Mode', fontweight='bold')
ax1.set_xlabel('Delay Rate (%)')

# Plot 2: Root cause delay impact (bar chart)
ax2 = fig.add_subplot(gs[0, 1])
risk_factors = ['weather_risk','port_congestion','supplier_issue','demand_spike']
risk_labels  = ['Weather','Port\nCongestion','Supplier\nIssue','Demand\nSpike']
with_rates   = [df[df[f]==1]['is_delayed'].mean()*100 for f in risk_factors]
without_rates= [df[df[f]==0]['is_delayed'].mean()*100 for f in risk_factors]
x_pos = np.arange(len(risk_factors))
w = 0.35
ax2.bar(x_pos - w/2, with_rates,    w, color='#E74C3C', label='With Risk Factor', alpha=0.85)
ax2.bar(x_pos + w/2, without_rates, w, color='#2ECC71', label='Without Risk Factor', alpha=0.85)
ax2.set_xticks(x_pos); ax2.set_xticklabels(risk_labels, fontsize=9)
ax2.set_title('Delay Rate: With vs Without Risk Factor', fontweight='bold')
ax2.set_ylabel('Delay Rate (%)'); ax2.legend()

# Plot 3: ROC curves for all models
ax3 = fig.add_subplot(gs[0, 2])
colors3 = ['#3498DB','#E74C3C','#2ECC71']
for (name, res), col in zip(results.items(), colors3):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    ax3.plot(fpr, tpr, color=col, linewidth=2,
             label=f"{name.split()[0]} (AUC={res['auc']:.3f})")
ax3.plot([0,1],[0,1],'k--', linewidth=1, label='Random')
ax3.set_title('ROC Curves — Model Comparison', fontweight='bold')
ax3.set_xlabel('False Positive Rate'); ax3.set_ylabel('True Positive Rate')
ax3.legend(fontsize=8); ax3.grid(True, alpha=0.25)

# Plot 4: Feature importance
ax4 = fig.add_subplot(gs[1, 0])
top_feats = feat_imp.head(10).sort_values()
colors4 = ['#E74C3C' if v > feat_imp.head(10).mean() else '#3498DB' for v in top_feats]
ax4.barh(top_feats.index, top_feats.values, color=colors4)
ax4.set_title(f'Top 10 Feature Importances\n({best_name})', fontweight='bold')
ax4.set_xlabel('Importance Score')

# Plot 5: Confusion matrix
ax5 = fig.add_subplot(gs[1, 1])
cm = confusion_matrix(y_test, best['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5,
            xticklabels=['On-Time','Delayed'],
            yticklabels=['On-Time','Delayed'])
ax5.set_title(f'Confusion Matrix\n({best_name})', fontweight='bold')
ax5.set_ylabel('Actual'); ax5.set_xlabel('Predicted')

# Plot 6: Delay probability distribution
ax6 = fig.add_subplot(gs[1, 2])
on_time_prob  = best['y_prob'][y_test == 0]
delayed_prob  = best['y_prob'][y_test == 1]
ax6.hist(on_time_prob, bins=30, color='#2ECC71', alpha=0.65, density=True, label='On-Time')
ax6.hist(delayed_prob, bins=30, color='#E74C3C', alpha=0.75, density=True, label='Delayed')
ax6.axvline(0.5, color='navy', linestyle='--', linewidth=1.5, label='Decision boundary')
ax6.set_title('AI Delay Probability Distribution', fontweight='bold')
ax6.set_xlabel('Predicted Delay Probability'); ax6.set_ylabel('Density')
ax6.legend()

plt.savefig(os.path.join(OUT, 'p3_supply_chain_dashboard.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n✅  Dashboard saved → p3_supply_chain_dashboard.png")

#  9. EXPORT OUTPUTS 
# Risk-scored shipments
risk_output = df_test[['shipment_id','supplier_id','transport_mode',
                        'order_value','expected_days','is_delayed']].copy()
risk_output['delay_probability'] = best['y_prob']
risk_output['risk_level'] = pd.cut(risk_output['delay_probability'],
                                    bins=[0,0.3,0.6,1.0], labels=['Low','Medium','High'])
risk_output.to_csv(os.path.join(CSV, 'shipment_risk_scores_output.csv'), index=False)

# Supplier risk table
sup_risk = (df.groupby(['supplier_id','country','tier','reliability_score'])
              .agg(total=('is_delayed','count'), delayed=('is_delayed','sum'))
              .reset_index())
sup_risk['delay_rate_pct'] = (sup_risk['delayed'] / sup_risk['total'] * 100).round(1)
sup_risk.sort_values('delay_rate_pct', ascending=False).to_csv(
    os.path.join(CSV, 'supplier_risk_output.csv'), index=False)

# Monthly delay trend
monthly_trend = (df.groupby(df['order_date'].dt.to_period('M'))
                   .agg(shipments=('is_delayed','count'),
                        delay_rate=('is_delayed','mean'))
                   .reset_index())
monthly_trend['order_date'] = monthly_trend['order_date'].astype(str)
monthly_trend['delay_rate_pct'] = (monthly_trend['delay_rate'] * 100).round(1)
monthly_trend.to_csv(os.path.join(CSV, 'monthly_delay_trend_output.csv'), index=False)

print("✅  3 output CSVs saved to /csv/")

print("\n--- INTERVIEW TALKING POINTS ---")
print(f"• Best model: {best_name}  (AUC = {best['auc']:.3f})")
print(f"• Top delay predictor: '{feat_imp.idxmax()}'  (importance = {feat_imp.max():.3f})")
print(f"• Caught {len(caught_early)} delays early → estimated ₹{savings:,.0f} in holding cost savings")
print(f"• Port congestion increases delay rate by +{with_rates[1]-without_rates[1]:.0f}pp vs normal")
print(f"• Architecture: CSV → merge 3 tables → feature eng → RF classifier → risk score")
