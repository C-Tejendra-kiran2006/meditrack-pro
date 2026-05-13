"""
MediTrack Pro — Model Trainer
Trains a Logistic Regression classifier on the Pima Indians Diabetes dataset
and saves both the model and scaler as .pkl files.
"""
import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def train_and_save():
    csv_path = 'diabetes.csv'
    if not os.path.exists(csv_path):
        print(f"❌ Error: '{csv_path}' not found!")
        return

    print("📊 Loading dataset...")
    df = pd.read_csv(csv_path)
    print(f"   Rows: {len(df)}, Columns: {list(df.columns)}")

    # Replace biologically impossible zeros with column medians
    zero_replace_cols = ['Glucose', 'BloodPressure', 'BMI']
    for col in zero_replace_cols:
        if col in df.columns:
            df[col] = df[col].replace(0, df[col].median())

    X = df[['Glucose', 'BloodPressure', 'BMI', 'Age']]
    y = df['Outcome']

    print(f"\n📈 Class distribution: Negative={sum(y==0)}, Positive={sum(y==1)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Try multiple models and pick the best
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    best_model = None
    best_auc = 0
    best_name = ''

    print("\n🔬 Evaluating models:")
    print("-" * 55)
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_proba = model.predict_proba(X_test_s)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        cv = cross_val_score(model, scaler.transform(X), y, cv=5, scoring='roc_auc').mean()
        print(f"  {name:<28} Acc: {acc:.3f}  AUC: {auc:.3f}  CV-AUC: {cv:.3f}")
        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_name = name

    print("-" * 55)
    print(f"\n✅ Best model: {best_name} (AUC: {best_auc:.3f})")

    # Save best model and scaler
    with open('diabetes_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print(f"\n💾 Saved to: {os.getcwd()}")
    print("   ✓ diabetes_model.pkl")
    print("   ✓ scaler.pkl")

    # Final report
    y_pred_final = best_model.predict(X_test_s)
    print(f"\n📋 Classification Report ({best_name}):")
    print(classification_report(y_test, y_pred_final, target_names=['No Diabetes', 'Diabetes']))

if __name__ == "__main__":
    train_and_save()
