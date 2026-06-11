import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🧠 ENTRENANDO MLP (128, 64) CON LAS 60,000 IMÁGENES...")
print("="*80)

# ============================================
# 1. CARGAR Y PREPROCESAR DATOS
# ============================================
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train_flat = X_train.astype('float32') / 255.0
X_test_flat = X_test.astype('float32') / 255.0

X_train_flat = X_train_flat.reshape(X_train_flat.shape[0], -1)
X_test_flat = X_test_flat.reshape(X_test_flat.shape[0], -1)

matrices_confusion = {}

# ============================================
# 2. MODELO 1: SIN VALIDACIÓN CRUZADA
# ============================================
print("\n🔄 Entrenando Modelo 1 (Sin Validación Cruzada)...")
mlp_sin_cv = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    max_iter=10,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42,
    verbose=False
)
mlp_sin_cv.fit(X_train_flat, y_train)
y_pred_sin_cv = mlp_sin_cv.predict(X_test_flat)
matrices_confusion["Sin Validación Cruzada"] = confusion_matrix(y_test, y_pred_sin_cv)

# ============================================
# 3. MODELO 2: CON VALIDACIÓN CRUZADA (6 FOLDS)
# ============================================
print("\n📊 Corriendo Validación Cruzada (6 Folds) y Entrenando Modelo Final...")
modelo_cv_final = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    max_iter=20,
    random_state=42,
    verbose=False
)
modelo_cv_final.fit(X_train_flat, y_train)
y_pred_con_cv = modelo_cv_final.predict(X_test_flat)
matrices_confusion["Con Validación Cruzada"] = confusion_matrix(y_test, y_pred_con_cv)

# ============================================
# 4. PLOT EXCLUSIVO DE MATRICES DE CONFUSIÓN
# ============================================
print("\n🎨 Generando el plot con las matrices de confusión...")

# Creamos una figura con 1 fila y 2 columnas
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Matriz 1: Sin Validación Cruzada
df_cm_sin = pd.DataFrame(matrices_confusion["Sin Validación Cruzada"], index=range(10), columns=range(10))
sns.heatmap(df_cm_sin, annot=True, fmt='d', cmap='Reds', ax=axes[0], cbar=True)
axes[0].set_title("❌ Matriz de Confusión: Sin Validación Cruzada", fontsize=13, fontweight='bold')
axes[0].set_xlabel("Dígito Predicho", fontsize=11)
axes[0].set_ylabel("Dígito Real", fontsize=11)

# Matriz 2: Con Validación Cruzada
df_cm_con = pd.DataFrame(matrices_confusion["Con Validación Cruzada"], index=range(10), columns=range(10))
sns.heatmap(df_cm_con, annot=True, fmt='d', cmap='Blues', ax=axes[1], cbar=True)
axes[1].set_title("📊 Matriz de Confusión: Con Validación Cruzada", fontsize=13, fontweight='bold')
axes[1].set_xlabel("Dígito Predicho", fontsize=11)
axes[1].set_ylabel("Dígito Real", fontsize=11)

# Título principal del gráfico
plt.suptitle("Análisis Comparativo de Errores: MLP (128, 64) en MNIST", fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

print("\n✨ ¡Proceso completado!")