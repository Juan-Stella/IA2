import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import pandas as pd
import joblib  # ← NUEVO: para guardar el modelo
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Silenciar mensajes de TensorFlow
import os as os_env
os_env.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Crear carpeta para guardar modelos si no existe
if not os.path.exists('modelos_guardados'):
    os.makedirs('modelos_guardados')

print("="*70)
print("EJERCICIO 2 - CLASIFICACIÓN DE DÍGITOS MANUSCRITOS MNIST")
print("="*70)

# 1. CARGAR DATOS
print("\n📥 Cargando dataset MNIST...")
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(f"   ✅ Entrenamiento: {X_train.shape[0]} imágenes")
print(f"   ✅ Prueba: {X_test.shape[0]} imágenes")

# 2. NORMALIZAR Y APLANAR
print("\n🔄 Normalizando y aplanando imágenes...")
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)
print(f"   ✅ Imágenes aplanadas a {X_train_flat.shape[1]} características")

# 3. MOSTRAR EJEMPLOS
print("\n🎨 Mostrando ejemplos del dataset...")
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    idx = np.random.randint(0, len(X_train))
    ax.imshow(X_train[idx], cmap='gray')
    ax.set_title(f"Dígito: {y_train[idx]}")
    ax.axis('off')
plt.suptitle("Ejemplos aleatorios del dataset MNIST", fontsize=14)
plt.tight_layout()
plt.show()

# 4. VALIDACIÓN CRUZADA (MÁS RÁPIDA)
print("\n" + "="*70)
print("📊 VALIDACIÓN CRUZADA CON 6 FOLDS")
print("="*70)

# Usar un modelo más simple para validación cruzada (más rápido)
modelo_rapido = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    max_iter=15,
    random_state=42,
    verbose=False
)

# Validación cruzada con 6 folds
kf = KFold(n_splits=6, shuffle=True, random_state=42)
cv_scores = cross_val_score(modelo_rapido, X_train_flat, y_train, cv=kf, n_jobs=-1)

print(f"\n✅ Resultados de validación cruzada (6 folds):")
for i, score in enumerate(cv_scores, 1):
    print(f"   Fold {i}: {score:.4f}")

print(f"\n📊 Estadísticas:")
print(f"   • Precisión media: {cv_scores.mean():.4f}")
print(f"   • Desviación estándar: {cv_scores.std():.4f}")
print(f"   • Rango: {cv_scores.min():.4f} - {cv_scores.max():.4f}")

# 5. ENTRENAR MODELO FINAL
print("\n" + "="*70)
print("🎯 ENTRENANDO MODELO FINAL")
print("="*70)

modelo_final = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    max_iter=20,
    random_state=42,
    verbose=True
)

print("\n🔄 Entrenando con todos los 60,000 datos...")
modelo_final.fit(X_train_flat, y_train)
print("✅ Modelo entrenado")

# 6. EVALUAR EN TEST
print("\n" + "="*70)
print("🧪 EVALUACIÓN EN CONJUNTO DE PRUEBA")
print("="*70)

y_pred = modelo_final.predict(X_test_flat)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Precisión en test: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# 7. REPORTE DE CLASIFICACIÓN
print("\n📋 Reporte de clasificación por dígito:")
print("-" * 50)
print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)]))

# 8. MATRIZ DE CONFUSIÓN
print("\n📊 Generando matriz de confusión...")
cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, index=[str(i) for i in range(10)], 
                     columns=[str(i) for i in range(10)])

plt.figure(figsize=(12, 10))
sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
plt.title("Matriz de Confusión - Clasificación de Dígitos MNIST", fontsize=16)
plt.xlabel("Dígito Predicho", fontsize=12)
plt.ylabel("Dígito Real", fontsize=12)
plt.tight_layout()
plt.show()

# 9. MOSTRAR ERRORES
errores = np.where(y_test != y_pred)[0]
print(f"\n🔍 Total de errores: {len(errores)} sobre {len(y_test)} ({len(errores)/len(y_test)*100:.2f}%)")

if len(errores) > 0:
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.flat):
        if i < len(errores):
            idx = errores[i]
            ax.imshow(X_test[idx], cmap='gray')
            ax.set_title(f"Real: {y_test[idx]} → Pred: {y_pred[idx]}", color='red')
            ax.axis('off')
    plt.suptitle("Ejemplos de errores de clasificación", fontsize=14)
    plt.tight_layout()
    plt.show()

# 10. VISUALIZAR VALIDACIÓN CRUZADA
plt.figure(figsize=(10, 5))
plt.bar(range(1, 7), cv_scores, color='skyblue', edgecolor='navy')
plt.axhline(y=cv_scores.mean(), color='red', linestyle='--', 
            label=f'Media: {cv_scores.mean():.4f}')
plt.xlabel('Fold', fontsize=12)
plt.ylabel('Precisión', fontsize=12)
plt.title('Validación Cruzada - 6 Folds', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
for i, score in enumerate(cv_scores, 1):
    plt.text(i, score + 0.002, f'{score:.4f}', ha='center', fontsize=9)
plt.tight_layout()
plt.show()

# ============================================
# 11. GUARDAR EL MODELO ENTRENADO
# ============================================
print("\n" + "="*70)
print("💾 GUARDANDO MODELO ENTRENADO")
print("="*70)

# Guardar con nombre que incluye la precisión y timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
nombre_modelo = f'modelos_guardados/mnist_model_cv_{test_accuracy:.4f}_{timestamp}.pkl'

joblib.dump(modelo_final, nombre_modelo)
print(f"✅ Modelo guardado en: {nombre_modelo}")

# También guardar una versión con nombre fijo
nombre_fijo = 'modelos_guardados/mnist_model_cv_ultimo.pkl'
joblib.dump(modelo_final, nombre_fijo)
print(f"✅ Modelo guardado en: {nombre_fijo}")

# Guardar también las métricas de validación cruzada
metricas = {
    'cv_scores': cv_scores,
    'cv_mean': cv_scores.mean(),
    'cv_std': cv_scores.std(),
    'test_accuracy': test_accuracy,
    'n_iter_': modelo_final.n_iter_,
    'loss_': modelo_final.loss_
}
joblib.dump(metricas, 'modelos_guardados/metricas_entrenamiento.pkl')
print("✅ Métricas de entrenamiento guardadas")

# Guardar datos de prueba normalizados (opcional)
np.save('modelos_guardados/X_test_flat.npy', X_test_flat)
np.save('modelos_guardados/y_test.npy', y_test)
print("✅ Datos de prueba guardados")

print("\n📊 Información del modelo guardado:")
print(f"   • Arquitectura: 784 → 128 → 64 → 10")
print(f"   • Precisión en test: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"   • Validación cruzada media: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"   • Iteraciones completadas: {modelo_final.n_iter_}")
print(f"   • Pérdida final: {modelo_final.loss_:.4f}")

# ============================================
# 12. DEMOSTRACIÓN DE CÓMO CARGAR EL MODELO
# ============================================
print("\n" + "="*70)
print("📂 DEMOSTRACIÓN: CARGANDO EL MODELO GUARDADO")
print("="*70)

# Cargar el modelo guardado
modelo_cargado = joblib.load(nombre_fijo)
print("✅ Modelo cargado correctamente")

# Verificar que funciona igual
predicciones_cargadas = modelo_cargado.predict(X_test_flat[:10])
print(f"🔢 Predicciones del modelo cargado: {predicciones_cargadas}")
print(f"🔢 Valores reales: {y_test[:10]}")

# Verificar que la precisión es la misma
precision_cargada = modelo_cargado.score(X_test_flat, y_test)
print(f"✅ Precisión del modelo cargado: {precision_cargada:.4f}")

# ============================================
# 13. RESUMEN FINAL
# ============================================
print("\n" + "="*70)
print("📋 RESUMEN FINAL")
print("="*70)
print(f"""
✅ EJERCICIO 2 COMPLETADO CON GUARDADO DE MODELO

📊 Resultados obtenidos:
   • Validación cruzada (6 folds): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}
   • Precisión en test: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)
   
💾 Modelos guardados:
   • {nombre_modelo}
   • {nombre_fijo}
   • modelos_guardados/metricas_entrenamiento.pkl
   • modelos_guardados/X_test_flat.npy
   • modelos_guardados/y_test.npy

🎯 Conclusión:
   El modelo clasifica correctamente el {test_accuracy*100:.2f}% de los dígitos manuscritos.
   El modelo ha sido guardado y puede ser reutilizado sin necesidad de reentrenar.
""")

print("✨ ¡Ejercicio 2 completado exitosamente con modelo guardado!")