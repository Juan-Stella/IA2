import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("📊 COMPARACIÓN DE MODELOS OPTIMIZADA: Con Matrices de Confusión")
print("="*80)

# ============================================
# 1. CARGAR Y PREPROCESAR DATOS COMPLETOS
# ============================================
print("\n📥 Cargando dataset MNIST completo...")
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalizar
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Aplanar para el conjunto TOTAL (60,000 imágenes)
X_train_total = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Crear subconjunto exclusivo para SVM (20,000 imágenes)
muestra_size = 20000
np.random.seed(42)
indices = np.random.choice(len(X_train_total), muestra_size, replace=False)
X_train_svm = X_train_total[indices]
y_train_svm = y_train[indices]

print(f"   ✅ Datos Totales (Resto de modelos): {X_train_total.shape[0]:,} imágenes")
print(f"   ✅ Muestra Reducida (Solo para SVM): {X_train_svm.shape[0]:,} imágenes")
print(f"   ✅ Datos de Prueba (Evaluación): {X_test_flat.shape[0]:,} imágenes")

# ============================================
# 2. DEFINIR LOS CLASIFICADORES
# ============================================
classifiers = {
    "AdaBoost (Decision Tree)": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=100,
        learning_rate=1.0,
        algorithm='SAMME',
        random_state=42
    ),
    "Gaussian Naive Bayes": GaussianNB(
        var_smoothing=1e-9
    ),
    "SVM (RBF Kernel)": SVC(
        kernel='rbf',
        C=5.0,
        gamma='scale',
        random_state=42,
        verbose=False,
        cache_size=500
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=100,
        C=1.0,
        solver='lbfgs',
        multi_class='multinomial',
        random_state=42,
        n_jobs=-1
    )
}

# ============================================
# 3. ENTRENAR Y EVALUAR
# ============================================
print("\n" + "="*80)
print("🔄 ENTRENANDO Y EVALUANDO MODELOS")
print("="*80)

resultados = []
modelos_entrenados = {}
predicciones = {} # Guardaremos las predicciones para las matrices de confusión

for nombre, modelo in classifiers.items():
    print(f"\n📌 Entrenando {nombre}...")
    inicio = time.time()
    
    if "SVM" in nombre:
        print("   ⚠️ Usando muestra reducida de 20,000 imágenes para SVM.")
        modelo.fit(X_train_svm, y_train_svm)
    else:
        print("   🚀 Usando el set COMPLETO de 60,000 imágenes.")
        modelo.fit(X_train_total, y_train)
    
    y_pred = modelo.predict(X_test_flat)
    predicciones[nombre] = y_pred # Almacenar para después
    
    precision = accuracy_score(y_test, y_pred)
    precision_por_clase = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall_por_clase = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1_por_clase = f1_score(y_test, y_pred, average=None, zero_division=0)
    tiempo = time.time() - inicio
    
    resultados.append({
        'Modelo': nombre,
        'Precisión Global': precision,
        'Tiempo (segundos)': tiempo,
        'Mejor Dígito': np.argmax(precision_por_clase),
        'Peor Dígito': np.argmin(precision_por_clase),
        'Precisiones_por_clase': precision_por_clase,
        'Recalls_por_clase': recall_por_clase,
        'F1_por_clase': f1_por_clase
    })
    modelos_entrenados[nombre] = modelo
    print(f"   ✅ Precisión global: {precision*100:.2f}% | ⏱️ Tiempo: {tiempo:.2f}s")

# ============================================
# 4. TABLA COMPARATIVA
# ============================================
df_resultados = pd.DataFrame(resultados).sort_values('Precisión Global', ascending=False)
print("\n" + "="*80)
print("📊 TABLA COMPARATIVA DE RENDIMIENTO")
print("="*80)
print(df_resultados[['Modelo', 'Precisión Global', 'Tiempo (segundos)', 'Mejor Dígito', 'Peor Dígito']].to_string(index=False))

# ============================================
# 5. PANEL DE MATRICES DE CONFUSIÓN (2x2)
# ============================================
print("\n📊 Generando panel de matrices de confusión...")
fig_cm, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel() # Aplanar matriz de subplots a una lista de 4 elementos

for idx, (nombre, y_pred) in enumerate(predicciones.items()):
    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index=range(10), columns=range(10))
    
    # Dibujar heatmap en su respectivo cuadrante
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False)
    axes[idx].set_title(f'Matriz de Confusión: {nombre}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Predicción')
    axes[idx].set_ylabel('Valor Real')

plt.tight_layout()
plt.suptitle('🔍 Análisis de Errores por Modelo (Dígito por Dígito)', fontsize=16, fontweight='bold', y=1.02)
plt.show()

# ============================================
# 6. GRÁFICOS COMPARATIVOS GENERALES
# ============================================
print("\n📊 Generando gráficos comparativos generales...")
fig = plt.figure(figsize=(16, 12))
colores = ['gold', 'silver', '#CD7F32', 'skyblue'][:len(df_resultados)]

# Gráfico 1: Precisión Global
ax1 = plt.subplot(2, 3, 1)
bars = ax1.barh(df_resultados['Modelo'], df_resultados['Precisión Global'], color=colores, edgecolor='black')
ax1.set_xlabel('Precisión')
ax1.set_title('🎯 Precisión Global por Modelo', fontsize=12, fontweight='bold')
ax1.set_xlim(0.7, 1.0)
ax1.grid(True, alpha=0.3, axis='x')
for bar, val in zip(bars, df_resultados['Precisión Global']):
    ax1.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center', fontsize=10)

# Gráfico 2: Tiempo de Entrenamiento
ax2 = plt.subplot(2, 3, 2)
bars2 = ax2.barh(df_resultados['Modelo'], df_resultados['Tiempo (segundos)'], color=colores, edgecolor='black')
ax2.set_xlabel('Tiempo (segundos)')
ax2.set_title('⏱️ Tiempo de Entrenamiento', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')
for bar, val in zip(bars2, df_resultados['Tiempo (segundos)']):
    ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}s', va='center', fontsize=10)

# Gráfico 3: Precisión por dígito
ax3 = plt.subplot(2, 3, 3)
for _, row in df_resultados.iterrows():
    ax3.plot(range(10), row['Precisiones_por_clase'], 'o-', linewidth=2, markersize=6, 
             label=row['Modelo'].replace(' (RBF Kernel)', '').replace(' (Decision Tree)', ''))
ax3.set_xlabel('Dígito')
ax3.set_ylabel('Precisión')
ax3.set_title('📈 Precisión por Dígito', fontsize=12, fontweight='bold')
ax3.set_xticks(range(10))
ax3.legend(loc='lower right', fontsize=8)
ax3.grid(True, alpha=0.3)
ax3.set_ylim(0.7, 1.0)

# Gráfico 4: Heatmap de precisión por dígito
ax4 = plt.subplot(2, 3, 4)
matriz_precisiones = np.array([row['Precisiones_por_clase'] for _, row in df_resultados.iterrows()])
modelos_nombres = [row['Modelo'].replace(' (RBF Kernel)', '').replace(' (Decision Tree)', '') for _, row in df_resultados.iterrows()]
im = ax4.imshow(matriz_precisiones, cmap='RdYlGn', aspect='auto', vmin=0.7, vmax=1.0)
ax4.set_xticks(range(10))
ax4.set_yticks(range(len(modelos_nombres)))
ax4.set_xticklabels(range(10))
ax4.set_yticklabels(modelos_nombres, fontsize=9)
ax4.set_xlabel('Dígito')
ax4.set_title('🎨 Precisión por Dígito (mapa de calor)', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax4, label='Precisión')

# Gráfico 5: Métricas promedio
ax5 = plt.subplot(2, 3, 5)
x = np.arange(len(df_resultados))
width = 0.25
precis_means = [row['Precisiones_por_clase'].mean() for _, row in df_resultados.iterrows()]
recall_means = [row['Recalls_por_clase'].mean() for _, row in df_resultados.iterrows()]
f1_means = [row['F1_por_clase'].mean() for _, row in df_resultados.iterrows()]

ax5.bar(x - width, precis_means, width, label='Precisión', color='skyblue', edgecolor='black')
ax5.bar(x, recall_means, width, label='Recall', color='lightgreen', edgecolor='black')
ax5.bar(x + width, f1_means, width, label='F1-Score', color='orange', edgecolor='black')
ax5.set_xticks(x)
ax5.set_xticklabels(modelos_nombres, rotation=15, ha='right')
ax5.set_title('📊 Métricas Promedio', fontsize=12, fontweight='bold')
ax5.legend()
ax5.set_ylim(0.7, 1.0)
ax5.grid(True, alpha=0.3, axis='y')

# Gráfico 6: Relación Precisión vs Tiempo
ax6 = plt.subplot(2, 3, 6)
for _, row in df_resultados.iterrows():
    ax6.scatter(row['Tiempo (segundos)'], row['Precisión Global'], s=200, c='skyblue', edgecolors='black', linewidth=2)
    ax6.annotate(row['Modelo'].replace(' (RBF Kernel)', '').replace(' (Decision Tree)', ''),
                 (row['Tiempo (segundos)'], row['Precisión Global']), xytext=(5, 5), textcoords='offset points', fontsize=8)
ax6.set_xlabel('Tiempo (segundos)')
ax6.set_ylabel('Precisión Global')
ax6.set_title('⚖️ Trade-off: Precisión vs Tiempo', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.set_ylim(0.8, 1.0)

plt.tight_layout()
plt.show()

# ============================================
# 7. VALIDACIÓN CRUZADA (Para el mejor modelo)
# ============================================
mejor_modelo_nombre = df_resultados.iloc[0]['Modelo']
mejor_modelo = modelos_entrenados[mejor_modelo_nombre]

print(f"\n📊 Realizando validación cruzada para el mejor modelo: {mejor_modelo_nombre}")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

if "SVM" in mejor_modelo_nombre:
    cv_scores = cross_val_score(mejor_modelo, X_train_svm, y_train_svm, cv=skf, n_jobs=-1)
else:
    cv_scores = cross_val_score(mejor_modelo, X_train_total, y_train, cv=skf, n_jobs=-1)

print(f"   • Scores por fold: {cv_scores}")
print(f"   • Media: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print("\n✨ ¡Estudio completo finalizado con éxito!")