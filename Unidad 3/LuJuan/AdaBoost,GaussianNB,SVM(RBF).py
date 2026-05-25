import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB  # ← Cambio: BernoulliNB → GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression  # ← NUEVO: Regresión Logística
import pandas as pd
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("📊 COMPARACIÓN DE MODELOS: AdaBoost vs GaussianNB vs SVM")
print("="*80)

# ============================================
# 1. CARGAR DATOS
# ============================================
print("\n📥 Cargando dataset MNIST...")
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalizar
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Aplanar
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Reducir tamaño para que SVM sea más rápido
muestra_size = 20000
indices = np.random.choice(len(X_train_flat), muestra_size, replace=False)
X_train_sample = X_train_flat[indices]
y_train_sample = y_train[indices]

print(f"   ✅ Entrenamiento: {X_train_sample.shape[0]:,} imágenes")
print(f"   ✅ Prueba: {X_test_flat.shape[0]:,} imágenes")
print(f"   ✅ Todos los modelos usan datos continuos (sin binarización)")

# ============================================
# 2. DEFINIR LOS CLASIFICADORES
# ============================================
print("\n" + "="*80)
print("🤖 CONFIGURACIÓN DE MODELOS")
print("="*80)

classifiers = {
    "AdaBoost (Decision Tree)": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=100,
        learning_rate=1.0,
        algorithm='SAMME',
        random_state=42
    ),
    "Gaussian Naive Bayes": GaussianNB(
        var_smoothing=1e-9  # Parámetro de suavizado para estabilidad
    ),
    "SVM (RBF Kernel)": SVC(
        kernel='rbf',
        C=5.0,
        gamma='scale',
        random_state=42,
        verbose=False,
        cache_size=500  # Aumentar caché para más velocidad
    ),
    "Logistic Regression": LogisticRegression(  # ← Modelo adicional
        max_iter=100,
        C=1.0,
        solver='lbfgs',
        multi_class='multinomial',
        random_state=42,
        n_jobs=-1
    )
}

# Mostrar configuración
for nombre, modelo in classifiers.items():
    print(f"\n📌 {nombre}")
    if "Gaussian" in nombre:
        print(f"   • Tipo: Naive Bayes con distribución Gaussiana")
        print(f"   • Ventaja: Trabaja directamente con datos continuos")
        print(f"   • Parámetro: var_smoothing={modelo.var_smoothing}")
    elif "AdaBoost" in nombre:
        print(f"   • Tipo: Ensemble boosting con árboles débiles")
        print(f"   • N° estimadores: {modelo.n_estimators}")
    elif "SVM" in nombre:
        print(f"   • Tipo: Máquina de vectores soporte")
        print(f"   • Kernel: RBF, C={modelo.C}")
    elif "Logistic" in nombre:
        print(f"   • Tipo: Regresión logística multinomial")
        print(f"   • Solver: {modelo.solver}")

# ============================================
# 3. ENTRENAR Y EVALUAR CADA MODELO
# ============================================
print("\n" + "="*80)
print("🔄 ENTRENANDO Y EVALUANDO MODELOS")
print("="*80)

resultados = []
modelos_entrenados = {}

for nombre, modelo in classifiers.items():
    print(f"\n📌 Entrenando {nombre}...")
    
    inicio = time.time()
    
    # Todos usan los mismos datos (continuos)
    modelo.fit(X_train_sample, y_train_sample)
    
    # Predecir
    y_pred = modelo.predict(X_test_flat)
    
    # Calcular métricas
    precision = accuracy_score(y_test, y_pred)
    precision_por_clase = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall_por_clase = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1_por_clase = f1_score(y_test, y_pred, average=None, zero_division=0)
    tiempo = time.time() - inicio
    
    # Guardar resultados
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
    
    print(f"   ✅ Precisión global: {precision:.4f} ({precision*100:.2f}%)")
    print(f"   ⏱️  Tiempo: {tiempo:.2f} segundos")
    print(f"   📊 Mejor dígito: {np.argmax(precision_por_clase)} ({precision_por_clase.max():.3f})")
    print(f"   📊 Peor dígito: {np.argmin(precision_por_clase)} ({precision_por_clase.min():.3f})")

# ============================================
# 4. TABLA COMPARATIVA
# ============================================
df_resultados = pd.DataFrame(resultados)
df_resultados = df_resultados.sort_values('Precisión Global', ascending=False)

print("\n" + "="*80)
print("📊 TABLA COMPARATIVA DE RENDIMIENTO")
print("="*80)
print(df_resultados[['Modelo', 'Precisión Global', 'Tiempo (segundos)', 'Mejor Dígito', 'Peor Dígito']].to_string(index=False))

# ============================================
# 5. GRÁFICOS COMPARATIVOS
# ============================================
print("\n📊 Generando gráficos comparativos...")

fig = plt.figure(figsize=(16, 12))

# Gráfico 1: Precisión Global
ax1 = plt.subplot(2, 3, 1)
colores = ['gold', 'silver', '#CD7F32', 'skyblue'][:len(df_resultados)]
bars = ax1.barh(df_resultados['Modelo'], df_resultados['Precisión Global'], color=colores, edgecolor='black')
ax1.set_xlabel('Precisión', fontsize=12)
ax1.set_title('🎯 Precisión Global por Modelo', fontsize=12, fontweight='bold')
ax1.set_xlim(0.7, 1.0)
ax1.grid(True, alpha=0.3, axis='x')
for bar, val in zip(bars, df_resultados['Precisión Global']):
    ax1.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
             f'{val:.4f}', va='center', fontsize=10)

# Gráfico 2: Tiempo de Entrenamiento
ax2 = plt.subplot(2, 3, 2)
bars2 = ax2.barh(df_resultados['Modelo'], df_resultados['Tiempo (segundos)'], 
                 color=colores, edgecolor='black')
ax2.set_xlabel('Tiempo (segundos)', fontsize=12)
ax2.set_title('⏱️ Tiempo de Entrenamiento', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')
for bar, val in zip(bars2, df_resultados['Tiempo (segundos)']):
    ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}s', va='center', fontsize=10)

# Gráfico 3: Precisión por dígito
ax3 = plt.subplot(2, 3, 3)
for _, row in df_resultados.iterrows():
    precisiones = row['Precisiones_por_clase']
    ax3.plot(range(10), precisiones, 'o-', linewidth=2, markersize=6, 
             label=row['Modelo'].replace(' (RBF Kernel)', '').replace(' (Decision Tree)', ''))

ax3.set_xlabel('Dígito', fontsize=12)
ax3.set_ylabel('Precisión', fontsize=12)
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
ax4.set_xlabel('Dígito', fontsize=10)
ax4.set_ylabel('Modelo', fontsize=10)
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
ax5.set_ylabel('Puntuación', fontsize=10)
ax5.set_title('📊 Métricas Promedio por Modelo', fontsize=12, fontweight='bold')
ax5.legend()
ax5.set_ylim(0.7, 1.0)
ax5.grid(True, alpha=0.3, axis='y')

# Gráfico 6: Relación Precisión vs Tiempo
ax6 = plt.subplot(2, 3, 6)
for _, row in df_resultados.iterrows():
    ax6.scatter(row['Tiempo (segundos)'], row['Precisión Global'], 
               s=200, c='skyblue', edgecolors='black', linewidth=2)
    ax6.annotate(row['Modelo'].replace(' (RBF Kernel)', '').replace(' (Decision Tree)', ''),
                (row['Tiempo (segundos)'], row['Precisión Global']),
                xytext=(5, 5), textcoords='offset points', fontsize=8)

ax6.set_xlabel('Tiempo de entrenamiento (segundos)', fontsize=11)
ax6.set_ylabel('Precisión Global', fontsize=11)
ax6.set_title('⚖️ Trade-off: Precisión vs Tiempo', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.set_ylim(0.8, 1.0)

plt.tight_layout()
plt.show()

# ============================================
# 6. MATRICES DE CONFUSIÓN
# ============================================
print("\n📊 Generando matrices de confusión...")

for nombre, modelo in modelos_entrenados.items():
    print(f"\n   • Matriz de confusión para {nombre}...")
    
    y_pred = modelo.predict(X_test_flat)
    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index=range(10), columns=range(10))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusión - {nombre}', fontsize=14, fontweight='bold')
    plt.xlabel('Predicción', fontsize=12)
    plt.ylabel('Valor Real', fontsize=12)
    plt.tight_layout()
    plt.show()

# ============================================
# 7. VALIDACIÓN CRUZADA
# ============================================
mejor_modelo_nombre = df_resultados.iloc[0]['Modelo']
mejor_modelo = modelos_entrenados[mejor_modelo_nombre]

print(f"\n📊 Realizando validación cruzada para el mejor modelo: {mejor_modelo_nombre}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(mejor_modelo, X_train_sample, y_train_sample, cv=skf, n_jobs=-1)

print(f"   • Scores por fold: {cv_scores}")
print(f"   • Media: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ============================================
# 8. CONCLUSIONES
# ============================================
print("\n" + "="*80)
print("📋 CONCLUSIONES DEL ESTUDIO COMPARATIVO")
print("="*80)

for i, row in df_resultados.iterrows():
    medalla = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📌"
    print(f"\n{medalla} **{row['Modelo']}**")
    print(f"   • Precisión: {row['Precisión Global']*100:.2f}%")
    print(f"   • Tiempo: {row['Tiempo (segundos)']:.2f} segundos")
    print(f"   • Mejor dígito: {row['Mejor Dígito']} ({row['Precisiones_por_clase'][row['Mejor Dígito']]*100:.1f}%)")
    print(f"   • Peor dígito: {row['Peor Dígito']} ({row['Precisiones_por_clase'][row['Peor Dígito']]*100:.1f}%)")

print(f"""
🔍 **ANÁLISIS COMPARATIVO DE MODELOS (DATOS CONTINUOS):**

| Característica | AdaBoost | GaussianNB | SVM | Logistic Reg. |
|----------------|----------|-------------|-----|---------------|
| Tipo | Boosting | Bayesiano | Kernel | Lineal |
| Datos | Continuos | Continuos | Continuos | Continuos |
| Interpretabilidad | Media | Alta | Baja | Muy Alta |
| Escalabilidad | Buena | Excelente | Pobre (O(n²)) | Buena |
| Resistencia ruido | Buena | Regular | Buena | Buena |
| Sobreajuste | Controlable | Bajo | Controlable | Controlable |

💡 **RECOMENDACIONES FINALES:**

1. **🥇 {df_resultados.iloc[0]['Modelo']}** → Mejor precisión
   {f"   • {df_resultados.iloc[0]['Precisión Global']*100:.2f}% de precisión"}

2. **🥈 {df_resultados.iloc[1]['Modelo'] if len(df_resultados) > 1 else 'N/A'}** → Mejor balance

3. **⚡ Para producción con muchos datos** → GaussianNB o Regresión Logística
   • Extremadamente rápidos
   • Escalan bien
   • Buenos para inferencia en tiempo real

4. **🎯 Para máxima precisión (offline)** → SVM
   • La mejor precisión pero lento
   • Ideal cuando el tiempo no es problema

5. **🤖 Para modelos interpretables** → Regresión Logística
   • Puedes ver pesos por característica
   • Fácil de explicar resultados
""")

print("\n✨ Estudio comparativo completado con modelos que NO requieren binarización!")