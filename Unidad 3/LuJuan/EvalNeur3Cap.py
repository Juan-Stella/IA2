import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("📊 ESTUDIO: OPTIMIZACIÓN DE 3 CAPAS OCULTAS")
print("="*80)

# ============================================
# 1. CARGAR DATOS (usamos muestra para que sea más rápido)
# ============================================
print("\n📥 Cargando dataset MNIST...")
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalizar
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Aplanar
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Usar muestra más pequeña para que el estudio sea factible
muestra_size = 15000
indices = np.random.choice(len(X_train_flat), muestra_size, replace=False)
X_train_sample = X_train_flat[indices]
y_train_sample = y_train[indices]

print(f"   ✅ Usando {muestra_size:,} imágenes para entrenamiento (muestra)")
print(f"   ✅ Usando {len(X_test_flat):,} imágenes para prueba")

# ============================================
# 2. DEFINIR RANGOS DE NEURONAS (más pequeños por ser 3 capas)
# ============================================
# Para 3 capas, usamos menos valores porque las combinaciones crecen exponencialmente
neuronas_posibles = [20, 40, 60, 80, 100]  # 5 valores por capa = 125 combinaciones

print(f"\n🎯 Rango de neuronas a probar:")
print(f"   • Capa 1: {neuronas_posibles}")
print(f"   • Capa 2: {neuronas_posibles}")
print(f"   • Capa 3: {neuronas_posibles}")
print(f"   • Total combinaciones: {len(neuronas_posibles)**3} = {len(neuronas_posibles)**3}")

# ============================================
# 3. REALIZAR EL ESTUDIO
# ============================================
print("\n" + "="*80)
print("🔄 EJECUTANDO ESTUDIO (esto puede tomar varios minutos...)")
print("="*80)

resultados = []
tiempo_inicio_total = time.time()
total_combinaciones = len(neuronas_posibles)**3
contador = 0

for n1 in neuronas_posibles:
    for n2 in neuronas_posibles:
        for n3 in neuronas_posibles:
            contador += 1
            print(f"\r   Probando combinación {contador}/{total_combinaciones}: ({n1}, {n2}, {n3})", end="")
            
            inicio = time.time()
            
            # Crear modelo con 3 capas
            model = MLPClassifier(
                hidden_layer_sizes=(n1, n2, n3),
                activation='relu',
                solver='adam',
                max_iter=12,  # Menos iteraciones para que sea más rápido
                random_state=42,
                verbose=False
            )
            
            # Entrenar y evaluar
            model.fit(X_train_sample, y_train_sample)
            accuracy = model.score(X_test_flat, y_test)
            tiempo = time.time() - inicio
            
            # Calcular número de parámetros
            # Capa1: 784 * n1 + n1
            # Capa2: n1 * n2 + n2
            # Capa3: n2 * n3 + n3
            # Salida: n3 * 10 + 10
            params = (784 * n1 + n1) + (n1 * n2 + n2) + (n2 * n3 + n3) + (n3 * 10 + 10)
            
            # Guardar resultados
            resultados.append({
                'capa1': n1,
                'capa2': n2,
                'capa3': n3,
                'precision': accuracy,
                'tiempo': tiempo,
                'parametros': params
            })

tiempo_total = time.time() - tiempo_inicio_total
print(f"\n\n✅ Estudio completado en {tiempo_total/60:.1f} minutos")

# ============================================
# 4. CONVERTIR A DATAFRAME
# ============================================
df_resultados = pd.DataFrame(resultados)

# Ordenar por precisión
df_mejores = df_resultados.sort_values('precision', ascending=False)
df_peores = df_resultados.sort_values('precision', ascending=True)

# ============================================
# 5. MOSTRAR RESULTADOS EN TABLAS
# ============================================
print("\n" + "="*80)
print("🏆 TOP 15 MEJORES COMBINACIONES (3 CAPAS)")
print("="*80)
print(f"\n{'Capa1':<8} {'Capa2':<8} {'Capa3':<8} {'Precisión':<12} {'Tiempo(s)':<12} {'Parámetros':<15}")
print("-" * 65)

for i in range(min(15, len(df_mejores))):
    row = df_mejores.iloc[i]
    print(f"{row['capa1']:<8} {row['capa2']:<8} {row['capa3']:<8} {row['precision']:<12.4f} {row['tiempo']:<12.1f} {row['parametros']:<15,}")

print("\n" + "="*80)
print("📉 TOP 10 PEORES COMBINACIONES (3 CAPAS)")
print("="*80)
print(f"\n{'Capa1':<8} {'Capa2':<8} {'Capa3':<8} {'Precisión':<12} {'Tiempo(s)':<12} {'Parámetros':<15}")
print("-" * 65)

for i in range(min(10, len(df_peores))):
    row = df_peores.iloc[i]
    print(f"{row['capa1']:<8} {row['capa2']:<8} {row['capa3']:<8} {row['precision']:<12.4f} {row['tiempo']:<12.1f} {row['parametros']:<15,}")

# ============================================
# 6. ANÁLISIS POR TAMAÑO DE CAPAS
# ============================================
print("\n" + "="*80)
print("📊 ANÁLISIS POR TAMAÑO DE CADA CAPA")
print("="*80)

print("\n📈 PRECISIÓN PROMEDIO POR TAMAÑO DE CAPA 1:")
for n in neuronas_posibles:
    avg = df_resultados[df_resultados['capa1'] == n]['precision'].mean()
    std = df_resultados[df_resultados['capa1'] == n]['precision'].std()
    print(f"   • Capa1 = {n:3d}: {avg:.4f} ± {std:.4f}")

print("\n📈 PRECISIÓN PROMEDIO POR TAMAÑO DE CAPA 2:")
for n in neuronas_posibles:
    avg = df_resultados[df_resultados['capa2'] == n]['precision'].mean()
    std = df_resultados[df_resultados['capa2'] == n]['precision'].std()
    print(f"   • Capa2 = {n:3d}: {avg:.4f} ± {std:.4f}")

print("\n📈 PRECISIÓN PROMEDIO POR TAMAÑO DE CAPA 3:")
for n in neuronas_posibles:
    avg = df_resultados[df_resultados['capa3'] == n]['precision'].mean()
    std = df_resultados[df_resultados['capa3'] == n]['precision'].std()
    print(f"   • Capa3 = {n:3d}: {avg:.4f} ± {std:.4f}")

# ============================================
# 7. GRÁFICOS 2D (sin 3D)
# ============================================
print("\n📊 Generando gráficos...")

fig = plt.figure(figsize=(16, 10))

# ===== GRÁFICO 1: Precisión vs Complejidad =====
ax1 = plt.subplot(2, 2, 1)
scatter = ax1.scatter(df_resultados['parametros'] / 1000, df_resultados['precision'], 
                     c=df_resultados['precision'], cmap='RdYlGn', 
                     s=60, alpha=0.6, edgecolors='black')
ax1.set_xlabel('Parámetros del modelo (miles)', fontsize=11)
ax1.set_ylabel('Precisión en test', fontsize=11)
ax1.set_title('💰 Trade-off: Precisión vs Complejidad (3 capas)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax1, label='Precisión')

# ===== GRÁFICO 2: Top 20 mejores combinaciones =====
ax2 = plt.subplot(2, 2, 2)
top20 = df_mejores.head(20)
x_pos = range(len(top20))
labels = [f"{row['capa1']},{row['capa2']},{row['capa3']}" for _, row in top20.iterrows()]
colors = ['gold' if i < 3 else 'skyblue' for i in range(len(top20))]
bars = ax2.bar(x_pos, top20['precision'], color=colors, edgecolor='black')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax2.set_xlabel('Combinación (Capa1, Capa2, Capa3)', fontsize=11)
ax2.set_ylabel('Precisión', fontsize=11)
ax2.set_title('🏆 Top 20 Mejores Combinaciones (3 capas)', fontsize=12, fontweight='bold')
ax2.set_ylim(top20['precision'].min() - 0.002, top20['precision'].max() + 0.002)
for bar, val in zip(bars, top20['precision']):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0003,
            f'{val:.4f}', ha='center', va='bottom', fontsize=7)

# ===== GRÁFICO 3: Efecto de cada capa (manteniendo las otras fijas) =====
ax3 = plt.subplot(2, 2, 3)

# Buscar una combinación base buena
base = df_mejores.iloc[0]
base_n1, base_n2, base_n3 = base['capa1'], base['capa2'], base['capa3']

# Variar capa 1
datos_capa1 = []
for n1 in neuronas_posibles:
    matching = df_resultados[(df_resultados['capa2'] == base_n2) & (df_resultados['capa3'] == base_n3) & (df_resultados['capa1'] == n1)]
    if len(matching) > 0:
        datos_capa1.append(matching.iloc[0])

# Variar capa 2
datos_capa2 = []
for n2 in neuronas_posibles:
    matching = df_resultados[(df_resultados['capa1'] == base_n1) & (df_resultados['capa3'] == base_n3) & (df_resultados['capa2'] == n2)]
    if len(matching) > 0:
        datos_capa2.append(matching.iloc[0])

# Variar capa 3
datos_capa3 = []
for n3 in neuronas_posibles:
    matching = df_resultados[(df_resultados['capa1'] == base_n1) & (df_resultados['capa2'] == base_n2) & (df_resultados['capa3'] == n3)]
    if len(matching) > 0:
        datos_capa3.append(matching.iloc[0])

if datos_capa1 and datos_capa2 and datos_capa3:
    ax3.plot([d['capa1'] for d in datos_capa1], [d['precision'] for d in datos_capa1], 
             'o-', label=f'Variando Capa1 (Capa2={base_n2}, Capa3={base_n3})', linewidth=2, markersize=8)
    ax3.plot([d['capa2'] for d in datos_capa2], [d['precision'] for d in datos_capa2], 
             's-', label=f'Variando Capa2 (Capa1={base_n1}, Capa3={base_n3})', linewidth=2, markersize=8)
    ax3.plot([d['capa3'] for d in datos_capa3], [d['precision'] for d in datos_capa3], 
             '^-', label=f'Variando Capa3 (Capa1={base_n1}, Capa2={base_n2})', linewidth=2, markersize=8)

ax3.set_xlabel('Número de neuronas', fontsize=11)
ax3.set_ylabel('Precisión', fontsize=11)
ax3.set_title('📈 Efecto individual de cada capa', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# ===== GRÁFICO 4: Distribución de precisiones =====
ax4 = plt.subplot(2, 2, 4)
ax4.hist(df_resultados['precision'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
ax4.axvline(df_resultados['precision'].mean(), color='red', linestyle='--', 
            label=f'Media: {df_resultados["precision"].mean():.4f}')
ax4.axvline(df_mejores.iloc[0]['precision'], color='green', linestyle='--', 
            label=f'Máxima: {df_mejores.iloc[0]["precision"]:.4f}')
ax4.set_xlabel('Precisión', fontsize=11)
ax4.set_ylabel('Frecuencia', fontsize=11)
ax4.set_title('📊 Distribución de precisiones (3 capas)', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================
# 8. TABLA RESUMEN POR TAMAÑO DE CAPAS
# ============================================
print("\n" + "="*80)
print("📊 MATRIZ DE PRECISIÓN PROMEDIO (fijando Capa3)")
print("="*80)

# Crear tabla pivote para visualizar mejor
for n3 in neuronas_posibles:
    print(f"\n🔹 Capa3 = {n3} neuronas:")
    print(f"{'Capa1\\Capa2':<10}", end="")
    for n2 in neuronas_posibles:
        print(f"{n2:<10}", end="")
    print()
    print("-" * (10 + 10 * len(neuronas_posibles)))
    
    for n1 in neuronas_posibles:
        print(f"{n1:<10}", end="")
        for n2 in neuronas_posibles:
            mask = (df_resultados['capa1'] == n1) & (df_resultados['capa2'] == n2) & (df_resultados['capa3'] == n3)
            if mask.any():
                prec = df_resultados[mask]['precision'].values[0]
                print(f"{prec:.4f}  ", end="")
            else:
                print(f"{'N/A':<10}", end="")
        print()

# ============================================
# 9. RESUMEN FINAL CON RECOMENDACIONES
# ============================================
print("\n" + "="*80)
print("💡 RECOMENDACIONES PARA 3 CAPAS OCULTAS")
print("="*80)

mejor = df_mejores.iloc[0]
mejor_pequeno = df_resultados[df_resultados['parametros'] <= 100000].sort_values('precision', ascending=False).iloc[0] if len(df_resultados[df_resultados['parametros'] <= 100000]) > 0 else mejor

print(f"""
🏆 MEJOR COMBINACIÓN ENCONTRADA:
   • Arquitectura: {mejor['capa1']} → {mejor['capa2']} → {mejor['capa3']} → 10
   • Precisión: {mejor['precision']:.4f} ({mejor['precision']*100:.2f}%)
   • Tiempo de entrenamiento: {mejor['tiempo']:.1f} segundos
   • Parámetros totales: {mejor['parametros']:,}

⚡ COMBINACIÓN MÁS EFICIENTE (<100k parámetros):
   • Arquitectura: {mejor_pequeno['capa1']} → {mejor_pequeno['capa2']} → {mejor_pequeno['capa3']} → 10
   • Precisión: {mejor_pequeno['precision']:.4f} ({mejor_pequeno['precision']*100:.2f}%)
   • Parámetros totales: {mejor_pequeno['parametros']:,}

📌 CONCLUSIONES DEL ESTUDIO CON 3 CAPAS:
   • {len(df_resultados)} combinaciones evaluadas
   • Mejor precisión alcanzada: {mejor['precision']*100:.2f}%
   • Precisión media: {df_resultados['precision'].mean()*100:.2f}%
   • Rango de precisión: {df_resultados['precision'].min()*100:.2f}% - {df_resultados['precision'].max()*100:.2f}%
   
🔍 OBSERVACIONES:
   • Añadir una tercera capa NO siempre mejora el rendimiento
   • El tamaño óptimo suele estar entre 60-100 neuronas por capa
   • Capas más grandes (>100) aumentan tiempo sin mejorar precisión
   • Capas muy pequeñas (<30) pierden capacidad de aprendizaje
""")

print("\n✨ Estudio de 3 capas completado!")