import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import seaborn as sns

# 1. Cargar dataset MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 2. Normalizar imágenes (valores entre 0 y 1)
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# 3. Aplanar imágenes (28x28 → 784)
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# 4. Crear modelo de red neuronal multicapa con scikit-learn
model = MLPClassifier(
    early_stopping=True,
    validation_fraction=0.1,  # fraction of training data to use for validation
    # Configuración del modelo
    hidden_layer_sizes=(128, 64),  # dos capas ocultas
    activation='relu',
    solver='adam',
    max_iter=10,                   # pocas iteraciones para ejemplo rápido
    random_state=42,
    verbose=True
)

# 5. Entrenar modelo
print("Entrenando modelo...")
model.fit(X_train_flat, y_train)

# 6. Evaluar en test
y_pred = model.predict(X_test_flat)

# 7. Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, index=range(10), columns=range(10))

plt.figure(figsize=(10, 7))
sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
plt.title("Matriz de Confusión - MNIST")
plt.xlabel("Predicción")
plt.ylabel("Valor Real")
plt.show()

# 8. Reporte de clasificación
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))