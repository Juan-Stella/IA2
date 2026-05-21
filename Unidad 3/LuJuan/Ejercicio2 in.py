import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import seaborn as sns
import joblib
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw
import os

print("="*70)
print("RECONOCEDOR DE DÍGITOS - Usando modelo pre-entrenado")
print("="*70)

# ============================================
# 1. CARGAR EL MODELO GUARDADO
# ============================================
print("\n📂 Cargando modelo previamente entrenado...")

nombre_modelo = 'modelos_guardados/mnist_model_cv_ultimo.pkl'

if not os.path.exists(nombre_modelo):
    print(f"❌ ERROR: No se encuentra el modelo en {nombre_modelo}")
    print("   Primero ejecuta E2convalidacioncruzada.py para entrenar y guardar el modelo")
    exit(1)

# Cargar el modelo
modelo = joblib.load(nombre_modelo)
print("✅ Modelo cargado correctamente")
print(f"   • Precisión del modelo: {modelo.score.__self__.__class__}")

# ============================================
# 2. CLASE DEL RECONOCEDOR CON DIBUJO
# ============================================

class DigitDrawRecognizer:
    def __init__(self, model):
        """
        Inicializa el reconocedor de dígitos con el modelo cargado
        """
        self.model = model
        
        # Configurar ventana de dibujo
        self.window = tk.Tk()
        self.window.title("✏️ Reconocedor de Dígitos - Dibuja un número")
        self.window.geometry("650x750")
        self.window.configure(bg='#f0f0f0')
        
        # Título
        title = tk.Label(self.window, text="RECONOCEDOR DE DÍGITOS MANUSCRITOS", 
                        font=('Arial', 16, 'bold'), bg='#f0f0f0', fg='#333')
        title.pack(pady=10)
        
        # Canvas para dibujar
        self.canvas_size = 280
        self.canvas = tk.Canvas(self.window, width=self.canvas_size, 
                                height=self.canvas_size, bg='white', 
                                bd=2, relief='solid')
        self.canvas.pack(pady=20)
        
        # Variables para el dibujo
        self.last_x = None
        self.last_y = None
        
        # Instrucciones
        instructions = tk.Label(self.window, 
                                text="✏️ Mantén presionado el mouse y dibuja un número",
                                font=('Arial', 10), bg='#f0f0f0', fg='gray')
        instructions.pack()
        
        # Frame para botones
        button_frame = tk.Frame(self.window, bg='#f0f0f0')
        button_frame.pack(pady=15)
        
        # Botón Predecir
        self.predict_btn = tk.Button(button_frame, text="🔍 PREDECIR", 
                                     command=self.predict_digit, 
                                     bg='#4CAF50', fg='white', 
                                     font=('Arial', 12, 'bold'), 
                                     padx=25, pady=10,
                                     relief='raised', bd=2)
        self.predict_btn.pack(side=tk.LEFT, padx=10)
        
        # Botón Limpiar
        self.clear_btn = tk.Button(button_frame, text="🗑️ LIMPIAR", 
                                   command=self.clear_canvas,
                                   bg='#f44336', fg='white', 
                                   font=('Arial', 12, 'bold'), 
                                   padx=25, pady=10,
                                   relief='raised', bd=2)
        self.clear_btn.pack(side=tk.LEFT, padx=10)
        
        # Botón Salir
        self.exit_btn = tk.Button(button_frame, text="❌ SALIR", 
                                  command=self.window.quit,
                                  bg='#555', fg='white', 
                                  font=('Arial', 12, 'bold'), 
                                  padx=25, pady=10,
                                  relief='raised', bd=2)
        self.exit_btn.pack(side=tk.LEFT, padx=10)
        
        # Label para mostrar el resultado
        self.result_label = tk.Label(self.window, text="✏️ Dibuja un número y presiona 'PREDECIR'", 
                                     font=('Arial', 14, 'bold'), bg='#f0f0f0', fg='#333', pady=15)
        self.result_label.pack()
        
        # Label para mostrar la confianza
        self.confidence_label = tk.Label(self.window, text="", 
                                        font=('Arial', 11), bg='#f0f0f0', fg='#666')
        self.confidence_label.pack()
        
        # Bindear eventos del mouse
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)
        
    def paint(self, event):
        """
        Dibuja en el canvas
        """
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                   width=25, fill='black', capstyle=tk.ROUND, 
                                   smooth=True)
        self.last_x = event.x
        self.last_y = event.y
    
    def reset(self, event):
        """
        Reinicia las coordenadas
        """
        self.last_x = None
        self.last_y = None
    
    def clear_canvas(self):
        """
        Limpia el canvas
        """
        self.canvas.delete("all")
        self.result_label.config(text="✏️ Dibuja un número y presiona 'PREDECIR'", fg='#333')
        self.confidence_label.config(text="")
    
    def preprocess_image(self):
        """
        Convierte el dibujo del canvas a formato MNIST (28x28)
        """
        # Guardar canvas como imagen
        self.canvas.postscript(file="temp_drawing.eps", colormode='color')
        
        # Abrir con PIL
        img = Image.open("temp_drawing.eps")
        img = img.convert('L')  # Escala de grises
        
        # Redimensionar a 28x28
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Invertir colores (MNIST fondo negro, dígito blanco)
        img = Image.eval(img, lambda x: 255 - x)
        
        # Convertir a numpy array y normalizar
        img_array = np.array(img) / 255.0
        
        # Aplanar para el modelo
        img_flat = img_array.reshape(1, -1)
        
        # Limpiar archivo temporal
        os.remove("temp_drawing.eps")
        
        return img_flat, img_array
    
    def predict_digit(self):
        """
        Predice el dígito dibujado usando el modelo cargado
        """
        try:
            # Preprocesar imagen
            img_flat, img_array = self.preprocess_image()
            
            # Predecir con el modelo cargado
            prediction = self.model.predict(img_flat)[0]
            probabilities = self.model.predict_proba(img_flat)[0]
            confidence = probabilities[prediction] * 100
            
            # Determinar color según la confianza
            if confidence > 80:
                color = '#4CAF50'  # Verde
                emoji = "🎉"
            elif confidence > 60:
                color = '#FF9800'  # Naranja
                emoji = "👍"
            else:
                color = '#f44336'  # Rojo
                emoji = "🤔"
            
            # Mostrar resultado
            self.result_label.config(
                text=f"{emoji} DÍGITO PREDICHO: {prediction} {emoji}",
                fg=color,
                font=('Arial', 18, 'bold')
            )
            self.confidence_label.config(
                text=f"Confianza: {confidence:.1f}%",
                fg=color,
                font=('Arial', 12)
            )
            
            # Mostrar la imagen preprocesada en una ventana aparte
            self.show_processed_image(img_array, prediction, confidence)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al procesar la imagen: {e}")
    
    def show_processed_image(self, img_array, prediction, confidence):
        """
        Muestra cómo ve el modelo el dibujo
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Mostrar la imagen preprocesada
        ax1.imshow(img_array, cmap='gray')
        ax1.set_title(f"Dígito detectado: {prediction}\nConfianza: {confidence:.1f}%", 
                     fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # Mostrar barras de probabilidad
        probabilities = self.model.predict_proba(img_array.reshape(1, -1))[0]
        digits = range(10)
        bars = ax2.bar(digits, probabilities, color='skyblue')
        ax2.set_xlabel('Dígito', fontsize=11)
        ax2.set_ylabel('Probabilidad', fontsize=11)
        ax2.set_title('Probabilidades por clase', fontsize=12, fontweight='bold')
        ax2.set_xticks(digits)
        ax2.set_ylim(0, 1)
        
        # Resaltar el dígito predicho
        bars[prediction].set_color('green')
        
        # Añadir valores sobre las barras
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            if prob > 0.05:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{prob:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.suptitle("¿Cómo ve el modelo tu dibujo?", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def run(self):
        """
        Ejecuta la aplicación
        """
        print("\n🎨 Aplicación iniciada. Dibuja un dígito en el canvas.")
        print("💡 Consejo: Dibuja números grandes y gruesos para mejor reconocimiento")
        print("   • Usa el mouse para dibujar")
        print("   • Presiona 'PREDECIR' para identificar el número")
        print("   • Presiona 'LIMPIAR' para borrar y dibujar otro")
        print("   • Presiona 'SALIR' para cerrar la aplicación\n")
        self.window.mainloop()

# ============================================
# 3. EJECUTAR LA APLICACIÓN
# ============================================
if __name__ == "__main__":
    print("\n🚀 Iniciando reconocedor de dígitos...")
    print("   (Usando modelo pre-entrenado - sin reentrenamiento)")
    
    # Crear y ejecutar la aplicación
    app = DigitDrawRecognizer(modelo)
    app.run()
    
    print("\n👋 Aplicación cerrada. ¡Hasta luego!")