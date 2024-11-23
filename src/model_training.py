import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import tensorflow as tf
import joblib
import os


class UNIDesercionPredictor:
    def __init__(self,
                 ruta_datos=r"C:\Users\Andriy\Documents\GitHub\streamlit-prediction-system\data\datos_estudiantes_uni.csv"):
        self.ruta_datos = ruta_datos
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.umbral_desercion = 0.5

        # Rutas para guardar/cargar el modelo y los transformadores
        self.model_path = '../models/mejor_modelo_desercion.h5'
        self.scaler_path = '../models/scaler.joblib'
        self.encoders_path = '../models/label_encoders.joblib'

        # Crear directorio models si no existe
        os.makedirs('../models', exist_ok=True)

    def guardar_transformadores(self):
        """Guarda el scaler y los label encoders"""
        print("Guardando transformadores...")
        joblib.dump(self.scaler, self.scaler_path)
        joblib.dump(self.label_encoders, self.encoders_path)

    def cargar_transformadores(self):
        """Carga el scaler y los label encoders guardados"""
        print("Cargando transformadores...")
        self.scaler = joblib.load(self.scaler_path)
        self.label_encoders = joblib.load(self.encoders_path)

    def cargar_modelo_guardado(self):
        """Carga el modelo guardado y los transformadores"""
        print("Cargando modelo y transformadores...")
        self.model = load_model(self.model_path)
        self.cargar_transformadores()

    def cargar_y_preparar_datos(self):
        """Carga y prepara los datos para el entrenamiento."""
        print("Cargando y preparando datos...")

        # Cargar datos
        df = pd.read_csv(self.ruta_datos)

        # Separar características y objetivo
        y = (df['Probabilidad_Desercion'] >= self.umbral_desercion).astype(int)

        # Definir características
        caracteristicas_numericas = [
            'Edad', 'Ciclo_Actual', 'Promedio_Ponderado', 'Creditos_Aprobados',
            'Cursos_Desaprobados', 'Asistencia_Porcentaje', 'Ranking_Facultad',
            'Ingreso_Familiar_Mensual', 'Tiempo_Transporte_Minutos',
            'Nivel_Estres', 'Satisfaccion_Carrera', 'Horas_Estudio_Diarias',
            'Participacion_Actividades', 'Actividades_Extracurriculares'
        ]

        caracteristicas_categoricas = [
            'Facultad', 'Carrera', 'Genero', 'Modalidad_Ingreso',
            'Distrito_Residencia'
        ]

        caracteristicas_binarias = [
            'Tiene_Internet', 'Tiene_Computadora', 'Trabaja',
            'Vive_Con_Padres', 'Uso_Biblioteca', 'Uso_Tutorias'
        ]

        # Preprocesar datos numéricos
        X_num = df[caracteristicas_numericas]
        X_num_scaled = self.scaler.fit_transform(X_num)

        # Preprocesar datos categóricos
        X_cat_encoded = []
        for col in caracteristicas_categoricas:
            le = LabelEncoder()
            encoded = le.fit_transform(df[col])
            self.label_encoders[col] = le
            X_cat_encoded.append(encoded)

        # Preprocesar datos binarios
        X_bin = df[caracteristicas_binarias].values

        # Combinar todas las características
        X = np.hstack([X_num_scaled] + [col.reshape(-1, 1) for col in X_cat_encoded] + [X_bin])

        # Guardar transformadores después del preprocesamiento
        self.guardar_transformadores()

        # Dividir en conjunto de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        return X_train, X_test, y_train, y_test, caracteristicas_numericas + caracteristicas_categoricas + caracteristicas_binarias

    def crear_modelo(self, input_dim):
        """Crea un modelo mejorado de red neuronal."""
        print("Creando modelo mejorado...")

        model = Sequential([
            Dense(256, activation='relu', input_dim=input_dim),
            Dropout(0.4),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )

        self.model = model
        return model

    def entrenar_modelo(self, X_train, y_train, X_test, y_test, epochs=150):
        """Entrena el modelo con configuración mejorada."""
        print("Entrenando modelo...")

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            mode='min'
        )

        checkpoint = ModelCheckpoint(
            self.model_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )

        # Usar class_weight para manejar desbalance de clases
        class_weights = dict(zip(
            np.unique(y_train),
            1 / np.bincount(y_train) * len(y_train) / 2
        ))

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stopping, checkpoint],
            class_weight=class_weights,
            verbose=1
        )

        return history

    def evaluar_modelo(self, X_test, y_test):
        """Evalúa el modelo y genera métricas y visualizaciones."""
        print("\nEvaluando modelo...")

        # Predicciones
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba >= self.umbral_desercion).astype(int)

        # Imprimir métricas
        print("\nReporte de Clasificación:")
        print(classification_report(y_test, y_pred))

        # Generar y guardar matriz de confusión
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matriz de Confusión')
        plt.ylabel('Real')
        plt.xlabel('Predicción')
        plt.savefig('../models/matriz_confusion_desercion.png')
        plt.close()

        # Generar y guardar curva ROC
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title('Curva ROC')
        plt.legend(loc="lower right")
        plt.savefig('../models/curva_roc_desercion.png')
        plt.close()

    def predecir_nuevo_caso(self, datos_estudiante):
        """Predice la probabilidad de deserción para un nuevo estudiante."""
        if not isinstance(datos_estudiante, pd.DataFrame):
            raise ValueError("Los datos deben ser un DataFrame de pandas")

        # Preparar datos usando las mismas transformaciones
        X = self._preparar_datos_prediccion(datos_estudiante)

        # Realizar predicción
        prob_desercion = self.model.predict(X)[0][0]
        return {
            'probabilidad_desercion': prob_desercion,
            'riesgo_desercion': 'Alto' if prob_desercion >= self.umbral_desercion else 'Bajo'
        }

    def _preparar_datos_prediccion(self, datos):
        """Prepara los datos de un nuevo estudiante para predicción."""
        datos_procesados = []

        # Asegurarse de que el scaler y los encoders estén cargados
        if not hasattr(self.scaler, 'feature_names_in_'):
            self.cargar_transformadores()

        for col in datos.columns:
            if col in self.scaler.feature_names_in_:
                datos_procesados.append(self.scaler.transform(datos[[col]]))
            elif col in self.label_encoders:
                le = self.label_encoders[col]
                encoded = le.transform(datos[col])
                datos_procesados.append(encoded.reshape(-1, 1))
            else:
                datos_procesados.append(datos[[col]].values)

        return np.hstack(datos_procesados)


def main():
    # Configurar semilla para reproducibilidad
    np.random.seed(42)
    tf.random.set_seed(42)

    try:
        # Crear instancia del predictor
        predictor = UNIDesercionPredictor()

        # Cargar y preparar datos
        X_train, X_test, y_train, y_test, nombres_caracteristicas = predictor.cargar_y_preparar_datos()

        # Crear y entrenar modelo
        predictor.crear_modelo(input_dim=X_train.shape[1])
        history = predictor.entrenar_modelo(X_train, y_train, X_test, y_test)

        # Evaluar modelo
        predictor.evaluar_modelo(X_test, y_test)

        # Ejemplo de predicción para un nuevo estudiante
        nuevo_estudiante = pd.DataFrame({
            'Edad': [20],
            'Ciclo_Actual': [5],
            'Promedio_Ponderado': [12.5],
            'Creditos_Aprobados': [100],
            'Cursos_Desaprobados': [2],
            'Asistencia_Porcentaje': [85],
            'Ranking_Facultad': [50],
            'Facultad': ['FIEE'],
            'Carrera': ['Ingeniería Electrónica'],
            'Genero': ['M'],
            'Modalidad_Ingreso': ['Ordinario'],
            'Ingreso_Familiar_Mensual': [3000],
            'Distrito_Residencia': ['San Miguel'],
            'Tiempo_Transporte_Minutos': [60],
            'Tiene_Internet': [1],
            'Tiene_Computadora': [1],
            'Trabaja': [0],
            'Vive_Con_Padres': [1],
            'Nivel_Estres': [3],
            'Satisfaccion_Carrera': [4],
            'Participacion_Actividades': [2],
            'Horas_Estudio_Diarias': [4],
            'Uso_Biblioteca': [1],
            'Uso_Tutorias': [1],
            'Actividades_Extracurriculares': [2]
        })

        # Ejemplo de cómo cargar el modelo guardado y hacer predicciones
        predictor_cargado = UNIDesercionPredictor()
        predictor_cargado.cargar_modelo_guardado()
        resultado = predictor_cargado.predecir_nuevo_caso(nuevo_estudiante)

        print("\nPredicción para nuevo estudiante:")
        print(f"Probabilidad de deserción: {resultado['probabilidad_desercion']:.2%}")
        print(f"Nivel de riesgo: {resultado['riesgo_desercion']}")

    except Exception as e:
        print(f"Error durante la ejecución: {str(e)}")


if __name__ == "__main__":
    main()