import json
import numpy as np
import pandas as pd
from datetime import datetime
import os


class UNIDataGenerator:
    def __init__(self, n_samples=1000):
        self.n_samples = n_samples
        self.seed = 42
        np.random.seed(self.seed)

        # Facultades y carreras específicas de la UNI
        self.FACULTADES = {
            "FIEE": {
                "nombre": "Facultad de Ingeniería Eléctrica y Electrónica",
                "carreras": ["Ingeniería Eléctrica", "Ingeniería Electrónica", "Ingeniería de Telecomunicaciones"]
            },
            "FIC": {
                "nombre": "Facultad de Ingeniería Civil",
                "carreras": ["Ingeniería Civil"]
            },
            "FIM": {
                "nombre": "Facultad de Ingeniería Mecánica",
                "carreras": ["Ingeniería Mecánica", "Ingeniería Mecatrónica"]
            },
            "FIQT": {
                "nombre": "Facultad de Ingeniería Química y Textil",
                "carreras": ["Ingeniería Química", "Ingeniería Textil"]
            },
            "FIIS": {
                "nombre": "Facultad de Ingeniería Industrial y de Sistemas",
                "carreras": ["Ingeniería Industrial", "Ingeniería de Sistemas"]
            },
            "FC": {
                "nombre": "Facultad de Ciencias",
                "carreras": ["Física", "Matemática", "Química"]
            },
            "FIAU": {
                "nombre": "Facultad de Ingeniería Ambiental y de Recursos Naturales",
                "carreras": ["Ingeniería Ambiental", "Ingeniería de Recursos Naturales"]
            },
            "FIP": {
                "nombre": "Facultad de Ingeniería de Petróleo, Gas Natural y Petroquímica",
                "carreras": ["Ingeniería de Petróleo", "Ingeniería Petroquímica"]
            },
            "FIEECS": {
                "nombre": "Facultad de Ingeniería Económica, Estadística y Ciencias Sociales",
                "carreras": ["Ingeniería Económica", "Estadística", "Ciencias Sociales"]
            },
            "FAU": {
                "nombre": "Facultad de Arquitectura, Urbanismo y Artes",
                "carreras": ["Arquitectura", "Urbanismo", "Artes"]
            }
        }

        self.DISTRITOS_LIMA = [
            "San Miguel", "Cercado de Lima", "Rímac", "San Martín de Porres",
            "Los Olivos", "Independencia", "San Juan de Lurigancho", "Ate",
            "Santa Anita", "La Victoria", "Breña", "Jesús María", "Callao"
        ]

    def _asignar_carrera(self, facultad):
        """Asigna una carrera específica según la facultad."""
        carreras = self.FACULTADES[facultad]['carreras']
        return np.random.choice(carreras)

    def _generar_datos_academicos(self):
        """Genera datos académicos realistas para estudiantes de la UNI."""
        return {
            "Ciclo_Actual": np.random.randint(1, 11, self.n_samples),
            "Promedio_Ponderado": np.round(np.random.normal(12.5, 2.0, self.n_samples).clip(0, 20), 2),
            "Creditos_Aprobados": np.random.randint(0, 220, self.n_samples),
            "Cursos_Desaprobados": np.random.poisson(2, self.n_samples),
            "Asistencia_Porcentaje": np.round(np.random.normal(85, 10, self.n_samples).clip(0, 100), 2),
            "Ranking_Facultad": np.random.randint(1, 101, self.n_samples),
            "Modalidad_Ingreso": np.random.choice(
                ["Ordinario", "Dos Primeros", "IEN", "Traslado Externo"],
                self.n_samples,
                p=[0.7, 0.2, 0.08, 0.02]
            )
        }

    def _generar_datos_socioeconomicos(self):
        """Genera datos socioeconómicos realistas para estudiantes de la UNI."""
        return {
            "Ingreso_Familiar_Mensual": np.round(np.random.normal(3000, 1000, self.n_samples).clip(930, 8000), 2),
            "Distrito_Residencia": np.random.choice(self.DISTRITOS_LIMA, self.n_samples),
            "Tiempo_Transporte_Minutos": np.round(np.random.normal(60, 20, self.n_samples).clip(15, 120), 0),
            "Tiene_Internet": np.random.choice([0, 1], self.n_samples, p=[0.1, 0.9]),
            "Tiene_Computadora": np.random.choice([0, 1], self.n_samples, p=[0.15, 0.85]),
            "Trabaja": np.random.choice([0, 1], self.n_samples, p=[0.7, 0.3]),
            "Vive_Con_Padres": np.random.choice([0, 1], self.n_samples, p=[0.2, 0.8])
        }

    def _generar_datos_psicosociales(self):
        """Genera datos psicosociales realistas."""
        return {
            "Nivel_Estres": np.random.randint(1, 6, self.n_samples),
            "Satisfaccion_Carrera": np.random.randint(1, 6, self.n_samples),
            "Participacion_Actividades": np.random.randint(0, 6, self.n_samples),
            "Horas_Estudio_Diarias": np.round(np.random.normal(4, 1, self.n_samples).clip(0, 12), 1),
            "Uso_Biblioteca": np.random.choice([0, 1], self.n_samples, p=[0.4, 0.6]),
            "Uso_Tutorias": np.random.choice([0, 1], self.n_samples, p=[0.7, 0.3]),
            "Actividades_Extracurriculares": np.random.randint(0, 4, self.n_samples)
        }

    def _calcular_riesgo_desercion(self, df):
        """Calcula el riesgo de deserción basado en múltiples factores."""
        riesgo = np.zeros(len(df))

        # Factores académicos (50% del peso total)
        riesgo += (df['Promedio_Ponderado'] < 10.5).astype(float) * 0.15
        riesgo += (df['Cursos_Desaprobados'] > 3).astype(float) * 0.15
        riesgo += (df['Asistencia_Porcentaje'] < 70).astype(float) * 0.10
        riesgo += (df['Ranking_Facultad'] > 80).astype(float) * 0.10

        # Factores socioeconómicos (30% del peso total)
        riesgo += (df['Ingreso_Familiar_Mensual'] < 1500).astype(float) * 0.10
        riesgo += (df['Tiempo_Transporte_Minutos'] > 90).astype(float) * 0.10
        riesgo += (df['Trabaja'] == 1).astype(float) * 0.10
        riesgo += (df['Vive_Con_Padres'] == 0).astype(float) * 0.10

        # Factores psicosociales (20% del peso total)
        riesgo += (df['Nivel_Estres'] > 4).astype(float) * 0.10
        riesgo += (df['Satisfaccion_Carrera'] < 3).astype(float) * 0.10
        riesgo += (df['Horas_Estudio_Diarias'] < 2).astype(float) * 0.10
        riesgo += (df['Actividades_Extracurriculares'] == 0).astype(float) * 0.10

        # Normalización y redondeo a 4 decimales
        return np.round(np.clip(riesgo, 0, 1), 4)

    def generar_dataset(self):
        """Genera el dataset completo con todos los factores."""
        data = {
            "ID_Estudiante": [f"UNI{str(i).zfill(6)}" for i in range(1, self.n_samples + 1)],
            "Facultad": np.random.choice(list(self.FACULTADES.keys()), self.n_samples)
        }

        # Asignar carreras basadas en facultad
        data["Carrera"] = [self._asignar_carrera(facultad) for facultad in data["Facultad"]]
        data["Edad"] = np.random.randint(16, 30, self.n_samples)
        data["Genero"] = np.random.choice(['M', 'F'], self.n_samples, p=[0.7, 0.3])

        # Agregar datos de cada categoría
        data.update(self._generar_datos_academicos())
        data.update(self._generar_datos_socioeconomicos())
        data.update(self._generar_datos_psicosociales())

        # Crear DataFrame
        df = pd.DataFrame(data)

        # Calcular riesgo de deserción
        df['Probabilidad_Desercion'] = self._calcular_riesgo_desercion(df)

        # Metadatos
        df.attrs['fecha_generacion'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        df.attrs['version'] = '3.0'
        df.attrs['institucion'] = 'Universidad Nacional de Ingeniería'

        return df

    def guardar_dataset(self, df, ruta="datos_estudiantes_uni.csv"):
        """Guarda el dataset con metadatos"""
        try:
            # Crear el directorio si no existe
            os.makedirs(os.path.dirname(ruta) or '.', exist_ok=True)

            # Guardar el DataFrame
            df.to_csv(ruta, index=False)

            # Guardar metadatos en un archivo separado
            metadata = {
                "fecha_generacion": df.attrs['fecha_generacion'],
                "version": df.attrs['version'],
                "institucion": df.attrs['institucion'],
                "num_registros": len(df),
                "columnas": list(df.columns),
                "facultades": list(self.FACULTADES.keys())
            }

            metadata_path = ruta.replace('.csv', '_metadata.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)

            print(f"Dataset guardado exitosamente en: {ruta}")
            print(f"Metadatos guardados en: {metadata_path}")

        except Exception as e:
            print(f"Error al guardar el dataset: {str(e)}")
            raise

    def generar_resumen(self, df):
        """Genera un resumen tipo prompt del dataset."""
        resumen = f"""
        Dataset generado para la Universidad Nacional de Ingeniería (UNI).
        Fecha de creación: {df.attrs['fecha_generacion']}.
        Total de registros: {len(df)}.
        Facultades: {', '.join(self.FACULTADES.keys())}.
        Variables principales: {', '.join(df.columns)}.
        Promedio de riesgo de deserción: {df['Probabilidad_Desercion'].mean():.4f}.
        """
        print(resumen)
        return resumen


# Script principal para ejecutar el generador
if __name__ == "__main__":
    try:
        # Crear instancia del generador
        generador = UNIDataGenerator(n_samples=2000)

        # Generar dataset
        print("Generando dataset...")
        df = generador.generar_dataset()

        # Guardar dataset
        generador.guardar_dataset(df)

        # Mostrar resumen
        generador.generar_resumen(df)

    except Exception as e:
        print(f"Error durante la ejecución: {str(e)}")