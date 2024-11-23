import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from datetime import datetime
import base64

# Configuración de la página
st.set_page_config(
    page_title="UNI - Sistema de Predicción de Deserción",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS mejorados
st.markdown("""
    <style>
    .main {
        padding: 2rem;
        background-color: #f8f9fa;
    }
    .stTitle {
        color: #2c3e50;
        font-size: 2.5rem !important;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .stSubheader {
        color: #34495e;
        font-size: 1.5rem !important;
        font-weight: 600;
        margin-top: 1.5rem;
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 0.5rem;
    }
    .stat-card {
        padding: 1.5rem;
        border-radius: 1rem;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
        margin-bottom: 1rem;
    }
    .stat-card:hover {
        transform: translateY(-5px);
    }
    .recommendation-card {
        background-color: #f8f9fa;
        border-left: 4px solid #2c3e50;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

class ModeloDesercion:
    # [Mantener la clase ModeloDesercion sin cambios]
    pass

@st.cache_data
def cargar_datos():
    try:
        df = pd.read_csv(r"C:\Users\Andriy\Documents\GitHub\streamlit-prediction-system\data\datos_estudiantes_uni.csv")
        # Agregar años (2015-2024)
        df['Año_Ingreso'] = np.random.choice(range(2015, 2025), size=len(df))
        return df
    except Exception as e:
        st.error(f"Error al cargar los datos: {str(e)}")
        return None

def aplicar_filtros(df, filtros):
    df_filtrado = df.copy()

    if filtros['facultades'] and 'Todos' not in filtros['facultades']:
        df_filtrado = df_filtrado[df_filtrado['Facultad'].isin(filtros['facultades'])]
    
    if filtros['generos']:
        df_filtrado = df_filtrado[df_filtrado['Genero'].isin(filtros['generos'])]

    if filtros['años']:
        año_min, año_max = filtros['años']
        df_filtrado = df_filtrado[
            (df_filtrado['Año_Ingreso'] >= año_min) &
            (df_filtrado['Año_Ingreso'] <= año_max)
        ]

    if filtros['ciclo']:
        ciclo_min, ciclo_max = filtros['ciclo']
        df_filtrado = df_filtrado[
            (df_filtrado['Ciclo_Actual'] >= ciclo_min) &
            (df_filtrado['Ciclo_Actual'] <= ciclo_max)
        ]

    if filtros['promedio']:
        prom_min, prom_max = filtros['promedio']
        df_filtrado = df_filtrado[
            (df_filtrado['Promedio_Ponderado'] >= prom_min) &
            (df_filtrado['Promedio_Ponderado'] <= prom_max)
        ]

    return df_filtrado

def generar_graficos_mejorados(df):
    # Tendencia temporal de deserción
    tendencia_anual = df.groupby('Año_Ingreso')['Probabilidad_Desercion'].mean().reset_index()
    fig_tendencia = px.line(
        tendencia_anual,
        x='Año_Ingreso',
        y='Probabilidad_Desercion',
        title='Evolución del Riesgo de Deserción por Año',
        labels={'Probabilidad_Desercion': 'Riesgo Promedio', 'Año_Ingreso': 'Año'},
        markers=True,
        line_shape='spline'
    )
    fig_tendencia.update_traces(line_color='#2c3e50', line_width=3)
    
    # Deserción por facultad y género
    desercion_facultad_genero = df.groupby(['Facultad', 'Genero'])['Probabilidad_Desercion'].mean().reset_index()
    fig_desercion = px.bar(
        desercion_facultad_genero,
        x='Facultad',
        y='Probabilidad_Desercion',
        color='Genero',
        barmode='group',
        title='Riesgo de Deserción por Facultad y Género',
        labels={'Probabilidad_Desercion': 'Riesgo de Deserción (%)', 'Facultad': 'Facultad'},
        color_discrete_map={'M': '#3498db', 'F': '#e74c3c'}
    )

    # Relación entre variables con animación temporal
    fig_scatter = px.scatter(
        df,
        x='Promedio_Ponderado',
        y='Probabilidad_Desercion',
        color='Facultad',
        size='Creditos_Aprobados',
        animation_frame='Año_Ingreso',
        title='Relación entre Promedio y Riesgo de Deserción a través del tiempo',
        labels={
            'Promedio_Ponderado': 'Promedio Ponderado',
            'Probabilidad_Desercion': 'Probabilidad de Deserción',
            'Creditos_Aprobados': 'Créditos Aprobados'
        }
    )

    return fig_tendencia, fig_desercion, fig_scatter

def main():
    # Header principal con logo UNI
    st.title("🎓 Universidad Nacional de Ingeniería")
    st.header("Sistema de Predicción de Deserción Estudiantil")

    # Cargar datos
    df = cargar_datos()
    if df is None:
        return

    # Panel de Control (antes Filtros)
    with st.sidebar:
        st.markdown("### 🎛️ Panel de Control")

        filtros = {
            'facultades': st.multiselect(
                "Facultades",
                options=['Todos'] + sorted(df['Facultad'].unique()),
                default=['Todos']
            ),
            
            'generos': st.multiselect(
                "Género",
                options=sorted(df['Genero'].unique()),
                default=sorted(df['Genero'].unique())
            ),

            'años': st.slider(
                "Período",
                min_value=2015,
                max_value=2024,
                value=(2015, 2024)
            ),

            'ciclo': st.slider(
                "Ciclo Académico",
                min_value=int(df['Ciclo_Actual'].min()),
                max_value=int(df['Ciclo_Actual'].max()),
                value=(1, 10)
            ),

            'promedio': st.slider(
                "Promedio Ponderado",
                min_value=float(df['Promedio_Ponderado'].min()),
                max_value=float(df['Promedio_Ponderado'].max()),
                value=(0.0, 20.0)
            )
        }

    # Tabs principales
    tabs = st.tabs(["📊 Dashboard", "🔍 Búsqueda Individual", "📈 Análisis", "💡 Recomendaciones Personalizadas"])

    # Tab Dashboard
    with tabs[0]:
        df_filtrado = aplicar_filtros(df, filtros)
        
        # Métricas principales en cards mejorados
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="stat-card">', unsafe_allow_html=True)
            st.metric(
                "Total Estudiantes",
                f"{len(df_filtrado):,}",
                f"de {len(df):,} total"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            riesgo_alto = (df_filtrado['Probabilidad_Desercion'] >= 0.5).mean() * 100
            st.markdown('<div class="stat-card">', unsafe_allow_html=True)
            st.metric(
                "Riesgo Alto",
                f"{riesgo_alto:.1f}%",
                f"{len(df_filtrado[df_filtrado['Probabilidad_Desercion'] >= 0.5])} estudiantes"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            promedio = df_filtrado['Promedio_Ponderado'].mean()
            st.markdown('<div class="stat-card">', unsafe_allow_html=True)
            st.metric(
                "Promedio General",
                f"{promedio:.2f}",
                f"DE: {df_filtrado['Promedio_Ponderado'].std():.2f}"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            asistencia = df_filtrado['Asistencia_Porcentaje'].mean()
            st.markdown('<div class="stat-card">', unsafe_allow_html=True)
            st.metric(
                "Asistencia Promedio",
                f"{asistencia:.1f}%",
                f"DE: {df_filtrado['Asistencia_Porcentaje'].std():.1f}%"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # Gráficos mejorados
        fig_tendencia, fig_desercion, fig_scatter = generar_graficos_mejorados(df_filtrado)
        
        st.plotly_chart(fig_tendencia, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_desercion, use_container_width=True)
        with col2:
            st.plotly_chart(fig_scatter, use_container_width=True)

 
    # Tab Búsqueda Individual
    with tabs[1]:
        st.markdown("### 🔍 Búsqueda de Estudiante")

        # Método de búsqueda
        busqueda_metodo = st.radio(
            "Método de búsqueda",
            options=["Código de Estudiante", "Filtros Avanzados"],
            horizontal=True
        )

        if busqueda_metodo == "Código de Estudiante":
            codigo = st.text_input("Ingrese el código de estudiante")
            if codigo:
                estudiante = df[df['ID_Estudiante'] == codigo]
                if not estudiante.empty:
                    st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("#### Datos del Estudiante")
                        st.write(f"**Facultad:** {estudiante['Facultad'].iloc[0]}")
                        st.write(f"**Carrera:** {estudiante['Carrera'].iloc[0]}")
                        st.write(f"**Ciclo:** {estudiante['Ciclo_Actual'].iloc[0]}")
                        st.write(f"**Promedio:** {estudiante['Promedio_Ponderado'].iloc[0]:.2f}")
                        st.write(f"**Créditos Aprobados:** {estudiante['Creditos_Aprobados'].iloc[0]}")

                    with col2:
                        st.markdown("#### Indicadores de Riesgo")
                        riesgo = estudiante['Probabilidad_Desercion'].iloc[0]
                        st.progress(riesgo)
                        if riesgo >= 0.5:
                            st.markdown('<p class="warning">⚠️ RIESGO ALTO</p>', unsafe_allow_html=True)
                        else:
                            st.markdown('<p class="success">✅ RIESGO BAJO</p>', unsafe_allow_html=True)

                        # Factores de riesgo
                        st.markdown("#### Factores de Riesgo")
                        if estudiante['Asistencia_Porcentaje'].iloc[0] < 70:
                            st.markdown("- ⚠️ Baja asistencia")
                        if estudiante['Promedio_Ponderado'].iloc[0] < 11:
                            st.markdown("- ⚠️ Bajo rendimiento académico")
                        if estudiante['Nivel_Estres'].iloc[0] > 7:
                            st.markdown("- ⚠️ Alto nivel de estrés")

                    st.markdown('</div>', unsafe_allow_html=True)

                    # Recomendaciones
                    recomendaciones = generar_recomendaciones(estudiante)
                    if recomendaciones:
                        st.markdown("### 📋 Recomendaciones")
                        for rec in recomendaciones:
                            with st.expander(rec['mensaje']):
                                for accion in rec['acciones']:
                                    st.markdown(f"- {accion}")
                else:
                    st.warning("No se encontró ningún estudiante con ese código")

        else:  # Filtros Avanzados
            col1, col2 = st.columns(2)
            with col1:
                facultad_filtro = st.selectbox("Facultad", options=sorted(df['Facultad'].unique()))
                carrera_filtro = st.selectbox("Carrera",
                                              options=sorted(df[df['Facultad'] == facultad_filtro]['Carrera'].unique()))

            with col2:
                ciclo_filtro = st.selectbox("Ciclo", options=sorted(df['Ciclo_Actual'].unique()))

            estudiantes_filtrados = df[
                (df['Facultad'] == facultad_filtro) &
                (df['Carrera'] == carrera_filtro) &
                (df['Ciclo_Actual'] == ciclo_filtro)
                ]

            if not estudiantes_filtrados.empty:
                st.markdown("### Resultados de la búsqueda")
                st.dataframe(
                    estudiantes_filtrados[[
                        'ID_Estudiante', 'Promedio_Ponderado', 'Creditos_Aprobados',
                        'Asistencia_Porcentaje', 'Probabilidad_Desercion'
                    ]].style.background_gradient(subset=['Probabilidad_Desercion'], cmap='RdYlGn_r')
                )
            else:
                st.warning("No se encontraron estudiantes con los filtros seleccionados")
      
    # Tab Análisis
    with tabs[2]:
        st.markdown("### 📈 Análisis Detallado")

        # Correlación entre variables
        st.subheader("Matriz de Correlación")
        variables_numericas = [
            'Edad', 'Ciclo_Actual', 'Promedio_Ponderado', 'Creditos_Aprobados',
            'Asistencia_Porcentaje', 'Nivel_Estres', 'Satisfaccion_Carrera',
            'Probabilidad_Desercion'
        ]

        corr_matrix = df_filtrado[variables_numericas].corr()
        fig_corr = px.imshow(
            corr_matrix,
            labels=dict(color="Correlación"),
            color_continuous_scale="RdBu",
            title="Matriz de Correlación entre Variables"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        # Análisis por facultad
        st.subheader("Análisis por Facultad")
        col1, col2 = st.columns(2)

        with col1:
            fig_prom_facultad = px.box(
                df_filtrado,
                x='Facultad',
                y='Promedio_Ponderado',
                color='Facultad',
                title='Distribución de Promedios por Facultad'
            )
            st.plotly_chart(fig_prom_facultad, use_container_width=True)

        with col2:
            desercion_facultad = df_filtrado.groupby('Facultad')['Probabilidad_Desercion'].agg(['mean', 'count'])
            fig_desercion_facultad = px.bar(
                desercion_facultad.reset_index(),
                x='Facultad',
                y='mean',
                title='Promedio de Riesgo de Deserción por Facultad',
                labels={'mean': 'Probabilidad Promedio', 'count': 'Cantidad de Estudiantes'},
                text='count'
            )
            st.plotly_chart(fig_desercion_facultad, use_container_width=True)

        # Tab Recomendaciones
    with tabs[3]:
        st.markdown("### 📑 Recomendaciones Generales")

        # Análisis general de la población en riesgo
        estudiantes_riesgo = df_filtrado[df_filtrado['Probabilidad_Desercion'] >= 0.5]
        total_riesgo = len(estudiantes_riesgo)

        st.markdown(f"""
                    #### Resumen de Población en Riesgo
                    - Total de estudiantes en riesgo alto: {total_riesgo}
                    - Porcentaje del total: {(total_riesgo / len(df_filtrado) * 100):.1f}%
                    """)

        # Factores principales de riesgo
        st.subheader("Factores Principales de Riesgo")
        col1, col2 = st.columns(2)

        with col1:
            bajo_rendimiento = len(estudiantes_riesgo[estudiantes_riesgo['Promedio_Ponderado'] < 11])
            baja_asistencia = len(estudiantes_riesgo[estudiantes_riesgo['Asistencia_Porcentaje'] < 70])
            alto_estres = len(estudiantes_riesgo[estudiantes_riesgo['Nivel_Estres'] > 7])

            fig_factores = px.bar(
                pd.DataFrame({
                    'Factor': ['Bajo Rendimiento', 'Baja Asistencia', 'Alto Estrés'],
                    'Cantidad': [bajo_rendimiento, baja_asistencia, alto_estres]
                }),
                x='Factor',
                y='Cantidad',
                title='Principales Factores de Riesgo'
            )
            st.plotly_chart(fig_factores, use_container_width=True)

        with col2:
            st.markdown("""
                        #### Recomendaciones Institucionales
                        1. **Programa de Tutorías**
                           - Implementar sistema de alerta temprana
                           - Asignar tutores a estudiantes en riesgo

                        2. **Apoyo Académico**
                           - Talleres de técnicas de estudio
                           - Asesorías académicas personalizadas

                        3. **Bienestar Estudiantil**
                           - Servicios de consejería
                           - Talleres de manejo del estrés

                        4. **Seguimiento**
                           - Monitoreo continuo del progreso
                           - Evaluación periódica de indicadores
                        """)
    
    # Nueva tab de Recomendaciones Personalizadas
    with tabs[3]:
        st.markdown("### 💡 Recomendaciones Personalizadas")
        
        # Selector de facultad y carrera
        col1, col2 = st.columns(2)
        with col1:
            facultad_sel = st.selectbox("Seleccione Facultad", options=sorted(df['Facultad'].unique()))
        with col2:
            carrera_sel = st.selectbox(
                "Seleccione Carrera",
                options=sorted(df[df['Facultad'] == facultad_sel]['Carrera'].unique())
            )

        # Análisis específico por facultad/carrera
        estudiantes_carrera = df[
            (df['Facultad'] == facultad_sel) &
            (df['Carrera'] == carrera_sel)
        ]

        # Mostrar recomendaciones personalizadas
        st.markdown("#### Análisis y Recomendaciones")
        
        col1, col2 = st.columns(2)
        with col1:
            # Métricas específicas
            promedio_carrera = estudiantes_carrera['Promedio_Ponderado'].mean()
            riesgo_carrera = estudiantes_carrera['Probabilidad_Desercion'].mean()
            
            st.markdown('<div class="recommendation-card">', unsafe_allow_html=True)
            st.markdown(f"""
                ##### Métricas de la Carrera
                - Promedio general: **{promedio_carrera:.2f}**
                - Riesgo de deserción promedio: **{riesgo_carrera:.2%}**
                - Total estudiantes: **{len(estudiantes_carrera)}**
            """)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            # Recomendaciones basadas en datos
            st.markdown('<div class="recommendation-card">', unsafe_allow_html=True)
            st.markdown("##### Recomendaciones Específicas")
            
            if riesgo_carrera > 0.4:
                st.warning("⚠️ Carrera con riesgo de deserción elevado")
                st.markdown("""
                    - Implementar programa de tutorías reforzado
                    - Realizar seguimiento individual a estudiantes
                    - Organizar grupos de estudio
                """)
            else:
                st.success("✅ Carrera con buen rendimiento general")
                st.markdown("""
                    - Mantener sistema actual de seguimiento
                    - Fortalecer actividades extracurriculares
                    - Promover participación en investigación
                """)
            st.markdown('</div>', unsafe_allow_html=True)

        # Gráficos específicos para la carrera
        st.markdown("#### Análisis Temporal")
        
        # Tendencia de rendimiento por año
        tendencia_carrera = estudiantes_carrera.groupby('Año_Ingreso').agg({
            'Promedio_Ponderado': 'mean',
            'Probabilidad_Desercion': 'mean'
        }).reset_index()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=tendencia_carrera['Año_Ingreso'],
            y=tendencia_carrera['Promedio_Ponderado'],
            name='Promedio',
            line=dict(color='#2ecc71', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=tendencia_carrera['Año_Ingreso'],
            y=tendencia_carrera['Probabilidad_Desercion'] * 20,  # Escalado para visualización
            name='Riesgo de Deserción (x20)',
            line=dict(color='#e74c3c', width=3, dash='dash')
        ))
        
        fig.update_layout(
            title=f'Evolución del Rendimiento en {carrera_sel}',
            xaxis_title='Año',
            yaxis_title='Valor',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Última actualización:** " + datetime.now().strftime("%Y-%m-%d %H:%M"))
    with col2:
        st.markdown("**Desarrollado por:** UNI")
    with col3:
        st.markdown("**Versión:** 2.0")

if __name__ == "__main__":
    main()