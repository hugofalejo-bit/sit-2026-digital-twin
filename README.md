SIT-2026 | Supply Chain Digital Twin (Enterprise Edition)

SIT-2026 es un Gemelo Digital logístico de grado empresarial desarrollado como Trabajo Final de Máster en la Universitat de Barcelona (UB). Diseñado para medir empíricamente el impacto financiero y operativo de relocalizar las cadenas de suministro globales (Nearshoring) hacia la Unión Europea y el área mediterránea.

La herramienta procesa millones de microdatos comerciales para cuantificar ineficiencias ocultas, calcular el impacto del impuesto al carbono (EU ETS), proyectar riesgos de inventario mediante modelos estocásticos y evaluar la tolerancia al estrés de la infraestructura portuaria.

Propuesta de Valor y Funcionalidad Principal

El núcleo del sistema es un simulador interactivo de Estrategia Global de Red. Al ajustar el porcentaje de relocalización hacia el Nearshoring, toda la arquitectura matemática de la aplicación se recalibra en tiempo real para proyectar los impactos en el Estado de Resultados (P&L), el flujo de caja y la saturación portuaria.

Toda la plataforma responde a un sistema de Filtros Inteligentes en Cascada. La selección de un sector macroeconómico actualiza dinámicamente los países de origen y los puertos de destino, garantizando la interoperabilidad absoluta de los datos físicos subyacentes. Además, permite aislar períodos históricos de crisis globales (ej. 2008, 2020) para analizar el comportamiento sistémico bajo disrupción.

Módulos Analíticos Core

-Panel Ejecutivo: Dashboard de control macro. Evalúa la salud estructural de la selección mediante el SIT Resilience Score (0-100). Advierte sobre concentraciones de riesgo o inmovilización de capital desproporcionada que justifiquen una estrategia de relocalización.

-Flujos Comerciales: Visualización topológica mediante diagramas de flujo de carga física (Sankey) y mapas de calor coropléticos, rastreando dependencias desde el país de origen hasta cada terminal portuaria europea.

-Gemelo Portuario (Simulador M/M/1): Motor predictivo basado en la Teoría Computacional de Colas. Evalúa si la infraestructura europea está preparada para recibir buques de menor tamaño y mayor frecuencia (Short Sea Shipping). Calcula la longitud de la cola logística y alerta sobre posibles fallos sistémicos o cuellos de botella.

-Análisis de Sobrecostos (TCO): Gráficos de cascada financiera que exponen la hemorragia de capital previa al desembarque de la mercancía en suelo europeo, desglosando la carga arancelaria, normativas de carbono (EU ETS) y fricción operativa (Alpha Nodal).

-Motor de Riesgo Estocástico (LVaR): Simulador Monte Carlo (10,000 iteraciones) que calcula el Logistics Value at Risk y el Tail Risk (CVaR). Determina matemáticamente las reservas de capital (Safety Stock financiero) necesarias para absorber el 5% de los peores escenarios históricos de disrupción marítima.

-Modelado de Escenarios (Simulador A/B): Matriz analítica para contrastar financieramente a dos proveedores (ej. Offshoring asiático vs Nearshoring norafricano), desglosando el ahorro en el Costo Total de Propiedad y la compresión de tiempos de terminal.

-Consola de Investigación Matemática: Entorno estadístico avanzado que incluye ecuaciones gravitacionales (Estimadores PPML) y detectores algorítmicos de perturbaciones (PELT) para aislar quiebres de varianza.

Stack Tecnológico y Arquitectura

-Frontend e Interfaz: Streamlit (Python) con inyección de CSS personalizado para garantizar diseño responsivo y estética corporativa.

-Motor de Base de Datos: DuckDB y Apache Arrow (Parquet) integrados para procesamiento columnar en memoria de grandes volúmenes de datos con latencia mínima.

-Integración Cloud: Conexión nativa a MotherDuck para el despliegue del almacenamiento de datos masivos en arquitectura de nube.

-Analítica Computacional: NumPy, Pandas, SciPy y Statsmodels empleados para inferencia estadística, regresiones no lineales y algoritmos estocásticos.

-Visualización de Datos: Plotly Express y Plotly Graph Objects para la generación de gráficas interactivas y dinámicas.

Instrucciones de Despliegue Local

1. Clona el repositorio en tu entorno local.

2. Crea y activa un entorno virtual de Python (se recomienda .venv).

3. Instala las dependencias requeridas ejecutando pip install -r requirements.txt en la terminal.

4. Asegúrate de contar con el archivo de base de datos Parquet estructurado en la ruta data/processed/ o, en su defecto, configura tu token de MotherDuck localmente dentro del directorio oculto .streamlit/secrets.toml.

5. Inicia el servidor de desarrollo ejecutando streamlit run app.py.

Contexto Académico e Institucional

Este sistema fue desarrollado por Hugo Francisco Alejo Cárdenas como Trabajo Final de Máster, bajo la supervisión y asesoría metodológica del Prof. Josep María Cervera.

El proyecto contó con el apoyo y la beca institucional de la Secretaría de Ciencia, Humanidades, Tecnología e Innovación (SECIHTI). Los registros físicos y bases de verificación integrados en el pipeline ETL provienen de Eurostat (COMEXT Analytics), CEPII Gravitational Data, Banco Central Europeo y la Administración de Información Energética (EIA).
