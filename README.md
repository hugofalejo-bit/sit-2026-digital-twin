# sit-2026-digital-twin
SIT-2026 | Supply Chain Digital Twin: An enterprise-grade analytical engine built with Streamlit and DuckDB to evaluate the financial and operational impact of nearshoring in EU maritime transport.
SIT-2026 | Supply Chain Twin (Edición Enterprise)
Resumen del Proyecto
SIT-2026 es un Gemelo Digital logístico diseñado para medir empíricamente el impacto financiero de relocalizar las cadenas de suministro (Nearshoring) hacia la Unión Europea. La herramienta procesa millones de microdatos comerciales para cuantificar ineficiencias ocultas, calcular el impacto del impuesto al carbono (EU ETS) y proyectar riesgos de inventario.

Instrucciones de Uso y Navegación
La plataforma está diseñada para ser intuitiva y responder en tiempo real. A continuación, se detalla cómo utilizar las funciones principales:

1. Panel de Control y Filtros en Cascada (Barra Lateral)
Toda la aplicación reacciona a los filtros ubicados en el menú lateral izquierdo.

Filtros Inteligentes: Funcionan "en cascada". Esto significa que si seleccionas un sector macroeconómico (ej. Textil), el filtro de "País de Origen" se actualizará automáticamente para mostrar solo los países que realmente exportan textiles a Europa.

Filtro de Crisis: Puedes activar la casilla "Aislar períodos de crisis globales" para ver cómo se comportan los costos y los retrasos únicamente durante años de disrupción (ej. 2008, 2020).

2. Panel Ejecutivo

Uso: Es tu vista panorámica. Al abrir la app, revisa aquí la salud estructural de tu selección.

Interpretación: Observa el SIT Resilience Score (del 0 al 100). Si el marcador está en rojo (Vulnerabilidad), indica una alta concentración de proveedores o un riesgo logístico (LVaR) desproporcionado, justificando una estrategia inmediata de relocalización.

3. Modelado de Escenarios (Simulador A/B)

Uso: Ve a esta pestaña para comparar financieramente a tu proveedor actual contra una nueva alternativa.

Paso a paso: 1. Ve a la sub-pestaña "Simulador de Relocalización (A/B)".
2. Selecciona el Proveedor A (ej. China - Offshoring).
3. Selecciona el Proveedor B (ej. Marruecos - Nearshoring).
4. Ingresa el Volumen del Contrato (en millones de euros).

Interpretación: El gráfico de doble eje desglosará el Costo Total de Propiedad (TCO). Podrás ver exactamente cuánto dinero ahorra la empresa al evitar aranceles e impuestos ambientales (EU ETS), y cuánto capital libera al acortar los tiempos de tránsito. Lee el "Dictamen Ejecutivo" generado en la parte inferior para la conclusión final.

4. Análisis de Sobrecostos (TCO)

Uso: Visualiza la hemorragia financiera macroeconómica.

Interpretación: El gráfico de Cascada (Waterfall) suma todos los sobrecostos de tu selección. Te permite demostrar cuánto se gasta en pura "fricción" (aranceles, emisiones y riesgo operativo) antes de que la mercancía toque suelo europeo.

5. Gemelo Portuario (Simulador de Congestión)

Uso: Evalúa si la infraestructura europea está preparada para recibir los buques pequeños y frecuentes que exige el Nearshoring.

Paso a paso: Selecciona un Estado Miembro y un Puerto específico. Juega con los sliders (deslizadores) para simular una "Temporada Alta" (Factor de Estrés) o para reducir la velocidad de descarga del puerto.

Interpretación: El modelo M/M/1 calculará la tensión de la infraestructura. Si la métrica de "Inventario Atascado" se dispara o el medidor muestra "Fallo Sistémico", significa que mover tu producción a ese puerto generará un cuello de botella logístico.

6. Motor de Riesgo (LVaR - Monte Carlo)

Uso: Calcula el peor escenario posible de retrasos estocásticos (aleatorios).

Interpretación: El modelo simula 10,000 futuros posibles. El KPI "Tail Risk (CVaR)" te indica exactamente cuánto dinero en efectivo debe guardar la empresa en el banco para sobrevivir al 5% de los peores retrasos marítimos.
