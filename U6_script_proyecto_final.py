import pulp
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# DATOS BASE DEL PROBLEMA (MODELO CONTINUO)
# ==============================================================================
# Conjunto de índices para los nodos de Origen (Plantas) y Destino (Centros)
plantas = ["Planta_A", "Planta_B"]
centros = ["Centro_1", "Centro_2", "Centro_3"]

# Vector de Recursos o Lados Derechos (RHS) para las restricciones de oferta
capacidad = {"Planta_A": 150, "Planta_B": 200}

# Vector de Requerimientos o Lados Derechos (RHS) para las restricciones de demanda
demanda = {"Centro_1": 80, "Centro_2": 120, "Centro_3": 100}

# Matriz de coeficientes de costos (c_ij) en la Función Objetivo para el transporte unitario
costos_transporte = {
    "Planta_A": {"Centro_1": 4, "Centro_2": 7, "Centro_3": 6},
    "Planta_B": {"Centro_1": 5, "Centro_2": 3, "Centro_3": 8}
}

def resolver_modelo_base(cap_A_dinamica=150, costo_A_C1_dinamico=4):
    """
    Función parametrizada que reconstruye y resuelve el modelo lineal continuo.
    Permite simular variaciones controladas en el RHS y en los coeficientes de la F.O.
    """
    # 1. Definición del problema Primal como un modelo de Minimización
    model = pulp.LpProblem("Sensibilidad_Supply_Chain", pulp.LpMinimize)
    
    # 2. Declaración de las variables continuas x_ij (Flujo) condicionadas a la No Negatividad (>= 0)
    x = pulp.LpVariable.dicts("x", (plantas, centros), lowBound=0, cat='Continuous')
    
    # 3. Construcción de la Función Objetivo con el coeficiente de costo c_11 modificado dinámicamente
    model += pulp.lpSum(
        (costo_A_C1_dinamico if i == "Planta_A" and j == "Centro_1" else costos_transporte[i][j]) * x[i][j]
        for i in plantas for j in centros
    )
    
    # 4. Restricciones de Capacidad (Oferta): El flujo total saliente no debe superar la capacidad disponible
    for i in plantas:
        cap_actual = cap_A_dinamica if i == "Planta_A" else capacidad[i] # Modifica b_1 si corresponde a la Planta_A
        model += pulp.lpSum(x[i][j] for j in centros) <= cap_actual, f"Capacidad_{i}"
    
    # 5. Restricciones de Demanda: El flujo total entrante debe satisfacer o superar los requisitos del mercado
    for j in centros:
        model += pulp.lpSum(x[i][j] for i in plantas) >= demanda[j], f"Demanda_{j}"
        
    # 6. Ejecución del algoritmo Simplex de forma silenciosa (sin mostrar los logs del terminal)
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # Retorna los resultados clave: Valor óptimo Z, diccionario de restricciones (para los duales) y las variables
    return pulp.value(model.objective), model.constraints, x

# ==============================================================================
# ANÁLISIS DE SENSIBILIDAD DEL RHS (b) - Variación de la Capacidad de Planta_A
# ==============================================================================
print("=== GENERANDO ANÁLISIS DE SENSIBILIDAD: VARIACIÓN DEL RHS (b) ===")

# Genera un vector con 21 valores equidistantes de capacidad (RHS b_1) desde 50 hasta 250 unidades
rangos_capacidad = np.linspace(50, 250, 21) 
costos_optimos_rhs = []  # Lista para almacenar la evolución del Costo Total (Z)
precios_sombra_A = []    # Lista para almacenar la evolución del Precio Sombra (Variable Dual \pi_1)

# Bucle para realizar el análisis paramétrico sobre la restricción de capacidad
for cap in rangos_capacidad:
    z_opt, restricciones, _ = resolver_modelo_base(cap_A_dinamica=cap)
    costos_optimos_rhs.append(z_opt)
    
    # Extrae el Precio Sombra de la restricción mediante el atributo '.pi' (Shadow Price / Multiplicador Dual)
    # Refleja el cambio marginal en Z por cada incremento unitario del RHS (b_1)
    precios_sombra_A.append(restricciones["Capacidad_Planta_A"].pi)

# --- Configuración y Graficación del Impacto de la Variación del RHS ---
fig, ax1 = plt.subplots(figsize=(10, 5))

# Eje primario (Izquierdo): Curva del costo total óptimo Z en función de la capacidad
color = '#1a365d'
ax1.set_xlabel('Capacidad de Planta A (RHS - $b_1$)', fontweight='bold')
ax1.set_ylabel('Costo Total de Operación ($Z$)', color=color, fontweight='bold')
ax1.plot(rangos_capacidad, costos_optimos_rhs, color=color, marker='o', linewidth=2, label='Costo Óptimo Z')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.6)

# Eje secundario (Derecho): Gráfica escalonada (step) del comportamiento del Precio Sombra dual
ax2 = ax1.twinx()  
color = '#e53e3e'
ax2.set_ylabel('Precio Sombra ($\pi_1$)', color=color, fontweight='bold')
ax2.step(rangos_capacidad, precios_sombra_A, color=color, where='mid', linewidth=2.5, linestyle=':', label='Precio Sombra')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Análisis de Sensibilidad del RHS ($b$)\nImpacto de la Capacidad de Planta_A en el Costo y su Valor Marginal', fontsize=12, fontweight='bold')
fig.tight_layout()
plt.savefig("sensibilidad_rhs.png", dpi=300) # Guarda la gráfica con calidad para reporte escrito
plt.show()

# ==============================================================================
# ANÁLISIS DE SENSIBILIDAD DE COEFICIENTES DE LA F.O. (c) - Ruta Planta_A -> Centro_1
# ==============================================================================
print("\n=== GENERANDO ANÁLISIS DE SENSIBILIDAD: COEFFICIENTES DE LA F.O. (c) ===")

# Genera un vector con 19 valores equidistantes para el costo c_11 de la ruta crítica, desde $1 hasta $10 por unidad
rangos_costo_ruta = np.linspace(1, 10, 19) 
costos_optimos_fo = []     # Lista para almacenar la variación del costo total del sistema (Z)
flujos_ruta_critica = []   # Lista para registrar la asignación de la variable primal x_11

# Bucle para evaluar los cambios en el coeficiente de la función objetivo
for c_dinamico in rangos_costo_ruta:
    z_opt, _, variables = resolver_modelo_base(costo_A_C1_dinamico=c_dinamico)
    costos_optimos_fo.append(z_opt)
    
    # Extrae el valor de la variable de decisión x_11 bajo el nuevo costo mediante el atributo '.varValue'
    # Monitorea si la base óptima cambia o si se mantiene estable en determinados intervalos de costo
    flujos_ruta_critica.append(variables["Planta_A"]["Centro_1"].varValue)

# --- Configuración y Graficación del Impacto de la Variación de los Coeficientes (c) ---
fig, ax1 = plt.subplots(figsize=(10, 5))

# Eje primario (Izquierdo): Curva del costo total global Z según sube el costo logístico de la ruta x_11
color = '#2b6cb0'
ax1.set_xlabel('Costo Unitario Ruta Planta_A -> Centro_1 ($c_{11}$)', fontweight='bold')
ax1.set_ylabel('Costo Total del Sistema ($Z$)', color=color, fontweight='bold')
ax1.plot(rangos_costo_ruta, costos_optimos_fo, color=color, marker='s', linewidth=2)
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.6)

# Eje secundario (Derecho): Gráfica de pasos que ilustra la cantidad óptima enviada por dicha ruta
ax2 = ax1.twinx()
color = '#2f855a'
ax2.set_ylabel('Flujo Asignado a la Ruta ($x_{11}$)', color=color, fontweight='bold')
# El dibujo de pasos (steps-post) representa perfectamente cómo el flujo cambia discretamente al saltar de una base óptima a otra
ax2.plot(rangos_costo_ruta, flujos_ruta_critica, color=color, linewidth=3, drawstyle='steps-post')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Análisis de Sensibilidad de la Función Objetivo ($c$)\nEfecto del Costo de Ruta en el Flujo Óptimo Asignado', fontsize=12, fontweight='bold')
fig.tight_layout()
plt.savefig("sensibilidad_coeficientes.png", dpi=300) # Guarda la gráfica con calidad para reporte escrito
plt.show()