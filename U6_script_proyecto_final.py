import pulp
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# DATOS BASE DEL PROBLEMA (MODELO CONTINUO)
# ==============================================================================
plantas = ["Planta_A", "Planta_B"]
centros = ["Centro_1", "Centro_2", "Centro_3"]
capacidad = {"Planta_A": 150, "Planta_B": 200}
demanda = {"Centro_1": 80, "Centro_2": 120, "Centro_3": 100}
costos_transporte = {
    "Planta_A": {"Centro_1": 4, "Centro_2": 7, "Centro_3": 6},
    "Planta_B": {"Centro_1": 5, "Centro_2": 3, "Centro_3": 8}
}

def resolver_modelo_base(cap_A_dinamica=150, costo_A_C1_dinamico=4):
    """Función auxiliar para resolver el modelo variando parámetros puntuales"""
    model = pulp.LpProblem("Sensibilidad_Supply_Chain", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", (plantas, centros), lowBound=0, cat='Continuous')
    
    # Función Objetivo con costo dinámico para la ruta Planta_A -> Centro_1
    model += pulp.lpSum(
        (costo_A_C1_dinamico if i == "Planta_A" and j == "Centro_1" else costos_transporte[i][j]) * x[i][j]
        for i in plantas for j in centros
    )
    
    # Restricciones con capacidad dinámica para Planta_A
    for i in plantas:
        cap_actual = cap_A_dinamica if i == "Planta_A" else capacidad[i]
        model += pulp.lpSum(x[i][j] for j in centros) <= cap_actual, f"Capacidad_{i}"
    
    for j in centros:
        model += pulp.lpSum(x[i][j] for i in plantas) >= demanda[j], f"Demanda_{j}"
        
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    return pulp.value(model.objective), model.constraints, x

# ==============================================================================
# ANÁLISIS DE SENSIBILIDAD DEL RHS (b) - Variación de la Capacidad de Planta_A
# ==============================================================================
print("=== GENERANDO ANÁLISIS DE SENSIBILIDAD: VARIACIÓN DEL RHS (b) ===")

rangos_capacidad = np.linspace(50, 250, 21) # Evaluar capacidad de Planta_A desde 50 hasta 250 unidades
costos_optimos_rhs = []
precios_sombra_A = []

for cap in rangos_capacidad:
    z_opt, restricciones, _ = resolver_modelo_base(cap_A_dinamica=cap)
    costos_optimos_rhs.append(z_opt)
    # Extraer el Precio Sombra (Dual) de la restricción de capacidad de Planta_A
    precios_sombra_A.append(restricciones["Capacidad_Planta_A"].pi)

# Gráfica 1: Impacto de la variación del RHS en el Costo Total y Precio Sombra
fig, ax1 = plt.subplots(figsize=(10, 5))

color = '#1a365d'
ax1.set_xlabel('Capacidad de Planta A (RHS - $b_1$)', fontweight='bold')
ax1.set_ylabel('Costo Total de Operación ($Z$)', color=color, fontweight='bold')
ax1.plot(rangos_capacidad, costos_optimos_rhs, color=color, marker='o', linewidth=2, label='Costo Óptimo Z')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.6)

ax2 = ax1.twinx()  
color = '#e53e3e'
ax2.set_ylabel('Precio Sombra ($\pi_1$)', color=color, fontweight='bold')
ax2.step(rangos_capacidad, precios_sombra_A, color=color, where='mid', linewidth=2.5, linestyle=':', label='Precio Sombra')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Análisis de Sensibilidad del RHS ($b$)\nImpacto de la Capacidad de Planta_A en el Costo y su Valor Marginal', fontsize=12, fontweight='bold')
fig.tight_layout()
plt.savefig("sensibilidad_rhs.png", dpi=300)
plt.show()

# ==============================================================================
# ANÁLISIS DE SENSIBILIDAD DE COEFICIENTES DE LA F.O. (c) - Ruta Planta_A -> Centro_1
# ==============================================================================
print("\n=== GENERANDO ANÁLISIS DE SENSIBILIDAD: COEFFICIENTES DE LA F.O. (c) ===")

rangos_costo_ruta = np.linspace(1, 10, 19) # Evaluar costo c_{11} de $1 a $10 por unidad
costos_optimos_fo = []
flujos_ruta_critica = []

for c_dinamico in rangos_costo_ruta:
    z_opt, _, variables = resolver_modelo_base(costo_A_C1_dinamico=c_dinamico)
    costos_optimos_fo.append(z_opt)
    # Monitorear cuántas unidades se envían por esa ruta según cambie su costo
    flujos_ruta_critica.append(variables["Planta_A"]["Centro_1"].varValue)

# Gráfica 2: Impacto del cambio de costo del coeficiente en el Flujo de la Ruta
fig, ax1 = plt.subplots(figsize=(10, 5))

color = '#2b6cb0'
ax1.set_xlabel('Costo Unitario Ruta Planta_A -> Centro_1 ($c_{11}$)', fontweight='bold')
ax1.set_ylabel('Costo Total del Sistema ($Z$)', color=color, fontweight='bold')
ax1.plot(rangos_costo_ruta, costos_optimos_fo, color=color, marker='s', linewidth=2)
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.6)

ax2 = ax1.twinx()
color = '#2f855a'
ax2.set_ylabel('Flujo Asignado a la Ruta ($x_{11}$)', color=color, fontweight='bold')
ax2.plot(rangos_costo_ruta, flujos_ruta_critica, color=color, linewidth=3, drawstyle='steps-post')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Análisis de Sensibilidad de la Función Objetivo ($c$)\nEfecto del Costo de Ruta en el Flujo Óptimo Asignado', fontsize=12, fontweight='bold')
fig.tight_layout()
plt.savefig("sensibilidad_coeficientes.png", dpi=300)
plt.show()