import pulp
from ortools.linear_solver import pywraplp
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 0. DEFINICIÓN DE PARAMETRIZACIÓN GENERAL DEL CASO DE ESTUDIO
# ==============================================================================
plantas = ["Planta_A", "Planta_B"]
centros = ["Centro_1", "Centro_2", "Centro_3"]

# Restricciones de capacidad de la infraestructura y costos asociados de apertura
capacidad = {"Planta_A": 150, "Planta_B": 200}
costo_fijo = {"Planta_A": 500, "Planta_B": 800}

# Restricciones estructurales del mercado objetivo (Demanda)
demanda = {"Centro_1": 80, "Centro_2": 120, "Centro_3": 100}

# Matriz de costos unitarios de transporte c_{ij}
costos_transporte = {
    "Planta_A": {"Centro_1": 4, "Centro_2": 7, "Centro_3": 6},
    "Planta_B": {"Centro_1": 5, "Centro_2": 3, "Centro_3": 8}
}

def resolver_modelo_base(cap_A_dinamica=150, costo_A_C1_dinamico=4):
    """
    Función auxiliar parametrizada para reconstruir y resolver el modelo lineal continuo.
    Permite realizar el análisis de sensibilidad iterativo tanto del RHS como de la F.O.
    """
    model = pulp.LpProblem("Sensibilidad_Supply_Chain", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", (plantas, centros), lowBound=0, cat='Continuous')
    
    # Función Objetivo con costo dinámico para la ruta Planta_A -> Centro_1
    model += pulp.lpSum(
        (costo_A_C1_dinamico if i == "Planta_A" and j == "Centro_1" else costos_transporte[i][j]) * x[i][j]
        for i in plantas for j in centros
    )
    
    # Restricciones con capacidad dinámica para Planta_A (Variación del RHS)
    for i in plantas:
        cap_actual = cap_A_dinamica if i == "Planta_A" else capacidad[i]
        model += pulp.lpSum(x[i][j] for j in centros) <= cap_actual, f"Capacidad_{i}"
    
    # Restricciones de Demanda
    for j in centros:
        model += pulp.lpSum(x[i][j] for i in plantas) >= demanda[j], f"Demanda_{j}"
        
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    return pulp.value(model.objective), model.constraints, x


# ==============================================================================
# 1. UNIDADES 1, 2 Y 3: MODELO CONTINUO, SIMPLEX Y DUALIDAD (Uso de PuLP)
# ==============================================================================
print("=" * 70)
print("=== MÓDULO 1: ANÁLISIS CONTINUO, ALGORITMO SIMPLEX Y DUALIDAD ===")
print("=" * 70)

model_pulp = pulp.LpProblem("Optimizacion_Supply_Chain_Continua", pulp.LpMinimize)
x_pulp = pulp.LpVariable.dicts("x", (plantas, centros), lowBound=0, cat='Continuous')

model_pulp += pulp.lpSum(costos_transporte[i][j] * x_pulp[i][j] for i in plantas for j in centros)

for i in plantas:
    model_pulp += pulp.lpSum(x_pulp[i][j] for j in centros) <= capacidad[i], f"Capacidad_{i}"
for j in centros:
    model_pulp += pulp.lpSum(x_pulp[i][j] for i in plantas) >= demanda[j], f"Demanda_{j}"

model_pulp.solve(pulp.PULP_CBC_CMD(msg=False))

print(f"Estado del Algoritmo Simplex: {pulp.LpStatus[model_pulp.status]}")
print(f"Costo Base de Operación de Transporte: ${pulp.value(model_pulp.objective):.2f}\n")

# Extracción de variables óptimas y Costo Reducido (Sensibilidad de Variables)
print(f"{'Ruta (Variable Primal)':<25} | {'Flujo Óptimo':<12} | {'Costo Reducido (dj)':<18}")
print("-" * 62)
for i in plantas:
    for j in centros:
        print(f"{i} -> {j:<12} | {x_pulp[i][j].varValue:<12.1f} | {x_pulp[i][j].dj:<18.2f}")

# Extracción de Multiplicadores Duales / Precios Sombra (Interpretación Económica)
print(f"\n{'Restricción Estructural':<25} | {'Precio Sombra (pi)':<18} | {'Holgura (Slack)':<15}")
print("-" * 62)
for name, c in model_pulp.constraints.items():
    print(f"{name:<25} | {c.pi:<18.2f} | {c.slack:<15.2f}")
print("\n")


# ==============================================================================
# 2. ANÁLISIS DE SENSIBILIDAD DEL RHS (b) - Variación de Capacidad con Impresión
# ==============================================================================
print("=" * 70)
print("=== MÓDULO 2: ANÁLISIS DE SENSIBILIDAD DEL RHS (b) ===")
print("=" * 70)
print(f"{'Capacidad A (b1)':<18} | {'Costo Óptimo Z':<16} | {'Precio Sombra (pi1)':<20}")
print("-" * 60)

rangos_capacidad = np.linspace(50, 250, 21) 
costos_optimos_rhs = []  
precios_sombra_A = []    

for cap in rangos_capacidad:
    z_opt, restricciones, _ = resolver_modelo_base(cap_A_dinamica=cap)
    costos_optimos_rhs.append(z_opt)
    p_sombra = restricciones["Capacidad_Planta_A"].pi
    precios_sombra_A.append(p_sombra)
    print(f"{cap:<18.1f} | ${z_opt:<15.2f} | {p_sombra:<20.2f}")

# Graficación del Impacto de la Variación del RHS
fig, ax1 = plt.subplots(figsize=(10, 5))
color = '#1a365d'
ax1.set_xlabel('Capacidad de Planta A (RHS - $b_1$)', fontweight='bold')
ax1.set_ylabel('Costo Total de Operación ($Z$)', color=color, fontweight='bold')
ax1.plot(rangos_capacidad, costos_optimos_rhs, color=color, marker='o', linewidth=2, label='Costo Óptimo Z')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.6)

ax2 = ax1.twinx()  
color = '#e53e3e'
ax2.set_ylabel(r'Precio Sombra ($\pi_1$)', color=color, fontweight='bold')  # r'...' corrige el escape string
ax2.step(rangos_capacidad, precios_sombra_A, color=color, where='mid', linewidth=2.5, linestyle=':', label='Precio Sombra')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Análisis de Sensibilidad del RHS ($b$)\nImpacto de la Capacidad de Planta_A en el Costo y su Valor Marginal', fontsize=12, fontweight='bold')
fig.tight_layout()
plt.savefig("sensibilidad_rhs.png", dpi=300)
plt.show(block=False) 


# ==============================================================================
# 3. ANÁLISIS DE SENSIBILIDAD DE LA F.O. (c) - Variación de Costos con Impresión
# ==============================================================================
print("\n" + "=" * 70)
print("=== MÓDULO 3: ANÁLISIS DE SENSIBILIDAD DE LA FUNCIÓN OBJETIVO (c) ===")
print("=" * 70)
print(f"{'Costo Ruta c11':<18} | {'Costo Óptimo Z':<16} | {'Flujo Asignado x11':<20}")
print("-" * 60)

rangos_costo_ruta = np.linspace(1, 10, 19) 
costos_optimos_fo = []     
flujos_ruta_critica = []   

for c_dinamico in rangos_costo_ruta:
    z_opt, _, variables = resolver_modelo_base(costo_A_C1_dinamico=c_dinamico)
    costos_optimos_fo.append(z_opt)
    flujo = variables["Planta_A"]["Centro_1"].varValue
    flujos_ruta_critica.append(flujo)
    print(f"${c_dinamico:<17.2f} | ${z_opt:<15.2f} | {flujo:<20.1f}")

# Graficación del Impacto de la Variación de Coeficientes de Costo (c)
fig2, ax1_f = plt.subplots(figsize=(10, 5))
color = '#2b6cb0'
ax1_f.set_xlabel('Costo Unitario Ruta Planta_A -> Centro_1 ($c_{11}$)', fontweight='bold')
ax1_f.set_ylabel('Costo Total del Sistema ($Z$)', color=color, fontweight='bold')
ax1_f.plot(rangos_costo_ruta, costos_optimos_fo, color=color, marker='s', linewidth=2)
ax1_f.tick_params(axis='y', labelcolor=color)
ax1_f.grid(True, linestyle='--', alpha=0.6)

ax2_f = ax1_f.twinx()
color = '#2f855a'
ax2_f.set_ylabel('Flujo Asignado a la Ruta ($x_{11}$)', color=color, fontweight='bold')
ax2_f.plot(rangos_costo_ruta, flujos_ruta_critica, color=color, linewidth=3, drawstyle='steps-post')
ax2_f.tick_params(axis='y', labelcolor=color)

plt.title('Análisis de Sensibilidad de la Función Objetivo ($c$)\nEfecto del Costo de Ruta en el Flujo Óptimo Asignado', fontsize=12, fontweight='bold')
fig2.tight_layout()
plt.savefig("sensibilidad_coeficientes.png", dpi=300)
plt.show(block=False)


# ==============================================================================
# 4. UNIDAD 4: PROGRAMACIÓN ENTERA MIXTA (MIP) CON COSTOS FIJOS (Uso de OR-Tools)
# ==============================================================================
print("\n" + "=" * 70)
print("=== MÓDULO 4: PROGRAMACIÓN ENTERA MIXTA (MIP) - COSTOS FIJOS ===")
print("=" * 70)

solver = pywraplp.Solver.CreateSolver('SCIP')

x_or = {}
for i in plantas:
    for j in centros:
        x_or[i, j] = solver.NumVar(0.0, solver.infinity(), f"x_{i}_{j}")

y_or = {}
for i in plantas:
    y_or[i] = solver.IntVar(0.0, 1.0, f"y_{i}")

objetivo = solver.Objective()
for i in plantas:
    objetivo.SetCoefficient(y_or[i], costo_fijo[i])
    for j in centros:
        objetivo.SetCoefficient(x_or[i, j], costos_transporte[i][j])
objetivo.SetMinimization()

for i in plantas:
    ct_cap = solver.Constraint(-solver.infinity(), 0.0, f"Capacidad_MIP_{i}")
    ct_cap.SetCoefficient(y_or[i], -capacidad[i])
    for j in centros:
        ct_cap.SetCoefficient(x_or[i, j], 1.0)

for j in centros:
    ct_dem = solver.Constraint(demanda[j], solver.infinity(), f"Demanda_MIP_{j}")
    for i in plantas:
        ct_dem.SetCoefficient(x_or[i, j], 1.0)

status = solver.Solve()

if status == pywraplp.Solver.OPTIMAL:
    print(f"Costo Total Óptimo de la Cadena (MIP): ${solver.Objective().Value():.2f}\n")
    print("Decisiones estratégicas de Infraestructura (Variables Binarias):")
    for i in plantas:
        estado = "ABIERTA (Incurre en costo fijo)" if y_or[i].solution_value() == 1.0 else "CERRADA"
        print(f"  - {i}: {estado} [y = {y_or[i].solution_value()}]")
else:
    print("Error: El espacio de soluciones del modelo MIP es infactible.")


# ==============================================================================
# 5. UNIDAD 4: REDES DE FLUJO E INTERPRETACIÓN GRAFICA TOPOLÓGICA (Uso de NetworkX)
# ==============================================================================
print("\n" + "=" * 70)
print("=== MÓDULO 5: MODELADO TOPOLÓGICO Y VALIDACIÓN DE REDES DE FLUJO ===")
print("=" * 70)

G = nx.DiGraph()

for i in plantas:
    G.add_node(i, layer=0)
for j in centros:
    G.add_node(j, layer=1)

for i in plantas:
    for j in centros:
        flujo_calculado = x_pulp[i][j].varValue
        if flujo_calculado > 0:
            G.add_edge(i, j, cost=costos_transporte[i][j], flow=flujo_calculado)

print("Estructura topológica de la red óptima calculada:")
print(f" Nodos de red validados: {list(G.nodes)}")
print(" Arcos y tasas de flujo asignadas:")
for nodo_origen, nodo_destino, data in G.edges(data=True):
    print(f"  Arco Dirigido [{nodo_origen} -> {nodo_destino}] | Costo Base: ${data['cost']} | Volumen: {data['flow']} uds")

# Generación del Diagrama de la Red de Distribución
pos = {}
for idx, node in enumerate(plantas):
    pos[node] = (0, float(idx) - len(plantas)/2.0 + 0.5)
for idx, node in enumerate(centros):
    pos[node] = (1, float(idx) - len(centros)/2.0 + 0.5)

plt.figure(figsize=(10, 6))
plt.title("Red de Flujo Óptima - Cadena de Suministro\n(Grosor de línea proporcional al flujo)", fontsize=14, fontweight='bold')

nx.draw_networkx_nodes(G, pos, nodelist=plantas, node_color='#1a365d', node_size=1200, label="Plantas")
nx.draw_networkx_nodes(G, pos, nodelist=centros, node_color='#2b6cb0', node_size=1200, label="Centros de Dist.")
nx.draw_networkx_labels(G, pos, font_color='white', font_size=9, font_weight='bold')

weights = [G[u][v]['flow'] / 20.0 for u, v in G.edges()] 
nx.draw_networkx_edges(G, pos, width=weights, edge_color='#4a5568', arrowsize=20, connectionstyle="arc3,rad=0.05")

labels_arcos = {(u, v): f"f={data['flow']:.0f}\n(c=${data['cost']})" for u, v, data in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels_arcos, font_size=9, font_color='#2d3748', label_pos=0.6)

plt.axis('off')
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig("red_flujo_optima.png", dpi=300)

print("\n-> Gráfica 'red_flujo_optima.png' generada con éxito.")
print("Mostrando todas las gráficas en pantalla. Cierra las ventanas de los gráficos para finalizar el programa de forma limpia.")
plt.show() 