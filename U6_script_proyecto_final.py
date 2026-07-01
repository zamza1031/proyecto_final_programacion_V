# ==============================================================================
# ANÁLISIS DE SENSIBILIDAD DEL RHS (b) - MODIFICADO CON IMPRESIÓN EN TERMINAL
# ==============================================================================
print("=== GENERANDO ANÁLISIS DE SENSIBILIDAD: VARIACIÓN DEL RHS (b) ===")
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
    
    # ESTA LÍNEA MUESTRA LOS RESULTADOS EN TU TERMINAL:
    print(f"{cap:<18.1f} | ${z_opt:<15.2f} | {p_sombra:<20.2f}")


# ==============================================================================
# ANÁLISIS DE SENSIBILIDAD DE LA F.O. (c) - MODIFICADO CON IMPRESIÓN EN TERMINAL
# ==============================================================================
print("\n=== GENERANDO ANÁLISIS DE SENSIBILIDAD: COEFFICIENTES DE LA F.O. (c) ===")
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
    
    # ESTA LÍNEA MUESTRA LOS RESULTADOS EN TU TERMINAL:
    print(f"${c_dinamico:<17.2f} | ${z_opt:<15.2f} | {flujo:<20.1f}")