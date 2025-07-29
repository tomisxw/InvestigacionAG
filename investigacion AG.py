import random
import math
import statistics
import matplotlib.pyplot as plt

# ------------------ PARÁMETROS ------------------
# Escenario actual (ajustable para hipótesis)
COSTO_POR_CASA = 6000000
CONSUMO_ANUAL = 3500  # kWh por casa
TARIFA_COMPRA = 200   # $/kWh
TARIFA_INYECCION_SUBSIDIO = 74  # $/kWh primeros 6 años
TARIFA_INYECCION_POST = 10      # $/kWh después
HSP = 4.5
EFICIENCIA = 0.8
LATITUD = 32
ANIOS_SIMULACION = 20
SUBSIDIO_ANIOS = 6

# Configuración barrio
N_CASAS = 50
CONSUMO_TOTAL = CONSUMO_ANUAL * N_CASAS

# Rangos GA
POTENCIA_MIN, POTENCIA_MAX = 1.0, 3.0
INCLINACION_MIN, INCLINACION_MAX = 0.0, 40.0
ORIENTACION_MIN, ORIENTACION_MAX = -30.0, 30.0

# ------------------ FUNCIÓN OBJETIVO ------------------
def calcular_fitness(individuo):
    potencia, inclinacion, orientacion = individuo

    # Penalización fuerte por orientación e inclinación malas
    desvio_orient = abs(orientacion)
    factor_orient = math.cos(math.radians(desvio_orient * 3))  # 30° off = casi cero
    factor_orient = max(0, factor_orient)

    desvio_inclin = abs(inclinacion - LATITUD)
    factor_inclin = max(0, 1 - 0.02 * desvio_inclin)  # 2% menos por grado fuera de óptimo

    # Generación anual ajustada
    generacion_casa = potencia * HSP * 365 * EFICIENCIA * factor_orient * factor_inclin
    generacion_total = generacion_casa * N_CASAS

    # Costo total
    costo_total = COSTO_POR_CASA * N_CASAS

    # Simulación 20 años con payback exacto
    ahorro_acumulado = 0
    payback_anio = None

    for anio in range(1, ANIOS_SIMULACION + 1):
        autoconsumo = min(generacion_total, CONSUMO_TOTAL)
        excedente = max(0, generacion_total - CONSUMO_TOTAL)

        if anio <= SUBSIDIO_ANIOS:
            ingreso_inyeccion = excedente * TARIFA_INYECCION_SUBSIDIO
        else:
            ingreso_inyeccion = excedente * TARIFA_INYECCION_POST

        ahorro_anual = (autoconsumo * TARIFA_COMPRA) + ingreso_inyeccion

        if ahorro_acumulado + ahorro_anual >= costo_total and payback_anio is None:
            restante = costo_total - ahorro_acumulado
            fraccion = restante / ahorro_anual
            payback_anio = (anio - 1) + fraccion  # Año con decimales

        ahorro_acumulado += ahorro_anual

    if payback_anio is None:
        payback_anio = 100  # Penalización si no recupera

    return 1 / payback_anio, payback_anio

# ------------------ GA OPERADORES ------------------
def generar_individuo():
    return [
        random.uniform(POTENCIA_MIN, POTENCIA_MAX),
        random.uniform(INCLINACION_MIN, INCLINACION_MAX),
        random.uniform(ORIENTACION_MIN, ORIENTACION_MAX)
    ]

def inicializar_poblacion(size):
    return [generar_individuo() for _ in range(size)]

def seleccion_torneo(poblacion, fitnesses, k=3):
    candidatos = random.sample(range(len(poblacion)), k)
    return poblacion[max(candidatos, key=lambda i: fitnesses[i][0])]

def crossover(p1, p2):
    punto = random.randint(1, len(p1) - 1)
    return p1[:punto] + p2[punto:], p2[:punto] + p1[punto:]

def mutar(individuo, tasa=0.3):
    if random.random() < tasa:
        individuo[0] = random.uniform(POTENCIA_MIN, POTENCIA_MAX)
    if random.random() < tasa:
        individuo[1] = random.uniform(INCLINACION_MIN, INCLINACION_MAX)
    if random.random() < tasa:
        individuo[2] = random.uniform(ORIENTACION_MIN, ORIENTACION_MAX)
    return individuo

# ------------------ GA PRINCIPAL ------------------
def algoritmo_genetico(generaciones=50, tam_poblacion=20):
    poblacion = inicializar_poblacion(tam_poblacion)
    mejor_fitness_hist = []

    for _ in range(generaciones):
        fitnesses = [calcular_fitness(ind) for ind in poblacion]
        mejor_fitness_hist.append(max(f[0] for f in fitnesses))

        nueva_poblacion = [poblacion[fitnesses.index(max(fitnesses, key=lambda x: x[0]))]]

        while len(nueva_poblacion) < tam_poblacion:
            p1 = seleccion_torneo(poblacion, fitnesses)
            p2 = seleccion_torneo(poblacion, fitnesses)
            hijo1, hijo2 = crossover(p1, p2)
            nueva_poblacion.extend([mutar(hijo1), mutar(hijo2)])

        poblacion = nueva_poblacion[:tam_poblacion]

    fitnesses = [calcular_fitness(ind) for ind in poblacion]
    mejor_idx = fitnesses.index(max(fitnesses, key=lambda x: x[0]))
    mejor = poblacion[mejor_idx]
    mejor_payback = fitnesses[mejor_idx][1]

    return mejor, mejor_payback, mejor_fitness_hist

# ------------------ MULTI-CORRIDAS ------------------
def correr_simulaciones(n_corridas=10):
    resultados = []
    for _ in range(n_corridas):
        mejor, payback, hist = algoritmo_genetico()
        resultados.append((mejor, payback, hist))

    paybacks = [r[1] for r in resultados]
    mejor_global = min(resultados, key=lambda x: x[1])

    print(f"Mejor configuración global: Potencia={mejor_global[0][0]:.2f} kWp, Inclinación={mejor_global[0][1]:.1f}°, Orientación={mejor_global[0][2]:.1f}°")
    print(f"Payback mínimo encontrado: {mejor_global[1]:.2f} años")
    print(f"Promedio Payback ({n_corridas} corridas): {statistics.mean(paybacks):.2f} años")

    # Histograma de paybacks
    plt.hist(paybacks, bins=10, edgecolor='black')
    plt.title("Distribución de Payback en Corridas")
    plt.xlabel("Payback (años)")
    plt.ylabel("Frecuencia")
    plt.show()

    # Evolución del mejor fitness en la primera corrida
    plt.plot(resultados[0][2])
    plt.title("Evolución del Mejor Fitness por Generación")
    plt.xlabel("Generación")
    plt.ylabel("Fitness")
    plt.show()

# Ejecutar simulación
correr_simulaciones(10)
