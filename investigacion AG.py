import random
import math
import json
import csv
import statistics
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
#                 PARÁMETROS
# ============================================================
COSTO_POR_CASA = 6_000_000
CONSUMO_ANUAL = 3500                 # kWh/año por casa
TARIFA_COMPRA = 200                  # $/kWh
TARIFA_INYECCION_SUBSIDIO = 74       # $/kWh primeros SUBSIDIO_ANIOS
TARIFA_INYECCION_POST = 10           # $/kWh después
HSP = 4.5
PR_BASE = 0.80
LATITUD = 32
ANIOS_SIMULACION = 20
SUBSIDIO_ANIOS = 6

# Configuración barrio
N_CASAS = 50
CONSUMO_TOTAL = CONSUMO_ANUAL * N_CASAS
COSTO_TOTAL = COSTO_POR_CASA * N_CASAS

# Crecimientos (escenarios)
GROWTH_COMPRA = 0.0
GROWTH_INY = 0.0

# Degradación anual
DEGRAD_ANUAL = 0.006  # 0.6%/año

# ============================================================

POT_MIN, POT_MAX = 1.0, 3.5
INCL_MIN, INCL_MAX = 0.0, 90.0
AZIM_MIN, AZIM_MAX = 0.0, 360.0
T_MIN, T_MAX = 0.0, 45.0

# ============================================================

GEN_MENSUAL_FORMA = [340,330,300,230,150,100,110,230,310,320,330,310]
peso_total = sum(GEN_MENSUAL_FORMA)
PESOS_MES = [x / peso_total for x in GEN_MENSUAL_FORMA]

SUC_MES = [0.02,0.03,0.04,0.06,0.08,0.10,0.10,0.08,0.06,0.04,0.03,0.02]

TEMP_COEF_PERC_PER_C = -0.004
NOCT = 45.0

# Disponibilidad (cortes verano)
OUTAGE_DIAS_MES = [3, 5, 5, 3, 1, 0, 0, 0, 1, 2, 3, 4]
OUTAGE_MIN_PROM = [25,30,28,18,12,8,8,8,10,12,18,22]
DIAS_MES = [31,28,31,30,31,30,31,31,30,31,30,31]
DISPON_BASE = 0.995

def factor_suciedad_anual():
    return sum((1.0 - SUC_MES[m]) * PESOS_MES[m] for m in range(12))

def factor_disponibilidad_mes(m):
    dias_corte = OUTAGE_DIAS_MES[m]
    minutos_dia = OUTAGE_MIN_PROM[m]
    perdida = (dias_corte * minutos_dia) / (DIAS_MES[m] * 24 * 60)
    f = (1.0 - perdida) * DISPON_BASE
    return max(0.90, min(1.0, f))

def factor_disponibilidad_anual():
    return sum(factor_disponibilidad_mes(m) * PESOS_MES[m] for m in range(12))

def factor_penal_total_estacional():
    return factor_suciedad_anual() * factor_disponibilidad_anual()

# ============================================================

def factor_inclinacion(inclin, lat_opt=LATITUD):
    desvio = abs(inclin - lat_opt)
    return max(0.60, 1.0 - 0.02 * desvio)

def factor_orientacion(azim):
    a = azim % 360.0
    d = min(a, 360.0 - a)
    return max(0.0, math.cos(math.radians(d)))

def eta_global(inclin, azim, T):
    T_cell = T + (NOCT - 20.0) * 1.0  
    f_temp = 1.0 + TEMP_COEF_PERC_PER_C * (T_cell - 25.0)
    f_temp = max(0.80, min(1.02, f_temp))
    return PR_BASE * factor_inclinacion(inclin) * factor_orientacion(azim) * factor_penal_total_estacional() * f_temp

# ============================================================

HOURS = 24
ARCH_P_DIURNO = 0.45  # % de casas con perfil diurno

BASE_PROFILES = {
    "res_diurno": [
        0.60,0.55,0.50,0.50,0.55,0.70,0.85,0.95,1.00,0.95,0.90,0.85,
        0.80,0.85,0.95,1.10,1.25,1.15,0.95,0.85,0.80,0.75,0.70,0.65
    ],
    "res_vespertino": [
        0.45,0.42,0.42,0.42,0.45,0.55,0.70,0.85,0.95,0.95,0.95,0.90,
        0.90,1.00,1.15,1.40,1.65,1.60,1.30,1.00,0.80,0.65,0.55,0.48
    ],
}

def sample_house_profile():
    kind = "res_diurno" if random.random() < ARCH_P_DIURNO else "res_vespertino"
    base = BASE_PROFILES[kind][:]
    prof = [max(0.05, v * (1.0 + random.uniform(-0.10, 0.10))) for v in base]
    s = sum(prof)
    return [v / s for v in prof]

def load_profiles_barrio(consumo_anual, n_casas):
    consumo_diario = consumo_anual / 365.0
    casas = [sample_house_profile() for _ in range(n_casas)]
    cargas_mes = {}
    for m in range(12):
        dias = DIAS_MES[m]
        hourly = [0.0]*HOURS
        for h in range(HOURS):
            total_h = sum(p[h]*consumo_diario for p in casas)
            hourly[h] = total_h * dias
        cargas_mes[m] = hourly
    return cargas_mes

# ============================================================

def pv_shape_hourly(az_deg):
    a = az_deg % 360.0
    shift = ((a - 180.0) / 180.0) * 1.5
    mu = 12.0 + shift
    sigma = 3.5
    out = [math.exp(-0.5*((h-mu)/sigma)**2) for h in range(HOURS)]
    s = sum(out) or 1.0
    return [v/s for v in out]

def pv_profiles_barrio(pot_kwp, inclin, azim, T):
    shape = pv_shape_hourly(azim)
    eta = eta_global(inclin, azim, T)
    pv_mes = {}
    for m in range(12):
        E_mes_casa = pot_kwp * HSP * 365 * eta * PESOS_MES[m]
        hourly = [E_mes_casa * sh for sh in shape]
        pv_mes[m] = [v * N_CASAS for v in hourly]
    return pv_mes

# ============================================================

def barrio_autoconsumo(pot_kwp, inclin, azim, T):
    load_m = load_profiles_barrio(CONSUMO_ANUAL, N_CASAS)
    pv_m   = pv_profiles_barrio(pot_kwp, inclin, azim, T)
    Eauto = Einj = Egen = 0.0
    for m in range(12):
        for h in range(HOURS):
            pv, ld = pv_m[m][h], load_m[m][h]
            Egen += pv
            Eauto += min(pv, ld)
            Einj  += max(pv-ld,0)
    return Eauto, Einj, Egen

# ============================================================

def evaluar_config(pot_kwp, inclin, azim, T):
    Eauto, Einj, Egen_total = barrio_autoconsumo(pot_kwp, inclin, azim, T)
    ahorro_acum = 0.0
    payback_acum = None
    for anio in range(1, ANIOS_SIMULACION+1):
        t_compra = TARIFA_COMPRA*((1+GROWTH_COMPRA)**(anio-1))
        t_iny = (TARIFA_INYECCION_SUBSIDIO if anio<=SUBSIDIO_ANIOS else TARIFA_INYECCION_POST)
        t_iny *= ((1+GROWTH_INY)**(anio-1))
        degr = (1-DEGRAD_ANUAL)**(anio-1)
        ahorro_anual = (Eauto*degr)*t_compra + (Einj*degr)*t_iny
        if ahorro_anual<=0: break
        if ahorro_acum+ahorro_anual>=COSTO_TOTAL and payback_acum is None:
            restante=COSTO_TOTAL-ahorro_acum
            fr=restante/ahorro_anual
            payback_acum=(anio-1)+fr
        ahorro_acum+=ahorro_anual
    if payback_acum is None: payback_acum=float('inf')
    eta = eta_global(inclin, azim, T)
    return {"payback_acum":payback_acum,"Egen_total":Egen_total,
            "Eauto":Eauto,"Einj":Einj,"eta":eta,"T":T}

def fitness_de(ind):
    det=evaluar_config(*ind)
    pay=det["payback_acum"]
    return ((1/pay) if (math.isfinite(pay) and pay>0) else 0.0,det)

# ============================================================

def nuevo_individuo():
    return [random.uniform(POT_MIN,POT_MAX),
            random.uniform(INCL_MIN,INCL_MAX),
            random.uniform(AZIM_MIN,AZIM_MAX),
            random.uniform(T_MIN,T_MAX)]

def inicializar_poblacion(n): return [nuevo_individuo() for _ in range(n)]

def seleccion_torneo(pobl,fits,k=3):
    ids=random.sample(range(len(pobl)),k)
    return pobl[max(ids,key=lambda i:fits[i][0])]

def crossover(p1,p2):
    punto=random.randint(1,len(p1)-1)
    return p1[:punto]+p2[punto:],p2[:punto]+p1[punto:]

def mutar(ind,tasa=0.25):
    if random.random()<tasa: ind[0]=random.uniform(POT_MIN,POT_MAX)
    if random.random()<tasa: ind[1]=random.uniform(INCL_MIN,INCL_MAX)
    if random.random()<tasa: ind[2]=random.uniform(AZIM_MIN,AZIM_MAX)
    if random.random()<tasa: ind[3]=random.uniform(T_MIN,T_MAX)
    return ind


def algoritmo_genetico(generaciones=100,tam_pob=50,elite=2):
    pobl=inicializar_poblacion(tam_pob)
    best_fit_hist,worst_fit_hist=[],[]
    best_pay_sofar_hist,worst_pay_sofar_hist=[],[]
    best_sofar=float('inf'); worst_sofar=0.0
    for _ in range(generaciones):
        fits=[fitness_de(ind) for ind in pobl]
        best=max(fits,key=lambda x:x[0])
        worst=min(fits,key=lambda x:x[0])
        best_fit_hist.append(best[0]); worst_fit_hist.append(worst[0])
        pays=[x[1]["payback_acum"] for x in fits if math.isfinite(x[1]["payback_acum"])]
        if pays:
            gen_best=min(pays); gen_worst=max(pays)
            best_sofar=min(best_sofar,gen_best)
            worst_sofar=max(worst_sofar,gen_worst)
        best_pay_sofar_hist.append(best_sofar)
        worst_pay_sofar_hist.append(worst_sofar)
        idx_sorted=sorted(range(len(pobl)),key=lambda i:fits[i][0],reverse=True)
        elite_inds=[pobl[i] for i in idx_sorted[:elite]]
        nueva=elite_inds.copy()
        while len(nueva)<tam_pob:
            p1=seleccion_torneo(pobl,fits); p2=seleccion_torneo(pobl,fits)
            h1,h2=crossover(p1,p2)
            nueva.extend([mutar(h1),mutar(h2)])
        pobl=nueva[:tam_pob]
    fits=[fitness_de(ind) for ind in pobl]
    idx_best=max(range(len(fits)),key=lambda i:fits[i][0])
    ind_best=pobl[idx_best]; det_best=fits[idx_best][1]
    return ind_best,det_best,best_fit_hist,worst_fit_hist,best_pay_sofar_hist,worst_pay_sofar_hist

# ============================================================
#                 Corridas + gráficos
# ============================================================
def correr(n_corridas=10,generaciones=100,tam_pob=50):
    resultados=[algoritmo_genetico(generaciones,tam_pob) for _ in range(n_corridas)]
    pay_finales=[det["payback_acum"] for _,det,*_ in resultados]
    finitos=[p for p in pay_finales if math.isfinite(p)]
    if finitos: print(f"Payback medio: {statistics.mean(finitos):.2f} años")
    plt.hist([p if math.isfinite(p) else ANIOS_SIMULACION+1 for p in pay_finales],
             bins=10,edgecolor='black'); plt.title("Distribución Payback (corridas)")
    plt.xlabel("Payback (años)"); plt.ylabel("Frecuencia"); plt.show()
    mejor_res=min(resultados,key=lambda r:r[1]["payback_acum"] if math.isfinite(r[1]["payback_acum"]) else float('inf'))
    (ind_best,det_best,best_fit_hist,worst_fit_hist,best_pay_sofar_hist,worst_pay_sofar_hist)=mejor_res
    print("\n=== Mejor corrida ===")
    print(f"Potencia = {ind_best[0]:.2f} kWp/casa | Tilt = {ind_best[1]:.1f}° | Azimut = {ind_best[2]:.1f}° | T = {ind_best[3]:.1f}°C")
    print(f"Payback = {det_best['payback_acum']:.2f} años" if math.isfinite(det_best['payback_acum']) else "NO recupera en 20 años")
    print(f"Egen = {det_best['Egen_total']:.0f} | Eauto = {det_best['Eauto']:.0f} | Einj = {det_best['Einj']:.0f}")

    load_m = load_profiles_barrio(CONSUMO_ANUAL, N_CASAS)                    
    pv_m   = pv_profiles_barrio(ind_best[0], ind_best[1], ind_best[2], ind_best[3])  

    annual_load = [sum(load_m[m][h] for m in range(12)) for h in range(HOURS)]
    annual_pv   = [sum(pv_m[m][h]   for m in range(12)) for h in range(HOURS)]

    plt.plot(range(HOURS), annual_load, label="Carga barrio (kWh/h, anual)")
    plt.plot(range(HOURS), annual_pv,   label="FV barrio (kWh/h, anual)")
    plt.title("Carga vs Generación FV — Perfil horario anual (mejor individuo)")
    plt.xlabel("Hora del día"); plt.ylabel("Energía (kWh acumulados en el año por hora)")
    plt.legend(); plt.show()
    plt.plot(best_pay_sofar_hist,label="Mejor histórico")
    plt.plot(worst_pay_sofar_hist,label="Peor histórico")
    plt.title("Evolución Payback (histórico)")
    plt.xlabel("Generación"); plt.ylabel("Payback (años)"); plt.legend(); plt.show()
    plt.plot(best_fit_hist,label="Mejor fitness"); plt.plot(worst_fit_hist,label="Peor fitness")
    plt.title("Evolución Fitness"); plt.xlabel("Generación"); plt.ylabel("Fitness"); plt.legend(); plt.show()
    payload={"mejor_configuracion":{"pot_kWp_casa":ind_best[0],"tilt":ind_best[1],"azimut":ind_best[2],"T_amb":ind_best[3]},"metricas":det_best}
    Path("mejor_configuracion.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    with open("evolucion_payback.csv","w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["gen","best_pay","worst_pay","best_fit","worst_fit"])
        for i,(bp,wp,bf,wf) in enumerate(zip(best_pay_sofar_hist,worst_pay_sofar_hist,best_fit_hist,worst_fit_hist),1):
            w.writerow([i,bp,wp,bf,wf])

# ============================================================
# MAIN
# ============================================================
if __name__=="__main__":
    random.seed(42)
    correr(n_corridas=5,generaciones=100,tam_pob=50)
