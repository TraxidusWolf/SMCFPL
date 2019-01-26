#!/usr/bin/env python3
"""
    Script de ejemplo para correr el modelo SMCFPL
"""
import smcfpl
import datetime as dt
print( "smcfpl version: {}".format(smcfpl.__version__) )

XLSX_FileName = 'DatosEntrada_39Bus_v6.xlsx'
InFilePath = "./DatosEntrada"    # (str) Ruta relativa de Planilla xls|xlsx con hojas con nombre de los archivos de entrada.
OutFilePath = "./DatosSalida"   # (str) Ruta relativa del Directorio que almacena las salidas. Debe existir previamente.

# Crea el caso de estudio ingresando los parámetros de la simulación
Simulacion = smcfpl.Simulacion(
    XLSX_FileName=XLSX_FileName,
    InFilePath = InFilePath,
    OutFilePath = OutFilePath,
    Sbase_MVA = 100,
    MaxItCongInter = 1,
    MaxItCongIntra = 1,
    FechaComienzo = '2018-06-01 00:00',  # formato "%Y-%m-%d %H:%M"
    FechaTermino = '2023-05-31 23:00',  # formato "%Y-%m-%d %H:%M"
    NumVecesDem = 2,
    NumVecesGen = 3,
    PerdCoseno = True,
    PEHidSeca = 0.8,  # 0 <= (float) <= 1
    PEHidMed = 0.5,  # 0 <= (float) <= 1
    PEHidHum = 0.2,  # 0 <= (float) <= 1
    DesvEstDespCenEyS = 0.1,  # desviación estándar considerada para el despacho de centrales Embalse y Serie
    DesvEstDespCenP = 0.2,  # desviación estándar considerada para el despacho de centrales Pasada
    NumParallelCPU = False,  # Puede ser False: No usa paralelismo ni lectura ni cálculo, 'Max' para
    # utilizar todos lo procesadores fisicos, o un integer para modificar el tamaño de la pool
    UsaSlurm=False,
    # UsaSlurm = dict(NumNodos=2, NodeWaittingTime=dt.timedelta(seconds=10), ntasks=1, cpu_per_tasks=2, borrar_cache_pre=False),  # False para no ser considerado
    Working_dir = '.',
)

# Simulacion.run(delete_TempData=False)
Simulacion.run(delete_TempData_post = False)
