#!/usr/bin/env python3
"""
    Script de ejemplo para correr el modelo SMCFPL
"""
import smcfpl
import datetime as dt
print( "smcfpl version: {}".format(smcfpl.__version__) )

XLSX_FileName = 'InputData_39Bus_v7.xlsx'  # note: for each maintanance row there is a Stage separation. Independent of duplicates.
InFilePath = "./InputData"    # (str) Ruta relativa de Planilla xls|xlsx con hojas con nombre de los archivos de entrada.
OutFilePath = "./OutputData"   # (str) Ruta relativa del Directorio que almacena las salidas. Debe existir previamente.

# Crea el caso de estudio ingresando los parámetros de la simulación
Sim = smcfpl.Simulation(
    XLSX_FileName=XLSX_FileName,
    InFilePath = InFilePath,
    OutFilePath = OutFilePath,
    Sbase_MVA = 100,
    MaxNumVecesSubRedes = 1,  # maximun number allowed up to create sub-nets from Inter-Congestions
    MaxItCongIntra = 10,
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
    NumParallelCPU = None,  # Puede ser None: No usa paralelismo en escritura ni lectura ni cálculo, 'Max' para
    # NumParallelCPU = 'Max',  # Puede ser False: No usa paralelismo en escritura ni lectura ni cálculo, 'Max' para
    # utilizar todos lo procesadores fisicos, o un integer para modificar el tamaño de la pool
    UsaSlurm=False,
    # UsaSlurm = dict(NumNodos=2, NodeWaittingTime=dt.timedelta(seconds=10), ntasks=1, cpu_per_tasks=2),  # False para no ser considerado
    Working_dir = '.',
    UseTempFolder = True,  # create a folder called 'TempData' in 'Working_dir'.
    RemovePreTempData = False,  # only considered if UseTempFolder == True. Beware! 'TempData' directory will be completyle errased.
    UseRandomSeed = 42_468,  # set randomness to predictable value (always same result) or None
    # UseRandomSeed = None,  # set randomness to predictable value (always same result) or None
)

# Simulacion.run(delete_TempData=False)
Sim.run()
