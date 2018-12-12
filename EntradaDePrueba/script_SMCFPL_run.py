#!/usr/bin/env python3
"""
    Script de ejemplo para correr el modelo SMCFPL
"""
import smcfpl
import datetime as dt
print( "smcfpl version: {}".format(smcfpl.__version__) )

InFilePath = "DatosEntrada_39Bus_v5.xlsx"    # (str) Ruta relativa de Planilla xls|xlsx con hojas con nombre de los archivos de entrada.
OutFilePath = "./DatosSalida"   # (str) Ruta relativa del Directorio que almacena las salidas. Debe existir previamente.

# Crea el caso de estudio ingresando los parámetros de la simulación
Simulacion = smcfpl.Simulacion(
    InFilePath=InFilePath,
    OutFilePath=OutFilePath,
    Sbase_MVA = 100,
    MaxItCongInter = 1,
    MaxItCongIntra = 1,
    FechaComienzo = '2018-06-01 00:00',  # formato "%Y-%m-%d %H:%M"
    FechaTermino = '2023-05-31 23:00',  # formato "%Y-%m-%d %H:%M"
    NumVecesDem = 5,
    NumVecesGen = 10,
    PerdCoseno = True,
    PEHidSeca = 0.8,
    PEHidMed = 0.5,
    PEHidHum = 0.2,
    ParallelMode = False,
    NumDespParall = 4,
    UsaArchivosParaEtapas=False,
    UsaSlurm=dict(NumNodos=2, NodeWaittingTime=dt.timedelta(seconds=10), ntasks=1, cpu_per_tasks=1),  # considera ssi UsaArchivosParaEtapas=True
)

# Simulacion.run()
