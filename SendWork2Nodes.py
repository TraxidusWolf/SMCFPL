"""
Script en python que hace llamada como tipo de script bash, para llamar otro script python en multiples nodos.
"""
from subprocess import call as sp__call
from datetime import timedelta as dt__timedelta, datetime as dt__datetime
from time import sleep as time__sleep
from os.path import exists as os__path__exists, abspath as os__path__abspath, dirname as os__path__dirname


def Send( NNodos=1, WTime=dt__timedelta(days=0, hours=0, minutes=5, seconds=0), NTasks=1, CPUxTask=1):
    pass


# file_name = "single_print.py"
# NumeroVeces = 2

# sbatch_cmd = """sbatch -J SlurmPy  \
# -D /data/clroa/GSeguel/slurm_pruebas/slurm_python_subprocessing  \
# -N 1  \
# -n {ntasks}  \
# -c {cpu_per_task}  \
# -o outFile_{file_name}-{NFile}_JId  \
# -e errFile_{file_name}-{NFile}_JId \
# --wrap "module load python/3.6.1; python {file_name}"
# """
# # --wait flag helps to wait for node to finish before continue. Version slurm 18.0

# # Buena práctica es asegurar que la cantidad de nodos están disponibles y reservarlos. (aquí no se hace)
# sinfo_cmd = "sinfo -N | grep idle | wc -l"  # cuenta el número de nodos habilitados.
# NodosDisponibles = int( sp.run([sinfo_cmd], shell=True, stdout=sp.PIPE).stdout )
# if NodosDisponibles < NumeroVeces:
#     print("No hay sufientes nodos para ejecutar trabajo al mismo tiempo! ..esperando")
# print("{} nodos disponibles para usar.".format(NodosDisponibles))

# ArchivosSalida = []
# for n in range(NumeroVeces):
#     # Repite el sbatch_cmd 'NumeroVeces' veces.
#     comando = sbatch_cmd.format(file_name=file_name,
#                                 ntasks=1,
#                                 cpu_per_task=1,
#                                 NFile=n,
#                                 )
#     OutFile = 'outFile_{file_name}-{NFile}_JId'.format(file_name=file_name, NFile=n)
#     ErrFile = 'errFile_{file_name}-{NFile}_JId'.format(file_name=file_name, NFile=n)
#     ArchivosSalida.append( (OutFile, ErrFile) )
#     #print(comando)
#     sp.call([comando], shell=True)  # ejecuta comando

# # Espera hasta un periodo determinado hasta que aparezcan todos los archivos de otros nodos. Revisa path cada algunos segundos.
# TAhora = dt.datetime.now()
# TMax = dt.timedelta(days=0, hours=0, minutes=5, seconds=0)
# print("Esperando respuesta de Nodos...")
# while dt.datetime.now() - TAhora < TMax:
#     CondiSalida = [ os.path.exists(ArchE) and os.path.exists(ArchE) for ArchO, ArchE in ArchivosSalida ]
#     if all(CondiSalida):    # todos son verdaderos es un ^
#         print("Tiempo de espera agotado")  # raise ??
#         break
#     else:
#         # Notar que existe un límite de tiempo para esperar los archivos.
#         time.sleep(1)   # segundos
# print("Script Finalizado!!")
