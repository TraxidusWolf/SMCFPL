"""
Script en python que hace llamada como tipo de script bash, para llamar otro script python en multiples nodos.
"""
from subprocess import run as sp__run, PIPE as sp__PIPE
from datetime import timedelta as dt__timedelta, datetime as dt__datetime
from os import sep as os__sep, getcwd as os__getcwd
from os import listdir as os__listdir
from os.path import exists as os__path__exists
from sys import path as sys__path
from multiprocessing import cpu_count as mu__cpu_count, Pool as mu__Pool
import time

import logging
logging.basicConfig(level=logging.DEBUG,
                    format="[%(levelname)s][%(asctime)s][%(filename)s:%(funcName)s] - %(message)s")
logger = logging.getLogger()


def send_work( NNodos=1, WTime=dt__timedelta(days=0, hours=0, minutes=5, seconds=0), NTasks=1,
               ntasks_per_node=1, CPUxTask=1, SMCFPL_dir=os__getcwd(), TempData_dir=os__getcwd(),
               DirsUsar=[], NumTrbjsHastaAhora=1, NTotalCasos=1, StageIndexesList=[],
               NumParallelCPU=False, MaxItCongInter=1, MaxItCongIntra=1 ):
    """
        Por cada llamada a esta función se ejecuta la linea de comando que llama al archivo
        'NucleoCalculo.py' de la biblioteca SMCFPL de python 3.6. a ejecutarse en los Nodos.
        Espera los resultados de su ejecución según los parámetros (argumentos) que acompañan
        a este función.
    """
    logger.debug("!Enviando grupo de casos {}/{} a nodos...".format(NumTrbjsHastaAhora + len(DirsUsar), NTotalCasos))
    # Agrega al path la ruta del directorio que contiene la biblioteca de SMCFPL python
    sys__path.append(SMCFPL_dir)

    # Inicializa precuela del comando para agregar trabajos a slurm. Lista de valores a pasar
    sbatch_cmd = ["sbatch", "-J", "SMCFPL"]

    # Inicializa contador de trabajos realizados
    ContadorCasos = NumTrbjsHastaAhora

    # Determina parámetros de paralelismo en caso de 'NumParallelCPU' no se False
    if bool(NumParallelCPU):
        if isinstance(NumParallelCPU, int):
            Ncpu = NumParallelCPU
        elif NumParallelCPU == 'Max':
            Ncpu = mu__cpu_count()
        logger.info("Enviando casos en paralelo. Utilizando {} procesos simultáneos.".format(Ncpu))
        Pool = mu__Pool(Ncpu)
        Results = []

    # Agrega al Pool simultáneamente varios casos, para enviarlos y esperarlos en paralelo
    # Envía simultáneamente varios casos en serie
    for FolderName in DirsUsar:
        print("FolderName:", FolderName)
        Hydrology = FolderName.split('_')[0]
        print("Hydrology:", Hydrology)
        # -- Crea argumentos de la función Calcular --
        # CasoNum
        Argument1 = str(ContadorCasos)
        # Hidrology
        Argument2 = '"' + Hydrology + '"'
        # Grillas
        Argument3 = "{{StageNum: from_pickle('{0}'+'{1}'+'Grid_Eta{{}}.p'.format(StageNum)) for StageNum in {2} }}".format(
            TempData_dir + os__sep + FolderName,
            os__sep,
            StageIndexesList)  # Double curly brackets for scape them
        # StageIndexesList
        Argument4 = str(StageIndexesList)
        # DF_ParamHidEmb_hid
        Argument5 = """read_csv("{}ParamHidEmb.csv", index_col=[0,1]).loc["{}", :]""".format(
            TempData_dir + os__sep,
            Hydrology)
        # DF_seriesconf
        Argument6 = """read_csv("{}seriesconf.csv", index_col=[0])""".format(
            TempData_dir + os__sep)
        # MaxItCongInter
        Argument7 = MaxItCongInter
        # MaxItCongIntra
        Argument8 = MaxItCongIntra
        # File_Caso
        Argument9 = '"' + TempData_dir + os__sep + FolderName + '"'
        # in_node
        Argument10 = 'True'
        Arguments = "{},{},{},{},{},{},{},{},{},{}".format(Argument1, Argument2, Argument3,
                                                           Argument4, Argument5, Argument6,
                                                           Argument7, Argument8, Argument9,
                                                           Argument10)
        # -- --

        # Complementa comando sbatch
        sbatch_cmd += ["-D", "{OsSep}data{OsSep}{cwd}".format(OsSep=os__sep, cwd=TempData_dir)]
        sbatch_cmd += ["-N", "{NNodos}".format(NNodos=NNodos)]
        sbatch_cmd += ["-n", "{ntasks}".format(ntasks=NTasks)]
        sbatch_cmd += ["---ntasks-per-node", "{ntasks_per_node}".format(ntasks_per_node=ntasks_per_node)]
        sbatch_cmd += ["-c", "{cpu_per_task}".format(cpu_per_task=CPUxTask)]
        sbatch_cmd += ["-o", "outFile_{file_name}-{FileNum}_JId".format(file_name='NucleoCalculo', FileNum=ContadorCasos)]
        sbatch_cmd += ["-e", "errFile_{file_name}-{FileNum}_JId".format(file_name='NucleoCalculo', FileNum=ContadorCasos)]
        sbatch_cmd += ["--wrap"]
        # Crea string de comando a ejecutar con sbatch en los nodos según caso de 'FolderName'
        CMD_execute = "module load python/3.6.1; python -c "
        CMD_execute += "'from smcfpl.NucleoCalculo import calc;"
        CMD_execute += "from pandas import read_csv;"  # requerido para leer archivos generales
        CMD_execute += "calc({Args})'".format(Args=Arguments)
        #
        #
        # debugg. Sobrescribe anteriores
        sbatch_cmd = ["python3", "-c"]
        CMD_execute = "from smcfpl.NucleoCalculo import calc;"
        CMD_execute += "from pandas import read_csv;"  # requerido para leer archivos generales
        CMD_execute += "from json import load;"  # requerido para leer archivos JSON (ExtraData)
        CMD_execute += "from pandapower import from_pickle;"  # requerido para leer grillas PandaPower
        CMD_execute += "calc({Args})".format(Args=Arguments)
        #
        #

        # junta listas de ejecución para comando completo
        CMD_final = sbatch_cmd + [CMD_execute]

        if bool(NumParallelCPU):  # Ejecución paralelo. Llena el Pool
            Results.append( Pool.apply_async( EjecutaComando,
                                              ( CMD_final, ContadorCasos,
                                                NTotalCasos, WTime,
                                                SMCFPL_dir, TempData_dir,
                                                FolderName)
                                              ))
        else:  # Ejecución serie. Corre directamente
            Salida_cmd = EjecutaComando( CMD_final, ContadorCasos,
                                         NTotalCasos, WTime,
                                         SMCFPL_dir, TempData_dir,
                                         FolderName)
            print("Salida_cmd:", Salida_cmd)

        ContadorCasos += 1

    if bool(NumParallelCPU):  # Ejecución paralelo. Obtiene resultados del Pool
        for result in Results:
            Salida_cmd = result.get()
            print("Salida_cmd:", Salida_cmd)
    logger.debug("!Grupo de casos {}/{} en nodos ha finalizado.".format(NumTrbjsHastaAhora + len(DirsUsar), NTotalCasos))
    return 1
# ##################################################################################################################
#     # Agrega al path la ruta del directorio que contiene la biblioteca de SMCFPL python
#     sys__path.append(SMCFPL_dir)
#     # Crea string de argumentos para la función Calcular
#     Arguments = ", ".join(ArgsFunc)
#     # Crea comando para agregar trabajos a slurm
#     sbatch_cmd = """sbatch -J SMCFPL  \
#     -D {OsSep}data{OsSep}{cwd}  \
#     -N {NNodos}  \
#     -n {ntasks}  \
#     -c {cpu_per_task}  \
#     -o outFile_{file_name}-{FileNum}_JId  \
#     -e errFile_{file_name}-{FileNum}_JId \
#     --wrap "module load python/3.6.1; python -c 'from smcfpl.NucleoCalculo import Calcular; Calcular({Args})'"
#     """
#     # --wrap "module load python/3.6.1; python {file_name}.py {Args}"
#     # --wait flag helps to wait for node to finish before continue. Version slurm 18.0
#     comando = sbatch_cmd.format( OsSep=os__sep,
#                                  cwd=os__sep.join( os__getcwd().split(os__sep) ),
#                                  NNodos=NNodos,  # múltiples nodos son usados para llamar varias veces
#                                  ntasks=NTasks,
#                                  cpu_per_task=CPUxTask,
#                                  file_name='NucleoCalculo',
#                                  FileNum=,
#                                  Args=Arguments,
#                                  )

#     # Buena práctica es asegurar que la cantidad de nodos están disponibles y reservarlos. (aquí no se hace)
#     sinfo_cmd = "sinfo -N | grep idle | wc -l"  # cuenta el número de nodos habilitados.
#     NodosDisponibles = int( sp__run([sinfo_cmd], shell=True, stdout=sp.PIPE).stdout )
#     if NodosDisponibles < NumeroVeces:
#         print("No hay suficientes nodos para ejecutar trabajo al mismo tiempo! ..esperando")
#     print("{} nodos disponibles para usar.".format(NodosDisponibles))
# ##################################################################################################################
# # file_name = "single_print.py"
# # NumeroVeces = 2

# # sbatch_cmd = """sbatch -J SlurmPy  \
# # -D /data/clroa/GSeguel/slurm_pruebas/slurm_python_subprocessing  \
# # -N 1  \
# # -n {ntasks}  \
# # -c {cpu_per_task}  \
# # -o outFile_{file_name}-{NFile}_JId  \
# # -e errFile_{file_name}-{NFile}_JId \
# # --wrap "module load python/3.6.1; python {file_name}"
# # """
# # # --wait flag helps to wait for node to finish before continue. Version slurm 18.0

# # # Buena práctica es asegurar que la cantidad de nodos están disponibles y reservarlos. (aquí no se hace)
# # sinfo_cmd = "sinfo -N | grep idle | wc -l"  # cuenta el número de nodos habilitados.
# # NodosDisponibles = int( sp.run([sinfo_cmd], shell=True, stdout=sp.PIPE).stdout )
# # if NodosDisponibles < NumeroVeces:
# #     print("No hay sufientes nodos para ejecutar trabajo al mismo tiempo! ..esperando")
# # print("{} nodos disponibles para usar.".format(NodosDisponibles))

# # ArchivosSalida = []
# # for n in range(NumeroVeces):
# #     # Repite el sbatch_cmd 'NumeroVeces' veces.
# #     comando = sbatch_cmd.format(file_name=file_name,
# #                                 ntasks=1,
# #                                 cpu_per_task=1,
# #                                 NFile=n,
# #                                 )
# #     OutFile = 'outFile_{file_name}-{NFile}_JId'.format(file_name=file_name, NFile=n)
# #     ErrFile = 'errFile_{file_name}-{NFile}_JId'.format(file_name=file_name, NFile=n)
# #     ArchivosSalida.append( (OutFile, ErrFile) )
# #     #print(comando)
# #     sp.call([comando], shell=True)  # ejecuta comando

# # # Espera hasta un periodo determinado hasta que aparezcan todos los archivos de otros nodos. Revisa path cada algunos segundos.
# # TAhora = dt.datetime.now()
# # TMax = dt.timedelta(days=0, hours=0, minutes=5, seconds=0)
# # print("Esperando respuesta de Nodos...")
# # while dt.datetime.now() - TAhora < TMax:
# #     CondiSalida = [ os.path.exists(ArchE) and os.path.exists(ArchE) for ArchO, ArchE in ArchivosSalida ]
# #     if all(CondiSalida):    # todos son verdaderos es un ^
# #         print("Tiempo de espera agotado")  # raise ??
# #         break
# #     else:
# #         # Notar que existe un límite de tiempo para esperar los archivos.
# #         time.sleep(1)   # segundos
# # print("Script Finalizado!!")


def EjecutaComando(comando, CasoNum, TotalCasos, WTime, SMCFPL_dir, TempData_dir, FolderName):
    """
        Utiliza biblioteca subprocess para ejecutar el comando con método run(..., shell=False)
        :param comando: Comando de shell, separado por espacios en formato de lista, i.e., ["ls", "-l", "-a"]
        :type comando: lista

        :param CasoNum: Número del caso que se está ejecutando. Sirve para debugging.
        :type CasoNum: integer

        :param TotalCasos: Número total de casos que se van a ejecutar. Sirve para debugging.
        :type TotalCasos: integer

        :param WTime: Tiempo máximo de espera antes de descartar el caso enviado.
        :type WTime: datetime.timedelta

        Retorna decoded output from comando.
    """
    logger.debug("Enviando caso a nodos {}/{} ...".format(CasoNum, TotalCasos))
    print("Caso:", CasoNum, "CMD:", comando)
    # Ejectuar comando sbatch. Output contiene stdout y strerr
    Output = sp__run(comando, shell=False, stdout=sp__PIPE, stderr=sp__PIPE)
    StdOut = Output.stdout.decode('utf-8')
    StdErr = Output.stderr.decode('utf-8')
    print("StdOut:", StdOut, "type:", type(StdOut))
    print("StdErr:", StdErr, "type:", type(StdErr))
    if StdErr:
        logger.warn("Caso {}/{} no se pudo enviar...".format(CasoNum, TotalCasos))

    if StdOut:
        logger.debug("Caso {}/{} enviado. Esperando respuesta...".format(CasoNum, TotalCasos))

    # Espera hasta un periodo determinado hasta que aparezcan todos los archivos en el directorio correspondiente.
    TAhora = dt__datetime.now()
    # Esperando respuesta de Nodos...
    while dt__datetime.now() - TAhora < WTime:
        # ArchivosSalida = os__listdir(TempData_dir + os__sep + FolderName)
        CondSalida = os__path__exists(TempData_dir + os__sep + str(CasoNum)) and os__path__exists(TempData_dir + os__sep + str(CasoNum))
        if CondSalida:    # todos son verdaderos es un ^
            print("Archivo encontrado!:", CondSalida)
            break
        else:
            # Notar que existe un límite de tiempo para esperar los archivos.
            time.sleep(1)   # segundos

    # read files outputed

    logger.debug("Caso {}/{} completados!".format(CasoNum, TotalCasos))
    return StdOut
