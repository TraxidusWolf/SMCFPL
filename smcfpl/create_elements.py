from smcfpl.in_out_proc import read_sheets_to_dataframes as smcfpl__in_out_proc__read_sheets_to_dataframes
from smcfpl.in_out_proc import ImprimeBDsGrales as smcfpl__in_out_proc__ImprimeBDsGrales
from smcfpl.in_out_proc import write_BDs_input_case as smcfpl__in_out_proc__write_BDs_input_case
from smcfpl.in_out_proc import dump_BDs_to_pickle as smcfpl__in_out_proc__dump_BDs_to_pickle
from smcfpl.smcfpl_exceptions import *
import smcfpl.aux_funcs as aux_smcfpl
import smcfpl.SendWork2Nodes as SendWork2Nodes
import smcfpl.core_calc as core_calc

from os.path import exists as os__path__exists, isdir as os__path__isdir
from os.path import abspath as os__path__abspath, isfile as os__path__isfile
from os import makedirs as os__makedirs, getcwd as os__getcwd
from os import listdir as os__listdir
from os import sep as os__sep
from sys import executable as sys__executable
from subprocess import run as sp__run, PIPE as sp__PIPE
from pandas import DataFrame as pd__DataFrame
from pandas import concat as pd__concat
from numpy import tile as np__tile
from datetime import datetime as dt
from datetime import timedelta as dt__timedelta
from dateutil import relativedelta as du__relativedelta
from shutil import rmtree as shutil__rmtree
from pickle import load as pickle__load
from pandapower import create_empty_network as pp__create_empty_network, create_buses as pp__create_buses
from pandapower import create_line as pp__create_line, create_std_types as pp__create_std_types
from pandapower import create_transformer as pp__create_transformer, create_transformer3w as pp__create_transformer3w
from pandapower import create_load as pp__create_load, create_gen as pp__create_gen
from pandapower import create_ext_grid as pp__create_ext_grid
from pandapower.topology import unsupplied_buses as pp__topology__unsupplied_buses
from pandapower import drop_inactive_elements as pp__drop_inactive_elements
from pandapower import to_pickle as pp__to_pickle, from_pickle as pp__from_pickle
from multiprocessing import cpu_count as mu__cpu_count, Pool as mu__Pool

import logging
Logging_level = logging.DEBUG
logging.basicConfig(level=Logging_level,
                    format="[%(levelname)s][%(asctime)s][%(filename)s:%(funcName)s] - %(message)s")
logger = logging.getLogger()

# Cambia logger level de pandapower a WARNING
# logging.getLogger("pandapower").setLevel(logging.WARNING)
# Cambia logger level de pandapower al actual
logging.getLogger("pandapower").setLevel(logger.level)
# logging.getLogger("pandapower").setLevel(Logging_level)


class Simulation(object):
    """
        Clase base que contiene los atributos y métodos de la simulación para ejecutar el modelo exitosamente.
        Guarda las base de datos en memoria (pandas dataframe, diccionarios, etc), desde los cuales adquiere los datos para cada etapa. Ojo, durante
        paralelismo se aumentarán los requerimientos de memoria según la cantidad de tareas.
    """

    def __init__(self, XLSX_FileName='', InFilePath='.', OutFilePath='.', Sbase_MVA=100, MaxNumVecesSubRedes=1,
                 MaxItCongIntra=1, FechaComienzo='2018-01-01 00:00', FechaTermino='2019-01-31 23:59',
                 NumVecesDem=1, NumVecesGen=1, PerdCoseno=True, PEHidSeca=0.8, PEHidMed=0.5, PEHidHum=0.2,
                 DesvEstDespCenEyS=0.1, DesvEstDespCenP=0.2, NumParallelCPU=False, UsaSlurm=False,
                 Working_dir=os__getcwd(), UseTempFolder = True, RemovePreTempData=True,
                 smcfpl_dir=os__getcwd(), TempFolderName='TempData', UseRandomSeed=False):
        """
            :param UsaSlurm: Diccionario con parámetros para ejecución en el sistema de colas de slurm. Hace que se ejecuten comandos (con biblioteca subprocess) sbatch
                            propios de slurm para ejecución en varios nodos. Se escriben BD datos en un directorio temporal para ser copiado a cada nodo.
                            Formato: {'NumNodos': (int), 'NodeWaittingTime': (datetime deltatime object), 'ntasks': (int), 'cpu_per_tasks': (int)}
                                    'NumNodos': Número de nodos a utilizar en el cluster.
                                    'NodeWaittingTime': Tiempo de espera máximo de ejecución de los procesos enviados a nodos.
                                    'ntasks': número de tareas a repartirse por nodo.
                                    'cpu-per-tasks': Número de cpu requeridas para cada tarea.
                            En caso de no utilizarse, se debe ingresar valor booleano 'False'.
            :type UsaSlurm: dict

        """
        logger.debug("! Initializating class Simulation(...)  (create_elements.py)...")
        STime = dt.now()
        #
        # Atributos desde entradas
        self.XLSX_FileName = XLSX_FileName  # (str)
        self.InFilePath = InFilePath    # (str)
        self.Sbase_MVA = Sbase_MVA  # (float)
        self.MaxNumVecesSubRedes = MaxNumVecesSubRedes    # (int)
        self.MaxItCongIntra = MaxItCongIntra    # (int)
        self.FechaComienzo = dt.strptime(FechaComienzo, "%Y-%m-%d %H:%M")  # (str) -> datetime
        self.FechaTermino = dt.strptime(FechaTermino, "%Y-%m-%d %H:%M")    # (str) -> datetime
        # verifica que 'FechaTermino' sea posterior en al menos un año a 'FechaComienzo'
        if self.FechaTermino < (self.FechaComienzo + du__relativedelta.relativedelta(years=1)):
            msg = "'FechaTermino' debe ser al menos 1 año posterior a 'FechaComienzo'."
            logger.error(msg)
            raise ValueError(msg)
        self.NumVecesDem = NumVecesDem  # (int)
        self.NumVecesGen = NumVecesGen  # (int)
        self.PerdCoseno = PerdCoseno    # (bool)
        self.PEHidSeca = PEHidSeca  # 0 <= (float) <= 1
        self.PEHidMed = PEHidMed    # 0 <= (float) <= 1
        self.PEHidHum = PEHidHum    # 0 <= (float) <= 1
        self.DesvEstDespCenEyS = DesvEstDespCenEyS  # 0 <= (float) <= 1
        self.DesvEstDespCenP = DesvEstDespCenP  # 0 <= (float) <= 1
        self.NumParallelCPU = NumParallelCPU  #
        if not (isinstance(self.NumParallelCPU, int) | (self.NumParallelCPU is False) | (self.NumParallelCPU == 'Max')):
            # Puede ser False: No usa paralelismo ni lectura ni cálculo, 'Max' para todos los procesadores, o un 'int' tamaño personalizado pool
            msg = "Input 'NumParallelCPU' debe ser integer, False, o 'Max'."
            logger.error(msg)
            raise IOError(msg)
        self.UsaSlurm = UsaSlurm  # (bool)
        self.UseTempFolder = UseTempFolder  # (bool)
        self.RemovePreTempData = RemovePreTempData  # (bool)
        self.TempFolderName = TempFolderName
        self.abs_path_temp = os__path__abspath(Working_dir + os__sep + TempFolderName)
        self.abs_path_smcfpl = os__path__abspath(smcfpl_dir)
        self.abs_InFilePath = os__path__abspath(InFilePath)
        self.abs_OutFilePath = os__path__abspath(OutFilePath)
        if isinstance(UseRandomSeed, int) and not isinstance(UseRandomSeed, bool):
            self.UseRandomSeed = UseRandomSeed
        else:
            msg = "'UseRandomSeed' must be an integer."
            raise IOError(msg)
            logger.error(msg)
        self.BD_file_exists = False

        # Verifies if it's necesary to import or calculate data. Added them to attributes
        FileFormat = 'pickle'  # tipes of files to be read
        _ReturnedBD = self.ManageTempData(FileFormat)
        # BD_Etapas
        # BD_DemProy
        # BD_Hidrologias_futuras
        # BD_TSFProy
        # BD_MantEnEta
        # BD_RedesXEtapa
        # BD_ParamHidEmb
        # BD_HistGenRenovable
        # BD_seriesconf
        self.BD_Etapas, self.BD_DemProy = _ReturnedBD[0], _ReturnedBD[1]
        self.BD_Hidrologias_futuras, self.BD_TSFProy = _ReturnedBD[2], _ReturnedBD[3]
        self.BD_MantEnEta, self.BD_RedesXEtapa = _ReturnedBD[4], _ReturnedBD[5]
        self.BD_ParamHidEmb, self.BD_HistGenRenovable = _ReturnedBD[6], _ReturnedBD[7]
        self.BD_seriesconf = _ReturnedBD[8]
        _ReturnedBD = None  # delete content for memory space

        if self.UseTempFolder and not self.BD_file_exists:
            msg = "Writing databases to file as requested (UseTempFolder=True) "
            self.write_DataBases_to_pickle(self.abs_path_temp)

        print('self.BD_Etapas:\n', self.BD_Etapas)
        # Numero total de etapas
        self.NEta = self.BD_Etapas.shape[0]

        #
        # Obtiene partes del diccionario ExtraData por etapa
        self.TecGenSlack = [d['ExtraData']['TecGenSlack'] for d in self.BD_RedesXEtapa.values()]
        # Crea lista del Número de Cargas en cada Etapa/Grid
        self.DictTypoCargasEta = { k: v['PandaPowerNet']['load'][['type']] for k, v in self.BD_RedesXEtapa.items() }
        # Crea lista del Número de Unidades de Generación en cada Etapa/Grid
        # self.ListNumGenNoSlack = [d['ExtraData']['NumGenNoSlack'] for d in self.BD_RedesXEtapa.values()]
        # Crea lista del tipos (Número de Unidades intrínseco) de Generación en cada Etapa/Grid
        self.DictTiposGenNoSlack = { k: d['ExtraData']['Tipos'] for k, d in self.BD_RedesXEtapa.items() }
        # # Obtiene Grillas de cada etapa en diccionario (1-indexed)
        # self.DictSEPBrutoXEtapa = {k: v['PandaPowerNet'] for k, v in self.BD_RedesXEtapa.items()}

        # checks for existance of OutFilePath directory
        if not os__path__isdir(self.abs_OutFilePath):
            # if not, create it
            msg = "'OutFilePath' directory does not exist. Creating it..."
            logger.info(msg)
            os__makedirs(self.abs_OutFilePath)

        # Timing
        RunTime = dt.now() - STime
        minutes, seconds = divmod(RunTime.seconds, 60)
        hours, minutes = divmod(minutes, 60)
        msg = "Inicialization of Simulation class finished after {} [hr], {} [min] and {} [s].".format(
            hours, minutes, seconds + RunTime.microseconds * 1e-6)
        logger.info(msg)
        logger.debug("! Initialization class Simulation(...) Finished! (create_elements.py)")

    def run(self, delete_TempData_post=True):
        logger.debug("Running method Simulation.run()...")
        STime = dt.now()

        # Comienza con la ejecución de lo cálculos
        if self.UsaSlurm:
            logger.info("Solving cases in NODE MODE.")
            """
                    ##    ##  #######  ########  ########    ##     ##  #######  ########  ########
                    ###   ## ##     ## ##     ## ##          ###   ### ##     ## ##     ## ##
                    ####  ## ##     ## ##     ## ##          #### #### ##     ## ##     ## ##
                    ## ## ## ##     ## ##     ## ######      ## ### ## ##     ## ##     ## ######
                    ##  #### ##     ## ##     ## ##          ##     ## ##     ## ##     ## ##
                    ##   ### ##     ## ##     ## ##          ##     ## ##     ## ##     ## ##
                    ##    ##  #######  ########  ########    ##     ##  #######  ########  ########

                # En caso de declarar uso de Slurm, se trabaja con archivos en lugar de mantener la base de datos
                general en memoria RAM, incluyendo todo lo necesario (Reduce velocidad). Estos archivos son
                necesario para ser copiado a los nodos. Escribe en un directorio de trabajo temporal (desde
                de donde se ejecutó el script).

                # Posibilidad de paralelismo mediante nodos en una configuración de un 'High-Performance Computer'
                con configuración beowulf y administrador de colas slurm.

                # Envía en paralelo múltiples grupos de casos (todas etapas bajo mismas condiciones de operación)
                según nodos se hubieran solicitado. Las potencias de grillas son alteradas con la probabilidad
                correspondiente justo antes de ser escritas.
            """

            # Se hace la suposición que de existir, contiene toda la información necesaria
            # if os__path__exists(self.abs_path_temp) and self.UsaSlurm['borrar_cache_pre']:
            #     # Si existe un directorio con el nombre del correspondiente temporal,
            #     # se elimina con warning. Se asegura que no existan conflictos con datos antiguos
            #     logger.warn("Eliminando directorio completo {}".format(self.TempFolderName))
            #     shutil__rmtree(self.abs_path_temp)
            #     # Crea el directorio temporal
            #     os__makedirs(self.abs_path_temp)
            # elif not os__path__exists(self.abs_path_temp):
            #     # verifica que exista directorio, de lo contrario lo crea.
            #     os__makedirs(self.abs_path_temp)
            ##########################################################
            # if os__path__exists(self.abs_path_temp):
            #     # Si existe un directorio con el nombre del correspondiente temporal,
            #     # se elimina con warning. Se asegura que no existan conflictos con datos antiguos
            #     if self.UsaSlurm['borrar_cache_pre']:
            #         logger.warn("Eliminando directorio completo {}".format(self.TempFolderName))
            #         shutil__rmtree(self.abs_path_temp)
            #         # Crea el directorio temporal
            #         os__makedirs(self.abs_path_temp)
            # else:
            #     # verifica que exista directorio, de lo contrario lo crea.
            #     os__makedirs(self.abs_path_temp)
            ##########################################################

            # Imprime las  BDs generales a los distintos casos. Usa directorio temporal 'self.abs_path_temp'
            smcfpl__in_out_proc__ImprimeBDsGrales(self)

            # Crea lista con hidrologías de interés en los datos
            ListaHidrologias = ['Humeda', 'Media', 'Seca']
            # Calcula el total se simulaciones de casos (Número de etapas según condición de operación)
            NTotalCasos = self.NumVecesDem * self.NumVecesGen * len(ListaHidrologias)
            # Calcula casos por nodo. Divide casos entre total de nodos pedidos (NCasos // NNodos; integer)
            NTrbjsXNodo = NTotalCasos // self.UsaSlurm['NumNodos']  # Número de casos por nodo
            # Calcula Número de casos restantes (NTrabajo % NNodos; resto, integer)
            NTrbjsRest = NTotalCasos % self.UsaSlurm['NumNodos']  # Número de casos por nodo

            # Ajuste parámetros para escribir archivos en paralelo
            if bool(self.NumParallelCPU):
                # Parámetros de paralelismo
                if isinstance(self.NumParallelCPU, int):
                    Ncpu = self.NumParallelCPU
                elif self.NumParallelCPU == 'Max':
                    Ncpu = mu__cpu_count()
                logger.info("Escribiendo en paralelo. Utilizando máximo {} procesos simultáneos.".format(Ncpu))
                Pool = mu__Pool(Ncpu)
                Results = []

            #
            # Para cada caso escribe los datos en directorio temporal
            #
            # Junta y escribe en sub-directorios cada caso (Serie de etapas) como un
            # directorio, el cual contiene las base de datos independientes (Grilla)
            for HidNom in ListaHidrologias:
                # ---- Ajusta base de datos según Hidrología ----
                # PEs por etapa asociadas a la Hidrología en cuestión
                DF_PEsXEtapa = self.BD_Hidrologias_futuras[['PE ' + HidNom + ' dec']]
                # Parámetros de cota-costo en hidrología actual. Indices: ['b', 'CVmin', 'CVmax', 'CotaMin', 'CotaMax']
                DF_ParamHidEmb_hid = self.BD_ParamHidEmb.loc[HidNom]  # Accede al DataFrame del DataFrame Multindex
                DF_CotasEmbalsesXEtapa = pd__DataFrame(np__tile(DF_PEsXEtapa, (1, len(DF_ParamHidEmb_hid.columns))),
                                                       columns = DF_ParamHidEmb_hid.columns,
                                                       index = DF_PEsXEtapa.index)
                # Obtiene las cotas Máximas y Mínimas de los embalses dada la hidrología actual
                CotasMax = DF_ParamHidEmb_hid.loc['CotaMax'].values
                CotasMin = DF_ParamHidEmb_hid.loc['CotaMin'].values
                # Calcula el porcentaje lineal dado por la PE en cada etapa, desde CotaMin hasta CotaMax. De cada Embalse
                DF_CotasEmbalsesXEtapa = (CotasMax - CotasMin) * (1 - DF_CotasEmbalsesXEtapa) + CotasMin
                # ----
                for NDem in range(1, self.NumVecesDem + 1):
                    for NGen in range(1, self.NumVecesGen + 1):
                        # Datos extra requeridos, no en instancia de clase
                        InputList = {  # permite incorporar más a futuro
                            'DF_PEsXEtapa': DF_PEsXEtapa  # pandas DataFrame
                        }

                        # permite generar nombre del sub-directorio '{HidNom}_D{NDem}_G{NGen}'
                        CaseIdentifier = (HidNom, NDem, NGen)  # post-morten tag
                        if bool(self.NumParallelCPU):  # En paralelo
                            # Agrega la función con sus argumentos al Pool para ejecutarla en paralelo
                            Results.append( Pool.apply_async(smcfpl__in_out_proc__write_BDs_input_case, (self, CaseIdentifier, InputList)) )
                        else:
                            # (En serie) Aplica directamente para cada caso
                            smcfpl__in_out_proc__write_BDs_input_case(self, CaseIdentifier, InputList)

            if bool(self.NumParallelCPU):  # En paralelo
                print("Ejecutando paralelismo escritura BDs...")
                # Obtiene los resultados del paralelismo, en caso de existir
                for result in Results:
                    # No retorna, pero ejecuta escribiendo a disco
                    result.get()

            #
            # ENVÍA A LOS NODOS EL COMANDO DE CALCULO PARA USARSE CON LA DATA ESCRITA
            #
            logger.info("Enviando casos a nodos...")
            # Crea lista de números de trabajos para iterar en el for loop
            ListaNumTrbjs = [NTrbjsXNodo] * self.UsaSlurm['NumNodos']
            if NTrbjsRest:  # si existe una división no entera (resto de división != 0)
                ListaNumTrbjs += [NTrbjsRest]
            # identifica cuantos sub-directorios existen. No existen más directorios de los que se escriben
            DirsList = []
            for name in os__listdir(self.abs_path_temp):
                if os__path__isdir(self.abs_path_temp + os__sep + name):
                    DirsList.append(name)
            c = 1  # contador necesario para mantener última posición en la lista de dirs
            logger.info("Dividiendo casos en {} nodos, cada uno con {} trabajos respectivamente.".format(len(ListaNumTrbjs), ListaNumTrbjs))
            # Envía grupos de trabajos/casos (en serie) a los nodos
            for NumTrbjs in ListaNumTrbjs:
                # Obtiene los 'NumTrbjs' primeros encontrados y utiliza sus BDs
                # DirsUsar = DirsList[c - 1: c - 1 + NumTrbjs]
                # Res = SendWork2Nodes.Send( NNodos=1,
                #                            WTime=self.UsaSlurm['NodeWaittingTime'],
                #                            NTasks=NumTrbjs,
                #                            ntasks_per_node=NumTrbjs  # define
                #                            CPUxTask=1  # self.UsaSlurm['cpu_per_tasks'],  # each task is single threaded
                #                            SMCFPL_dir=self.abs_path_smcfpl,
                #                            TempData_dir=self.abs_path_temp,
                #                            DirsUsar=DirsUsar,
                #                            NumTrbjsHastaAhora=c,
                #                            NTotalCasos=NTotalCasos,
                #                            StageIndexesList = self.BD_Etapas.index.tolist(),
                #                            NumParallelCPU=self.NumParallelCPU,
                #                            MaxNumVecesSubRedes=self.MaxNumVecesSubRedes,
                #                            MaxItCongIntra=self.MaxItCongIntra,
                #                            CaseID=CaseIdentifier  )
                c += NumTrbjs
                print("NumTrbjs:", NumTrbjs)
                # print("Res:", Res)

        else:
            """
                    ##        #######   ######     ###    ##          ##     ##  #######  ########  ########
                    ##       ##     ## ##    ##   ## ##   ##          ###   ### ##     ## ##     ## ##
                    ##       ##     ## ##        ##   ##  ##          #### #### ##     ## ##     ## ##
                    ##       ##     ## ##       ##     ## ##          ## ### ## ##     ## ##     ## ######
                    ##       ##     ## ##       ######### ##          ##     ## ##     ## ##     ## ##
                    ##       ##     ## ##    ## ##     ## ##          ##     ## ##     ## ##     ## ##
                    ########  #######   ######  ##     ## ########    ##     ##  #######  ########  ########

                # En caso contrario trabaja con la información en memoria. Proceso no compatible con paralelización por nodo - cluster.
                # Procedimiento jerárquico:
                    .- Hidrología
                    .- Definir Demanda
                    .- Definir Despacho
                    .- Resolución Etapas
            """
            logger.info("Solving cases in LOCAL MODE.")
            # Total number of stages across the cases that converged to something
            TotalSuccededStages = 0
            # Crea lista con hidrologías de interés en los datos
            ListaHidrologias = ['Humeda', 'Media', 'Seca']

            # Ajuste parámetros para escribir archivos en paralelo
            if bool(self.NumParallelCPU):
                # Parámetros de paralelismo
                if isinstance(self.NumParallelCPU, int):
                    Ncpu = self.NumParallelCPU
                elif self.NumParallelCPU == 'Max':
                    Ncpu = mu__cpu_count()
                logger.info("Escribiendo en paralelo. Utilizando máximo {} procesos simultáneos.".format(Ncpu))
                Pool = mu__Pool(Ncpu)
                Results = []

            #
            # Para cada caso escribe los datos en directorio temporal
            #
            # Inicializa contador de casos
            ContadorCasos = 1
            # Junta y escribe en sub-directorios cada caso (Serie de etapas) como un
            # directorio, el cual contiene las base de datos independientes (Grilla)
            for HidNom in ListaHidrologias:
                # ---- Ajusta base de datos según Hidrología ----
                # PEs por etapa asociadas a la Hidrología en cuestión
                DF_PEsXEtapa = self.BD_Hidrologias_futuras[['PE ' + HidNom + ' dec']]
                # Parámetros de cota-costo en hidrología actual. Indices: ['b', 'CVmin', 'CVmax', 'CotaMin', 'CotaMax']
                DF_ParamHidEmb_hid = self.BD_ParamHidEmb.loc[HidNom]  # Accede al DataFrame del DataFrame Multindex
                DF_CotasEmbalsesXEtapa = pd__DataFrame(np__tile(DF_PEsXEtapa, (1, len(DF_ParamHidEmb_hid.columns))),
                                                       columns = DF_ParamHidEmb_hid.columns,
                                                       index = DF_PEsXEtapa.index)
                # Obtiene las cotas Máximas y Mínimas de los embalses dada la hidrología actual
                CotasMax = DF_ParamHidEmb_hid.loc['CotaMax'].values
                CotasMin = DF_ParamHidEmb_hid.loc['CotaMin'].values
                # Calcula el porcentaje lineal dado por la PE en cada etapa, desde CotaMin hasta CotaMax. De cada Embalse
                DF_CotasEmbalsesXEtapa = (CotasMax - CotasMin) * (1 - DF_CotasEmbalsesXEtapa) + CotasMin
                # ----
                for NDem in range(1, self.NumVecesDem + 1):
                    for NGen in range(1, self.NumVecesGen + 1):
                        # Crea los generadores de demanda y despacho por caso
                        PyGeneratorDemand = aux_smcfpl.GeneradorDemanda(StageIndexesList=self.BD_Etapas.index.tolist(),
                                                                        DF_TasaCLib=self.BD_DemProy[['TasaCliLib']],  # pandas DataFrame
                                                                        DF_TasaCReg=self.BD_DemProy[['TasaCliReg']],  # pandas DataFrame
                                                                        DF_DesvDec=self.BD_DemProy[['Desv_decimal']],  # pandas DataFrame
                                                                        DictTypoCargasEta=self.DictTypoCargasEta,  # diccionario
                                                                        seed=self.UseRandomSeed)  # int
                        PyGeneratorDispatched = aux_smcfpl.GeneradorDespacho(StageIndexesList=self.BD_Etapas.index.tolist(),
                                                                             Dict_TiposGen=self.DictTiposGenNoSlack,  # lista
                                                                             DF_HistGenERNC=self.BD_HistGenRenovable,  # tupla de dos pandas DataFrame
                                                                             DF_TSF=self.BD_TSFProy,  # para cada tecnología que recurra con falla se asigna
                                                                             DF_PE_Hid=DF_PEsXEtapa,  # pandas DataFrame
                                                                             DesvEstDespCenEyS=self.DesvEstDespCenEyS,  # float
                                                                             DesvEstDespCenP=self.DesvEstDespCenP,  # float
                                                                             seed=self.UseRandomSeed)  # int

                        # permite generar nombre del sub-directorio '{HidNom}_D{NDem}_G{NGen}'
                        CaseIdentifier = (HidNom, NDem, NGen)  # post-morten tag

                        if bool(self.NumParallelCPU):  # En paralelo
                            # Agrega la función con sus argumentos al Pool para ejecutarla en paralelo
                            Results.append( Pool.apply_async( core_calc.calc,
                                                              ( ContadorCasos, HidNom, self.BD_RedesXEtapa,
                                                                self.BD_Etapas.index, DF_ParamHidEmb_hid,
                                                                self.BD_seriesconf, self.MaxNumVecesSubRedes,
                                                                self.MaxItCongIntra,
                                                                ),
                                                              # No se pueden pasar argumentos en generadores en paralelo
                                                              { 'abs_OutFilePath': self.abs_OutFilePath,
                                                                'DemGenerator_Dict': {k: v for k, v in PyGeneratorDemand},
                                                                'DispatchGenerator_Dict': {k: v for k, v in PyGeneratorDispatched},
                                                                'in_node': False, 'CaseID': CaseIdentifier,
                                                                }
                                                              )
                                            )
                        else:
                            # (En serie) Aplica directamente para cada caso
                            NumSuccededStages = core_calc.calc( ContadorCasos, HidNom, self.BD_RedesXEtapa,
                                                                self.BD_Etapas.index, DF_ParamHidEmb_hid,
                                                                self.BD_seriesconf, self.MaxNumVecesSubRedes,
                                                                self.MaxItCongIntra,
                                                                abs_OutFilePath= self.abs_OutFilePath,
                                                                DemGenerator_Dict={k: v for k, v in PyGeneratorDemand},
                                                                DispatchGenerator_Dict={k: v for k, v in PyGeneratorDispatched},
                                                                in_node=False, CaseID=CaseIdentifier,
                                                                )
                            TotalSuccededStages += NumSuccededStages
                        ContadorCasos += 1

            if bool(self.NumParallelCPU):  # En paralelo
                print("Ejecutando paralelismo calculo SEPs...")
                # Obtiene los resultados del paralelismo, en caso de existir
                for result in Results:
                    NumSuccededStages = result.get()
                    TotalSuccededStages += NumSuccededStages
                    # (CaseNum, RelevantData) = result.get()
                    # Ejecuta escribiendo a disco
                    # Dict_Casos[CaseIdentifier] = RelevantData

        TotalStagesCases = self.NEta * ContadorCasos
        RunTime = dt.now() - STime
        minutes, seconds = divmod(RunTime.seconds, 60)
        hours, minutes = divmod(minutes, 60)
        msg = "Finished successfully {} stages across {} cases ({:.2f}%), after {} [hr], {} [min] and {} [s].".format(
            TotalSuccededStages, ContadorCasos - 1, TotalSuccededStages / TotalStagesCases,
            hours, minutes, seconds + RunTime.microseconds * 1e-6)
        logger.info(msg)
        logger.debug("Ran of method Simulation.run(...) finished!")
        return None

    def ManageTempData(self, FileFormat):
        """ Manage the way to detect temp folder, in order to create temporarly files if requested or create them from scratch.

            Returns tuple of all databases requiered.
        """
        if self.UseTempFolder:
            if os__path__isdir(self.abs_path_temp):
                if self.RemovePreTempData:
                    try:
                        # remove temp folder and all it's contents
                        shutil__rmtree(self.abs_path_temp)
                        logger.warn("Eliminando directorio completo {}".format(self.TempFolderName))
                    except Exception:
                        logger.warn("Directory '{}' doesn't exists... New one created.".format(self.TempFolderName))
                    # create temporarly folder
                    os__makedirs(self.abs_path_temp)
            else:
                os__makedirs(self.abs_path_temp)
            if self.check_for_DataBases_in_temp_folder(self.abs_path_temp, formatFile=FileFormat):
                msg = "All databases files exist within '{}' folder.".format(self.TempFolderName)
                logger.info(msg)
                self.BD_file_exists = True
                return self.import_DataBases_from_folder(self.abs_path_temp, formatFile=FileFormat)
            else:
                msg = "At least one doesn't o None database file exist within '{}' folder.".format(self.TempFolderName)
                logger.info(msg)
                return self.Create_DataBases()
        else:
            return self.Create_DataBases()

    def check_for_DataBases_in_temp_folder(self, abs_path_folder, formatFile='pickle'):
        """ Checks for the folowwing databases to exist within abs_path_folder:
                    BD_Etapas.p
                    BD_DemProy.p
                    BD_Hidrologias_futuras.p
                    BD_TSFProy.p
                    BD_MantEnEta.p
                    BD_RedesXEtapa.p
                    BD_ParamHidEmb.p
                    BD_HistGenRenovable.p
                    BD_seriesconf.p
            If at least one is missing, returns false.
        """
        Laux = []
        Names2Look = ('BD_Etapas', 'BD_DemProy', 'BD_Hidrologias_futuras',
                      'BD_TSFProy', 'BD_MantEnEta', 'BD_RedesXEtapa',
                      'BD_ParamHidEmb', 'BD_HistGenRenovable', 'BD_seriesconf')
        if formatFile == 'pickle':
            postfix = 'p'
        else:
            raise IOError("'{}' format not implemented yet or des not exists.". format(formatFile))

        for filename in Names2Look:
            filename += '.{}'.format(postfix)
            Laux.append( os__path__isfile(abs_path_folder + os__sep + filename) )
        return all(Laux)

    def import_DataBases_from_folder(self, abs_path_folder, formatFile='pickle'):
        """ Import all Database files in abs_path_folder:
                BD_Etapas.p
                BD_DemProy.p
                BD_Hidrologias_futuras.p
                BD_TSFProy.p
                BD_MantEnEta.p
                BD_RedesXEtapa.p
                BD_ParamHidEmb.p
                BD_HistGenRenovable.p
                BD_seriesconf.p
            Return list with each database ordered as mentioned.
        """
        msg = "Importing databases from existing files in '{}' folder.".format(self.TempFolderName)
        logger.info(msg)
        # initialize return list
        Return_list = []
        Names2Look = ('BD_Etapas', 'BD_DemProy', 'BD_Hidrologias_futuras',
                      'BD_TSFProy', 'BD_MantEnEta', 'BD_RedesXEtapa',
                      'BD_ParamHidEmb', 'BD_HistGenRenovable', 'BD_seriesconf')
        if formatFile == 'pickle':
            postfix = 'p'
        else:
            raise IOError("'{}' format not implemented yet or des not exists.". format(formatFile))

        for name in Names2Look:
            with open(abs_path_folder + os__sep + "{}.{}".format(name, postfix), 'rb') as f:
                Return_list.append( pickle__load(f) )
        return Return_list

    def Create_DataBases(self):
        """
            Creates the Databases. These are store in memory while running.
            Relevant Databases are:
                BD_Etapas.p
                BD_DemProy.p
                BD_Hidrologias_futuras.p
                BD_TSFProy.p
                BD_MantEnEta.p
                BD_RedesXEtapa.p
                BD_ParamHidEmb
                BD_HistGenRenovable.p
                BD_seriesconf.p

            Returns a list with all relevant databases, which are added to the ReturnList after creation.
        """
        msg = "Creating Databases on memory..."
        logger.info(msg)
        # initialize return list
        ReturnList = []
        # lee archivos de entrada (dict of dataframes)
        DFs_Entradas = smcfpl__in_out_proc__read_sheets_to_dataframes(self.abs_InFilePath,
                                                                      self.XLSX_FileName,
                                                                      self.NumParallelCPU)  # only exists here
        # Determina duración de las etapas  (1-indexed)
        BD_Etapas = Crea_Etapas( DFs_Entradas['df_in_smcfpl_mantbarras'],
                                 DFs_Entradas['df_in_smcfpl_manttx'],
                                 DFs_Entradas['df_in_smcfpl_mantgen'],
                                 DFs_Entradas['df_in_smcfpl_mantcargas'],
                                 DFs_Entradas['df_in_smcfpl_histsolar'],
                                 DFs_Entradas['df_in_smcfpl_histeolicas'],
                                 self.FechaComienzo, self.FechaTermino)
        ReturnList.append(BD_Etapas)

        #
        # IDENTIFICA LA INFORMACIÓN QUE LE CORRESPONDE A CADA ETAPA (Siguiente BD son todas en etapas):
        #
        # Calcula y convierte valor a etapas de la desviación histórica de la demanda... (pandas Dataframe)
        BD_DemSistDesv = aux_smcfpl.DesvDemandaHistoricaSistema_a_Etapa( DFs_Entradas['df_in_scmfpl_histdemsist'],
                                                                         BD_Etapas)
        # Obtiene y convierte la demanda proyectada a cada etapas... (pandas Dataframe)
        BD_DemTasaCrecEsp = aux_smcfpl.TasaDemandaEsperada_a_Etapa( DFs_Entradas['df_in_smcfpl_proydem'],
                                                                    BD_Etapas, self.FechaComienzo,
                                                                    self.FechaTermino)
        # Unifica datos de demanda anteriores por etapa (pandas Dataframe)
        BD_DemProy = pd__concat([BD_DemTasaCrecEsp, BD_DemSistDesv.abs()], axis = 'columns')
        ReturnList.append(BD_DemProy)
        # Almacena la PE de cada año para cada hidrología (pandas Dataframe)
        BD_Hidrologias_futuras = aux_smcfpl.Crea_hidrologias_futuras( DFs_Entradas['df_in_smcfpl_histhid'],
                                                                      BD_Etapas, self.PEHidSeca, self.PEHidMed,
                                                                      self.PEHidHum, self.FechaComienzo, self.FechaTermino,
                                                                      seed=self.UseRandomSeed)
        ReturnList.append(BD_Hidrologias_futuras)
        # Respecto a la base de datos 'in_smcfpl_ParamHidEmb' en DFs_Entradas['df_in_smcfpl_ParamHidEmb'], ésta es dependiente de hidrologías solamente
        # Respecto a la base de datos 'in_smcfpl_seriesconf' en DFs_Entradas['df_in_smcfpl_seriesconf'], ésta define configuración hidráulica fija
        # Almacena la TSF por etapa de las tecnologías (pandas Dataframe)
        BD_TSFProy = aux_smcfpl.TSF_Proyectada_a_Etapa( DFs_Entradas['df_in_smcfpl_tsfproy'],
                                                        BD_Etapas, self.FechaComienzo)
        ReturnList.append(BD_TSFProy)
        # Convierte los dataframe de mantenimientos a etapas dentro de un diccionario con su nombre como key
        BD_MantEnEta = aux_smcfpl.Mantenimientos_a_etapas( DFs_Entradas['df_in_smcfpl_mantbarras'],
                                                           DFs_Entradas['df_in_smcfpl_manttx'],
                                                           DFs_Entradas['df_in_smcfpl_mantgen'],
                                                           DFs_Entradas['df_in_smcfpl_mantcargas'],
                                                           BD_Etapas)
        ReturnList.append(BD_MantEnEta)
        # Por cada etapa crea el SEP correspondiente (...paralelizable...) (dict of pandaNetworks and extradata)
        BD_RedesXEtapa = Crea_SEPxEtapa( DFs_Entradas['df_in_smcfpl_tecbarras'],
                                         DFs_Entradas['df_in_smcfpl_teclineas'],
                                         DFs_Entradas['df_in_smcfpl_tectrafos2w'],
                                         DFs_Entradas['df_in_smcfpl_tectrafos3w'],
                                         DFs_Entradas['df_in_smcfpl_tipolineas'],
                                         DFs_Entradas['df_in_smcfpl_tipotrafos2w'],
                                         DFs_Entradas['df_in_smcfpl_tipotrafos3w'],
                                         DFs_Entradas['df_in_smcfpl_tecgen'],
                                         DFs_Entradas['df_in_smcfpl_teccargas'],
                                         BD_MantEnEta, BD_Etapas, self.Sbase_MVA,
                                         self.NumParallelCPU)
        ReturnList.append(BD_RedesXEtapa)
        # print("BD_RedesXEtapa:", BD_RedesXEtapa)
        BD_ParamHidEmb = DFs_Entradas['df_in_smcfpl_ParamHidEmb']
        ReturnList.append(BD_ParamHidEmb)
        # print('BD_HistGenRenovable:', BD_HistGenRenovable)
        BD_HistGenRenovable = aux_smcfpl.GenHistorica_a_Etapa(BD_Etapas,
                                                              DFs_Entradas['df_in_smcfpl_histsolar'],
                                                              DFs_Entradas['df_in_smcfpl_histeolicas'])
        ReturnList.append(BD_HistGenRenovable)
        # print("BD_HistGenRenovable:", BD_HistGenRenovable)
        BD_seriesconf = DFs_Entradas['df_in_smcfpl_seriesconf']
        ReturnList.append(BD_seriesconf)
        # print('BD_seriesconf:', BD_seriesconf)
        return ReturnList

    def write_DataBases_to_pickle(self, abs_path_temp, FileFormat='pickle'):
        """ Write the folowwing databases into abs_path_temp folder in pickle format:
                BD_Etapas.p
                BD_DemProy.p
                BD_Hidrologias_futuras.p
                BD_TSFProy.p
                BD_MantEnEta.p
                BD_RedesXEtapa.p
                BD_ParamHidEmb
                BD_HistGenRenovable.p
                BD_seriesconf.p
        """
        Names_Variables = { 'BD_Etapas': self.BD_Etapas,
                            'BD_DemProy': self.BD_DemProy,
                            'BD_Hidrologias_futuras': self.BD_Hidrologias_futuras,
                            'BD_TSFProy': self.BD_TSFProy,
                            'BD_MantEnEta': self.BD_MantEnEta,
                            'BD_RedesXEtapa': self.BD_RedesXEtapa,
                            'BD_HistGenRenovable': self.BD_HistGenRenovable,
                            'BD_ParamHidEmb': self.BD_ParamHidEmb,
                            'BD_seriesconf': self.BD_seriesconf,
                            }
        smcfpl__in_out_proc__dump_BDs_to_pickle( Names_Variables,
                                                 pathto=abs_path_temp,
                                                 FileFormat=FileFormat)


def Crea_Etapas(DF_MantBarras, DF_MantTx, DF_MantGen, DF_MantLoad, DF_Solar, DF_Eolicas, FechaComienzo, FechaTermino):
    """
    Función que organiza la creación de etapas en base de las series de tiempo almacenadas en las base de datos de los argumentos.
    Esta función hace llamado a las funciones 'Crea_Etapas_Topologicas()' y 'Crea_2ra_div_Etapas()' (encontradas en el mismo archivo fuente), siendo
    aquellas las que realizan la acción.

    Syntax:
        Etapas = Crea_Etapas(DF_MantBarras, DF_MantGen, DF_MantTx, DF_MantLoad, DF_Solar, DF_Eolicas, FechaComienzo, FechaTermino)

    Etapas: Pandas DataFrame con las siguientes columnas
            'EtaNum': (int),
            'FechaIni': (str),
            'FechaFin': (str),
            'HoraDiaIni':(int),
            'HoraDiaFin': (int),
            'TotalHoras': (int)

    :type DF_MantBarras: Pandas DataFrame
    :param DF_MantBarras: DataFrame de los mantenimientos futuros a las barras simuladas.

    :type DF_MantGen: Pandas DataFrame
    :param DF_MantGen: DataFrame de los mantenimientos futuros a las unidades generadoras simuladas.

    :type DF_MantTx: Pandas DataFrame
    :param DF_MantTx: DataFrame de los mantenimientos futuros a los elementos del sistema de transmisión simulados.

    :type DF_MantLoad: Pandas DataFrame
    :param DF_MantLoad: DataFrame de los mantenimientos futuros a las cargas simuladas.

    :type DF_Solar: Pandas DataFrame
    :param DF_Solar: DataFrame del historial para la(s) unidad(es) tipo representativas para las unidades solares.

    :type DF_Eolicas: Pandas DataFrame
    :param DF_Eolicas: DataFrame del historial para la(s) unidad(es) tipo representativas para las unidades eólicos.

    :type FechaComienzo: Datetime object
    :param FechaComienzo: Fecha y hora de la primera hora de la simulación.

    :type FechaTermino: Datetime object
    :param FechaTermino: Fecha y hora de la última hora de la simulación.

    """
    logger.debug("! entrando en función: 'Crea_Etapas' (create_elements.py) ...")
    Etapas = Crea_Etapas_Topologicas(DF_MantBarras, DF_MantGen, DF_MantTx, DF_MantLoad, FechaComienzo, FechaTermino)
    logger.info("Se crearon {} etapas topológicas.".format(Etapas.shape[0]))
    Etapas = Crea_Etapas_Renovables(Etapas, DF_Solar, DF_Eolicas)
    logger.info("Se creó un total de {} etapas.".format(Etapas.shape[0]))
    logger.debug("! saliendo de función: 'Crea_Etapas' (create_elements.py) ...")
    return Etapas


def Crea_Etapas_Topologicas(DF_MantBarras, DF_MantGen, DF_MantTx, DF_MantLoad, FechaComienzo, FechaTermino):
    """
        Crea el DataFrame de las etapas topológicas. Identifica los cambio de fechas (con diferencia mayor a un día) que
        cambian "considerablemente" la topología del SEP mediante lo informado en los programas de mantenimiento (prácticamente mensual).

        1.- Filtra y ordena en forma ascendente las fechas de los mantenimientos.
        2.- Con el DF_CambioFechas identifica y crea las etapas bajo dos Metodologías y la función 'aux_smcfpl.Crea_Etapas_desde_Cambio_Mant'.
    """
    logger.debug("! entrando en función: 'Crea_Etapas_Topologicas' (create_elements.py) ...")
    # Juntar todos los cambios DE FECHAS en un único pandas series. (Inicializa única columna)
    DF_CambioFechas = pd__DataFrame(data=[FechaComienzo, FechaTermino], columns=[0])
    for df in (DF_MantBarras, DF_MantGen, DF_MantTx, DF_MantLoad):
        DF_CambioFechas = pd__concat([ DF_CambioFechas, df['FechaIni'], df['FechaFin'] ], axis='index', join='outer', ignore_index=True)
    # Elimina las fechas duplicadas
    DF_CambioFechas.drop_duplicates(keep='first', inplace=True)
    # Ordena en forma ascendente el pandas series
    DF_CambioFechas.sort_values(by=[0], ascending=True, inplace=True)
    # Resetea los indices
    DF_CambioFechas.reset_index(drop=True, inplace=True)    # Es un DataFrame con datetime (detalle horario) de la existencia de todos los cambios.
    # print('DF_CambioFechas:\n', DF_CambioFechas)

    Metodo_RefFila_EtaTopo = 2
    msg = "Utilizando Método {} para diferencia de filas en Creación Etapas Topológicas.".format(Metodo_RefFila_EtaTopo)
    logger.info(msg)
    if Metodo_RefFila_EtaTopo == 1:
        """ Método 1: Metodo_RefFila_EtaTopo = 1 (Diferencia entre filas - referencia móvil)
        En caso de NO habilitarse 'ref_fija':
            Se fija primera fila como referencia. En caso de existir
            un día o más de diferencia con respecto a la siguiente
            fecha, éstas se marcan como fechas límite. De lo contrario,
            se avanza la referencia a la siguiente y se mide con
            respecto a la que le sigue desde aquí. Proceso finaliza
            luego de cuando la referencia llega a la penúltima fecha
            disponible.

        Notar que el último valor no es considerado (por reducción de indice en comparación y ser éste la fecha de termino de simulación).
        While es necesario para facilitar el salto de filas en iteración.

        ¿Qué hace en el caso de existir una fecha con menos de un día c/r a fecha termino, sin previa etapa limitante?
        """
        logger.debug("! saliendo de función: 'Crea_Etapas_Topologicas' (create_elements.py) ...")
        return aux_smcfpl.Crea_Etapas_desde_Cambio_Mant(DF_CambioFechas, ref_fija=False)
    elif Metodo_RefFila_EtaTopo == 2:
        """ Método 2: Metodo_RefFila_EtaTopo (Diferencia respecto fila referencia)
        En caso de habilitarse 'ref_fija':
            Se fija primera fila como referencia. En caso de existir
            un día o más de diferencia con respecto a la siguiente
            fecha, éstas se marcan como fechas límite y se desplaza
            la referencia a la última de las fechas comparadas.
            De lo contrario, se mide con respecto a la subsiguiente.
            Proceso finaliza luego de cuando la referencia llega a
            la penúltima fecha disponible.

        Notar que el valor de la fila saltado no es considerado a futuro, por lo que se considera como si no existiese.
        While es necesario para facilitar el salto de filas en iteración.

        ¿Qué hace en el caso de existir una fecha con menos de un día c/r a fecha termino, sin previa etapa limitante?
        """
        logger.debug("! saliendo de función: 'Crea_Etapas_Topologicas' (create_elements.py) ...")
        return aux_smcfpl.Crea_Etapas_desde_Cambio_Mant(DF_CambioFechas, ref_fija=True)
    else:
        msg = "Método_RefFila_EtaTopo No fue ingresado válidamente en función 'Crea_Etapas_Topologicas' (create_elements.py)."
        logger.error(msg)
        raise ValueError(msg)


def Crea_Etapas_Renovables(Etapas, DF_Solar, DF_Eolicas):
    """
        Función para crear las etapas renovables a partir de las topológicas y los cambios de potencia en los datos históricos agrupados
        anualmente de las centrales tipo solar y eólicas.
        1.- Por cada etapa topológica, Identifica las fechas que la limitan.
            1.1.- Obtiene los meses del año que componen la etapa, eliminando columna meses.
            1.2.- Agrupa los valores mensuales a 'día' y 'hora' entre los valores mensuales.
            1.3.- Obtiene los días que componen los datos resultantes, eliminando columnas días.
            1.4.- Agrupa los valores diarios a 'hora', éstos son los representativos del "día equivalente" de la etapa topológica.
            1.5.- Normaliza valores respecto del máximo encontrado en la compresión anual. Es de interés ver quien es mayor cada hora del día equivalente.
            1.6.- Encuentra las horas del día en que existen los cambios de máximo.
            1.7.- Crea un DataFrame con función 'Lista2DF_consecutivo', identificando horas de la etapa topológica correspondiente y las horas que
                  limitan los cambios (limites etapa renovable).
            1.8.- Embellece y ordena el DataFrame de salida

        Retorna un pandas DataFrame con columnas: 'FechaIni_EtaTopo', 'FechaFin_EtaTopo', 'HoraDiaIni', 'HoraDiaFin', 'TotalHoras'; e índice
        del numero de cada etapa.
    """
    logger.debug("! entrando en función: 'Crea_Etapas_Renovables' (create_elements.py) ...")
    # inicializa DataFrame de salida
    DF_Eta = pd__DataFrame(columns=['FechaIni_EtaTopo', 'FechaFin_EtaTopo', 'HoraDiaIni', 'HoraDiaFin', 'TotalHoras'])
    # print('Etapas:\n', Etapas)
    for row in Etapas.iterrows():
        FInicioEta = row[1]['FechaIni']
        FTerminoEta = row[1]['FechaFin']
        # NumAniosNecesarios = FTerminoEta.year - FInicioEta.year + 1

        """ Unifica los DataFrame ERNC en DF_aux con índices datetime (año,mes,dia,hora ) asignándole cada año existente en la división de
        etapas, desde la inicial. Posteriormente es filtrado según los mismo número de etapas eliminando los primeros y los últimos
        (conservándose estos últimos)

        Notar que al existir una etapa existente entre más de año, la BD ERNC se considera cíclica.
        """
        # inicializa dataframe para incorporar generación previa de las 4 unidades tipo.
        DF_ERNC = DF_Solar.join(DF_Eolicas)

        # Calcula el máximo anual para normalizar posteriormente
        MaximoAnual = DF_ERNC.max(axis=0)

        # Filtra por meses y posteriormente elimina dicha columna
        Cond1 = FInicioEta.month <= DF_ERNC.index.get_level_values('Mes')
        Cond2 = DF_ERNC.index.get_level_values('Mes') <= FTerminoEta.month
        # Asegura de filtrar adecuadamente los DataFrame (debe ser cíclico si existe cambio de año)
        if FInicioEta.month <= FTerminoEta.month:
            DF_ERNC = DF_ERNC[ Cond1 & Cond2 ]
        elif FTerminoEta.month < FInicioEta.month:
            DF_ERNC = DF_ERNC[ Cond1 | Cond2 ]
        DF_ERNC = DF_ERNC.groupby(level=['Dia', 'Hora']).mean()  # Obtiene el PROMEDIO de los valores inter-mensuales (de existir)

        # Filtra por Días y posteriormente elimina dicha columna
        Cond1 = FInicioEta.day <= DF_ERNC.index.get_level_values('Dia')
        Cond2 = DF_ERNC.index.get_level_values('Dia') <= FTerminoEta.day
        # Asegura de filtrar adecuadamente los DataFrame (debe ser cíclico si existe cambio de año)
        if FInicioEta.day <= FTerminoEta.day:
            DF_ERNC = DF_ERNC[ Cond1 & Cond2 ]
        elif FTerminoEta.day < FInicioEta.day:
            DF_ERNC = DF_ERNC[ Cond1 | Cond2 ]
        DF_ERNC = DF_ERNC.groupby(level=['Hora']).mean()  # Obtiene el PROMEDIO de los valores inter-diario (de existir)
        # DF_ERNC ya es un DataFrame con las 24 horas de un día

        #
        # Escribe valores en archivo para estudio
        #
        # DF_ERNC.divide(MaximoAnual, axis='columns').to_csv("EtaTopo{EtaTopo}_ValDiaEquiv.csv".format(EtaTopo=row[0]))  # genera valores de cambio para posterior estudio
        #

        # Warning en caso de estar todo el DataFrame vacío
        if 0 in DF_ERNC.max(axis=0):
            msg = "El máximo de uno de los datos de las columnas es 0!\n Valor encontrado entre fechas {} y {}.".format(FInicioEta, FTerminoEta)
            logger.warn(msg)

        # Normaliza los valores respecto al máximo de aquellos existentes en la etapa.
        # DF_ERNC = DF_ERNC.divide(DF_ERNC.max(axis=0), axis='columns')  # 49 etapas total aprox SEP 39us
        #
        # usa máximo anual en lugar del de la etapa (representación anual)
        # 27 etapas total aprox SEP 39us
        DF_ERNC = DF_ERNC.divide(MaximoAnual, axis='columns')

        # En DF_Cambios obtiene el nombre de columna de aquel con mayor valor para las horas de la etapa
        DF_Cambios = DF_ERNC.idxmax(axis=1)  # axis=1 | axis='columns' : columns-wise

        # Encuentra los índices que son distintos. Los desfaza hacia abajo (1 periodo), y rellena el vació con el valor siguiente encontrado
        DF_Cambios = DF_Cambios.ne(DF_Cambios.shift(periods=1).fillna(method='bfill'))  # boolean single-column
        # obtiene los elementos que son de cambio, según lo encontrado previamente. Misma variable ya que son lo que finalmente importa
        DF_ERNC = DF_ERNC[ DF_Cambios.values ]

        #
        # Escribe valores en archivo para estudio
        #
        # DF_ERNC.to_csv("EtaTopo{EtaTopo}_ValEtaCambios.csv".format(EtaTopo=row[0]))  # genera valores de cambio para posterior estudio
        #

        # Convierte horas dentro de cada etapa en dataframe
        Horas_Cambio = [0] + (DF_ERNC.index.tolist()) + [23]
        Horas_Cambio = sorted(set(Horas_Cambio))    # transforma a set para evitar duplicidades, posteriormente transforma a lista ordenada ascendente
        # Agrega columnas de las horas límite de las etapas renovables
        DF_etapas2 = aux_smcfpl.Lista2DF_consecutivo(Lista=Horas_Cambio, incremento=1, NombreColumnas=['HoraDiaIni', 'HoraDiaFin'])
        # crea una nueva columna al inicio con el mismo valor para 'FechaFin_EtaTopo' (fechas temporales límite de etapas topológicas)
        DF_etapas2.insert(loc=0, column='FechaFin_EtaTopo', value=FTerminoEta)
        # crea una nueva columna al nuevo inicio con el mismo valor para 'FechaIni_EtaTopo' (fechas temporales límite de etapas topológicas)
        DF_etapas2.insert(loc=0, column='FechaIni_EtaTopo', value=FInicioEta)
        # agrega columna (al final) con la cantidad de horas diarias equivalentes de cada etapa
        DF_etapas2 = DF_etapas2.assign( **{'TotalHoras': DF_etapas2['HoraDiaFin'] - DF_etapas2['HoraDiaIni'] + 1} )

        DF_Eta = DF_Eta.append( DF_etapas2 )
    # reinicia los indices, les suma uno y asigna nombre de 'EtaNum'. Todo para facilidad de comprensión (1-indexed)
    DF_Eta.reset_index(drop=True, inplace=True)
    DF_Eta.index += 1
    DF_Eta.index.name = 'EtaNum'

    logger.debug("! saliendo de función: 'Crea_Etapas_Renovables' (create_elements.py) ...")
    return DF_Eta


def Crea_SEPxEtapa( DF_TecBarras, DF_TecLineas, DF_TecTrafos2w, DF_TecTrafos3w,
                    DF_TipoLineas, DF_TipoTrafos2w, DF_TipoTrafos3w, DF_TecGen,
                    DF_TecCargas, Dict_DF_Mantenimientos, DF_Etapas, Sbase_MVA,
                    NumParallelCPU):
    """
        Identifica los elementos del SEP que se encuentran habilitados en cada etapa topológica y estén estén disponibles en las etapas renovables.
        Utiliza la misma idea de la metodología de conversión tiempo-etapa para filtrar los elementos con menos de un día de duración. "Esto puede existir debido
        a la forma en que se ingresan las fechas de mantención, tal que las fechas límite de cada evento se encuentran dentro del mismo. Existe entonces un límite
        de etapa dado por la fecha inicial de cada cambio."
        Se consideran las siguientes condiciones:
            a) Los eventos con duración menor a un día no son considerados en mantención, por lo que se mantienen operativos.
            b) Los eventos en BD 'df_in_smcfpl_manttx', 'df_in_smcfpl_mantgen', y 'df_in_smcfpl_mantcargas', que presenten flag operativa='True' sobrescriben los parámetros
               del elemento correspondiente en la duración de la etapa.
            c) Los eventos en BD 'df_in_smcfpl_manttx', 'df_in_smcfpl_mantgen', y 'df_in_smcfpl_mantcargas', que presenten flag operativa='False' hace que el elemento
               de la BD correspondiente no se utilice en la duración de la etapa.
            d) Los eventos en BD 'df_in_smcfpl_mantbarras' no posee el parámetro/columna operativa, por lo que de existir en el DataFrame se considerada en
               mantención a excepción de la regla temporal diaria.
            e) Cuando una barra está fuera de servicio, se desconectan de igual medida los elementos que conecta, i.e., líneas, trafos, cargas, gen.

        Retorna un diccionario con los NetWorks PandaPower para cada etapa, más diccionario que no podía incorporarse directamente a la red
    """
    logger.debug("! entrando en función: 'Crea_SEPxEtapa' (CrearElementos.py) ...")
    # Cuenta cantidad de etapas
    TotalEtas = DF_Etapas.shape[0]
    # Inicializa diccionario de salida con los índices de las etapas
    DictSalida = dict.fromkeys( DF_Etapas.index.tolist() )
    if not NumParallelCPU:
        for EtaNum, Etapa in DF_Etapas.iterrows():
            # print("EtaNum:", EtaNum)
            Grid, ExtraData = CompletaSEP_PandaPower(DF_TecBarras, DF_TecLineas, DF_TecTrafos2w,
                                                     DF_TecTrafos3w, DF_TipoLineas, DF_TipoTrafos2w,
                                                     DF_TipoTrafos3w, DF_TecGen, DF_TecCargas,
                                                     Dict_DF_Mantenimientos, EtaNum, Sbase_MVA,
                                                     TotalEtas)
            # Agrega información creada al DictSalida
            DictSalida[EtaNum] = {'PandaPowerNet': Grid}
            DictSalida[EtaNum]['ExtraData'] = ExtraData
    else:   # en parallel
        if isinstance(NumParallelCPU, int):
            Ncpu = NumParallelCPU
        elif NumParallelCPU == 'Max':
            Ncpu = mu__cpu_count()
        logger.info("Creando SEPs en paralelo para etapas. Utilizando máximo {} procesos simultáneos.".format(Ncpu))
        # Inicializa Pool de resultados
        Pool = mu__Pool(Ncpu)
        Results = []
        # Por cada Etapa rellena el Pool
        for EtaNum, Etapa in DF_Etapas.iterrows():
            # Rellena el Pool con los tasks correspondientes
            Results.append( [
                Pool.apply_async(CompletaSEP_PandaPower, (DF_TecBarras, DF_TecLineas,
                                                          DF_TecTrafos2w, DF_TecTrafos3w,
                                                          DF_TipoLineas,  DF_TipoTrafos2w,
                                                          DF_TipoTrafos3w, DF_TecGen,
                                                          DF_TecCargas, Dict_DF_Mantenimientos,
                                                          EtaNum, Sbase_MVA, TotalEtas),
                                 ),
                EtaNum])
        # Obtiene los resultados del paralelismo y asigna a variables de interés
        for result, EtaNum in Results:
            Grid, ExtraData = result.get()
            DictSalida[EtaNum] = {'PandaPowerNet': Grid}
            DictSalida[EtaNum]['ExtraData'] = ExtraData

    logger.debug("! saliendo en función: 'Crea_SEPxEtapa' (CrearElementos.py) ...")
    return DictSalida


def CompletaSEP_PandaPower(DF_TecBarras, DF_TecLineas, DF_TecTrafos2w,
                           DF_TecTrafos3w, DF_TipoLineas, DF_TipoTrafos2w,
                           DF_TipoTrafos3w, DF_TecGen, DF_TecCargas,
                           Dict_DF_Mantenimientos, EtaNum, Sbase_MVA,
                           TotalEtas):
    """
        Función que recibe la pelota para el completado de la red PandaPower en una etapa determinada.
        1.- Por cada etapa inicializa el SEP PandaPower
        2.- Agrega los tipos a la red (lineas|trafos2w|trafos3w)
        3.- Identifica el grupo (pandas DataFrame) de mantenimientos de barra en la etapa y, no las considerada como disponibles
            3.1.- Crea barras del SEP como elemento de PandaPower
        4.- Verifica si existen mantenimientos de líneas para la etapa
            4.1.- Obtiene los elementos de Tx en mantención
            4.2.- Identifica el grupo (pandas DataFrame) con flag Operativa == True para sobrescribir parámetros, dejando primera coincidencia en caso de duplicados
            4.3.- Identifica el grupo (pandas DataFrame) con flag Operativa == False para eliminarles de las disponibles
            4.4.- Desde las barras que están en mantención durante la etapa, se eliminan no se consideran al DataFrame aquellas lineas conectadas a la barra en mantención
            4.5.- Por cada línea, crea las lineas del SEP según disponibles
        5.- Verifica si existen elementos en mantenimiento
            5.1.- identifica los trafos que se definan operativos, con flag Operativa == True para sobrescribir parámetros
            5.2.- sobrescribe nuevos parámetros de los operativos
            5.3.- identifica los 'no operativos', con flag Operativa == False para eliminarles
            5.4.- No considera aquellos trafos2w 'no operativos'
            5.5.- En caso de existir barras en mantenimiento, remueve los trafos2w conectados a ellas
            5.6.- Por cada transformador de dos devanados disponibles, se crean los elementos
        6.- Verifica si existen elementos en mantenimiento
            6.1.- identifica los trafos que se definan operativos, con flag Operativa == True para sobrescribir parámetros
            6.2.- sobrescribe nuevos parámetros de los operativos
            6.3.- identifica los 'no operativos', con flag Operativa == False para eliminarles
            6.4.- No considera aquellos trafos3w 'no operativos'
            6.5.- En caso de existir barras en mantenimiento, remueve los trafos3w conectados a ellas. Cualquiera de las 3 barras
            6.6.- Por cada transformador de dos devanados disponibles, se crean los elementos
        7.- Verifica si existen elementos en mantenimiento
            7.1.- identifica los trafos que se definan operativos, con flag Operativa == True para sobrescribir parámetros
            7.2.- sobrescribe nuevos parámetros de los operativos
            7.3.- identifica los 'no operativos', con flag Operativa == False para eliminarles
            7.4.- No considera aquellas Cargas 'no operativos'
            7.5.- En caso de existir barras en mantenimiento, remueve las cargas conectadas a ellas
            7.6.- Por cada transformador de dos devanados disponibles, se crean los elementos
        8.- Crea lista de parámetros que se modifican con los mantenimiento
            8.1.- Verifica si existen elementos en mantenimiento
            8.2.- identifica los trafos que se definan operativos, con flag Operativa == True para sobrescribir parámetros
            8.3.- sobrescribe nuevos parámetros de los operativos (en 'ColumnasGen')
            8.4.- identifica los 'no operativos', con flag Operativa == False para eliminarles
            8.5.- No considera aquellas Cargas 'no operativos'
            8.6.- En caso de existir barras en mantenimiento, remueve las cargas conectadas a ellas
            8.7.- Identifica si NO se definió alguna unidad de referencia
            8.8.- Asigna primera coincidencia de generador dentro del pandas DataFrame como referencia
            8.9.- Identifica que exista solo una coincidencia en el pandas DataFrame 'GenDisp' de barras Slack
            8.10.- Restablece todos los flag a 'False'
            8.11.- Asigna primera coincidencia de generador dentro del pandas DataFrame como referencia
        9.- Única unidad de referencia existente debe ser ingresada como Red Externa (Requerimientos de PandaPower)
        10.- Elimina el generador de referencia del DataFrame de disponibles
        11.- Por cada Generador disponible crea el correspondiente elemento en la RED
        12.- Elimina los elementos del sistema en caso de quedan sin aislados (Sin conexión a Gen Ref) producto de mantención

        Retorna una tupla con (PandaPower Grid filled, ExtraData)
    """
    logger.debug("! Creando SEP en etapa {}/{} ...".format(EtaNum, TotalEtas))
    #
    # 1.- Inicializa el SEP PandaPower y diccionario ExtraData con datos que no pueden incorporarse en el Grid
    Grid = pp__create_empty_network(name='EtaNum {}'.format(EtaNum), f_hz=50, sn_kva=Sbase_MVA * 1e3)
    ExtraData = {}

    """
          ###                                #       #
         #   #                               #
         #      # ##    ###    ###          ####    ##    # ##    ###    ###
         #      ##  #  #   #      #          #       #    ##  #  #   #  #
         #      #      #####   ####          #       #    ##  #  #   #   ###
         #   #  #      #      #   #          #  #    #    # ##   #   #      #
          ###   #       ###    ####           ##    ###   #       ###   ####
                                                          #
                                                          #
    """
    # 2.- Agrega los tipos a la red (lineas|trafos2w|trafos3w)
    pp__create_std_types(Grid, data=DF_TipoLineas.T.to_dict(), element='line')  # tipos de lineas
    pp__create_std_types(Grid, data=DF_TipoTrafos2w.T.to_dict(), element='trafo')  # tipos de trafos2w
    pp__create_std_types(Grid, data=DF_TipoTrafos3w.T.to_dict(), element='trafo3w')  # tipos de trafos3w

    """
          ###                               ####
         #   #                               #  #
         #      # ##    ###    ###           #  #   ###   # ##   # ##    ###    ###
         #      ##  #  #   #      #          ###       #  ##  #  ##  #      #  #
         #      #      #####   ####          #  #   ####  #      #       ####   ###
         #   #  #      #      #   #          #  #  #   #  #      #      #   #      #
          ###   #       ###    ####         ####    ####  #      #       ####  ####
    """
    # logger.debug("! 'Crea_SEPxEtapa':Creando barras en EtaNum {} ...".format(EtaNum))
    # Verifica si existen mantenimientos de barras para la etapa, a modo de filtrarlas
    if EtaNum in Dict_DF_Mantenimientos['df_in_smcfpl_mantbarras'].index:
        # 3.- Identifica el grupo (pandas DataFrame) de mantenimientos de barra en la etapa y, no las considerada como disponibles
        BarsEnMant = Dict_DF_Mantenimientos['df_in_smcfpl_mantbarras'].loc[[EtaNum], ['BarNom']]  # Una sola columna de interés
        BarrasDisp = DF_TecBarras.drop(labels=BarsEnMant['BarNom'], axis='index')
    else:   # No hay mantenimientos de barras
        BarrasDisp = DF_TecBarras
        BarsEnMant = pd__DataFrame(columns=['BarNom'])

    # 3.1- Crea barras del SEP como elemento de PandaPower
    pp__create_buses(Grid, nr_buses=BarrasDisp.shape[0], vn_kv=BarrasDisp['Vnom'].values, name=BarrasDisp.index)

    # Crea lista de columnas a ser consideradas para líneas, trafos2w y trafos3w
    ColumnasLineas = ['ElmTxNom', 'TipoElmn', 'Parallel', 'Largo_km', 'Pmax_AB_MW', 'Pmax_BA_MW', 'Operativa', 'BarraA', 'BarraB', 'TipoNom']
    ColumnasTrafos2w = ['ElmTxNom', 'TipoElmn', 'Parallel', 'Pmax_AB_MW', 'Pmax_BA_MW', 'Operativa', 'BarraA_HV', 'BarraB_LV', 'TipoNom']
    ColumnasTrafos3w = ['ElmTxNom', 'TipoElmn', 'Operativa', 'BarraA_HV', 'BarraB_MV', 'BarraC_LV', 'Pmax_inA_MW', 'Pmax_outA_MW', 'Pmax_inB_MW', 'Pmax_outB_MW', 'Pmax_inC_MW', 'Pmax_outC_MW', 'TipoNom']

    """
          ###                               #        #
         #   #                              #
         #      # ##    ###    ###          #       ##    # ##    ###    ###    ###
         #      ##  #  #   #      #         #        #    ##  #  #   #      #  #
         #      #      #####   ####         #        #    #   #  #####   ####   ###
         #   #  #      #      #   #         #        #    #   #  #      #   #      #
          ###   #       ###    ####         #####   ###   #   #   ###    ####  ####
    """
    # logger.debug("! 'Crea_SEPxEtapa':Creando Lineas en EtaNum {} ...".format(EtaNum))
    # 4.- Verifica si existen mantenimientos de líneas para la etapa
    LinsDisp = DF_TecLineas.copy(deep=True)  # innocent until proven guilty
    CondCalcMant = (EtaNum in Dict_DF_Mantenimientos['df_in_smcfpl_manttx'][
        Dict_DF_Mantenimientos['df_in_smcfpl_manttx']['TipoElmn'] == 'Linea' ].index) & (not LinsDisp.empty)
    if CondCalcMant:
        # 4.1- Obtiene los elementos de Tx en mantención
        LinsEnMant = Dict_DF_Mantenimientos['df_in_smcfpl_manttx'].loc[[EtaNum], ColumnasLineas]
        # 4.2- Identifica el grupo (pandas DataFrame) con flag Operativa == True para sobrescribir parámetros, dejando primera coincidencia en caso de duplicados
        LinsEnMantOp = LinsEnMant[ LinsEnMant['Operativa'] & (LinsEnMant['TipoElmn'] == 'Linea') ].drop_duplicates(keep='first')
        # Genera DataFrame de líneas disponibles en la etapa filtrado
        LinsDisp.loc[ LinsEnMantOp['ElmTxNom'], : ] = LinsEnMantOp.reset_index(drop=True).set_index('ElmTxNom')
        # 4.3- Identifica el grupo (pandas DataFrame) con flag Operativa == False para eliminarles de las disponibles
        LinsEnMantNoOp = LinsEnMant[ (~LinsEnMant['Operativa']) & (LinsEnMant['TipoElmn'] == 'Linea') ].drop_duplicates(keep='first')
        LinsDisp.drop(labels=LinsEnMantNoOp['ElmTxNom'], axis='index', inplace=True)

    # 4.4- Desde las barras que están en mantención durante la etapa, se eliminan no se consideran al DataFrame aquellas lineas conectadas a la barra en mantención
    if not BarsEnMant.empty:
        LinsDisp.drop( labels=LinsDisp[ (LinsDisp['BarraA'].isin(BarsEnMant['BarNom'])) | (LinsDisp['BarraB'].isin(BarsEnMant['BarNom'])) ].index,
                       axis='index', inplace=True)

    # 4.5- Por cada línea, crea las lineas del SEP según disponibles
    for LinNom, Linea in LinsDisp.iterrows():
        IndBarraA = Grid['bus'][ Grid['bus']['name'] == Linea['BarraA'] ].index[0]
        IndBarraB = Grid['bus'][ Grid['bus']['name'] == Linea['BarraB'] ].index[0]
        pp__create_line(Grid, from_bus=IndBarraA, to_bus=IndBarraB, length_km=Linea['Largo_km'],
                        std_type=Linea['TipoNom'], name=LinNom, parallel=Linea['Parallel'])

    """
          ###                               #####                  ##           ###
         #   #                                #                   #  #         #   #
         #      # ##    ###    ###            #    # ##    ###    #      ###       #  #   #
         #      ##  #  #   #      #           #    ##  #      #  ####   #   #    ##   #   #
         #      #      #####   ####           #    #       ####   #     #   #   #     # # #
         #   #  #      #      #   #           #    #      #   #   #     #   #  #      # # #
          ###   #       ###    ####           #    #       ####   #      ###   #####   # #
    """
    # logger.debug("! 'Crea_SEPxEtapa':Creando Trf2w en EtaNum {} ...".format(EtaNum))
    Trafo2wDisp = DF_TecTrafos2w.copy(deep=True)  # innocent until proven guilty
    # 5.- Verifica si existen elementos en mantenimiento
    CondCalcMant = (EtaNum in Dict_DF_Mantenimientos['df_in_smcfpl_manttx'][
        Dict_DF_Mantenimientos['df_in_smcfpl_manttx']['TipoElmn'] == 'Trafo2w' ].index) & (not Trafo2wDisp.empty)
    if CondCalcMant:
        Trafo2wEnMant = Dict_DF_Mantenimientos['df_in_smcfpl_manttx'].loc[[EtaNum], ColumnasTrafos2w]  # DataFrame de Trafos2w y respectivas columnas
        # 5.1-identifica los trafos que se definan operativos, con flag Operativa == True para sobrescribir parámetros
        Trafo2wEnMantOp = Trafo2wEnMant[ Trafo2wEnMant['Operativa'] & (Trafo2wEnMant['TipoElmn'] == 'Trafo2w') ].drop_duplicates(keep='first')
        # 5.2-sobrescribe nuevos parámetros de los operativos
        Trafo2wDisp.loc[ Trafo2wEnMantOp['ElmTxNom'], : ] = Trafo2wEnMantOp.reset_index(drop=True).set_index('ElmTxNom')
        # 5.3-identifica los 'no operativos', con flag Operativa == False para eliminarles
        Trafo2wEnMantNoOp = Trafo2wEnMant[ ( ~Trafo2wEnMant['Operativa'] ) & (Trafo2wEnMant['TipoElmn'] == 'Trafo2w') ].drop_duplicates(keep='first')
        # 5.4-No considera aquellos trafos2w 'no operativos'
        Trafo2wDisp.drop(labels=Trafo2wEnMantNoOp['ElmTxNom'], axis='index', inplace=True)

    # 5.5-En caso de existir barras en mantenimiento, remueve los trafos2w conectados a ellas
    if not BarsEnMant.empty:
        Trafo2wDisp.drop( labels=Trafo2wDisp[ (Trafo2wDisp['BarraA_HV'].isin(BarsEnMant['BarNom'])) | (Trafo2wDisp['BarraB_LV'].isin(BarsEnMant['BarNom'])) ].index,
                          axis='index', inplace=True)

    # 5.6- Por cada transformador de dos devanados disponibles, se crean los elementos
    for Trf2wNom, Trf2w in Trafo2wDisp.iterrows():
        IndBarraA = Grid['bus'][ Grid['bus']['name'] == Trf2w['BarraA_HV'] ].index[0]
        IndBarraB = Grid['bus'][ Grid['bus']['name'] == Trf2w['BarraB_LV'] ].index[0]
        pp__create_transformer(Grid, hv_bus=IndBarraA, lv_bus=IndBarraB, std_type=Trf2w['TipoNom'],
                               name=Trf2wNom, parallel=Trf2w['Parallel'])

    """
          ###                               #####                  ##          #####
         #   #                                #                   #  #             #
         #      # ##    ###    ###            #    # ##    ###    #      ###      #   #   #
         #      ##  #  #   #      #           #    ##  #      #  ####   #   #    ##   #   #
         #      #      #####   ####           #    #       ####   #     #   #      #  # # #
         #   #  #      #      #   #           #    #      #   #   #     #   #  #   #  # # #
          ###   #       ###    ####           #    #       ####   #      ###    ###    # #
    """
    # logger.debug("! 'Crea_SEPxEtapa':Creando Trf3w en EtaNum {} ...".format(EtaNum))
    Trafo3wDisp = DF_TecTrafos3w.copy(deep=True)  # innocent until proven guilty
    # 6.- Verifica si existen elementos en mantenimiento
    CondCalcMant = (EtaNum in Dict_DF_Mantenimientos['df_in_smcfpl_manttx'][
        Dict_DF_Mantenimientos['df_in_smcfpl_manttx']['TipoElmn'] == 'Trafo3w'].index) & (not Trafo3wDisp.empty)
    if CondCalcMant:
        Trafo3wEnMant = Dict_DF_Mantenimientos['df_in_smcfpl_manttx'].loc[[EtaNum], ColumnasTrafos3w]  # DataFrame de Trafos3w y respectivas columnas
        # 6.1- identifica los trafos que se definan operativos, con flag Operativa == True para sobrescribir parámetros
        Trafo3wEnMantOp = Trafo3wEnMant[ Trafo3wEnMant['Operativa'] & (Trafo3wEnMant['TipoElmn'] == 'Trafo3w') ].drop_duplicates(keep='first')
        # 6.2- sobrescribe nuevos parámetros de los operativos
        Trafo3wDisp.loc[ Trafo3wEnMantOp['ElmTxNom'], : ] = Trafo3wEnMantOp.reset_index(drop=True).set_index('ElmTxNom')
        # 6.3- identifica los 'no operativos', con flag Operativa == False para eliminarles
        Trafo3wEnMantNoOp = Trafo3wEnMant[ ( ~Trafo3wEnMant['Operativa'] ) & (Trafo3wEnMant['TipoElmn'] == 'Trafo3w') ].drop_duplicates(keep='first')
        # 6.4- No considera aquellos trafos3w 'no operativos'
        Trafo3wDisp.drop(labels=Trafo3wEnMantNoOp['ElmTxNom'], axis='index', inplace=True)

    # 6.5- En caso de existir barras en mantenimiento, remueve los trafos3w conectados a ellas. Cualquiera de las 3 barras
    if not BarsEnMant.empty:
        Trafo3wDisp.drop( labels=Trafo3wDisp[
            (Trafo3wDisp['BarraA_HV'].isin(BarsEnMant['BarNom'])) |
            (Trafo3wDisp['BarraB_MV'].isin(BarsEnMant['BarNom'])) |
            (Trafo3wDisp['BarraC_LV'].isin(BarsEnMant['BarNom']))
        ].index,
            axis='index', inplace=True)

    # 6.6- Por cada transformador de dos devanados disponibles, se crean los elementos
    for Trf3wNom, Trf3w in Trafo3wDisp.iterrows():
        IndBarraA = Grid['bus'][ Grid['bus']['name'] == Trf3w['BarraA_HV'] ].index[0]
        IndBarraB = Grid['bus'][ Grid['bus']['name'] == Trf3w['BarraB_MV'] ].index[0]
        IndBarraC = Grid['bus'][ Grid['bus']['name'] == Trf3w['BarraC_LV'] ].index[0]
        pp__create_transformer3w(Grid, hv_bus=IndBarraA, mv_bus=IndBarraB, lv_bus=IndBarraC, std_type=Trf3w['TipoNom'], name=Trf3wNom)

    """
          ###                                ###
         #   #                              #   #
         #      # ##    ###    ###          #       ###   # ##    ## #   ###    ###
         #      ##  #  #   #      #         #          #  ##  #  #  #       #  #
         #      #      #####   ####         #       ####  #       ##     ####   ###
         #   #  #      #      #   #         #   #  #   #  #      #      #   #      #
          ###   #       ###    ####          ###    ####  #       ###    ####  ####
                                                                 #   #
                                                                  ###
    """
    # logger.debug("! 'Crea_SEPxEtapa':Creando Cargas en EtaNum {} ...".format(EtaNum))
    CargasDisp = DF_TecCargas.copy(deep=True)  # innocent until proven guilty
    # 7.- Verifica si existen elementos en mantenimiento
    CondCalcMant = (EtaNum in Dict_DF_Mantenimientos['df_in_smcfpl_mantcargas'].index) & (not CargasDisp.empty)
    if CondCalcMant:
        CargasEnMant = Dict_DF_Mantenimientos['df_in_smcfpl_mantcargas'].loc[[EtaNum], :]  # DataFrame de cargas y respectivas columnas
        # 7.1- identifica las cargas que se definan operativos, con flag Operativa == True para sobrescribir parámetros
        CargasEnMantOp = CargasEnMant[ CargasEnMant['Operativa'] ].drop_duplicates(keep='first')
        # 7.2- sobrescribe nuevos parámetros de los operativos
        CargasDisp.loc[ CargasEnMantOp['LoadNom'], : ] = CargasEnMantOp.reset_index(drop=True).set_index('LoadNom')
        # 7.3- identifica los 'no operativos', con flag Operativa == False para eliminarles
        CargasEnMantNoOp = CargasEnMant[ ~CargasEnMant['Operativa'] ].drop_duplicates(keep='first')
        # 7.4- No considera aquellas Cargas 'no operativos'
        CargasDisp.drop(labels=CargasEnMantNoOp['LoadNom'], axis='index', inplace=True)

    # 7.5- En caso de existir barras en mantenimiento, remueve las cargas conectadas a ellas
    if not BarsEnMant.empty:
        CargasDisp.drop( labels=CargasDisp[ CargasDisp['NomBarConn'].isin(BarsEnMant['BarNom'])  ].index,
                         axis='index', inplace=True)

    # 7.6- Por cada transformador de dos devanados disponibles, se crean los elementos
    for CargaNom, Load in CargasDisp.iterrows():
        IndBarraConn = Grid['bus'][ Grid['bus']['name'] == Load['NomBarConn'] ].index[0]
        # Notar que se le asigna la potencia nominal a la carga. Ésta es posteriormente modificada según los parámetros de la etapa en cada proceso
        pp__create_load(Grid, bus=IndBarraConn, name=CargaNom, p_kw=Load['DemNom_MW'] * 1e3, type=Load['LoadTyp'])

    """
          ###                                ###
         #   #                              #   #
         #      # ##    ###    ###          #       ###   # ##
         #      ##  #  #   #      #         #      #   #  ##  #
         #      #      #####   ####         #  ##  #####  #   #
         #   #  #      #      #   #         #   #  #      #   #
          ###   #       ###    ####          ###    ###   #   #
    """
    # logger.debug("! 'Crea_SEPxEtapa':Creando Unidades en EtaNum {} ...".format(EtaNum))
    GenDisp = DF_TecGen.copy(deep=True)  # innocent until proven guilty
    # 8.- Crea lista de parámetros que se modifican con los mantenimiento
    ColumnasGen = ['PmaxMW', 'PminMW', 'NomBarConn', 'CVar', 'EsSlack']
    # 8.1.- Verifica si existen elementos en mantenimiento
    CondCalcMant = (EtaNum in Dict_DF_Mantenimientos['df_in_smcfpl_mantgen'].index) & (not GenDisp.empty)
    if CondCalcMant:
        GenEnMant = Dict_DF_Mantenimientos['df_in_smcfpl_mantgen'].loc[[EtaNum], :]  # DataFrame de cargas y respectivas columnas
        # 8.2.- identifica los trafos que se definan operativos, con flag Operativa == True para sobrescribir parámetros
        GenEnMantOp = GenEnMant[ GenEnMant['Operativa'] ].drop_duplicates(keep='first')
        # 8.3.- sobrescribe nuevos parámetros de los operativos (en 'ColumnasGen')
        GenDisp.loc[ GenEnMantOp['GenNom'], ColumnasGen ] = GenEnMantOp.reset_index(drop=True).set_index('GenNom')
        # 8.4.- identifica los 'no operativos', con flag Operativa == False para eliminarles
        GenEnMantNoOp = GenEnMant[ ~GenEnMant['Operativa'] ].drop_duplicates(keep='first')
        # 8.5.- No considera aquellas Cargas 'no operativos'
        GenDisp.drop(labels=GenEnMantNoOp['GenNom'], axis='index', inplace=True)

    # 8.6.- En caso de existir barras en mantenimiento, remueve las cargas conectadas a ellas
    if not BarsEnMant.empty:
        GenDisp.drop( labels=GenDisp[ GenDisp['NomBarConn'].isin(BarsEnMant['BarNom'])  ].index,
                      axis='index', inplace=True)

    # 8.7.- Identifica si NO se definió alguna unidad de referencia
    if not GenDisp['EsSlack'].any():
        msg = "NO existe Unidad definida de referencia para la Etapa {}! ...".format(EtaNum)
        logger.warning(msg)
        GenRef = GenDisp.index[0]
        # 8.8.- Asigna primera coincidencia de generador dentro del pandas DataFrame como referencia
        GenDisp.loc[GenRef, 'EsSlack'] = True
        msg = "Fijando Unidad: '{}' como referencia.".format(GenRef)
        logger.warning(msg)
    # 8.9.- Identifica que exista solo una coincidencia en el pandas DataFrame 'GenDisp' de barras Slack
    elif GenDisp['EsSlack'].sum() > 1:
        msg = "Existe más de una Unidad definida de referencia para la Etapa {}! Restableciendo flags 'EsSlack' ...".format(EtaNum)
        logger.warning(msg)
        GenRef = GenDisp.index[0]
        # 8.10.- Restablece todos los flag a 'False'
        GenDisp.loc[:, 'EsSlack'] = False
        # 8.11.- Asigna primera coincidencia de generador dentro del pandas DataFrame como referencia
        GenDisp.loc[GenRef, 'EsSlack'] = True
        msg = "Fijando Unidad: '{}' como referencia.".format(GenRef)
        logger.warning(msg)

    # 9.- Única unidad de referencia existente debe ser ingresada como Red Externa (Requerimientos de PandaPower)
    pdSerie_GenRef = GenDisp[GenDisp['EsSlack']].squeeze()  # convert single row pandas DataFrame to Series
    IndBarraConn = Grid['bus'][ Grid['bus']['name'] == pdSerie_GenRef['NomBarConn'] ].index[0]  # toma la primera coincidencia
    pp__create_ext_grid( Grid, bus=IndBarraConn, vm_pu=1.0, va_degree=0.0, name=pdSerie_GenRef.name,
                         max_p_kw=-pdSerie_GenRef['PmaxMW'] * 1e3, min_p_kw=-pdSerie_GenRef['PminMW'] * 1e3 )  # negativo para generación
    # 10.- Elimina el generador de referencia del DataFrame de disponibles
    GenDisp.drop(labels=pdSerie_GenRef.name, axis='index', inplace=True)

    # 11.- Por cada Generador disponible crea el correspondiente elemento en la RED
    for GenNom, Generador in GenDisp.iterrows():
        IndBarraConn = Grid['bus'][ Grid['bus']['name'] == Generador['NomBarConn'] ].index[0]
        # Notar que se le asigna la potencia nominal a la carga. Ésta es posteriormente modificada según los parámetros de la etapa en cada proceso
        pp__create_gen(Grid, bus=IndBarraConn, name=GenNom, p_kw=-Generador['PmaxMW'] * 1e3,
                       max_p_kw=-Generador['PmaxMW'] * 1e3, min_p_kw=-Generador['PminMW'] * 1e3,
                       type=Generador['GenTec'])  # p_kw es negativo para generación

    # 12.- Elimina los elementos del sistema en caso de quedan sin aislados (Sin conexión a Gen Ref) producto de mantención
    #      Notar que pueden definirse múltiples Generadores de referencia, los cuales podrán quedar separados eléctricamente.
    SetBarrasAisladas = pp__topology__unsupplied_buses(Grid)
    if bool(SetBarrasAisladas):
        logger.warn("Existe sistema aislado sin conexión con Gen Ref. Eliminándolo...")
        # Notar como los sistemas quedan separados cuando existe más de una Barra de Referencia,
        # dada por el generador de referencia (angV = 0). No da Warning cuando ésto ocurre.
        pp__drop_inactive_elements(Grid)  # Incluye logger Info level. From pandapower
        # Descarta de los TrafosDisp y LinsDisp, aquellos que no se
        # encuentran activos luego de eliminar los desenergizados
        Trafo2wDisp = Trafo2wDisp.loc[ Grid['trafo'].name, :]
        Trafo3wDisp = Trafo3wDisp.loc[ Grid['trafo3w'].name, :]
        LinsDisp = LinsDisp.loc[ Grid['line'].name, :]

    # 13.- Actualiza el diccionario ExtraData con información adicional
    # Costo variable unidad de referencia (Red Externa)
    ExtraData['CVarGenRef'] = float(pdSerie_GenRef['CVar'])
    # Nombre de la tecnología del generador de referencia
    ExtraData['TecGenSlack'] = str(pdSerie_GenRef['GenTec'])
    # Número de cargas existentes por etapa
    ExtraData['NumLoads'] = Grid['load'].shape[0]
    # pandas DataFrame del índice de generadores en la Grilla y Tipo de tecnología
    ExtraData['Tipos'] = Grid['gen'][['type']]
    # pandas DataFrame del índice de generadores en la Grilla y CVar
    ExtraData['CVarGenNoRef'] = pd__DataFrame( data=GenDisp.loc[ Grid['gen']['name'], 'CVar'].values,
                                               index=Grid['gen'].index,
                                               columns=['CVar'])
    ExtraData['CVarGenNoRef'] = ExtraData['CVarGenNoRef'].astype({'CVar': float})
    # potencia permitida por transformador 'trafo2w'
    ExtraData['PmaxMW_trafo2w'] = Trafo2wDisp[['Pmax_AB_MW', 'Pmax_BA_MW']]
    # potencia permitida por transformador 'trafo3w'
    ExtraData['PmaxMW_trafo3w'] = Trafo3wDisp[['Pmax_inA_MW', 'Pmax_outA_MW', 'Pmax_inB_MW', 'Pmax_outB_MW', 'Pmax_inC_MW', 'Pmax_outC_MW']]
    # potencia permitida por lineas 'line'
    ExtraData['PmaxMW_line'] = LinsDisp[['Pmax_AB_MW', 'Pmax_BA_MW']]

    logger.debug("! SEP en etapa {}/{} creado.".format(EtaNum, TotalEtas))

    return (Grid, ExtraData)
