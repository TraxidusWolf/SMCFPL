from smcfpl.in_out_proc import read_sheets_to_dataframes as smcfpl__in_out_proc__read_sheets_to_dataframes
from smcfpl.in_out_proc import ImprimeBDsGrales as smcfpl__in_out_proc__ImprimeBDsGrales
from smcfpl.in_out_proc import write_BDs_input_case as smcfpl__in_out_proc__write_BDs_input_case
from smcfpl.in_out_proc import dump_BDs_to_pickle as smcfpl__in_out_proc__dump_BDs_to_pickle
from smcfpl.smcfpl_exceptions import *
import smcfpl.aux_funcs as aux_funcs
from smcfpl.send_cases_to_nodes import send_work
import smcfpl.core_calc as core_calc
from itertools import tee as it__tee
from os.path import exists as os__path__exists, isdir as os__path__isdir
from os.path import abspath as os__path__abspath, isfile as os__path__isfile
from os import makedirs as os__makedirs, getcwd as os__getcwd
from os import listdir as os__listdir
from os import sep as os__sep
from sys import executable as sys__executable
from subprocess import run as sp__run, PIPE as sp__PIPE
from pandas import DataFrame as pd__DataFrame
from pandas import concat as pd__concat
from numpy import tile as np__tile, float128 as np__float128
from numpy import exp as np__exp
from numpy import ceil as np__ceil
from datetime import datetime as dt
from datetime import timedelta as dt__timedelta
from dateutil import relativedelta as du__relativedelta
from shutil import rmtree as shutil__rmtree
from pickle import load as pickle__load, dump as pickle__dump
from copy import deepcopy as copy__deepcopy
from pandapower import create_empty_network as pp__create_empty_network, create_buses as pp__create_buses
from pandapower import create_line as pp__create_line, create_std_types as pp__create_std_types
from pandapower import create_transformer as pp__create_transformer, create_transformer3w as pp__create_transformer3w
from pandapower import create_load as pp__create_load, create_gen as pp__create_gen
from pandapower import create_ext_grid as pp__create_ext_grid
from pandapower.topology import unsupplied_buses as pp__topology__unsupplied_buses
from pandapower import drop_inactive_elements as pp__drop_inactive_elements
from multiprocessing import cpu_count as mu__cpu_count, Pool as mu__Pool

import logging
# define global loggers
aux_funcs.setup_logger('stdout_only', level=logging.DEBUG)
aux_funcs.setup_logger('Intra_congestion', log_file=r'IntraCongs.log',
                       )

# create local logger variable
logger = logging.getLogger('stdout_only')
logger_IntraCong = logging.getLogger('stdout_only')
# Cambia logger level de pandapower al actual
logging.getLogger("pandapower").setLevel(logger.level)


class Simulation(object):
    """
        Clase base que contiene los atributos y métodos de la simulación para ejecutar el modelo exitosamente.
        Guarda las base de datos en memoria (pandas dataframe, diccionarios, etc), desde los cuales adquiere los datos para cada etapa. Ojo, durante
        paralelismo se aumentarán los requerimientos de memoria según la cantidad de tareas.
    """

    def __init__(self, simulation_name, XLSX_FileName='', InFilePath='.', OutFilePath='.', Sbase_MVA=100, MaxNumVecesSubRedes=1,
                 MaxItCongIntra=1, FechaComienzo='2018-01-01 00:00', FechaTermino='2019-01-31 23:59',
                 NumVecesDem=1, NumVecesGen=1, PerdCoseno=True, PEHidSeca=0.8, PEHidMed=0.5, PEHidHum=0.2,
                 DesvEstDespCenEyS=0.1, DesvEstDespCenP=0.2, NumParallelCPU=None, UsaSlurm=False,
                 Working_dir=os__getcwd(), UseTempFolder = True, RemovePreTempData=True,
                 smcfpl_dir=os__getcwd(), TempFolderName='TempData', UseRandomSeed=None):
        """
            :param UsaSlurm: Diccionario con parámetros para ejecución en el sistema de colas de slurm. Hace que se ejecuten comandos (con biblioteca subprocess) sbatch
                            propios de slurm para ejecución en varios nodos. Se escriben BD datos en un directorio temporal para ser copiado a cada nodo.
                            Formato: {'NumNodes': (int), 'NodeWaittingTime': (datetime deltatime object), 'ntasks': (int), 'cpu_per_tasks': (int)}
                                    'NumNodes': Número de nodos a utilizar en el cluster.
                                    'NodeWaittingTime': Tiempo de espera máximo de ejecución de los procesos enviados a nodos.
                                    'ntasks': número de tareas a repartirse por nodo.
                                    'cpu-per-tasks': Número de cpu requeridas para cada tarea.
                            En caso de no utilizarse, se debe ingresar valor booleano 'False'.
            :type UsaSlurm: dict

        """

        logger.debug("! Initializating class Simulation(...)")
        STime = dt.now()

        #
        # Atributos desde entradas
        self.simulation_name = simulation_name  # (str)
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
        # Crea lista con hidrologías de interés en los datos
        self.ListaHidrologias = ['Humeda', 'Media', 'Seca']
        self.NumCasesExpected = NumVecesDem * NumVecesGen * len(self.ListaHidrologias)
        self.DesvEstDespCenEyS = DesvEstDespCenEyS  # 0 <= (float) <= 1
        self.DesvEstDespCenP = DesvEstDespCenP  # 0 <= (float) <= 1
        self.Working_dir = Working_dir
        if isinstance(NumParallelCPU, int) | (NumParallelCPU is None) | (NumParallelCPU == 'Max'):
            self.NumParallelCPU = NumParallelCPU
        else:
            # Puede ser False: No usa paralelismo ni lectura ni cálculo, 'Max' para todos los procesadores, o un 'int' tamaño personalizado pool
            msg = "Input 'NumParallelCPU' must be integer, False, or 'Max'."
            logger.error(msg)
            raise IOError(msg)
        # checks for integrity of dictionary keys
        if UsaSlurm:  # not False nor None
            if not isinstance(UsaSlurm, dict):
                msg = "UsaSlurm argument must be 'None' or populated dictionary."
                logger.error(msg)
                raise IOError(msg)
            else:  # it's a dictionary, check for the keys!
                #  {'var name': {'types possibility': 'str true condition to format' }}
                required_keys = {
                    'NumNodes': {int: '{}>0', str: "'{}'=='Max'", type(None): '{} is None'},
                    'NodeWaittingTime': {dt__timedelta: 'True'},  # any timedelta value
                    'ntasks': {int: '{}>0'},
                    'cpu_per_tasks': {int: '{}>0'},
                }
                for k, sk in required_keys.items():
                    # stores or condition for type
                    is_any_type = False
                    # checks for existence of all keys
                    if k not in UsaSlurm:
                        msg = "Input argument UsaSlurm (dict) missing key: '{}'.".format(k)
                        logger.error(msg); raise IOError(msg)
                    # checks for key type and true value conditions
                    for t, cond in sk.items():
                        # checks for type
                        if isinstance(UsaSlurm[k], t):
                            is_any_type = True
                            # checks for allowed values
                            evaluate = cond.format(UsaSlurm[k])
                            if not eval(evaluate):  # should evaluate to boolean condition
                                varname = "UsaSlurm['{}']".format(k)
                                fullcond = cond.format(varname)
                                msg = "Input {} value must fulfill condition: {}".format(varname, fullcond)
                                logger.error(msg); raise IOError(msg)
                        else:
                            # stores message if variable types do not match
                            msg = "UsaSlurm['{}'] type must be: '{}'.".format(k, t)
                    # after all types checks, if non is matched lunch error
                    if not is_any_type:
                        logger.error(msg); raise IOError(msg)
        self.UsaSlurm = UsaSlurm  # (bool)
        self.UseTempFolder = UseTempFolder  # (bool)
        self.RemovePreTempData = RemovePreTempData  # (bool)
        self.TempFolderName = str(TempFolderName) + '_' + str(simulation_name)
        self.abs_path_temp = os__path__abspath(Working_dir + os__sep + self.TempFolderName)
        self.abs_path_smcfpl = os__path__abspath(smcfpl_dir)
        self.abs_InFilePath = os__path__abspath(InFilePath)
        self.abs_OutFilePath = os__path__abspath(OutFilePath + '_' + simulation_name)
        if isinstance(UseRandomSeed, int) | (UseRandomSeed is None):
            self.UseRandomSeed = UseRandomSeed
        else:
            msg = "'UseRandomSeed' must be an integer or None."
            raise IOError(msg)
            logger.error(msg)
        self.BD_file_exists = False

        FileFormat = 'pickle'  # tipes of files to be read
        # Verifies if it's necesary to import or calculate data. Added them to attributes
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

        #
        # Usefull debugging
        print('self.BD_Etapas:\n', self.BD_Etapas)
        #

        """
             ####           #            #####    #     ##     #                     #
              #  #          #            #               #     #
              #  #   ###   ####    ###   #       ##      #    ####    ###   # ##    ##    # ##    ## #
              #  #      #   #         #  ####     #      #     #     #   #  ##  #    #    ##  #  #  #
              #  #   ####   #      ####  #        #      #     #     #####  #        #    #   #   ##
              #  #  #   #   #  #  #   #  #        #      #     #  #  #      #        #    #   #  #
             ####    ####    ##    ####  #       ###    ###     ##    ###   #       ###   #   #   ###
                                                                                                 #   #
                                                                                                  ###
        """
        #
        # Numero total de etapas
        self.NEta = self.BD_Etapas.shape[0]
        # Number os spected stages (across cases)
        self.total_stages_per_cases = self.NEta * self.NumCasesExpected

        # Gets a set of all generator's name (including ext_grid) available across stages (to set curve cost)
        self.AllGenNames = set()
        for NStg, val in self.BD_RedesXEtapa.items():
            for n in val['PandaPowerNet'].gen.name.values:
                self.AllGenNames.add(n)
            for n in val['PandaPowerNet'].ext_grid.name.values:
                self.AllGenNames.add(n)

        # FILTERS the Reservoirs not asociated to generation along simulation. From self.AllGenNames
        self.BD_seriesconf_filtered = self.BD_seriesconf[ self.BD_seriesconf.CenNom.isin(self.AllGenNames) ]
        # Gets type of cost curve for each Reservoir
        self.CostCurves = 'LogisticaS'  # TODO: use list for each reservoir: self.BD_seriesconf_filtered['FuncCosto'].tolist()
        # Find all Reservoirs' names used along the simulation
        self.ReservoirNames = set(self.BD_seriesconf['NombreEmbalse'])
        # FILTERS Reservoirs that are not used within the simulation
        self.BD_ParamHidEmb_filtered = self.BD_ParamHidEmb.loc[:, self.BD_ParamHidEmb.columns.isin( self.ReservoirNames )]

        #
        # Obtiene partes del diccionario ExtraData por etapa
        self.TecGenSlack = [d['ExtraData']['TecGenSlack'] for d in self.BD_RedesXEtapa.values()]
        # Crea lista del Número de Cargas en cada Etapa/Grid
        self.DictTypoCargasEta = { k: v['PandaPowerNet']['load'][['type']] for k, v in self.BD_RedesXEtapa.items() }
        # Crea lista del Número de Unidades de Generación en cada Etapa/Grid
        # self.ListNumGenNoSlack = [d['ExtraData']['NumGenNoSlack'] for d in self.BD_RedesXEtapa.values()]
        # Crea lista del tipos (Número de Unidades intrínseco) de Generación en cada Etapa/Grid
        self.DictTiposGenNoSlack = { k: d['ExtraData']['Tipos'] for k, d in self.BD_RedesXEtapa.items() }

        #
        # Creates the Demand base generator for cases
        # self.Base_PyGeneratorDemand = aux_funcs.GeneradorDemanda(StageIndexesList=self.BD_Etapas.index.tolist(),
        #                                                          DF_TasaCLib=self.BD_DemProy[['TasaCliLib']],  # pandas DataFrame
        #                                                          DF_TasaCReg=self.BD_DemProy[['TasaCliReg']],  # pandas DataFrame
        #                                                          DF_DesvDec=self.BD_DemProy[['Desv_decimal']],  # pandas DataFrame
        #                                                          DictTypoCargasEta=self.DictTypoCargasEta,  # dictionary
        #                                                          seed=self.UseRandomSeed)  # int, None
        # Creates a hydrology databases and saves it in:
        #    BD_Hydro = {HidNom: { 'DF_PEsXEtapa': DF_PEsXEtapa,
        #                          'DF_ParamHidEmb_hid': DF_ParamHidEmb_hid,
        #                          'DF_CotasEmbalsesXEtapa': DF_CotasEmbalsesXEtapa,
        #                          'DF_CVarReservoir_hid': DF_CVarReservoir_hid,
        #                          'DF_CostoCentrales': DF_CostoCentrales,
        #                          'PyGeneratorDispatched': PyGeneratorDispatched},
        #                ...}
        self.BD_Hydro = dict()
        self.BD_BaseGenDisp = dict()
        for HidNom in self.ListaHidrologias:
            self.BD_Hydro[HidNom] = dict()
            # ---- Ajusta base de datos según Hidrología ----
            # PEs por etapa asociadas a la Hidrología en cuestión (DF_PEsXEtapa)
            DF_PEsXEtapa = self.BD_Hidrologias_futuras[['PE ' + HidNom + ' dec']]
            # Parámetros de cota-costo en hidrología actual. Indices: ['b', 'CVmin', 'CVmax', 'CotaMin', 'CotaMax']
            DF_ParamHidEmb_hid = self.BD_ParamHidEmb_filtered.loc[HidNom]  # Accede al DF del DF Multindex
            # Initialize reservoir's levels with PE's (excedence probability) as decimal across stages
            DF_CotasEmbalsesXEtapa = pd__DataFrame(np__tile(DF_PEsXEtapa, (1, len(DF_ParamHidEmb_hid.columns))),
                                                   columns = DF_ParamHidEmb_hid.columns,
                                                   index = DF_PEsXEtapa.index)
            # Obtiene las cotas Máximas y Mínimas de los embalses dada la hidrología actual
            CotasMax = DF_ParamHidEmb_hid.loc['CotaMax'].values
            CotasMin = DF_ParamHidEmb_hid.loc['CotaMin'].values
            # Calcula el valor lineal dado por la PE en cada etapa, desde CotaMin hasta CotaMax. De cada Embalse
            DF_CotasEmbalsesXEtapa = (CotasMax - CotasMin) * (1 - DF_CotasEmbalsesXEtapa) + CotasMin
            DF_CVarReservoir_hid = calc_reservoir_costs_from_cota( DF_CotasEmbalsesXEtapa,
                                                                   DF_ParamHidEmb_hid,
                                                                   CostCurves = self.CostCurves)
            #
            # Find generation costs at given hidrology
            DF_CostoCentrales = pd__DataFrame(index=DF_CVarReservoir_hid.index)
            for indx, (Reservoir, GenNom, FuncCost) in self.BD_seriesconf_filtered.iterrows():
                DF_CostoCentrales = DF_CostoCentrales.assign(**{GenNom: DF_CVarReservoir_hid[Reservoir]})
            # Updates the dictionary
            self.BD_Hydro[HidNom]['DF_PEsXEtapa'] = DF_PEsXEtapa
            self.BD_Hydro[HidNom]['DF_ParamHidEmb_hid'] = DF_ParamHidEmb_hid
            self.BD_Hydro[HidNom]['DF_CotasEmbalsesXEtapa'] = DF_CotasEmbalsesXEtapa
            self.BD_Hydro[HidNom]['DF_CVarReservoir_hid'] = DF_CVarReservoir_hid
            self.BD_Hydro[HidNom]['DF_CostoCentrales'] = DF_CostoCentrales

        #
        # checks for temporal directory. Mainly debugging process
        if self.UseTempFolder and not self.BD_file_exists:
            msg = "Writing databases to file as requested (UseTempFolder=True)"
            self.write_DataBases_to_pickle(self.abs_path_temp)
        # dont forget about BD_Hydro
        if self.UseTempFolder and not os__path__isfile(self.abs_path_temp + os__sep + 'BD_Hydro.p'):
            # checks for hydro database
            smcfpl__in_out_proc__dump_BDs_to_pickle( {'BD_Hydro': self.BD_Hydro},
                                                     pathto=self.abs_path_temp,
                                                     FileFormat='pickle')

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
        msg = "Initialization of Simulation class finished after {} [hr], {} [min] and {} [s].".format(
            hours, minutes, seconds + RunTime.microseconds * 1e-6)
        logger.info(msg)
        logger.debug("! Initialization class Simulation(...) Finished!")

    def run(self):
        logger.debug("Running method Simulation.run()...")
        STime = dt.now()

        # initialize first row of logger_IntraCong if headers were declared
        headers = 'LogInfo,StageNum,CaseNum,TypeElmnt,IndTable,loading_percent'
        logger_IntraCong.info(headers)

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

                If UsaSlurm['NumNodes'] is bigger than nodes, it will wait unit cluster has those nodes.
                Variable UsaSlurm['NumNodes'] also defines the number of parallel jobs in head node to send
                cases accordingly to node number.

                Per each group (in 'hydro_dict_cases_list'), function 'send_work()' is executed. This creates and
                executes a bash script to run a certain amount of cases in one one. The same functions run
                the cases IN PARALLEL within each node requested.

                Data files must be read from TempData* folder.

                # Everything should be fine if NumNodes approx NumCPU head node.
            """
            # base database files
            BD_fnames = [
                'BD_DemProy.p', 'BD_Etapas.p', 'BD_Hidrologias_futuras.p',
                'BD_HistGenRenovable.p', 'BD_Hydro.p', 'BD_MantEnEta.p',
                'BD_ParamHidEmb.p', 'BD_RedesXEtapa.p', 'BD_seriesconf.p',
                'BD_TSFProy.p']
            # Total number of stages across the cases that converged to something
            total_cases_sent = 0
            total_cases_succeded = 0
            total_stages_succeded = 0

            # each position:
            #    0-random_seed
            #    1-DesvEstDespCenEyS
            #    2-DesvEstDespCenP
            #    3-DictTypoCargasEta
            #    4-DF_GenType_per_unit
            #    5-abs_OutFilePath
            #    6-NumVecesDem
            #    7-NumVecesGen
            gral_params = [
                self.UseRandomSeed,
                self.DesvEstDespCenEyS,
                self.DesvEstDespCenP,
                self.DictTypoCargasEta,
                self.DF_GenType_per_unit,
                self.abs_OutFilePath,
                self.NumVecesDem,
                self.NumVecesGen,
            ]

            # find the number of available nodes ('idle' status)
            max_av_nodes = find_maximum_nodes_available()

            # Interpreters the number of nodes to use
            nodes_to_use = aux_funcs.configure_input_nodes(self.UsaSlurm['NumNodes'], max_av_nodes)

            if nodes_to_use > self.NumCasesExpected:
                # Necessary, otherwise nodes will have no work to do.
                nodes_to_use = self.NumCasesExpected
                msg = "Nodes number > number of cases. Forcing use to {} nodes.".format(nodes_to_use)
                logger.warn(msg)

            # when no nodes are available raise error
            if nodes_to_use == 0:
                msg = "No nodes available at the time. Please try later."
                logger.error(msg); raise NoNodesAvailable(msg)

            # (Logging only) Split cases as groups into nodes as most equal ratio possibly (across 25 nodes; 24 core each)
            cases_per_groups = aux_funcs.calc_sending_to_nodes_matrix( self.NumCasesExpected,
                                                                       nodes_to_use,
                                                                       self.ListaHidrologias)
            cases_per_groups = cases_per_groups.sum(axis=0).astype(int).tolist()  # sum cols
            list_interpreted_msg = aux_funcs.interprete_list_of_groups(cases_per_groups)
            msg = "Dividing {} cases into {}, across {} nodes...".format( self.NumCasesExpected,
                                                                          list_interpreted_msg,
                                                                          nodes_to_use)
            logger.info(msg)

            # Truly divides the cases according to hydrologies. Get a list of dicts
            hydro_dict_cases_list = aux_funcs.split_num_into_hydrologies( self.NumCasesExpected,
                                                                          nodes_to_use,
                                                                          self.ListaHidrologias)

            n_groups = len(cases_per_groups)  # == len(hydro_dict_cases_list)
            w_time = self.UsaSlurm['NodeWaittingTime']  # timeout for node response

            # Parallel parameters
            if self.UsaSlurm['NumNodes']:
                msg = "Cases will be sent in parallel. Using {} groups of cases according to NumNodes."
                if isinstance(self.UsaSlurm['NumNodes'], int):
                    Ncpu = nodes_to_use
                elif self.UsaSlurm['NumNodes'] == 'Max':
                    Ncpu = mu__cpu_count()
                logger.info( msg.format(Ncpu) )
                Pool = mu__Pool(Ncpu)
                results = []

            # group processing
            iterable = enumerate(zip(cases_per_groups, hydro_dict_cases_list), start=1)
            for nth_group, (cases_per_group, group_details) in iterable:
                total_cases_sent += cases_per_group
                # prepares input data for case classifier on sending. Compress data as single argument to function
                group_info = (
                    nth_group,
                    cases_per_group,
                    n_groups,
                    n_cases,
                    group_details)

                if self.UsaSlurm['NumNodes']:  # In parallel
                    results.append( Pool.apply_async(
                        send_work,
                        (
                            self,
                            group_info,
                            BD_fnames,
                            gral_params,
                            w_time,
                        )
                    )
                    )
                else:
                    n_cases_succeded, n_stages_succeded = send_work(
                        self,
                        group_info,
                        BD_fnames,
                        gral_params,
                        w_time,
                    )
                    total_cases_succeded += n_cases_succeded
                    total_stages_succeded += n_stages_succeded

            if self.UsaSlurm['NumNodes']:  # En paralelo
                # Get results from parallelism , if it exists
                for result in results:
                    # n_cases_succeded, n_stages_succeded, n_group = result.get()
                    n_cases_succeded = 0
                    n_stages_succeded = 0
                    result.get()
                    total_cases_succeded += n_cases_succeded
                    total_stages_succeded += n_stages_succeded

            # change variable name for logging proposes
            case_num_counter = total_cases_succeded

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
            total_cases_succeded = 0
            total_stages_succeded = 0
            # Initialize case executed number
            case_num_counter = 0

            # Ajuste parámetros de paralelismo para escribir archivos
            if self.NumParallelCPU:
                msg = "Parallel mode activated. Using maximum of {} simultaneous processes."
                if isinstance(self.NumParallelCPU, int):
                    Ncpu = self.NumParallelCPU
                elif self.NumParallelCPU == 'Max':
                    Ncpu = mu__cpu_count()
                logger.info( msg.format(Ncpu) )
                Pool = mu__Pool(Ncpu)
                Results = []

            # Start case generation
            for HidNom in self.ListaHidrologias:
                for NDem in range(1, self.NumVecesDem + 1):
                    for NGen in range(1, self.NumVecesGen + 1):
                        case_num_counter += 1
                        # get relevant values form hydrology
                        # DF_CotasEmbalsesXEtapa = self.BD_Hydro[HidNom]['DF_CotasEmbalsesXEtapa']
                        # DF_CostoCentrales = self.BD_Hydro[HidNom]['DF_CostoCentrales']

                        # Creates an iterator (class type with __next__ dunder) for each loop (different values)
                        instance_IterDem = aux_funcs.IteratorDemand(StageIndexesList=self.BD_Etapas.index.tolist(),
                                                                    DF_TasaCLib=self.BD_DemProy[['TasaCliLib']],  # pandas DataFrame
                                                                    DF_TasaCReg=self.BD_DemProy[['TasaCliReg']],  # pandas DataFrame
                                                                    DF_DesvDec=self.BD_DemProy[['Desv_decimal']],  # pandas DataFrame
                                                                    DictTypoCargasEta=self.DictTypoCargasEta,  # diccionario
                                                                    seed=self.UseRandomSeed)  # int, None
                        instance_IterDispatched = aux_funcs.IteratorDespatch(StageIndexesList=self.BD_Etapas.index.tolist(),
                                                                             DF_GenType_per_unit=self.DictTiposGenNoSlack,  # dict of numpy array
                                                                             DF_HistGenERNC=self.BD_HistGenRenovable,  # tupla de dos pandas DataFrame
                                                                             DF_TSF=self.BD_TSFProy,  # para cada tecnología que recurra con falla se asigna
                                                                             DF_PE_Hid=self.BD_Hydro[HidNom]['DF_PEsXEtapa'],  # pandas DataFrame
                                                                             DesvEstDespCenEyS=self.DesvEstDespCenEyS,  # float
                                                                             DesvEstDespCenP=self.DesvEstDespCenP,  # float
                                                                             seed=self.UseRandomSeed)  # int, None
                        # permite generar nombre del sub-directorio '{HidNom}_D{NDem}_G{NGen}'
                        case_identifier = (HidNom, NDem, NGen)  # post-morten tag

                        if bool(self.NumParallelCPU):  # En paralelo
                            # Agrega la función con sus argumentos al Pool para ejecutarla en paralelo
                            Results.append( Pool.apply_async( core_calc.calc,
                                                              ( case_num_counter, HidNom, self.BD_RedesXEtapa,
                                                                self.BD_Etapas.index,
                                                                self.BD_Hydro[HidNom]['DF_ParamHidEmb_hid'],
                                                                self.BD_seriesconf,
                                                                self.BD_Hydro[HidNom]['DF_CVarReservoir_hid'],
                                                                self.MaxNumVecesSubRedes, self.MaxItCongIntra,
                                                                ),
                                                              # No se pueden pasar argumentos en generadores en paralelo
                                                              { 'abs_OutFilePath': self.abs_OutFilePath,
                                                                'DemGenerator': instance_IterDem,
                                                                'DispatchGenerator': instance_IterDispatched,
                                                                'in_node': False, 'CaseID': case_identifier,
                                                                }
                                                              )
                                            )
                        else:
                            # (En serie) Aplica directamente para cada caso
                            NumSuccededStages = core_calc.calc( case_num_counter, HidNom, self.BD_RedesXEtapa,
                                                                self.BD_Etapas.index,
                                                                self.BD_Hydro[HidNom]['DF_ParamHidEmb_hid'],
                                                                self.BD_seriesconf,
                                                                self.BD_Hydro[HidNom]['DF_CVarReservoir_hid'],
                                                                self.MaxNumVecesSubRedes, self.MaxItCongIntra,
                                                                abs_OutFilePath= self.abs_OutFilePath,
                                                                DemGenerator=instance_IterDem,
                                                                DispatchGenerator=instance_IterDispatched,
                                                                in_node=False, CaseID=case_identifier,
                                                                )
                            total_stages_succeded += NumSuccededStages

            if self.NumParallelCPU:  # En paralelo
                logger.info("Executing paralelism calculations on power system cases...")
                # Obtiene los resultados del paralelismo, en caso de existir
                for result in Results:
                    NumSuccededStages = result.get()
                    total_cases_succeded += NumSuccededStages

        #
        # FINISHING STEP AFTER ALL WORK
        #
        RunTime = dt.now() - STime
        minutes, seconds = divmod(RunTime.seconds, 60)
        hours, minutes = divmod(minutes, 60)
        msg = "Finished successfully {}/{} stages ({:.2f}%) across {} cases with {} stages each, "
        msg += "after {} [hr], {} [min] and {} [s]."
        msg = msg.format( total_stages_succeded, self.total_stages_per_cases,
                          total_stages_succeded / self.total_stages_per_cases * 100,
                          case_num_counter, self.NEta,
                          hours, minutes, seconds + RunTime.microseconds * 1e-6)
        logger.info(msg)
        logger.debug("Ran of method Simulation.run(...) finished!")

    def ManageTempData(self, FileFormat):
        """ Manage the way to detect temp folder, in order to create temporarily files if requested or create them from scratch.

            Returns tuple of all databases required.
        """
        if self.UseTempFolder:
            if os__path__isdir(self.abs_path_temp):
                if self.RemovePreTempData:
                    try:
                        # remove temp folder and all it's contents
                        shutil__rmtree(self.abs_path_temp)
                        logger.warn("Deleting full directory {}".format(self.TempFolderName))
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
        """ Checks for the following databases to exist within abs_path_folder:
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
        FileExists = True
        Names2Look = ('BD_Etapas', 'BD_DemProy', 'BD_Hidrologias_futuras',
                      'BD_TSFProy', 'BD_MantEnEta', 'BD_RedesXEtapa',
                      'BD_ParamHidEmb', 'BD_HistGenRenovable', 'BD_seriesconf')
        if formatFile == 'pickle':
            postfix = 'p'
        else:
            raise IOError("'{}' format not implemented yet or des not exists.". format(formatFile))

        # checks for database files
        for filename in Names2Look:
            filename += '.{}'.format(postfix)
            FileExists &= os__path__isfile(abs_path_folder + os__sep + filename)
            if not FileExists:
                break
        return FileExists

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
        BD_DemSistDesv = aux_funcs.DesvDemandaHistoricaSistema_a_Etapa( DFs_Entradas['df_in_scmfpl_histdemsist'],
                                                                        BD_Etapas)
        # Obtiene y convierte la demanda proyectada a cada etapas... (pandas Dataframe)
        BD_DemTasaCrecEsp = aux_funcs.TasaDemandaEsperada_a_Etapa( DFs_Entradas['df_in_smcfpl_proydem'],
                                                                   BD_Etapas, self.FechaComienzo,
                                                                   self.FechaTermino)
        # Unifica datos de demanda anteriores por etapa (pandas Dataframe)
        BD_DemProy = pd__concat([BD_DemTasaCrecEsp, BD_DemSistDesv.abs()], axis = 'columns')
        ReturnList.append(BD_DemProy)
        # Almacena la PE de cada año para cada hidrología (pandas Dataframe)
        BD_Hidrologias_futuras = aux_funcs.Crea_hidrologias_futuras( DFs_Entradas['df_in_smcfpl_histhid'],
                                                                     BD_Etapas, self.PEHidSeca, self.PEHidMed,
                                                                     self.PEHidHum, self.FechaComienzo, self.FechaTermino,
                                                                     seed=self.UseRandomSeed)
        ReturnList.append(BD_Hidrologias_futuras)
        # Respecto a la base de datos 'in_smcfpl_ParamHidEmb' en DFs_Entradas['df_in_smcfpl_ParamHidEmb'], ésta es dependiente de hidrologías solamente
        # Respecto a la base de datos 'in_smcfpl_seriesconf' en DFs_Entradas['df_in_smcfpl_seriesconf'], ésta define configuración hidráulica fija
        # Almacena la TSF por etapa de las tecnologías (pandas Dataframe)
        BD_TSFProy = aux_funcs.TSF_Proyectada_a_Etapa( DFs_Entradas['df_in_smcfpl_tsfproy'],
                                                       BD_Etapas, self.FechaComienzo)
        ReturnList.append(BD_TSFProy)
        # Convierte los dataframe de mantenimientos a etapas dentro de un diccionario con su nombre como key
        BD_MantEnEta = aux_funcs.Mantenimientos_a_etapas( DFs_Entradas['df_in_smcfpl_mantbarras'],
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
        BD_HistGenRenovable = aux_funcs.GenHistorica_a_Etapa(BD_Etapas,
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
            'StageNum': (int),
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
        2.- Con el DF_CambioFechas identifica y crea las etapas bajo dos Metodologías y la función 'aux_funcs.Crea_Etapas_desde_Cambio_Mant'.
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
        return aux_funcs.Crea_Etapas_desde_Cambio_Mant(DF_CambioFechas, ref_fija=False)
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
        return aux_funcs.Crea_Etapas_desde_Cambio_Mant(DF_CambioFechas, ref_fija=True)
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
        DF_etapas2 = aux_funcs.Lista2DF_consecutivo(Lista=Horas_Cambio, incremento=1, NombreColumnas=['HoraDiaIni', 'HoraDiaFin'])
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
    logger.debug("! entrando en función: 'Crea_SEPxEtapa' ...")
    # Cuenta cantidad de etapas
    TotalEtas = DF_Etapas.shape[0]
    # Inicializa diccionario de salida con los índices de las etapas
    DictSalida = dict.fromkeys( DF_Etapas.index.tolist() )
    if not NumParallelCPU:
        for StageNum, Etapa in DF_Etapas.iterrows():
            # print("StageNum:", StageNum)
            Grid, ExtraData = CompletaSEP_PandaPower(DF_TecBarras, DF_TecLineas, DF_TecTrafos2w,
                                                     DF_TecTrafos3w, DF_TipoLineas, DF_TipoTrafos2w,
                                                     DF_TipoTrafos3w, DF_TecGen, DF_TecCargas,
                                                     Dict_DF_Mantenimientos, StageNum, Sbase_MVA,
                                                     TotalEtas)
            # Agrega información creada al DictSalida
            DictSalida[StageNum] = {'PandaPowerNet': Grid}
            DictSalida[StageNum]['ExtraData'] = ExtraData
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
        for StageNum, Etapa in DF_Etapas.iterrows():
            # Rellena el Pool con los tasks correspondientes
            Results.append( [
                Pool.apply_async(CompletaSEP_PandaPower, (DF_TecBarras, DF_TecLineas,
                                                          DF_TecTrafos2w, DF_TecTrafos3w,
                                                          DF_TipoLineas,  DF_TipoTrafos2w,
                                                          DF_TipoTrafos3w, DF_TecGen,
                                                          DF_TecCargas, Dict_DF_Mantenimientos,
                                                          StageNum, Sbase_MVA, TotalEtas),
                                 ),
                StageNum])
        # Obtiene los resultados del paralelismo y asigna a variables de interés
        for result, StageNum in Results:
            Grid, ExtraData = result.get()
            DictSalida[StageNum] = {'PandaPowerNet': Grid}
            DictSalida[StageNum]['ExtraData'] = ExtraData

    logger.debug("! saliendo en función: 'Crea_SEPxEtapa' ...")
    return DictSalida


def CompletaSEP_PandaPower(DF_TecBarras, DF_TecLineas, DF_TecTrafos2w,
                           DF_TecTrafos3w, DF_TipoLineas, DF_TipoTrafos2w,
                           DF_TipoTrafos3w, DF_TecGen, DF_TecCargas,
                           Dict_DF_Mantenimientos, StageNum, Sbase_MVA,
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
    logger.debug("! Creating PS for Stage {}/{} ...".format(StageNum, TotalEtas))
    #
    # 1.- Inicializa el SEP PandaPower y diccionario ExtraData con datos que no pueden incorporarse en el Grid
    Grid = pp__create_empty_network(name='StageNum {}'.format(StageNum), f_hz=50, sn_kva=Sbase_MVA * 1e3, add_stdtypes=False)
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
    logger.debug("!! Adding branch types to stage {}/{} ...".format(StageNum, TotalEtas))
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
    logger.debug("!! Creating buses in StageNum {} ...".format(StageNum))
    # Verifica si existen mantenimientos de barras para la etapa, a modo de filtrarlas
    if StageNum in Dict_DF_Mantenimientos['df_in_smcfpl_mantbarras'].index:
        # 3.- Identifica el grupo (pandas DataFrame) de mantenimientos de barra en la etapa y, no las considerada como disponibles
        BarsEnMant = Dict_DF_Mantenimientos['df_in_smcfpl_mantbarras'].loc[[StageNum], ['BarNom']]  # Una sola columna de interés
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
    logger.debug("!! Creating Lines in StageNum {} ...".format(StageNum))
    # 4.- Verifica si existen mantenimientos de líneas para la etapa
    LinsDisp = DF_TecLineas.copy(deep=True)  # innocent until proven guilty
    CondCalcMant = (StageNum in Dict_DF_Mantenimientos['df_in_smcfpl_manttx'][
        Dict_DF_Mantenimientos['df_in_smcfpl_manttx']['TipoElmn'] == 'Linea' ].index) & (not LinsDisp.empty)
    if CondCalcMant:
        # 4.1- Obtiene los elementos de Tx en mantención
        LinsEnMant = Dict_DF_Mantenimientos['df_in_smcfpl_manttx'].loc[[StageNum], ColumnasLineas]
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
    logger.debug("!! Creating Trf2w in StageNum {} ...".format(StageNum))
    Trafo2wDisp = DF_TecTrafos2w.copy(deep=True)  # innocent until proven guilty
    # 5.- Verifica si existen elementos en mantenimiento
    CondCalcMant = (StageNum in Dict_DF_Mantenimientos['df_in_smcfpl_manttx'][
        Dict_DF_Mantenimientos['df_in_smcfpl_manttx']['TipoElmn'] == 'Trafo2w' ].index) & (not Trafo2wDisp.empty)
    if CondCalcMant:
        Trafo2wEnMant = Dict_DF_Mantenimientos['df_in_smcfpl_manttx'].loc[[StageNum], ColumnasTrafos2w]  # DataFrame de Trafos2w y respectivas columnas
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
    logger.debug("!! Creating Trf3w in StageNum {} ...".format(StageNum))
    Trafo3wDisp = DF_TecTrafos3w.copy(deep=True)  # innocent until proven guilty
    # 6.- Verifica si existen elementos en mantenimiento
    CondCalcMant = (StageNum in Dict_DF_Mantenimientos['df_in_smcfpl_manttx'][
        Dict_DF_Mantenimientos['df_in_smcfpl_manttx']['TipoElmn'] == 'Trafo3w' ].index) & (not Trafo3wDisp.empty)
    if CondCalcMant:
        Trafo3wEnMant = Dict_DF_Mantenimientos['df_in_smcfpl_manttx'].loc[[StageNum], ColumnasTrafos3w]  # DataFrame de Trafos3w y respectivas columnas
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
    logger.debug("!! Creating Loads in StageNum {} ...".format(StageNum))
    CargasDisp = DF_TecCargas.copy(deep=True)  # innocent until proven guilty
    # 7.- Verifica si existen elementos en mantenimiento
    CondCalcMant = (StageNum in Dict_DF_Mantenimientos['df_in_smcfpl_mantcargas'].index) & (not CargasDisp.empty)
    if CondCalcMant:
        CargasEnMant = Dict_DF_Mantenimientos['df_in_smcfpl_mantcargas'].loc[[StageNum], :]  # DataFrame de cargas y respectivas columnas
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
    logger.debug("!! Creating Units in StageNum {} ...".format(StageNum))
    GenDisp = DF_TecGen.copy(deep=True)  # innocent until proven guilty
    # 8.- Crea lista de parámetros que se modifican con los mantenimiento
    ColumnasGen = ['PmaxMW', 'PminMW', 'NomBarConn', 'CVar', 'EsSlack']
    # 8.1.- Verifica si existen elementos en mantenimiento
    CondCalcMant = (StageNum in Dict_DF_Mantenimientos['df_in_smcfpl_mantgen'].index) & (not GenDisp.empty)
    if CondCalcMant:
        GenEnMant = Dict_DF_Mantenimientos['df_in_smcfpl_mantgen'].loc[[StageNum], :]  # DataFrame de cargas y respectivas columnas
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
        msg = "NO existe Unidad definida de referencia para la Etapa {}! ...".format(StageNum)
        logger.warning(msg)
        GenRef = GenDisp.index[0]
        # 8.8.- Asigna primera coincidencia de generador dentro del pandas DataFrame como referencia
        GenDisp.loc[GenRef, 'EsSlack'] = True
        msg = "Fijando Unidad: '{}' como referencia.".format(GenRef)
        logger.warning(msg)
    # 8.9.- Identifica que exista solo una coincidencia en el pandas DataFrame 'GenDisp' de barras Slack
    elif GenDisp['EsSlack'].sum() > 1:
        msg = "Existe más de una Unidad definida de referencia para la Etapa {}! Restableciendo flags 'EsSlack' ...".format(StageNum)
        logger.warning(msg)
        GenRef = GenDisp.index[0]
        # 8.10.- Restablece todos los flag a 'False'
        GenDisp.loc[:, 'EsSlack'] = False
        # 8.11.- Asigna primera coincidencia de generador dentro del pandas DataFrame como referencia
        GenDisp.loc[GenRef, 'EsSlack'] = True
        msg = "Fijando Unidad: '{}' como referencia.".format(GenRef)
        logger.warning(msg)

    # 9.- Única unidad de referencia existente debe ser ingresada como Red Externa (Requerimientos de PandaPower)
    pdDF_GenRef = GenDisp[GenDisp['EsSlack']]
    for gen_nom, row_pdser in pdDF_GenRef.iterrows():  # there should be only one iteration. Done for completeness
        # gets the index bus number from connected bus (only one bus should exist)
        IndBarraConn = Grid['bus'][ Grid['bus']['name'] == row_pdser['NomBarConn'] ].index[0]
        pp__create_ext_grid( Grid, bus=IndBarraConn, vm_pu=1.0, va_degree=0.0, name=gen_nom,
                             max_p_kw=-row_pdser['PmaxMW'] * 1e3, min_p_kw=-row_pdser['PminMW'] * 1e3 )  # negativo para generación
        # 10.- Elimina el generador de referencia del DataFrame de disponibles para no ser reconocido como no referencia
        GenDisp.drop(labels=gen_nom, axis='index', inplace=True)

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
        logger.warn("Unssuplied system buses from Gen Ref. Deleting it...")
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
    ExtraData['CVarGenRef'] = pdDF_GenRef[['CVar']]  # DF
    # Nombre de la tecnología del generador de referencia
    ExtraData['TecGenSlack'] = pdDF_GenRef[['GenTec']]  # DF
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

    logger.info("! PS for stage {}/{} created.".format(StageNum, TotalEtas))

    return (Grid, ExtraData)


def calc_reservoir_costs_from_cota(DF_CotasEmbalsesXEtapa, DF_ParamHidEmb_hid, CostCurves = 'LogisticaS'):
    """
        Returns a DataFrame with stages as indices and reservoir names
        as column names with respective variable cost by given reservoir level.

        Variable cost units are the same as CVmin, CVmax of inputs variables (DF_ParamHidEmb_hid)

    Inputs:
        **DF_CotasEmbalsesXEtapa** (pd.DF)
        **DF_ParamHidEmb_hid** (pd.DF)
        **CostCurves** (str)
    """
    if CostCurves == 'LogisticaS':
        # get parameters needed as pandas series
        b = DF_ParamHidEmb_hid.loc['b', :]
        CVmin = DF_ParamHidEmb_hid.loc['CVmin', :]
        CVmax = DF_ParamHidEmb_hid.loc['CVmax', :]
        aux = DF_CotasEmbalsesXEtapa.astype(np__float128) + b  # overflow can occur!! Maximum precision requiered.
        DF = (CVmax - CVmin) / (1 + aux.apply(np__exp)) + CVmin
    else:
        msg = "CostCurves type '{}' are not supported for the moment."
        logger.error(msg)
        raise ValueError(msg)
    return DF


def find_maximum_nodes_available():
    """
        Uses predefined bash command to find the number of nodes
        available in the cluster, if 'sinfo' is found.
    """
    sinfo_cmd = "sinfo -N | grep idle | wc -l"  # slurm must be installed
    res_cmd = sp__run([sinfo_cmd], shell=True, stdout=sp__PIPE, stderr=sp__PIPE)
    # convert to string the outputs. If empty use empty string.
    res_cmd_stdout = res_cmd.stdout.decode() if res_cmd.stdout else ''   # str
    res_cmd_stderr = res_cmd.stderr.decode() if res_cmd.stderr else ''  # str
    if 'sinfo: not found' in res_cmd_stderr:
        msg = "sinfo binary (slurm) not found."
        logger.error(msg); raise sinfoNotAvailable(msg)
    elif not res_cmd_stderr:  # no errors found in bash prompt
        try:
            max_av_nodes = int(res_cmd_stdout)
        except ValueError:
            msg = "Result from sinfo filtered were not an integer."
            logger.error(msg); raise SlurmCallError(msg)
        except Exception:
            msg = "Could not convert stdout results from batch command to integer."
            msg += "Command results: {}".format(res_cmd_stdout)
            logger.error(msg); raise SlurmCallError(msg)
    else:  # Other errors appeared when looking for available nodes
        msg = "something else happend when looking for available nodes. Stderr was not empty."
        logger.error(msg); raise SlurmCallError(msg)

    return max_av_nodes
