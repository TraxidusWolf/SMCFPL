from smcfpl.in_out_files import read_sheets_to_dataframes
from os import sep as os__sep
from os.path import exists as os__path__exists, sep as os__path__sep
from os import makedirs as os__makedirs
from sys import executable as sys__executable
from subprocess import run as sp__run, PIPE as sp__PIPE
from pandas import DataFrame as pd__DataFrame
from pandas import concat as pd__concat
from datetime import datetime as dt
from dateutil import relativedelta as du__relativedelta
from shutil import rmtree as shutil__rmtree
from pandapower import create_empty_network as pp__create_empty_network, create_buses as pp__create_buses
from pandapower import create_line as pp__create_line, create_std_types as pp__create_std_types
from pandapower import create_transformer as pp__create_transformer, create_transformer3w as pp__create_transformer3w
from pandapower import create_load as pp__create_load, create_gen as pp__create_gen
from pandapower import create_ext_grid as pp__create_ext_grid, to_pickle as pp__to_pickle
from multiprocessing import cpu_count as mu__cpu_count, Pool as mu__Pool
from json import dump as json__dump
import smcfpl.aux_funcs as aux_smcfpl
import smcfpl.SendWork2Nodes as SendWork2Nodes
import smcfpl.NucleoCalculo as NucleoCalculo

import logging
logging.basicConfig(level=logging.DEBUG,
                    format="[%(levelname)s][%(asctime)s] - %(message)s")
logger = logging.getLogger()


class Simulacion(object):
    """
        Clase base que contiene los atributos y métodos de la simulación para ejecutar el modelo exitosamente.
        Guarda las base de datos en memoria (pandas dataframe, diccionarios, etc), desde los cuales adquiere los datos para cada etapa. Ojo, durante
        paralelismo se aumentarán los requerimientos de memoria según la cantidad de tareas.
    """

    def __init__(self, InFilePath, OutFilePath, Sbase_MVA, MaxItCongInter, MaxItCongIntra, FechaComienzo, FechaTermino,
                 NumVecesDem, NumVecesGen, PerdCoseno, PEHidSeca, PEHidMed, PEHidHum, DesvEstDespCenEyS, DesvEstDespCenP,
                 ParallelMode, NumParallelCPU, UsaSlurm):
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
        logger.debug("! inicializando clase Simulacion(...)  (CreaElementos.py)...")
        #
        # Atributos desde entradas
        self.InFilePath = InFilePath    # (str)
        self.Sbase_MVA = Sbase_MVA  # (float)
        self.MaxItCongInter = MaxItCongInter    # (int)
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
        self.ParallelMode = ParallelMode    # (bool)
        if (isinstance(NumParallelCPU, int)) | (NumParallelCPU is False) | (NumParallelCPU == 'Max'):
            self.NumParallelCPU = NumParallelCPU  # (int)
        else:
            msg = "Input 'NumParallelCPU' debe ser integer, False, o 'Max'."
            logger.error(msg)
            raise ValueError(msg)
        self.UsaSlurm = UsaSlurm  # (bool)

        #
        # Atributos extra
        # self.ModulePath = os__path__dirname(os__path__abspath(__file__))

        FileName = self.InFilePath.split(os__sep)[-1]
        PathInput = self.InFilePath.split(os__sep)[:-1]
        # lee archivos de entrada
        self.DFs_Entradas = read_sheets_to_dataframes(os__sep.join(PathInput), FileName, NumParallelCPU)
        # Determina duración de las etapas  (1-indexed)
        self.BD_Etapas = Crea_Etapas(self.DFs_Entradas['df_in_smcfpl_mantbarras'],
                                     self.DFs_Entradas['df_in_smcfpl_manttx'],
                                     self.DFs_Entradas['df_in_smcfpl_mantgen'],
                                     self.DFs_Entradas['df_in_smcfpl_mantcargas'],
                                     self.DFs_Entradas['df_in_smcfpl_histsolar'],
                                     self.DFs_Entradas['df_in_smcfpl_histeolicas'],
                                     self.FechaComienzo,
                                     self.FechaTermino)
        # print('self.BD_Etapas:', self.BD_Etapas)
        # Numero total de etapas
        self.NEta = self.BD_Etapas.shape[0]
        #
        # IDENTIFICA LA INFORMACIÓN QUE LE CORRESPONDE A CADA ETAPA (Siguiente BD son todas en etapas):
        #
        # Calcula y convierte valor a etapas de la desviación histórica de la demanda... (pandas Dataframe)
        self.BD_DemSistDesv = aux_smcfpl.DesvDemandaHistoricaSistema_a_Etapa(self.DFs_Entradas['df_in_scmfpl_histdemsist'], self.BD_Etapas)
        # Obtiene y convierte la demanda proyectada a cada etapas... (pandas Dataframe)
        self.BD_DemTasaCrecEsp = aux_smcfpl.TasaDemandaEsperada_a_Etapa(self.DFs_Entradas['df_in_smcfpl_proydem'], self.BD_Etapas, self.FechaComienzo, self.FechaTermino)
        # Unifica datos de demanda anteriores por etapa (pandas Dataframe)
        self.BD_DemProy = pd__concat([self.BD_DemTasaCrecEsp, self.BD_DemSistDesv.abs()], axis = 'columns')
        # Almacena la PE de cada año para cada hidrología (pandas Dataframe)
        self.BD_Hidrologias_futuras = aux_smcfpl.Crea_hidrologias_futuras(self.DFs_Entradas['df_in_smcfpl_histhid'], self.BD_Etapas, self.PEHidSeca, self.PEHidMed, self.PEHidHum, self.FechaComienzo, self.FechaTermino)
        # Respecto a la base de datos 'in_smcfpl_ParamHidEmb' en self.DFs_Entradas['df_in_smcfpl_ParamHidEmb'], ésta es dependiente de hidrologías solamente
        # Respecto a la base de datos 'in_smcfpl_seriesconf' en self.DFs_Entradas['df_in_smcfpl_seriesconf'], ésta define configuración hidráulica fija
        # Almacena la TSF por etapa de las tecnologías (pandas Dataframe)
        self.BD_TSFProy = aux_smcfpl.TSF_Proyectada_a_Etapa(self.DFs_Entradas['df_in_smcfpl_tsfproy'], self.BD_Etapas, self.FechaComienzo)
        # Convierte los dataframe de mantenimientos a etapas dentro de un diccionario con su nombre como key
        self.BD_MantEnEta = aux_smcfpl.Mantenimientos_a_etapas( self.DFs_Entradas['df_in_smcfpl_mantbarras'], self.DFs_Entradas['df_in_smcfpl_manttx'],
                                                                self.DFs_Entradas['df_in_smcfpl_mantgen'], self.DFs_Entradas['df_in_smcfpl_mantcargas'],
                                                                self.BD_Etapas)
        # Por cada etapa crea el SEP correspondiente (...paralelizable...) (dict of pandaNetworks and extradata)
        self.BD_RedesXEtapa = Crea_SEPxEtapa( self.DFs_Entradas['df_in_smcfpl_tecbarras'], self.DFs_Entradas['df_in_smcfpl_teclineas'],
                                              self.DFs_Entradas['df_in_smcfpl_tectrafos2w'], self.DFs_Entradas['df_in_smcfpl_tectrafos3w'],
                                              self.DFs_Entradas['df_in_smcfpl_tipolineas'], self.DFs_Entradas['df_in_smcfpl_tipotrafos2w'],
                                              self.DFs_Entradas['df_in_smcfpl_tipotrafos3w'], self.DFs_Entradas['df_in_smcfpl_tecgen'],
                                              self.DFs_Entradas['df_in_smcfpl_teccargas'], self.BD_MantEnEta, self.BD_Etapas, self.Sbase_MVA,
                                              NumParallelCPU)
        # print("self.BD_RedesXEtapa:", self.BD_RedesXEtapa)
        self.BD_HistGenRenovable = aux_smcfpl.GenHistorica_a_Etapa(self.BD_Etapas,
                                                                   self.DFs_Entradas['df_in_smcfpl_histsolar'],
                                                                   self.DFs_Entradas['df_in_smcfpl_histeolicas'])
        # print('self.BD_HistGenRenovable:', self.BD_HistGenRenovable)
        #
        # Obtiene partes del diccionario ExtraData por etapa
        self.TecGenSlack = [d['ExtraData']['TecGenSlack'] for d in self.BD_RedesXEtapa.values()]
        # Crea lista del Número de Cargas en cada Etapa/Grid
        self.ListNumLoads = [d['ExtraData']['NumLoads'] for d in self.BD_RedesXEtapa.values()]
        # Crea lista del Número de Unidades de Generación en cada Etapa/Grid
        # self.ListNumGenNoSlack = [d['ExtraData']['NumGenNoSlack'] for d in self.BD_RedesXEtapa.values()]
        # Crea lista del tipos (Número de Unidades intrínseco) de Generación en cada Etapa/Grid
        self.ListTiposGenNoSlack = [d['ExtraData']['Tipos'] for d in self.BD_RedesXEtapa.values()]

        logger.debug("! inicialización clase Simulacion(...) (CreaElementos.py) Finalizada!")

    def ImprimeBDs(self):
        """
            Imprime en el directorio temporal definido 'self.TempFolderName', las siguientes base de datos.
        """
        logger.info("Exportando a archivos temporales ...")
        # Guarda una copia base de datos de etapas en los archivos
        self.BD_Etapas.to_csv(self.TempFolderName + os__path__sep + 'BD_Etapas.csv')
        # Guarda una copia base de datos de Parámetros de hidrologías de los embalses en los archivos
        self.DFs_Entradas['df_in_smcfpl_ParamHidEmb'].to_csv(self.TempFolderName + os__path__sep + 'ParamHidEmb.csv')
        # Guarda una copia base de datos de configuración hidráulica en los archivos
        self.DFs_Entradas['df_in_smcfpl_seriesconf'].to_csv(self.TempFolderName + os__path__sep + 'seriesconf.csv')
        # Guarda una copia base de datos de Proyección de la Demanda Sistema
        self.BD_DemProy.to_csv(self.TempFolderName + os__path__sep + 'BD_DemProy.csv')
        # Guarda una copia base de datos de la Probabilidad de Excedencia (PE) por etapa
        self.BD_Hidrologias_futuras.to_csv(self.TempFolderName + os__path__sep + 'BD_Hidrologias_futuras.csv')
        # Guarda una copia base de datos de la Tasa de Falla/Salida Forzada en el directorio temporal
        self.BD_TSFProy.to_csv(self.TempFolderName + os__path__sep + 'BD_TSFProy.csv')

        # Imprime las los archivos de Redes/Grids de cada etapa, para luego ser leídos por los nodos.
        for EtaNum in self.BD_Etapas.index:
            """ Por cada etapa imprime dos archivos, uno llamado '#.json' (donde # es el número de la etapa) con info extra de la etapa y, otro
            llamado 'Grid_#.json' que contiene la red asociada a la etapa casi lista para simular lpf.
            """
            BD_RedesXEtapa_ExtraData = self.BD_RedesXEtapa[EtaNum]['ExtraData']

            # Guarda Datos de etapa en archivo JSON
            with open(self.TempFolderName + os__path__sep + "{}.json".format(EtaNum), 'w') as f:
                json__dump(BD_RedesXEtapa_ExtraData, f)

                # Exporta la red a archivo pickle. Necesario para exportar tipos de lineas. Más pesado que JSON y levemente más lento pero funcional... :c
                pp__to_pickle( self.BD_RedesXEtapa[EtaNum]['PandaPowerNet'], self.TempFolderName + os__path__sep + "Grid_Eta{}.p".format(EtaNum) )
        logger.info("Exportando completado.")

    def run(self, delete_TempData=True):
        logger.debug("Corriendo método Simulacion.run()...")
        if bool(self.UsaSlurm) & delete_TempData:  # OJO: bool({}) -> False
            # Cuando finaliza se borran los archivos temporales
            shutil__rmtree(self.TempFolderName)

        # Comienza con la ejecución de lo cálculos
        if self.UsaSlurm:
            """
                En caso de declarar uso de Slurm, se trabaja con archivos en lugar de mantener la base de datos general en memoria RAM, incluyendo todo lo
                necesario (Reduce velocidad). Estos archivos son necesario para ser copiado a los nodos.
                Escribe en un directorio de trabajo temporal (desde de donde se ejecutó el script).

                Posibilidad de paralelismo mediante nodos en una configuración de un 'High-Performance Computer' con configuración beowulf y
                administrador de colas slurm.
            """
            self.TempFolderName = 'TempData'
            if not os__path__exists(self.TempFolderName):  # verifica que exista directorio, de lo contrario lo crea.
                os__makedirs(self.TempFolderName)

            # Método de clase creado para imprimir BDs más importantes
            self.ImprimeBDs()

            # Ejecuta el archivo que pone en cola de slurm el archivo 'NucleoCalculo' en los nodos, y resuelve cada
            # serie de etapas (caso) dada una hidrología y Dem-Gen. Información necesaria ya esta escrita en disco.
            SendWork2Nodes.Send( NNodos=self.UsaSlurm['NumNodos'], WTime=self.UsaSlurm['NodeWaittingTime'],
                                 NTasks=self.UsaSlurm['ntasks'], CPUxTask=self.UsaSlurm['cpu_per_tasks'])  # RETORNAR??
            # dividir trabajos entre total de nodos pedidos (NNodos)
            # Número de trabajos por nodo: (NTrabajo // NNodos = resultado entero)
            # Número de trabajos restantes para enviar en última corrida (en un nodo): (NTrabajo % NNodos = resto entero)
            #
            # COMPLETAR CUANDO FUNCIONE SLURM !!!
            #

        else:
            """
                En caso contrario trabaja con la información en memoria. Proceso no compatible con paralelización por nodo - cluster.
                Procedimiento jerárquico:
                    1.- Hidrología
                    1.- Definir Demanda
                    1.- Definir Despacho
                    1.- Resolución Etapas
            """
            # Paralelizar de alguna forma
            for HidNom in ['Humeda', 'Media', 'Seca']:
                # PEs por etapa asociadas a la Hidrología en cuestión
                DF_PEsXEtapa = self.BD_Hidrologias_futuras[['PE ' + HidNom + ' dec']]
                # Numero total Demandas: NumVecesDem
                for NDem in range(self.NumVecesDem):
                    #
                    # Numero total Despachos: NumVecesGen
                    for NGen in range(self.NumVecesGen):
                        PyGeneratorDemand_FreeCust = aux_smcfpl.GeneradorDemanda(Medias=self.BD_DemProy['TasaCliLib'].tolist(),
                                                                                 Sigmas=self.BD_DemProy['Desv_decimal'].tolist(),
                                                                                 NCargas=self.ListNumLoads)
                        PyGeneratorDemand_RegCust = aux_smcfpl.GeneradorDemanda(Medias=self.BD_DemProy['TasaCliReg'].tolist(),
                                                                                Sigmas=self.BD_DemProy['Desv_decimal'].tolist(),
                                                                                NCargas=self.ListNumLoads)
                        PyGeneratorDispatched = aux_smcfpl.GeneradorDespacho(Lista_TiposGen=self.ListTiposGenNoSlack,  # lista
                                                                             DF_HistGenERNC=self.BD_HistGenRenovable,  # tupla de dos pandas DataFrame
                                                                             DF_TSF=self.BD_TSFProy,  # para cada tecnología que recurra con falla se asigna
                                                                             DF_PE_Hid=DF_PEsXEtapa,  # pandas DataFrame
                                                                             DesvEstDespCenEyS=self.DesvEstDespCenEyS,
                                                                             DesvEstDespCenP=self.DesvEstDespCenP)
                        # Medias=,  # val histo (ERNC), pdf Uniforme (termo), o PE Hid (hidro)
                        # Sigmas=,  # val histo (ERNC), pdf Uniforme (termo), o 10%|20% (hidro)

                        # Corre directamente (Sin escribir archivos a disco) llama al archivo de resolución por caso o serie de etapas dada una hidrología y Dem-Gen
                        # -- Llamar al Núcleo de cálculo mediante linea de comando
                        # Out = sp__run([sys__executable, 'NucleoCalculo.py', 'Humeda'], stdout=sp__PIPE).stdout.decode('utf-8')  # Retorna string  <---- DEBE SER SCRIPT EJECUTABLE argv POR LOS NODOS
                        # -- Llama al Núcleo de cálculo como python function. Ejecuta 'el caso' o 'serie de etapas' dadas la condiciones de entrada
                        NucleoCalculo.Calcular(self.NEta, HidNom, PyGeneratorDemand_FreeCust, PyGeneratorDemand_RegCust, PyGeneratorDispatched,
                                               self.DesvEstDespCenEyS, self.DesvEstDespCenP, self.DFs_Entradas['df_in_smcfpl_ParamHidEmb'],
                                               self.DFs_Entradas['df_in_smcfpl_seriesconf'], self.BD_Etapas)
                        break
                    break
                break

        logger.debug("Corrida método Simulacion.run() finalizada!")
        return


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
    :param DF_MantBarras: DataFrame de los mantnimientos futuros a las barras simuladas.
    :type DF_MantGen: Pandas DataFrame
    :param DF_MantGen: DataFrame de los mantenimientos futuros a las unidades generadoras simuladas.
    :type DF_MantTx: Pandas DataFrame
    :param DF_MantTx: DataFrame de los mantnimientos futuros a los elementos del sistema de transmisión simulados.
    :type DF_MantLoad: Pandas DataFrame
    :param DF_MantLoad: DataFrame de los mantnimientos futuros a las cargas simuladas.
    :type DF_Solar: Pandas DataFrame
    :param DF_Solar: DataFrame del historial para la(s) unidad(es) tipo representativas para las unidades solares.
    :type DF_Eolicas: Pandas DataFrame
    :param DF_Eolicas: DataFrame del historial para la(s) unidad(es) tipo representativas para las unidades eólicos.
    :type FechaComienzo: Datetime object
    :param FechaComienzo: Fecha y hora de la primera hora de la simulación.
    :type FechaTermino: Datetime object
    :param FechaTermino: Fecha y hora de la última hora de la simulación.
    """
    logger.debug("! entrando en función: 'Crea_Etapas' (CreaElementos.py) ...")
    Etapas = Crea_Etapas_Topologicas(DF_MantBarras, DF_MantGen, DF_MantTx, DF_MantLoad, FechaComienzo, FechaTermino)
    logger.info("Se crearon {} etapas topológicas.".format(Etapas.shape[0]))
    Etapas = Crea_Etapas_Renovables(Etapas, DF_Solar, DF_Eolicas)
    logger.info("Se creó un total de {} etapas.".format(Etapas.shape[0]))
    logger.debug("! saliendo de función: 'Crea_Etapas' (CreaElementos.py) ...")
    return Etapas


def Crea_Etapas_Topologicas(DF_MantBarras, DF_MantGen, DF_MantTx, DF_MantLoad, FechaComienzo, FechaTermino):
    """
        Crea el DataFrame de las etapas topológicas. Identifica los cambio de fechas (con diferencia mayor a un día) que
        cambian "considerablemente" la topología del SEP mediante lo informado en los programas de mantenimiento (prácticamente mensual).

        1.- Filtra y ordena en forma ascendente las fechas de los mantenimientos.
        2.- Con el DF_CambioFechas identifica y crea las etapas bajo dos Metodologías y la función 'aux_smcfpl.Crea_Etapas_desde_Cambio_Mant'.
    """
    logger.debug("! entrando en función: 'Crea_Etapas_Topologicas' (CreaElementos.py) ...")
    # Juntar todos los cambios DE FECHAS en un único pandas series. (Inicializa única columna)
    DF_CambioFechas = pd__DataFrame(data=[FechaComienzo, FechaTermino], columns=[0])
    for df in (DF_MantBarras, DF_MantGen, DF_MantTx, DF_MantLoad):
        DF_CambioFechas = pd__concat([ DF_CambioFechas, df['FechaIni'], df['FechaFin'] ], axis=0, join='outer', ignore_index=True)
    # Elimina las fechas duplicadas
    DF_CambioFechas.drop_duplicates(keep='first', inplace=True)
    # Ordena en forma ascendente el pandas series
    DF_CambioFechas.sort_values(by=[0], ascending=True, inplace=True)
    # Resetea los indices
    DF_CambioFechas.reset_index(drop=True, inplace=True)    # Es un DataFrame con datetime (detalle horario) de la existencia de todos los cambios.
    # print('DF_CambioFechas:\n', DF_CambioFechas)

    Metodo_RefFila_EtaTopo = 2
    print('Metodo_RefFila_EtaTopo:', Metodo_RefFila_EtaTopo)
    logger.info("Utilizando Método {} para diferencia de filas en Creación Etapas Topológicas.".format(Metodo_RefFila_EtaTopo))
    if Metodo_RefFila_EtaTopo == 1:
        """ Método 1: Metodo_RefFila_EtaTopo = 1 (Diferencia entre filas - referencia móvil)
        En forma progresiva, desde la primera hasta la penúltima fila de cambios topológicos, se observa la diferencia de días
        existentes entre la fila de referencia o fila actual y la siguiente. De ser ésta mayor a un día, se mueve la referencia
        a la siguiente fila y se mide con respecto a la nueva siguiente. Se continúa observando los cambios entre filas hasta
        cumplir con la condición requerida para así asignar la fecha como límite de etapa.

        Notar que el último valor no es considerado (por reducción de indice en comparación y ser éste la fecha de termino de simulación).
        While es necesario para facilitar el salto de filas en iteración.

        ¿Qué hace en el caso de existir una fecha con menos de un día c/r a fecha termino, sin previa etapa limitante?
        """
        logger.debug("! saliendo de función: 'Crea_Etapas_Topologicas' (CreaElementos.py) ...")
        return aux_smcfpl.Crea_Etapas_desde_Cambio_Mant(DF_CambioFechas, ref_fija=False)
    elif Metodo_RefFila_EtaTopo == 2:
        """ Método 2: Metodo_RefFila_EtaTopo (Diferencia respecto fila referencia)
        En forma progresiva, desde la primera hasta la penúltima fila de cambios topológicos, se observa la
        diferencia de cías existentes entre la fila actual (o de referencia) y la que le sigue. De ser ésta
        menor a un día, se calcula la diferencia de la fila de referencia con respecto a la sub-siguiente hasta encontrarse
        una diferencia mayor al límite requerido.

        Notar que el valor de la fila saltado no es considerado a futuro, por lo que se considera como si no existiese.
        While es necesario para facilitar el salto de filas en iteración.

        ¿Qué hace en el caso de existir una fecha con menos de un día c/r a fecha termino, sin previa etapa limitante?
        """
        logger.debug("! saliendo de función: 'Crea_Etapas_Topologicas' (CreaElementos.py) ...")
        return aux_smcfpl.Crea_Etapas_desde_Cambio_Mant(DF_CambioFechas, ref_fija=True)
    else:
        msg = "Metodo_RefFila_EtaTopo No fue ingresado válidamente en función 'Crea_Etapas_Topologicas' (CreaElementos.py)."
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
    logger.debug("! entrando en función: 'Crea_Etapas_Renovables' (CreaElementos.py) ...")
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
        DF_ERNC = DF_ERNC.divide(MaximoAnual, axis='columns')  # usa máximo anual en lugar del de la etapa (representación anual)  # 27 etapas total aprox SEP 39us

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

    logger.debug("! saliendo de función: 'Crea_Etapas_Renovables' (CreaElementos.py) ...")
    return DF_Eta


def Crea_SEPxEtapa( DF_TecBarras, DF_TecLineas, DF_TecTrafos2w, DF_TecTrafos3w, DF_TipoLineas, DF_TipoTrafos2w, DF_TipoTrafos3w, DF_TecGen,
                    DF_TecCargas, Dict_DF_Mantenimientos, DF_Etapas, Sbase_MVA, NumParallelCPU):
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
    # Inicializa diccionario de salida con los índices de las etapas
    DictSalida = dict.fromkeys( DF_Etapas.index.tolist() )
    if not NumParallelCPU:
        for EtaNum, Etapa in DF_Etapas.iterrows():
            print("EtaNum:", EtaNum)
            Grid, ExtraData = CompletaSEP_PandaPower(DF_TecBarras, DF_TecLineas, DF_TecTrafos2w, DF_TecTrafos3w, DF_TipoLineas, DF_TipoTrafos2w,
                                                     DF_TipoTrafos3w, DF_TecGen, DF_TecCargas, Dict_DF_Mantenimientos, EtaNum, Sbase_MVA)
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
                Pool.apply_async(CompletaSEP_PandaPower, (DF_TecBarras, DF_TecLineas, DF_TecTrafos2w, DF_TecTrafos3w, DF_TipoLineas, DF_TipoTrafos2w,
                                                          DF_TipoTrafos3w, DF_TecGen, DF_TecCargas, Dict_DF_Mantenimientos, EtaNum, Sbase_MVA)),
                EtaNum])
        # Obtiene los resultados del paralelismo y asigna a variables de interés
        for result, EtaNum in Results:
            Grid, ExtraData = result.get()
            DictSalida[EtaNum] = {'PandaPowerNet': Grid}
            DictSalida[EtaNum]['ExtraData'] = ExtraData

    logger.debug("! saliendo en función: 'Crea_SEPxEtapa' (CrearElementos.py) ...")
    return DictSalida


def CompletaSEP_PandaPower(DF_TecBarras, DF_TecLineas, DF_TecTrafos2w, DF_TecTrafos3w, DF_TipoLineas, DF_TipoTrafos2w, DF_TipoTrafos3w, DF_TecGen,
                           DF_TecCargas, Dict_DF_Mantenimientos, EtaNum, Sbase_MVA):
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

        Retorna una tupla con (PandaPower Grid filled, ExtraInfo)
    """
    logger.debug("! Creando SEP en etapa {} ...".format(EtaNum))
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
        # 7.1- identifica los trafos que se definan operativos, con flag Operativa == True para sobrescribir parámetros
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
        msg = "NO Existe Unidad definida de referencia para la Etapa {}! ...".format(EtaNum)
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
    else:
        pass

    # 9.- Única unidad de referencia existente debe ser ingresada como Red Externa (Requerimientos de PandaPower)
    pdSerie_GenRef = GenDisp[GenDisp['EsSlack']].squeeze()  # convert single row pandas DataFrame to Series
    IndBarraConn = Grid['bus'][ Grid['bus']['name'] == pdSerie_GenRef['NomBarConn'] ].index[0]
    pp__create_ext_grid( Grid, bus=IndBarraConn, vm_pu=1.0, va_degree=0.0, name=pdSerie_GenRef.name,
                         max_p_kw=pdSerie_GenRef['PmaxMW'] * 1e3, min_p_kw=pdSerie_GenRef['PminMW'] * 1e3 )
    # 10.- Elimina el generador de referencia del DataFrame de disponibles
    GenDisp.drop(labels=pdSerie_GenRef.name, axis='index', inplace=True)

    # 11.- Por cada Generador disponible crea el correspondiente elemento en la RED
    for GenNom, Generador in GenDisp.iterrows():
        IndBarraConn = Grid['bus'][ Grid['bus']['name'] == Generador['NomBarConn'] ].index[0]
        # Notar que se le asigna la potencia nominal a la carga. Ésta es posteriormente modificada según los parámetros de la etapa en cada proceso
        pp__create_gen(Grid, bus=IndBarraConn, name=GenNom, p_kw=-Generador['PmaxMW'] * 1e3,
                       min_p_kw=Generador['PminMW'] * 1e3, type=Generador['GenTec'])  # p_kw es negativo para generación

    # 12.- Actualiza el diccionario ExtraData con información adicional
    ExtraData['CVarGenRef'] = float(pdSerie_GenRef['CVar'])  # costo variable unidad de referencia (Red Externa)
    ExtraData['TecGenSlack'] = str(pdSerie_GenRef['GenTec'])  # Nombre de la tecnología del generador de referencia
    ExtraData['NumLoads'] = Grid['load'].shape[0]  # Número de cargas existentes por etapa
    # ExtraData['NumGenNoSlack'] = Grid['gen'].shape[0]  # Número de generadores (no slack) existentes por etapa
    ExtraData['Tipos'] = Grid['gen'][['type']]  # pandas DataFrame del índice de generadores en la Grilla y Tipo de tecnología

    logger.debug("! SEP en etapa {} creado.".format(EtaNum))
    return (Grid, ExtraData)
