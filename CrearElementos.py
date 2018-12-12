from smcfpl.in_out_files import read_sheets_to_dataframes
from os import sep as os__sep
from os.path import exists as os__path_exists
from os import makedirs as os__makedirs
from pandas import DataFrame as pd__DataFrame
from pandas import concat as pd__concat
from datetime import datetime as dt
from dateutil import relativedelta as du__relativedelta
import smcfpl.aux_funcs as aux_smcfpl

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

    def __init__(self, InFilePath, OutFilePath, Sbase_MVA, MaxItCongInter, MaxItCongIntra,
                 FechaComienzo, FechaTermino, NumVecesDem, NumVecesGen, PerdCoseno,
                 PEHidSeca, PEHidMed, PEHidHum, ParallelMode, NumDespParall, UsaArchivosParaEtapas, UsaSlurm):
        """
            :param UsaArchivosParaEtapas: Valor verdadero define si escribe un archivo (eliminado al final de la ejecución) por cada etapa para trabajar en paralelismo cada uno
            de ellos sin la posibilidad de coincidencia. Valor es considerado verdadero si: UsaSlurm is not None.
            :type UsaArchivosParaEtapas: bool.

            :param UsaSlurm: Diccionario con parámetros para ejecución en el sistema de colas de slurm. Hace que se ejecuten comandos (con biblioteca subprocess) sbatch propios de slurm para ejecución en varios nodos.
                            Formato: {'NumNodos': (int), 'NodeWaittingTime': (datetime deltatime object), 'ntasks': (int), 'cpu_per_tasks': (int)}
                                    'NumNodos': Número de nodos a utilizar en el cluster.
                                    'NodeWaittingTime': Tiempo de espera máximo de ejecución de los procesos enviados a nodos.
                                    'ntasks': número de tareas a repartirse por nodo.
                                    'cpu-per-tasks': Número de cpu requeridas para cada tarea.
            :type UsaSlurm: dict

        """
        logger.debug("! inicializando clase Simulacion(...)  (CreaElementos.py)...")
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
        self.PEHidSeca = PEHidSeca  # (float)
        self.PEHidMed = PEHidMed    # (float)
        self.PEHidHum = PEHidHum    # (float)
        self.ParallelMode = ParallelMode    # (bool)
        self.NumDespParall = NumDespParall  # (int)

        FileName = self.InFilePath.split(os__sep)[-1]
        PathInput = self.InFilePath.split(os__sep)[:-1]
        # lee archivos de entrada
        self.DFs_Entradas = read_sheets_to_dataframes(os__sep.join(PathInput), FileName)
        # Determina duración de las etapas  (1-indexed)
        self.BD_Etapas = Crea_Etapas(self.DFs_Entradas['df_in_smcfpl_mantbarras'],
                                     self.DFs_Entradas['df_in_smcfpl_manttx'],
                                     self.DFs_Entradas['df_in_smcfpl_mantgen'],
                                     self.DFs_Entradas['df_in_smcfpl_mantcargas'],
                                     self.DFs_Entradas['df_in_smcfpl_histsolar'],
                                     self.DFs_Entradas['df_in_smcfpl_histeolicas'],
                                     self.FechaComienzo,
                                     self.FechaTermino)
        print('self.BD_Etapas:', self.BD_Etapas)
        #
        # IDENTIFICA LA INFORMACIÓN QUE LE CORRESPONDE A CADA ETAPA (Siguiente BD son todas en etapas):
        #
        # Calcula y convierte valor a etapas de la desviación histórica de la demanda... (pandas Dataframe)
        self.BD_DemSistDesv = aux_smcfpl.DesvDemandaHistoricaSistema_a_Etapa(self.DFs_Entradas['df_in_scmfpl_histdemsist'], self.BD_Etapas)
        # Obtiene y convierte la demanda proyectada a cada etapas... (pandas Dataframe)
        self.BD_DemTasaCrecEsp = aux_smcfpl.TasaDemandaEsperada_a_Etapa(self.DFs_Entradas['df_in_smcfpl_proydem'], self.BD_Etapas, self.FechaComienzo, self.FechaTermino)
        # Unifica datos de demanda por etapa (pandas Dataframe)
        self.BD_DemProy = pd__concat([self.BD_DemTasaCrecEsp, self.BD_DemSistDesv], axis='columns')
        # Almacena la PE de cada año para cada hidrología (pandas Dataframe)
        self.BD_Hidrologias_futuras = aux_smcfpl.Crea_hidrologias_futuras(self.DFs_Entradas['df_in_smcfpl_histhid'], self.BD_Etapas, self.PEHidSeca, self.PEHidMed, self.PEHidHum, self.FechaComienzo, self.FechaTermino)
        # Almacena la TSF por etapa de las tecnologías (pandas Dataframe)
        self.BD_TSFProy = aux_smcfpl.TSF_Proyectada_a_Etapa(self.DFs_Entradas['df_in_smcfpl_tsfproy'], self.BD_Etapas, self.FechaComienzo)
        # Convierte los dataframe de mantenimientos a etapas dentro de un diccionario con su nombre como key
        self.BD_MantEnEta = aux_smcfpl.Mantenimientos_a_etapas( self.DFs_Entradas['df_in_smcfpl_mantbarras'], self.DFs_Entradas['df_in_smcfpl_manttx'],
                                                                self.DFs_Entradas['df_in_smcfpl_mantgen'], self.DFs_Entradas['df_in_smcfpl_mantcargas'],
                                                                self.BD_Etapas)
        # Por cada etapa crea el SEP correspondiente (...paralelizable...) (dict of pandaNetworks and extradata)
        self.BD_RedesXEtapa = aux_smcfpl.Crea_SEPxEtapa( self.DFs_Entradas['df_in_smcfpl_tecbarras'], self.DFs_Entradas['df_in_smcfpl_teclineas'],
                                                         self.DFs_Entradas['df_in_smcfpl_tectrafos2w'], self.DFs_Entradas['df_in_smcfpl_tectrafos3w'],
                                                         self.DFs_Entradas['df_in_smcfpl_tipolineas'], self.DFs_Entradas['df_in_smcfpl_tipotrafos2w'],
                                                         self.DFs_Entradas['df_in_smcfpl_tipotrafos3w'], self.DFs_Entradas['df_in_smcfpl_tecgen'],
                                                         self.DFs_Entradas['df_in_smcfpl_teccargas'], self.BD_MantEnEta, self.BD_Etapas, self.Sbase_MVA)
        print("self.BD_RedesXEtapa:", self.BD_RedesXEtapa)

        if UsaArchivosParaEtapas:
            """
                En caso de solicitarse trabajar con archivos en lugar de mantener la base de datos general en memoria ram más todo lo
                necesario. (Reduce velocidad)
                Escribe en un directorio de trabajo temporal (desde de donde se ejecutó el script).
            """
            if not os__path_exists('TempData'):  # verifica que exista directorio, de lo contrario lo crea.
                os__makedirs('TempData')
            # escribe base de datos a archivos
            self.BD_Etapas.to_csv('./TempData/BD_Etapas.csv')
        else:
            """En caso contrario deja toda la información en memoria mientras la obtiene. Proceso no compatible con paralelización por nodo"""
            pass
        # if UsaSlurm:
        #     """Posibilidad de paralelismo mediante nodos en una configuración de un 'High-Performance Computer' con configuración beowulf y administrador de colas slurm"""
        #     self.BDxEtapa()
        # else:
        #     """Ejecuta todo en un solo computador."""
        #     pass
        logger.debug("! inicialización clase Simulacion(...) (CreaElementos.py) Finalizada!")

    # def BDxEtapa(self):
        """Por cada etapa existente en self.BD_Etapas, filtra la información correspondientes a ésta. Esto es, obtiene los elementos existentes por mantenciones y valores de demanda y generación"""
        # for Eta in self.BD_Etapas.iterrows():
        #    print("Eta:", Eta)

    # def Prepara_archivos_multinodo():
    #     pass

    def run(self):
        logger.debug("Corriendo método Simulacion.run()")
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
    logger.debug("! entrando en función: 'Crea_Etapas_Topologicas' (CreaElementos.py) ...")
    # Juntar todos los cambios de fechas en un único pandas series. (Inicializa única columna)
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
    logger.debug("! entrando en función: 'Crea_Etapas_Renovables' (CreaElementos.py) ...")
    # inicializa DataFrame de salida
    DF_Eta = pd__DataFrame(columns=['FechaIni_EtaTopo', 'FechaFin_EtaTopo', 'HoraDiaIni', 'HoraDiaFin', 'TotalHoras'])
    # print('Etapas:\n', Etapas)
    for row in Etapas.iterrows():
        FInicioEta = row[1]['FechaIni']
        FTerminoEta = row[1]['FechaFin']
        # NumAniosNecesarios = FTerminoEta.year - FInicioEta.year + 1
        # print('FInicioEta', FInicioEta)
        # print('FTerminoEta', FTerminoEta)
        # print('NumAniosNecesarios', NumAniosNecesarios)

        """ Unifica los DataFrame ERNC en DF_aux con índices datetime (año,mes,dia,hora ) asignándole cada año existente en la división de
        etapas, desde la inicial. Posteriormente es filtrado según los mismo número de etapas eliminando los primeros y los últimos
        (conservándose estos últimos)

        Notar que al existir una etapa existente entre más de año, la BD ERNC se considera cíclica.
        """
        # inicializa dataframe para incorporar generación previa de las 4 unidades tipo.
        DF_ERNC = DF_Solar.join(DF_Eolicas)
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

        # Warning en caso de ser todo el dataframe vacío
        if 0 in DF_ERNC.max(axis=0):
            msg = "El máximo de uno de los datos de las columnas es 0!\n Valor encontrado entre fechas {} y {}.".format(FInicioEta, FTerminoEta)
            logger.warn(msg)

        # Normaliza los valores respecto al máximo de aquellos existentes en la etapa.
        DF_ERNC = DF_ERNC.divide(DF_ERNC.max(axis=0), axis='columns')

        # print('DF_ERNC:\n', DF_ERNC)

        # En DF_Cambios obtiene el nombre de columna de aquel con mayor valor para las horas de la etapa
        DF_Cambios = DF_ERNC.idxmax(axis=1)  # axis=1 | axis='columns' : columns-wise
        # print('DF_Cambios:\n', DF_Cambios)

        # Encuentra los índices que son distintos. Los desfaza hacia abajo (1 periodo), y rellena el vació con el valor siguiente encontrado
        DF_Cambios = DF_Cambios.ne(DF_Cambios.shift(periods=1).fillna(method='bfill'))  # boolean single-column
        # print('DF_Cambios:\n', DF_Cambios)
        # obtiene los elementos que son de cambio, según lo encontrado previamente. Misma variable ya que son lo que finalmente importa
        DF_ERNC = DF_ERNC[ DF_Cambios.values ]
        # print('DF_ERNC:\n', DF_ERNC)
        # print('DF_ERNC.shape:\n', DF_ERNC.shape)
        # print()

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
        # print('DF_etapas2:\n', DF_etapas2)
        # print()

        DF_Eta = DF_Eta.append( DF_etapas2 )
    # reinicia los indices, les suma uno y asigna nombre de 'EtaNum'. Todo para facilidad de comprensión
    DF_Eta.reset_index(drop=True, inplace=True)
    DF_Eta.index += 1
    DF_Eta.index.name = 'EtaNum'
    # print('DF_Eta:\n', DF_Eta)

    logger.debug("! saliendo de función: 'Crea_Etapas_Renovables' (CreaElementos.py) ...")
    return DF_Eta
