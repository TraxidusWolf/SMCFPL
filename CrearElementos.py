from smcfpl.in_out_files import read_sheets_to_dataframes

from os import sep as os__sep
from pandas import DataFrame as pd__DataFrame
from pandas import concat as pd__concat
from datetime import datetime as dt
from dateutil import relativedelta as du__relativedelta
from smcfpl.aux_funcs import *

import logging
logging.basicConfig(level=logging.DEBUG,
                    format="[%(levelname)s][%(asctime)s] - %(message)s")
logger = logging.getLogger()


class Simulacion(object):
    """Clase que contiene los parámetros de simulación para ejecutar el modelo en cada instancia de la simulación."""

    def __init__(self, InFilePath, OutFilePath, Sbase_MVA, MaxItCongInter, MaxItCongIntra,
                 FechaComienzo, FechaTermino, NumVecesDem, NumVecesGen, PerdCoseno,
                 PEHidSeca, PEHidMed, PEHidHum, ParallelMode, NumDespParall):
        logger.debug("! inicializando clase Simulacion(...) ...")
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
        # Determina duración de las etapas
        self.BD_Etapas = Crea_Etapas(self.DFs_Entradas['df_in_smcfpl_mantbarras'],
                                     self.DFs_Entradas['df_in_smcfpl_mantgen'],
                                     self.DFs_Entradas['df_in_smcfpl_manttx'],
                                     self.DFs_Entradas['df_in_smcfpl_mantcargas'],
                                     self.DFs_Entradas['df_in_smcfpl_histsolar'],
                                     self.DFs_Entradas['df_in_smcfpl_histeolicas'],
                                     self.FechaComienzo,
                                     self.FechaTermino)
        # transforma temporalidad de entradas al rango especificado por las etapas
        # print('self.BD_Etapas:\n', self.BD_Etapas)

        logger.debug("! inicialización clase Simulacion(...) Finalizada!")

    def run(self):
        logger.debug("Corriendo método Simulacion.run()")
        return


def Crea_Etapas(DF_MantBarras, DF_MantGen, DF_MantTx, DF_MantLoad, DF_Solar, DF_Eolicas, FechaComienzo, FechaTermino):
    logger.debug("! entrando en función: 'Crea_Etapas' ...")
    Etapas = Crea_1ra_div_Etapas(DF_MantBarras, DF_MantGen, DF_MantTx, DF_MantLoad, FechaComienzo, FechaTermino)
    Etapas = Crea_2da_div_Etapas(Etapas, DF_Solar, DF_Eolicas)
    logger.debug("! saliendo de función: 'Crea_Etapas' ...")
    return Etapas


def Crea_1ra_div_Etapas(DF_MantBarras, DF_MantGen, DF_MantTx, DF_MantLoad, FechaComienzo, FechaTermino):
    logger.debug("! entrando en función: 'Crea_1ra_div_Etapas' ...")
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

    MetodoUsar = 2
    print('MetodoUsar:', MetodoUsar)
    if MetodoUsar == 1:
        """ Método 1 (Diferencia entre filas)
        Para cada fila del DataFrame observa la diferencia con siguiente fecha. En
        caso de se menor a 1 día se sigue observando el siguiente a partir de este último. Finalmente,
        se selecciona el primer valor como aquel de referencia. Notar que el último valor no es
        considerado (por reducción de indice en comparación y ser éste la fecha de termino de simulación).
        While es necesario para facilitar el salto de filas en iteración."""
        logger.debug("! saliendo de función: 'Crea_1ra_div_Etapas' ...")
        return Crea_Etapas_desde_Cambio_Mant(DF_CambioFechas, ref=False)
    elif MetodoUsar == 2:
        """ Método 2 (Diferencia respecto fila referencia)
        Para cada fila 'fila' del DataFrame observa la diferencia con siguiente fecha. En
        caso de poseer menor diferencia a 1 día se sigue observando el siguiente respecto desde la misma fila. Notar que
        el valor de la fila saltado no es considerado a futuro, por lo que se considera como si no existiese.
        While es necesario para facilitar el salto de filas en iteración."""
        logger.debug("! saliendo de función: 'Crea_1ra_div_Etapas' ...")
        return Crea_Etapas_desde_Cambio_Mant(DF_CambioFechas, ref=True)
    else:
        msg = "MetodoUsar No fue ingresado válidamente en función 'Crea_1ra_div_Etapas'."
        logger.error(msg)
        raise ValueError(msg)


def Crea_2da_div_Etapas(Etapas, DF_Solar, DF_Eolicas):
    logger.debug("! entrando en función: 'Crea_2da_div_Etapas' ...")
    # inicializa DataFrame de salida
    DF_Eta = pd__DataFrame(columns=['FechaIni', 'FechaFin', 'HoraDiaIni', 'HoraDiaFin', 'TotalHoras'])
    print('Etapas:\n', Etapas)
    for row in Etapas.iterrows():
        FInicioEta = row[1]['FechaIni']
        FTerminoEta = row[1]['FechaFin']
        NumAniosNecesarios = FTerminoEta.year - FInicioEta.year + 1
        print('FInicioEta', FInicioEta)
        print('FTerminoEta', FTerminoEta)
        print('NumAniosNecesarios', NumAniosNecesarios)

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
        print('Horas_Cambio:', Horas_Cambio)
        DF_etapas2 = Lista2DF_consecutivo(Lista=Horas_Cambio, incremento=1, NombreColumnas=['HoraDiaIni', 'HoraDiaFin'])
        # crea una nueva columna al inicio con el mismo valor para 'FechaFin'
        DF_etapas2.insert(loc=0, column='FechaFin', value=FTerminoEta)
        # crea una nueva columna al nuevo inicio con el mismo valor para 'FechaIni'
        DF_etapas2.insert(loc=0, column='FechaIni', value=FInicioEta)
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

    logger.debug("! saliendo de función: 'Crea_2da_div_Etapas' ...")
    return DF_Eta
