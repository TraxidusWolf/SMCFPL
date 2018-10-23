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
    # Juntar todos los cambios de fechas en un único pandas series.
    DF_CambioFechas = pd__DataFrame(data=[FechaComienzo, FechaTermino], columns=[0])
    for df in (DF_MantBarras, DF_MantGen, DF_MantTx, DF_MantLoad):
        DF_CambioFechas = pd__concat([ DF_CambioFechas, df['FechaIni'], df['FechaFin'] ], axis=0, join='outer', ignore_index=True)
    # Elimina las fechas duplicadas
    DF_CambioFechas.drop_duplicates(keep='first', inplace=True)
    # Ordena en forma ascendente el pandas series
    DF_CambioFechas.sort_values(by=[0], ascending=True, inplace=True)
    # Resetea los indices
    DF_CambioFechas.reset_index(drop=True, inplace=True)

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
    # print(DF_Solar)
    # print(DF_Eolicas)
    logger.debug("! saliendo de función: 'Crea_2da_div_Etapas' ...")
    return Etapas
