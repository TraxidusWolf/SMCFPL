from smcfpl.in_out_files import read_sheets_to_dataframes

from os import sep as os__sep
from pandas import DataFrame as pd__DataFrame
from pandas import concat as pd__concat
from pandas import to_datetime as pd__to_datetime
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
    DF_Etapas2 = pd__DataFrame()
    print('Etapas:\n', Etapas)
    # inicializa un contador de etapas
    ContadorEtapas = 1
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
        DF_ERNC = DF_Solar.join(DF_Eolicas).reset_index()
        # inicializa dataframe para incorporar generación previa de las 4 unidades tipo.
        DF_aux = pd__DataFrame()
        # Agrega el dataframe de generación ERNC previa al DF_aux considerando cada año de las etapas
        for DeltaAnio in range(NumAniosNecesarios):
            # Duplica DF_ERNC agregando indices con datetime. Cada año tiene un DF_aux
            DF_aux = DF_aux.append(
                DF_ERNC.set_index(
                    pd__to_datetime(
                        pd__DataFrame(
                            {
                                'year': FInicioEta.year + DeltaAnio,
                                'month': DF_ERNC['Mes'],
                                'day': DF_ERNC['Dia'],
                                'hour': DF_ERNC['Hora']
                            }
                        )
                    )
                )
            )
        # Limpia el DataFrame. Elimina columnas 'Mes', 'Dia', y 'Hora'
        DF_aux.drop(labels=['Mes', 'Dia', 'Hora'], inplace=True, axis='columns')
        # Utiliza el DF_aux para filtrarlo según las fechas de las etapas. Elimina fechas anteriores
        DF_aux = DF_aux[ FInicioEta <= DF_aux.index ]
        # Ahora elimina fechas posteriores
        DF_aux = DF_aux[ DF_aux.index <= FTerminoEta ]
        print( 'DF_aux.max(axis=0):\n', DF_aux.max(axis=0) )

        # Warning en caso de ser todo el dataframe vacío
        if 0 in DF_aux.max(axis=0):
            msg = "El máximo de uno de los datos de las columnas es 0!\n Valor encontrado entre fechas {} y {}.".format(FInicioEta, FTerminoEta)
            logger.warn(msg)

        # Normaliza los valores respecto al máximo de aquellos existentes en la etapa.
        DF_aux = DF_aux.divide(DF_aux.max(axis=0), axis='columns')
        # DF_aux = DF_aux/DF_aux.max(axis=0)
        # print('DF_aux:\n', DF_aux)

        # Inicializa una lista para transformar a un DataFrame de salida
        Lista_Cambios = []

        # RE-corrobora que cada 1ra separación de etapa no sea menor que un día.
        if DF_ERNC.shape[0] < 24:
            msg = "Etapa {} posee menos de 24 hrs.".format(row[0])
            logger.error(msg)
            raise ValueError(msg)

        # recorre todas las horas del DF_ERNC en la etapa para identificar el indice de aquel con mayor valor
        # for h in range(DF_ERNC.shape[0]):
        for FilaDF_Aux in DF_aux.iterrows():
            # Nombre del indice que posee mayor valor: row[1].idxmax()
            # print( 'row[1].idxmax():', row[1].idxmax() )
            Lista_Cambios.append( FilaDF_Aux[1].idxmax() )
            # print()

        # print('Lista_Cambios:\n', Lista_Cambios)
        # Crea dataframe a partir de la lista de cambios.
        DF_Cambios = pd__DataFrame( Lista_Cambios )
        # Encuentra los índices que son distintos. Los desfaza hacia abajo (1 periodo), y rellena el vació con el valor siguiente encontrado
        DF_Cambios = DF_Cambios.ne(DF_Cambios.shift(periods=1).fillna(method='bfill'))  # boolean single-column
        # print('DF_Cambios:\n', DF_Cambios)
        # obtiene los elementos que son de cambio, según lo encontrado previamente.
        DF_aux = DF_aux[ DF_Cambios.values ]
        print('DF_aux:\n', DF_aux)
        print('DF_aux.shape:\n', DF_aux.shape)
        print()

        # Por cada fila agrega al DataFrame de salida las etapas intermedias con 'FechaIni' y 'FechaFin'
        LAux = []
        RangoCambios = range(DF_aux.shape[0] - 1)
        for FilaNum in RangoCambios:
            if FilaNum == 0:
                # excepción para primer valor
                LAux.append( [ DF_aux.index[FilaNum], DF_aux.index[FilaNum + 1] ] )
            elif FilaNum == RangoCambios[-1]:
                # Finalmente agrega última fecha (última iteración)
                LAux.append( [DF_aux.index[FilaNum] + dt__timedelta(hours=1), row[1]['FechaFin']] )
            else:
                LAux.append( [ DF_aux.index[FilaNum] + dt__timedelta(hours=1), DF_aux.index[FilaNum + 1] ] )
            # print('LAux:\n', LAux)
        # crea otro dataframe auxiliar para transformar la lista de fechas de etapas al formato tradicional de dataframe_etapas
        DF_AuxEta = pd__DataFrame(data=LAux, columns=['FechaIni', 'FechaFin']).reset_index()
        DF_AuxEta.columns = ['EtaNum', 'FechaIni', 'FechaFin']
        DF_AuxEta.set_index('EtaNum', inplace=True)
        DF_AuxEta.index += 1    # comienza con etapa 1
        # print('DF_AuxEta:\n', DF_AuxEta)
        DF_Etapas2 = DF_Etapas2.append( DF_AuxEta )

        ContadorEtapas += 1
        break
    print('DF_Etapas2:\n', DF_Etapas2)

    logger.debug("! saliendo de función: 'Crea_2da_div_Etapas' ...")
    return Etapas
