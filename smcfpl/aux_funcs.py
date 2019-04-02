from datetime import timedelta as dt__timedelta
from pandapower import overloaded_lines as pp__overloaded_lines
from pandapower.topology import create_nxgraph as pp__create_nxgraph
from pandapower import select_subnet as pp__select_subnet
from networkx import has_path as nx__has_path
from networkx import connected_component_subgraphs as nx__connected_component_subgraphs
from pandas import DataFrame as pd__DataFrame, date_range as pd__date_range, Series as pd__Series
from pandas import datetime as pd__datetime, set_option as pd__set_option
from pandas import concat as pd__concat
from numpy import mean as np__mean, nan as np__NaN, arange as np__arange, bool_ as np__bool_
from numpy import zeros as np__zeros
from numpy.random import uniform as np_random__uniform, normal as np__random__normal
from numpy.random import seed as np__random__seed
from numpy.random import choice as np__random__choice
from dateutil.relativedelta import relativedelta as du__relativedelta
from collections import OrderedDict as collections__OrderedDict

import locale
import logging

locale.setlocale(locale.LC_ALL, 'es_ES.utf8')
# logger = logging.getLogger('stdout_only')


def date_parser(x):
    return pd__datetime.strptime(x, '%Y-%m-%d %H:%M')


def print_full_df():
    # allow print all panda columns
    pd__set_option('precision', 4)
    pd__set_option('expand_frame_repr', False)


def setup_logger(logger_name, log_file='.', level=logging.DEBUG):
    """ If log_file is declared (different than '.') log file is used.
    """
    logg = logging.getLogger(logger_name)
    formatter = logging.Formatter('[%(levelname)s][%(asctime)s][%(filename)s:%(funcName)s] - %(message)s')
    if log_file != '.':
        fileHandler = logging.FileHandler(log_file, mode='w')
        fileHandler.setFormatter(formatter)
        logg.addHandler(fileHandler)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    logg.setLevel(level)
    logg.addHandler(streamHandler)
    logg.propagate = False  # avoid multiple logging messages


def Crea_Etapas_desde_Cambio_Mant(DF_CambioFechas, ref_fija=True):
    """
        En caso de habilitarse 'ref_fija':
            Se fija primera fila como referencia. En caso de existir
            un día o más de diferencia con respecto a la siguiente
            fecha, éstas se marcan como fechas límite y se desplaza
            la referencia a la última de las fechas comparadas.
            De lo contrario, se mide con respecto a la subsiguiente.
            Proceso finaliza luego de cuando la referencia llega a
            la penúltima fecha disponible.

        En caso de NO habilitarse 'ref_fija':
            Se fija primera fila como referencia. En caso de existir
            un día o más de diferencia con respecto a la siguiente
            fecha, éstas se marcan como fechas límite. De lo contrario,
            se avanza la referencia a la siguiente y se mide con
            respecto a la que le sigue desde aquí. Proceso finaliza
            luego de cuando la referencia llega a la penúltima fecha
            disponible.

        Notar que el último valor no es considerado (por reducción de indice en comparación y
        ser éste la fecha de termino de simulación). While es necesario para facilitar el
        salto de filas en iteración.

        Retorna un DataFrama de columnas 'FechaIni' y 'FechaFin'. Con Index.name = 'EtaNum'.
    """
    logger.debug("! entrando en función: 'Crea_Etapas_desde_Cambio_Mant' (aux_funcs.py) ...")
    fila = 0
    NumFilas = len(DF_CambioFechas.index) - 1
    # Inicializa la primera fecha (inicio simulación, ya que está ordenada)
    ListaFechasFinales = [ DF_CambioFechas.loc[fila, 0] ]
    while fila < NumFilas:
        # recorre las filas, tal que la fila actual depende de fila + i
        i = 0   # indicador de cuantas filas hay que desplazarse desde 'fila' para siguiente valor no cercano.
        Continuar = True
        while Continuar:    # parecido a un 'do-while'
            if ref_fija:
                # calcula la diferencia temporal entre el de referencia y el siguiente
                NextHorasDiff = DF_CambioFechas.loc[fila + i + 1, 0] - DF_CambioFechas.loc[fila, 0]
            else:
                # calcula la diferencia temporal entre el instante de tiempo actual y el siguiente
                NextHorasDiff = DF_CambioFechas.loc[fila + i + 1, 0] - DF_CambioFechas.loc[fila + i, 0]
            Condicion1 = NextHorasDiff >= dt__timedelta(days=1)  # condición: diferencia de tiempo sea mayor o igual a 1 día (type: timedelta)
            Condicion2 = fila + i > NumFilas    # condición: supere el número de filas DataFrame
            i += 1
            if Condicion1 | Condicion2:
                Continuar = False
        """
        Agrega a una lista la primera coincidencia de las fechas cercanas. Notar que el largo de 'ListaFechasFinales'
        es el Número de etapas resultantes + 1
        """
        ListaFechasFinales.append( DF_CambioFechas.loc[fila + i, 0] )
        fila += i

    # print('ListaFechasFinales:', ListaFechasFinales)
    # Convierte la lista de fechas finales en una lista para ser ingresada al data del DataFrame de salida.
    LAux = []
    for IndFecha in range(len(ListaFechasFinales) - 1):
        """ Recordar que las fechas datetime son indicativas de la hora completa que le siguen, i.e., si se menciona que un evento ocurrió
        a determinada hora significa que ocurrió durante o al menos dentro de dicha hora. """
        if IndFecha == 0:
            # En caso de ser el primer elemento agrega tal cual la fecha divisoria.
            LAux.append( [IndFecha + 1, ListaFechasFinales[IndFecha], ListaFechasFinales[IndFecha + 1]] )
        else:
            # En caso de presentarse las siguientes 'FechaIni_EtaTopo', estas se les agrega una hora c/r al de la fila superior 'FechaFin'.
            LAux.append( [IndFecha + 1, ListaFechasFinales[IndFecha] + dt__timedelta(hours=1), ListaFechasFinales[IndFecha + 1]] )

    logger.debug("! saliendo en función: 'Crea_Etapas_desde_Cambio_Mant' (aux_funcs.py) ...")
    return pd__DataFrame(data=LAux, columns=['EtaNum', 'FechaIni', 'FechaFin']).set_index('EtaNum')


def Lista2DF_consecutivo(Lista, incremento, NombreColumnas):
    """
        Crea una un dataframe de 2 columnas con los itemes separados, a partir de una lista unidimensional.
        Notar que el tipo de la variable incremento debe ser del mismo tipo que los item de la lista (todos mismo tipo).
        Ejemplo:
        >>> Lista2DF_consecutivo(Lista = [10,20,33,45], Incremento = 1, NombreColumnas=['a','b'])
             a    b
        0   10   20
        1   21   33
        2   34   45
    """
    # logger.debug("! entrando en función: 'Lista2DF_consecutivo' (aux_funcs.py) ...")
    LAux = []
    for IndElmn in range(len(Lista) - 1):
        if IndElmn == 0:
            LAux.append( [ Lista[IndElmn], Lista[IndElmn + 1] ] )
        else:
            LAux.append( [ Lista[IndElmn] + incremento, Lista[IndElmn + 1] ] )

    # logger.debug("! saliendo en función: 'Lista2DF_consecutivo' (aux_funcs.py) ...")
    return pd__DataFrame(data=LAux, columns=NombreColumnas)


def DesvDemandaHistoricaSistema_a_Etapa(BD_Historica, BD_Etapas):
    """Retorna un DataFrame con la desviación de la demanda sistema para todas las etapas futuras (ajustando temporalidad por promedio). En función de los datos históricos de in_smcfpl_histdem"""
    logger.debug("! entrando en función: 'DesvDemandaHistoricaSistema_a_Etapa' (aux_funcs.py) ...")
    # Inicializa DataFrame de salida
    DF_Salida = pd__DataFrame(columns=['EtaNum', 'Desv_decimal'])
    # Por cada etapa existente filtra los datos de BD_Historica
    logger.debug("! para cada etapa entra en función: 'Filtra_DataFrame_Agrupado' (aux_funcs.py) ...")
    for Num, Etapa in BD_Etapas.iterrows():
        MesInicio = Etapa['FechaIni_EtaTopo'].month
        DiaInicio = Etapa['FechaIni_EtaTopo'].day
        HoraInicio = Etapa['HoraDiaIni']
        MesFin = Etapa['FechaFin_EtaTopo'].month
        DiaFin = Etapa['FechaFin_EtaTopo'].day
        HoraFin = Etapa['HoraDiaFin']
        Desv_dec = Filtra_DataFrame_Agrupado(DF_Entrante=BD_Historica,
                                             MesInicio=MesInicio, DiaInicio=DiaInicio, HoraInicio=HoraInicio,
                                             MesFin=MesFin, DiaFin=DiaFin, HoraFin=HoraFin)
        # Calcular desviación mean(programado - real / real?)
        # Notar es muy difícil que demanda sistema programada sea cero.
        Desv_dec = np__mean( (Desv_dec['programado'] - Desv_dec['real']) / Desv_dec['programado'])
        # print("Desv_dec:", Desv_dec)

        # Asigna valores al DataFrame de salida
        DF_Salida.loc[ Num, ['EtaNum', 'Desv_decimal'] ] = [Num, abs(Desv_dec)]  # asegura scale/desv. est. positivo
    logger.debug("! para cada etapa sale de función: 'Filtra_DataFrame_Agrupado' (aux_funcs.py) ...")

    logger.debug("! saliendo en función: 'DesvDemandaHistoricaSistema_a_Etapa' (aux_funcs.py) ...")
    return DF_Salida.set_index('EtaNum')


def Filtra_DataFrame_Agrupado(DF_Entrante, MesInicio, DiaInicio, HoraInicio, MesFin, DiaFin, HoraFin):
    """
        Filtra un DataFrame multindex que tiene como nombre de índices ('Mes', 'Dia', 'Hora').
        Selecciona las filas que corresponden según los argumentos de '*Inicio' y '*Fin' requeridos.
        Recordar que el filtrado directo es de meses y días, mientras que las horas son dependientes de lo que permiten las etapas renovables.
        Por esta razón se obtiene un promedio entre las horas equivalentes de un día.
    """
    # logger.debug("! entrando en función: 'Filtra_DataFrame_Agrupado' (aux_funcs.py) ...")
    #
    # FILTRA NIVEL DE MESES
    # Obtiene promedio de valores entre las fechas de la etapa actual
    if MesInicio <= MesFin:
        CondMes = MesInicio <= DF_Entrante.index.get_level_values('Mes')
        CondMes &= DF_Entrante.index.get_level_values('Mes') <= MesFin
    else:  # uno de los caso cuando etapas recorren de un año a otro
        CondMes = MesInicio <= DF_Entrante.index.get_level_values('Mes')
        CondMes |= DF_Entrante.index.get_level_values('Mes') <= MesFin
    DF_Salida = DF_Entrante[ CondMes ]

    #
    # FILTRA NIVEL DE DIAS (Fija mes)
    # identifica los días que se quieren extraer del primer mes
    CondDia_1 = ( DF_Salida.index.get_level_values('Mes') == MesInicio ) & ( DF_Salida.index.get_level_values('Dia') < DiaInicio )
    # selecciona los días del primer mes que no correspondan
    CondDia_1 = ~ CondDia_1
    #
    # identifica los días que se quieren extraer del último mes
    CondDia_2 = ( DF_Salida.index.get_level_values('Mes') == MesFin ) & ( DiaFin < DF_Salida.index.get_level_values('Dia') )
    # selecciona los días del ultimo mes que no correspondan
    CondDia_2 = ~ CondDia_2
    CondDia = CondDia_1 & CondDia_2  # deben perdurar los True (Fechas buscadas)
    DF_Salida = DF_Salida[ CondDia ]

    #
    # "FILTRADO NIVEL DE HORAS" (promedio por Etapa Renovable)
    # Encuentra el promedio de un día equivalente entre las fechas de la etapa topológica, para luego escoger las horas correspondiente a la etapa renovable
    DF_Salida = DF_Salida.groupby('Hora').mean()
    DF_Salida = DF_Salida[ (HoraInicio <= DF_Salida.index) & (DF_Salida.index <= HoraFin) ]

    # logger.debug("! saliendo en función: 'Filtra_DataFrame_Agrupado' (aux_funcs.py) ...")
    return DF_Salida


def TasaDemandaEsperada_a_Etapa(DF_ProyDem, BD_Etapas, FechaIniSim, FechaFinSim):
    """
        Transforma la información del dataframe de demanda proyectada (temporal - mensual) a sus correspondientes etapa dentro del horizonte de simulación.
        Notar que para los meses que no sean informados, mantendrán la misma tasa que el último informado. La tasa de crecimiento para a referenciarse respecto
        al valor de demanda inicial.

        Pasos:
        1.- Desde la fecha inicial de simulación (FechaIniSim) hasta la fecha final de simulación (FechaFinSim), genera pandas date_range mensual.
        2.- Recorta el dataframe en demanda proyectada a las fechas que le corresponden del horizonte de simulación
        3.- Asignar valores coincidentes con lo ingresado en fechas de DF_ProyDem. Reindex rellenando con nan los vacíos
        4.- Los valores faltantes (de existir) se rellenan con el primer valor encontrado anteriormente.
        5.- Calcula la tasa de crecimiento (CLibre y CRegulado) acumulativa respecto a la primera fecha del DF. Recordar que el valor en cada mes es la tasa desde el mes anterior #hacia el próximo mes.
        6.- Para cada etapa en BD_Etapas encuentra el promedio de tasas.
        Retorna un pandas DataFrame con la tasa de crecimiento acumulada esperada en etapas para esa etapa c/r la demanda inicial

        DF_ProyDem: Posee la tasa de crecimiento mensual de clientes libres y regulados de cada mes respecto al mes anterior.
        BD_Etapas: (pandas dataframe)
        FechaIniSim: (Datetime object)
        FechaFinSim: (Datetime object)
    """
    logger.debug("! entrando en función: 'DemandaEsperada_a_Etapa' (aux_funcs.py) ...")
    # 1.- Desde la fecha inicial de simulación (FechaIniSim) hasta la fecha final de simulación (FechaFinSim), genera pandas date_range mensual.
    # TimeIndx = pd__date_range(start=FechaIniSim, end=FechaFinSim, freq='MS', name='fecha')  # 'MS' mensual primer día del mes
    TimeIndx = pd__date_range(start=FechaIniSim, end=FechaFinSim, freq='D', name='Fecha')   # Se requiere detalle diario para aquellas etapas menores a un mes y comiencen después del primer día.
    # 2.- Recorta el dataframe en demanda proyectada a las fechas que le corresponden del horizonte de simulación
    DF_ProyDem = DF_ProyDem[ (FechaIniSim <= DF_ProyDem['Fecha']) & (DF_ProyDem['Fecha'] <= FechaFinSim) ].set_index('Fecha')
    # 3.- Asignar valores coincidentes con lo ingresado en fechas de DF_ProyDem. Reindex rellenando con nan los vacíos
    DF_ProyDem = DF_ProyDem.reindex(index=TimeIndx, fill_value=np__NaN)
    # 4.- Los valores faltantes (de existir) se rellenan con el primer valor encontrado anteriormente.
    DF_ProyDem.fillna(method='ffill', inplace=True)  # 'forward fill': reemplaza nan values con el valor anterior válido. De no existir queda como nan.
    DF_ProyDem.fillna(0, inplace=True)  # En caso de no existir primer valor válido de tasa de crecimiento se considera nula,
    # 5.- Calcula la tasa de crecimiento (CLibre y CRegulado) acumulativa respecto a la primera fecha del DF. Recordar que el valor en cada mes es la tasa hacia el próximo mes.
    DF_TasaAcumulativa = DF_ProyDem + 1  # agrega constante a todos los valores del DataFrame (para hacer modificación c/r valor inicial)
    DF_TasaAcumulativa = DF_TasaAcumulativa.cumprod(axis='index')   # recordar que para referenciar todos al valor inicial: D_fin = D_ini*Prod i=1 N (1+tasa_i)
    # 6.- Para cada etapa en BD_Etapas encuentra el promedio de tasas.
    DF_Salida = pd__DataFrame(columns=['EtaNum', 'TasaCliLib', 'TasaCliReg']).set_index('EtaNum')
    for Num, Etapa in BD_Etapas.iterrows():
        FechaInicio = Etapa['FechaIni_EtaTopo']
        FechaFin = Etapa['FechaFin_EtaTopo']

        # filtra las fecha en DF_TasaAcumulativa
        DF_aux = DF_TasaAcumulativa[ (FechaInicio <= DF_TasaAcumulativa.index) & (DF_TasaAcumulativa.index <= FechaFin) ].mean(axis='index')
        # asigna valores promedios al DataFrame de salida
        DF_Salida.loc[Num, :] = DF_aux.values
    #
    logger.debug("! saliendo en función: 'DemandaEsperada_a_Etapa' (aux_funcs.py) ...")
    return DF_Salida


def Crea_hidrologias_futuras(DF_HistHid, DF_Etapas, PE_HidSeca, PE_HidMedia, PE_HidHumeda, FechaIniSim, FechaFinSim, seed=None):
    """
        Desde la tabla de entrada de 'in_smcfpl_histhid', calcula la probabilidad de excedencia (PE) anual de los años de la muestra que luego es adaptada (por promedio en caso de ambigüedad) a las etapas de la simulación.
        Notar como en las etapas renovables de cada etapa topológica, la PE es constante debido a la misma definición.
        El procedimiento de cálculo es el numerado a continuación:
            1.- Obtiene la PE anual de cada año de la muestra. Generando un DataFrame anual con PE que es actualizado posteriormente
            2.- Identifica los años en DF_PE_anual con PE más cercana a lo ingresado
            3.- Obtiene la energía afluente de cada año desde la muestra identificado (detalle mensual), además del mes que le sigue para posterior variación.
            4.- Encuentra la variación inter-mensual en el año hidrológico desde la muestra. Identificar la variación al mes siguiente respecto del próximo.
            5.- Obtiene el máximo y mínimo valor de variación inter-mensual del año representativo de la hidrología y obtiene los valores entre los que limita
            6.- Modifica la energía afluente, UNIFORMEMENTE ALEATORIOS entre máximo y mínimo (anteriores) dado por rango variación inter-mensual
                6.1.- En caso de hidrología media
                6.2.- En caso de hidrología seca
                6.3.- En caso de hidrología húmeda
            7.- Se agrega la energía de la hidrología Húmeda/Media/Seca al DF_HidrologiasFuturas en respectivo lugar
            8.- calcula la PE de la energía afluente del año ocurrente, en la hidrología de estudio, como si éste participara de la muestra
            9.- obtiene la PE del año futuro a la asigna en la hidrología ocurrente al DataFrame de salida
            10.- Asigna las PE anuales a todas etapas en simulación (renovables - todas). En caso de existir etapas topológicas entre años, se asigna la PE promedio de los años en cuestión

        Retorna un pandas DataFrame con la probabilidad de excedencia de cada etapa (indice) en cada una de las tres hidrologías.

        INPUT:
            **DF_HistHid**
            **DF_Etapas**
            **PE_HidSeca**
            **PE_HidMedia**
            **PE_HidHumeda**
            **FechaIniSim**
            **FechaFinSim**
        Optional:
            **seed** (int, None) - set random number seed to this value.

        :param seed: Sets the random number to be tha same always
        :type seed: int
    """
    logger.debug("! entrando en función: 'Crea_hidrologias_futuras' (aux_funcs.py) ...")
    # calcula años en horizonte de simulación
    NAniosSimulacion = du__relativedelta(FechaFinSim, FechaIniSim).years + 2  # cuenta el primero y el último
    ListaAniosSim = [FechaIniSim.year + i for i in range(NAniosSimulacion)]

    # 1.- Obtiene la PE anual de cada año de la muestra. Generando un DataFrame anual con PE según los años de la muestra
    DF_PE_anual = CalculaPE_DataFrame(DF_HistHid)

    DF_HidrologiasFuturas = pd__DataFrame(columns=['PE HidSeca', 'E HidSeca', 'PE HidMedia', 'E HidMedia', 'PE HidHumeda', 'E HidHumeda'], index=[ListaAniosSim])
    # para cada año de la simulación se determinan las PE respecto a lo ocurrido en años pasados, finalmente obteniéndose un dataframe año-PE
    for Anio in ListaAniosSim:
        # Inicializa diccionario de hidrologías (de querer utilizar más hidrologías modificar aquí). Debe ser ordenado para asegurar correcto funcionamiento
        Dict_Hidrologia = collections__OrderedDict({ 'HidMedia': PE_HidMedia,
                                                     'HidSeca': PE_HidSeca,
                                                     'HidHumeda': PE_HidHumeda})   # diccionario con info de hidrologías
        # utiliza el Dict_Hidrologia para trabajar simultáneamente las hidrologías
        for HidNom, PE_Hid_In in Dict_Hidrologia.items():
            # 2.- Identifica los años en DF_PE_anual con PE más cercana a lo ingresado
            AnioCercano = (DF_PE_anual['PE_decimal_calc'] - PE_Hid_In).abs().argsort()[:2]
            AnioCercano = DF_PE_anual.iloc[ AnioCercano ]
            AnioCercano = AnioCercano.loc[AnioCercano.index[0], :]   # pandas Series del año más cercano (primera fila)
            # AnioCercano['Año']    # el nombre del año más cercano
            # AnioCercano['TOTAL']    # energía afluente anual del año más cercano
            # AnioCercano['PE_decimal_calc']    # la PE obtenida del año más cercano

            # 3.- Obtiene la energía afluente de cada año desde la muestra identificado (detalle mensual), además del mes que le sigue para posterior variación.
            IndAnio = AnioCercano.name
            DF_actual = DF_HistHid.loc[IndAnio, DF_HistHid.columns != 'TOTAL'].to_frame()
            DF_actual = DF_actual.loc[ DF_actual.index != 'Año' ]  # remueve fila con nombre de años correspondientes
            DF_actual.columns = ['E_aflu [GWh]']
            try:    # intenta obtener valores del próximo año
                valor_abril_siguiente = DF_HistHid.loc[IndAnio + 1, 'abril']
            except Exception:
                # En caso de tratarse del último año de la muestra, se repite el último valor del último mes a modo que no afecte en valores de variación intermensual
                valor_abril_siguiente = DF_actual['E_aflu [GWh]'].iloc[-1]  # repite el último valor
                print("'valor_abril_siguiente' no posee valores históricos, i.e., no hay datos después del año hidrológico {}".format(DF_actual.name))
            # expande el DF_actual agregando un mes más al final
            DF_actual = DF_actual.append( pd__Series({'E_aflu [GWh]': valor_abril_siguiente}, name='abril'),
                                          ignore_index=False, verify_integrity=False, sort=None)  # agrega df una fila después.

            # 4.- Encuentra la variación intermensual en el año hidrológico desde la muestra. Identificar la variación al mes siguiente respecto del próximo.
            DF_VarMens = DF_actual.diff(periods=1, axis='index')  # calcula la diferencia de cada fila siguiente respecto a la 'actual'
            ColNames = DF_VarMens.index.values  # guarda los nombres de los indices
            DF_VarMens.dropna(axis='index', inplace=True)   # elimina la primera fila con el NaN
            DF_VarMens.index = ColNames[:-1]  # reasigna indices acorde meses (con shift de 1 mes atrás)
            DF_VarMens = DF_VarMens / DF_actual[:-1]  # encuentra la variación decimal mensual
            # NOTAR que a esta altura cada mes tiene el valor de la variación respecto al siguiente mes

            # 5.- Obtiene el máximo y mínimo valor de variación intermensual del año representativo de la hidrología y obtiene los valores entre los que limita
            MaxE_Anio = ( 1 + min(DF_VarMens.values.max(), .5) ) * AnioCercano['TOTAL']  # limita la variación límite a no más de un 150% de la energía
            MinE_Anio = ( 1 + max(DF_VarMens.values.min(), -.5) ) * AnioCercano['TOTAL']  # limita la variación límite a no menos de un -150% de la energía

            # In case a seed is 'int', it's used to generate same numbers from seed. Otherwise makes it more random.
            np__random__seed(seed)

            # 6.- Modifica la energía afluente, UNIFORMEMENTE ALEATORIOS entre máx y mín (anteriores) dado por rango variación intermensual
            # 6.1.- En caso de hidrología media
            if HidNom == 'HidMedia':
                NuevaEAflu = np_random__uniform(MinE_Anio, MaxE_Anio)  # valor en [GWh]
            # 6.2.- En caso de hidrología seca
            elif HidNom == 'HidSeca':
                E_HidMedia = DF_HidrologiasFuturas.loc[Anio, 'E HidMedia'].values[0]
                NuevaEAflu = np_random__uniform(MinE_Anio, E_HidMedia)  # valor en [GWh]
            # 6.3.- En caso de hidrología humeda
            elif HidNom == 'HidHumeda':
                E_HidMedia = DF_HidrologiasFuturas.loc[Anio, 'E HidMedia'].values[0]
                NuevaEAflu = np_random__uniform(E_HidMedia, MaxE_Anio)  # valor en [GWh]
            else:
                raise ValueError("'HidNom' no posee nombre válido.")
            # 7.- Se agrega la energia de la hidrología Humeda/Media/Seca al DF_HidrologiasFuturas en respectivo lugar
            NuevaEAflu = max(NuevaEAflu, 0)  # asegura que el valor no se negativo (no tiene sentido de lo contrario)
            DF_HidrologiasFuturas.loc[Anio, 'E ' + HidNom] = NuevaEAflu

            # 8.- calcula la PE de la energía afluente del año ocurrente, en la hidrología de estudio, como si éste participara de la muestra
            DF_temp = DF_PE_anual[['Año', 'TOTAL']].append( pd__DataFrame({'TOTAL': [NuevaEAflu], 'Año': ['futuro']}),
                                                            ignore_index=True, verify_integrity=False, sort=False )
            DF_temp = CalculaPE_DataFrame(DF_temp)

            # 9.- obtiene la PE del año futuro a la asigna en la hidrología ocurrente al DataFrame de salida
            DF_HidrologiasFuturas.loc[Anio, 'PE ' + HidNom] = DF_temp[ DF_temp['Año'] == 'futuro' ]['PE_decimal_calc'].values[0]

    # 10.- Asigna las PE anuales a todas etapas en simulación (renovables - todas). En caso de existir etapas topológicas entre años, se asigna la PE promedio de los años en cuestión
    DF_Salida = pd__DataFrame(columns=['EtaNum', 'PE Seca dec', 'PE Media dec', 'PE Humeda dec'])
    for EtaNum, Etapa in DF_Etapas.iterrows():
        # Obtiene los años que representan la etapa. De ser mayor a
        NAniosSimulacion = du__relativedelta(Etapa['FechaFin_EtaTopo'], Etapa['FechaFin_EtaTopo']).years + 1  # cuenta número de años
        ListaAniosEta = [Etapa['FechaFin_EtaTopo'].year + i for i in range(NAniosSimulacion)]   # comienza desde el año inicial de la etapa
        if len(ListaAniosEta) > 1:
            DF_aux = pd__DataFrame( columns=['PE HidSeca', 'PE HidMedia', 'PE HidHumeda'] )
            for Anio in ListaAniosEta:
                ValoresPE = DF_HidrologiasFuturas.loc[Anio, ['PE HidSeca', 'PE HidMedia', 'PE HidHumeda']]
                DF_aux = pd__concat([DF_aux, ValoresPE], axis='index')

            # calcula el promedio a lo largo de las hidrologías
            ValoresPE = DF_aux.mean().values.tolist()
            DF_Salida.loc[EtaNum, :] = [int(EtaNum), *ValoresPE]
        else:
            Anio = ListaAniosEta[0]
            ValoresPE = DF_HidrologiasFuturas.loc[Anio, ['PE HidSeca', 'PE HidMedia', 'PE HidHumeda']].values.flatten().tolist()
            DF_Salida.loc[EtaNum, :] = [int(EtaNum), *ValoresPE]

    DF_Salida.set_index('EtaNum', inplace=True)

    logger.debug("! saliendo en función: 'Crea_hidrologias_futuras' (aux_funcs.py) ...")
    return DF_Salida


def CalculaPE_DataFrame(DF_HistHid):
    """
        Calcula y retorna la probabilidad de execedecia tipo weibull (PE) de un Dataframe que posee columnas 'Año' (str) y 'TOTAL' (float).
        Retorna mismo dataframe pero con nueva columna llamada 'PE_decimal_calc'
    """
    DF_PE_anual = DF_HistHid[['Año', 'TOTAL']]  # identifica los datos anuales
    NAniosMuestra = DF_PE_anual.shape[0]  # identifica cantidad de años de la muestra
    DF_PE_anual_aux = DF_PE_anual.sort_values(by='TOTAL', ascending=False).reset_index(drop=True)  # ordena en orden descendente para cálculo de PE en nuevo DataFrame
    DF_PE_anual_aux = DF_PE_anual_aux.assign(**{'PE_decimal_calc': DF_PE_anual_aux.index / (NAniosMuestra + 1)})  # asigna nueva columna con PE calculada
    DF_PE_anual_aux = DF_PE_anual_aux.drop('TOTAL', axis=1)  # elimina fila totales (para este DF no necesaria)
    return DF_PE_anual.merge(DF_PE_anual_aux, on='Año')  # agrega en orden por índice (indice numérico dado a años) al DF anual (original) las PE correspondientes


def TSF_Proyectada_a_Etapa(DF_TSFProy, DF_Etapas, FechaIniSim):
    """
        Convierte los datos de la tabla (pandas Dataframe) de entrada en su correspondiente de etapas. Se obtienen las Tasa de Salida Forzada (TSF) para cada tipo
        de generación. Se evalúa la fecha de la primera fila a modo que exista una TSF a comienzo de la simulación para utilizase en el despacho, de no existir la fila
        o no existir datos, se asume que la tasa de falla es cero para todas la tecnologías en todas las etapas. Este procedimiento se alerta con warning en ventana salida.
        De no existir al menos un valor en una de las filas siguientes, el valor es considerado cero.
            1.- Inicializa Dataframe de salida con TSF NaN values e indice de etapas (después rellena NaN values)
            2.- Fija index 'Fecha' de DF_TSFProy
            3.- Verifica si el archivo está vacío. En caso de estarlo se termina retornando todas las etapas con TSF nulos.
            4.- Verifica primera fecha del archivo sea igual o anterior a la fecha de inicio de simulación. En tal caso se mantienen cero aquellas no especificadas
            5.- Cuando hay datos, rellena etapas según las fechas informadas en cada fila del DF_TSFProy
            6.- Para cada cambio de TSF, identifica etapa topológicas (y etapas renovables asociadas) con fecha perteneciente
            7.- Asigna los valores de TSF de la fecha en cuestión al DF de salida
            8.- Rellena NaN siguientes con ultimo valor válido (method='ffill') y NaN restantes con cero.
        Debido a que se itera desde la primera fila de in_smcfpl_tsfproy hasta la última, se sobrescriben dejando el último valor de TSF en fechas que pertenezcan
        a la misma etapa topológica. Antes de ingresar los datos es recomendable verificar las posibles etapas topológicas a generarse para definir las fechas de
        cambio de TSF.

        Retorna un pandas DataFrame de columnas provenientes de 'DF_TSFProy'
    """
    logger.debug("! entrando en función: 'TSF_Proyectada_a_Etapa' (aux_funcs.py) ...")
    # 1.- Inicializa Dataframe de salida con TSF NaN values e indice de etapas (después rellena Nan values)
    DF_Salida = pd__DataFrame( np__NaN, index=DF_Etapas.index, columns=DF_TSFProy.columns.tolist()[1:] )
    # 2.- fija index 'Fecha' de DF_TSFProy
    DF_TSFProy.set_index('Fecha', inplace=True)

    if DF_TSFProy.empty:
        # 3.- Verifica si el archivo está vacío. En caso de estarlo se termina retornando todas las etapas con TSF nulos.
        logger.warn("Archivo de entrada 'in_smcfpl_tsfproy' no posee datos! ...")
    else:  # No está vacío
        if DF_TSFProy.index[0] > FechaIniSim:
            # 4.- Verifica primera fecha del archivo sea igual o anterior a la fecha de inicio de simulación. En tal caso se mantienen cero aquellas no especificadas
            logger.warn("Fecha inicial en entrada 'in_smcfpl_tsfproy' es posterior al inicio de simulación! Considerado TSF nulas hasta próximo cambio ...")
        # 5.- Cuando hay datos, rellena etapas según las fechas informadas en cada fila del DF_TSFProy
        for FechaTSF, pdSerieTSF in DF_TSFProy.iterrows():
            # 6.- Para cada cambio de TSF, identifica etapa topológicas (y etapas renovables asociadas) con fecha perteneciente
            Ind_EtapasCorresp = DF_Etapas[ (DF_Etapas['FechaIni_EtaTopo'] <= FechaTSF) & (FechaTSF <= DF_Etapas['FechaFin_EtaTopo']) ].index
            # 7.- Asigna los valores de TSF de la fecha en cuestión al DF de salida
            DF_Salida.loc[Ind_EtapasCorresp, :] = pdSerieTSF.values
        # 8.- Rellena NaN siguientes con ultimo valor válido (method='ffill') y NaN restantes con cero
        DF_Salida.fillna(method='ffill', axis='index', inplace=True)
        DF_Salida.fillna(value=0, axis='index', inplace=True)
    logger.debug("! saliendo en función: 'TSF_Proyectada_a_Etapa' (aux_funcs.py) ...")
    return DF_Salida


def Mantenimientos_a_etapas(DF_MantBar, DF_MantTx, DF_MantGen, DF_MantLoad, DF_Etapas):
    """
        Convierte los distintos dataframe de mantenimientos temporales, en un diccionario de dataframes dependientes de los índices de las etapas.
            1.- Por cada DataFrame de mantenimiento, realiza las siguientes acciones
                1.2.- Identifica las columnas del DF que no sean las fechas
                1.3.- Inicializa un DF_Salida con dichas columnas y nombre de indices 'EtaNum'
            2.- Por cada evento de mantenimiento del dataframe de mantenimiento, identifica los indices de las etapas entre las que
                se encuentran las fechas de inicio y término.
            3.- Se crea un rango de enteros entre los indices límite. En el caso que se trate de una sola etapa renovable dentro de la
                etapa topológica, se ajusta el comienzo partiendo del entero anterior.
            4.- Se mide la distancia temporal entre 'FIni' (fecha inicial del mantenimiento) y la de término de la etapa topológica, de se menor a 1 día es considerado
                no representativo y se comienza desde la siguiente etapa topológica.
            5.- Análogamente para caso opuesto, se mide la distancia temporal entre 'FFin' (fecha final del mantenimiento) y la de inicio de la etapa topológica, de se menor
                a 1 día es considerado no representativo y se comienza desde la etapa topológica anterior.
            6.- Para cada numero del rango anterior (etapas ya están ingresadas en orden), se agrega nueva fila al DF_Salida con los datos del mantenimiento.
        Retorna un diccionario con DataFrame de indices EtaNum, solo para aquellas filas que no sean completamente nulas.
    """
    logger.debug("! entrando en función: 'Mantenimientos_a_etapas' (aux_funcs.py) ...")
    Dict_DF_Salida = { 'df_in_smcfpl_mantbarras': DF_MantBar, 'df_in_smcfpl_manttx': DF_MantTx,
                       'df_in_smcfpl_mantgen': DF_MantGen, 'df_in_smcfpl_mantcargas': DF_MantLoad}

    # Por cada dataframe de mantenimiento, opera similarmente en cada DataFrame
    for DFName, df in Dict_DF_Salida.items():
        # obtiene las columnas del DatFrame (sin las fechas)
        ColumnasSalida = df.columns[(df.columns != 'FechaFin') & (df.columns != 'FechaIni')]
        # Inicializa el DataFrame relacionado con el tipo de mantenimiento
        DF_Salida = pd__DataFrame(columns=ColumnasSalida)

        # Por cada evento de mantenimiento
        for IndMant, pdSerie_Mant in df.iterrows():
            FIni = pdSerie_Mant['FechaIni']
            FFin = pdSerie_Mant['FechaFin']
            # Verifica cuales son los números etapas posteriores a fecha de inicio
            cond1 = (DF_Etapas['FechaIni_EtaTopo'] <= FIni) & (FIni <= DF_Etapas['FechaFin_EtaTopo'])  # En que etapas está FIni de la mantención
            cond2 = (DF_Etapas['FechaIni_EtaTopo'] <= FFin) & (FFin <= DF_Etapas['FechaFin_EtaTopo'])  # En que etapas está FFin de la mantención
            # Debido a que las etapas se encuentran en orden creciente en el tiempo, se toma el primer índice y el último verdadero
            FisrtIndx = cond1[cond1].index[0]  # start stage time
            LastIndx = cond2[cond2].index[-1]   # stop stage time

            # De no poseer al menos un día de distancia, se considera no representativo
            # Verifica que desde la fecha de inicio 'FIni' hasta el fin de la primera etapa topológica exista al menos un día, de lo contrario pasa a la etapa siguiente
            Time2NextStage = DF_Etapas.loc[FisrtIndx, 'FechaFin_EtaTopo'] - FIni
            if Time2NextStage < dt__timedelta(days=1):
                # Modifica las fechas FIni
                FIni += Time2NextStage
                cond1 = (DF_Etapas['FechaIni_EtaTopo'] <= FIni) & (FIni <= DF_Etapas['FechaFin_EtaTopo'])
                FisrtIndx = cond1[cond1].index[0]
            # Verifica que desde la fecha de término hasta el inicio de la ultima etapa topológica exista al menos un día, de lo contrario pasa a la etapa anterior
            Time2PrevStage = FFin - DF_Etapas.loc[LastIndx, 'FechaIni_EtaTopo']
            if Time2PrevStage < dt__timedelta(days=1):
                # Modifica las fechas FFin
                FFin -= Time2PrevStage
                cond1 = (DF_Etapas['FechaIni_EtaTopo'] <= FIni) & (FIni <= DF_Etapas['FechaFin_EtaTopo'])
                FisrtIndx = cond1[cond1].index[0]

            # Very rare case when Both are the same
            if FisrtIndx == LastIndx:
                # set FisrtIndx to be one smaller (worst case FisrtIndx results 0)
                FisrtIndx -= 1

            # Por cada indice del BD_Etapa que coincide
            for indx in np__arange(FisrtIndx, LastIndx + 1):
                # DF_Salida = DF_Salida.append( pdSerie_Mant[ColumnasSalida] )
                ToConcat = pdSerie_Mant[ColumnasSalida].to_frame().T
                ToConcat.index = [indx]
                DF_Salida = pd__concat( [DF_Salida, ToConcat], axis='index',  join='outer', join_axes=None, ignore_index=False,
                                        keys=None, levels=None, names=None, verify_integrity=False, sort=None, copy=True)

        DF_Salida.index.name = 'EtaNum'  # agrega nombre de indices
        # Da formato (casting) a las columnas especiales (booleanas)
        if 'Operativa' in DF_Salida.columns:
            DF_Salida['Operativa'] = DF_Salida['Operativa'].astype(np__bool_)
        if 'EsSlack' in DF_Salida.columns:
            DF_Salida['EsSlack'] = DF_Salida['EsSlack'].astype(np__bool_)
        Dict_DF_Salida[DFName] = DF_Salida
    logger.debug("! saliendo en función: 'Mantenimientos_a_etapas' (aux_funcs.py) ...")
    return Dict_DF_Salida


def GenHistorica_a_Etapa(DF_Etapas, DF_histsolar, DF_histeolicas):
    """
        Crea una tupla de dos pandas DataFrame (Solar y eólico, respectivamente) con los datos del año fijo (comprimido de histórico).
        Para cada Etapa renovable se identifica: El promedio de valores de potencias y, la desviación estándar de dichas potencias de generación
        renovable solar (1) y eólicas (4).
            1.- Crea nuevas columnas para ambos DataFrame Renovables, llamados según su tipo + número (desde cero). Cada una duplicada y agregada '_mean' o '_std'.
            2.- Por cada etapa renovable, resuelve para ambos DataFrame.
                2.1.- Recorta los meses, dias, y horas que representan la etapa renovable desde el año fijo.
                2.2.- Normaliza valores respecto al máximo encontrado en la serie anual (año fijo).
                2.3.- Calcula el promedio y desviación estándar de todas las columnas con datos.
                2.4.- Asigna respectivas columnas de salida y retorna tupla.
        Retorna una tupla con el pandas DataFrame creado para Solar, y la de Eólicos.
    """
    logger.debug("! entrando en función: 'GenHistorica_a_Etapa' (aux_funcs.py) ...")
    # Obtiene nombre columnas y renombra como Solar0. Las duplica y agrega '_mean' y '_std' a cada una.
    # ColumnasSolar = [ "Solar{}_{}".format(Num + 1, ParamNom) for Num in range(len(DF_histsolar.columns)) for ParamNom in ['mean', 'std'] ]  # opción genérica
    ColumnasSolar = [ "Solar_{}".format(ParamNom) for Num in range(len(DF_histsolar.columns)) for ParamNom in ['mean', 'std'] ]  # permite solo un tipo de solar
    # Obtiene nombre columnas y renombra como EolicaZ0, EolicaZ1, ... . Las duplica y agrega '_mean' y '_std' a cada una.
    ColumnasEolicas = [ "EólicaZ{}_{}".format(Num + 1, ParamNom) for Num in range(len(DF_histeolicas.columns)) for ParamNom in ['mean', 'std'] ]
    # Inicializa los pandas DataFrame de Salida
    DF_Salida_Solar = pd__DataFrame(columns=ColumnasSolar, index=DF_Etapas.index)
    DF_Salida_Eolico = pd__DataFrame(columns=[ColumnasEolicas], index=DF_Etapas.index)
    # Calcula el máximo anual para normalizar posteriormente
    for EtaNum, Etapa in DF_Etapas.iterrows():
        for DF_ERNC in [DF_histsolar, DF_histeolicas]:  # calcula para ambos DataFrame
            MaximoAnual = DF_ERNC.max(axis=0)
            # OBTIENE HORAS REPRESENTATIVAS
            FInicioEta = Etapa['FechaIni_EtaTopo']
            FTerminoEta = Etapa['FechaFin_EtaTopo']
            #
            # Identifica de la EtaTopo
            Cond1 = FInicioEta.month <= DF_ERNC.index.get_level_values('Mes')
            Cond2 = DF_ERNC.index.get_level_values('Mes') <= FTerminoEta.month
            # Asegura de filtrar adecuadamente los DataFrame (debe ser cíclico si existe cambio de año)
            if FInicioEta.month <= FTerminoEta.month:
                DF_ERNC = DF_ERNC[ Cond1 & Cond2 ]
            elif FTerminoEta.month < FInicioEta.month:
                DF_ERNC = DF_ERNC[ Cond1 | Cond2 ]

            #
            # identifica los días de los meses límite que pertenecen a la etapa
            Cond1 = DF_ERNC.index.get_level_values('Dia') < FInicioEta.day  # días de los meses fuera
            Cond1 &= FInicioEta.month == DF_ERNC.index.get_level_values('Mes')  # mes límite
            Cond2 = FTerminoEta.day < DF_ERNC.index.get_level_values('Dia')  # días de los meses fuera
            Cond2 &= FTerminoEta.month == DF_ERNC.index.get_level_values('Mes')  # mes límite
            # Asegura de filtrar adecuadamente los DataFrame (debe ser cíclico si existe cambio de año)
            if FInicioEta.day <= FTerminoEta.day:
                DF_ERNC = DF_ERNC[ ~Cond1 & ~Cond2 ]
            elif FTerminoEta.day < FInicioEta.day:
                DF_ERNC = DF_ERNC[ ~Cond1 | ~Cond2 ]

            #
            # Filtra el DF_ERNC por las horas de los DÍA límite
            Cond1 = DF_ERNC.index.get_level_values('Hora') < FInicioEta.hour  # horas de los todos los días no incorporadas
            Cond1 &= FInicioEta.day == DF_ERNC.index.get_level_values('Dia')  # día límite
            Cond2 = FTerminoEta.hour < DF_ERNC.index.get_level_values('Hora')  # horas de los todos los días no incorporadas
            Cond2 &= FTerminoEta.day == DF_ERNC.index.get_level_values('Dia')  # día límite
            if FInicioEta.day <= FTerminoEta.day:
                DF_ERNC = DF_ERNC[ ~Cond1 & ~Cond2 ]
            elif FTerminoEta.day < FInicioEta.day:
                DF_ERNC = DF_ERNC[ ~Cond1 | ~Cond2 ]

            #
            # Filtra DataFrame dejando las horas de interés de la etapa renovable
            DF_ERNC = DF_ERNC[ (Etapa['HoraDiaIni'] <= DF_ERNC.index.get_level_values('Hora')) &
                               (DF_ERNC.index.get_level_values('Hora') <= Etapa['HoraDiaFin']) ]
            # Se eliminan multindex y columnas previamente indices ('Mes' y 'Dia'). Horas Se dejan como index
            DF_ERNC = DF_ERNC.reset_index().drop(labels=['Mes', 'Dia'], axis='columns').set_index('Hora')
            # Se normalizan potencias c/r al máximo total
            DF_ERNC = DF_ERNC.divide(MaximoAnual)

            # chekcs if only one row (otherwise nan are returned for std)
            if DF_ERNC.shape[0] > 1:
                # Encuentra el promedio de los valores
                Arr_Mean = DF_ERNC.mean(axis='index').values
                # Encuentra la desviación estándar
                Arr_Std = DF_ERNC.std(axis='index').values
            else:
                # Encuentra el promedio de los valores
                Arr_Mean = DF_ERNC.loc[DF_ERNC.index[0], :].values
                # Encuentra la desviación estándar
                Arr_Std = np__zeros(DF_ERNC.shape[1])  # If single evalues, means it is the only posible. Std = 0
            Arr_Mean[Arr_Mean < 0] = 0  # makes sure there isn't negative mean for power

            # Asigna valores a los DataFrame de salida según la etapa
            if DF_ERNC.columns.tolist() == DF_histsolar.columns.tolist():
                DF_Salida_Solar.loc[EtaNum, DF_Salida_Solar.filter(regex=r'_mean', axis='columns').columns] = Arr_Mean
                DF_Salida_Solar.loc[EtaNum, DF_Salida_Solar.filter(regex=r'_std', axis='columns').columns] = Arr_Std
                # DF_Salida_Solar.loc[EtaNum, DF_Salida_Solar.columns.str.contains('_mean$')] = Arr_Mean
                # DF_Salida_Solar.loc[EtaNum, DF_Salida_Solar.columns.str.contains('_std$')] = Arr_Std
            elif DF_ERNC.columns.tolist() == DF_histeolicas.columns.tolist():
                DF_Salida_Eolico.loc[EtaNum, DF_Salida_Eolico.filter(regex=r'_mean', axis='columns').columns] = Arr_Mean
                DF_Salida_Eolico.loc[EtaNum, DF_Salida_Eolico.filter(regex=r'_std', axis='columns').columns] = Arr_Std
    logger.debug("! saliendo en función: 'GenHistorica_a_Etapa' (aux_funcs.py) ...")
    return (DF_Salida_Solar, DF_Salida_Eolico)


def GeneradorDemanda( StageIndexesList=[], DF_TasaCLib = pd__DataFrame(), DF_TasaCReg = pd__DataFrame(),
                      DF_DesvDec = pd__DataFrame(), DictTypoCargasEta = {}, seed=None):
    """
        Genera un iterador de valor p.u. de las demandas en cada carga (Ésta
        debe ser multiplicada por el valor inicial nominal de la carga al
        momento de implementarse). El iterador crea un arreglo por cada etapa
        para cada carga (largo variable), por lo que cada vez que se itere se
        generan los valores en cada etapa, con valores de tasa y desviación
        absolutos (asegura signo positivo).

        Notar que se debe obtener previamente el número de cargas en cada
        etapa, en este caso está implícito en DictTypoCargasEta para cada
        etapa (en indices).

        Lo que se busca finalmente, y después de esta función, es obtienes
        una demanda para cada carga que sea variable según la pdf normal.
        Notar que aquí se varía la tasa de crecimiento en lugar de la
        demanda bruta. Variar esta última produce el mismo efecto que
        variar directamente la tasa ya que la desviación estándar esta
        normalizada.

        :param DF_TasaCLib: DataFrame con valores promedio de la tasa de crecimiento de Clientes
                       Libres. Posee indices de etapas.
        :type DF_TasaCLib: pandas DataFrame.

        :param DF_TasaCReg: DataFrame con valores promedio de la tasa de crecimiento de Clientes
                       Regulados. Posee indices de etapas.
        :type DF_TasaCReg: pandas DataFrame.

        :param DF_DesvDec: DataFrame con valores de desviación estándar en la predicción de la
                       demanda, que es utilizada para la distribución de los clientes Libres y
                       Regulados. Posee indices de etapas.
        :type DF_DesvDec: pandas DataFrame.

        :param DictTypoCargasEta: Tipo de las cargas ('L' o 'R') que posee la Grilla en cada etapa.
        :type DictTypoCargasEta: Diccionario {EtaNum: pandas DataFrame}

        :param seed: Sets the random number to be tha same always
        :type seed: int, None

        Cada vez que se llama retorna una tupla con: (EtaNum, pandas DataFrame)

        Pasos:
            1.- Verifica consistencia del número de etapas en entradas
            2.- Para cada Etapa (0-indexed),
                2.1.- Obtiene indices de cargas tipo 'L'
                2.2.- Obtiene indices de cargas tipo 'R'
                2.3.- Genera un arreglo aleatorio de largo del Número de índices.
                      La media (loc) y desviación estándar (scale) según el tipo: DF_TasaCLib | DF_TasaCReg,
                      incluyéndose lo de DF_DesvDec.
                2.4.- Agrega arreglo al DataFrame de Salida.
                2.5.- Retorna la tupla (EtaNum 1-indexed, pandas DataFrame)
    """
    # Verifica que el largo de Etapas sean coincidentes, de lo contrario retorna ValueError
    if DF_TasaCLib.shape[0] != DF_TasaCReg.shape[0] != DF_DesvDec.shape[0] != len(DictTypoCargasEta):
        msg = "El numero de etapas en DF_TasaCLib, DF_TasaCReg y DF_DesvDec son diferentes del tamaño de DictTypoCargasEta."
        logger.error(msg)
        raise ValueError(msg)

    for EtaNum in StageIndexesList:
        DF_Salida = pd__DataFrame( index=DictTypoCargasEta[EtaNum].index,
                                   columns=['PDem_pu'])
        # Obtiene los indices de las cargas en la etapa actual que sean Clientes Libres
        IndCLib = DictTypoCargasEta[EtaNum][ DictTypoCargasEta[EtaNum]['type'] == 'L' ].index
        # Obtiene los indices de las cargas en la etapa actual que sean Clientes Regulados
        IndCReg = DictTypoCargasEta[EtaNum][ DictTypoCargasEta[EtaNum]['type'] == 'R' ].index

        # In case a seed is 'int', it's used to generate same numbers from seed. Otherwise makes it more random.
        np__random__seed(seed)

        #
        # Utiliza crecimiento esperado (DF_TasaCLib | DF_TasaCReg) como valor promedio para cada cliente,
        # sujeto a la desviación en la proyección en la proyección de demanda pasada (referida a esta etapa)
        #
        # Genera nueva tasa de los Clientes Libres.
        dataCLib = np__random__normal( loc=float(DF_TasaCLib.loc[EtaNum, :]),
                                       scale=float(DF_DesvDec.loc[EtaNum, :]),
                                       size=len(IndCLib))
        # Genera nueva tasa de los Clientes Regulados.
        dataCReg = np__random__normal( loc=float(DF_TasaCReg.loc[EtaNum, :]),
                                       scale=float(DF_DesvDec.loc[EtaNum, :]),
                                       size=len(IndCReg))
        # Asigna los datos Regulados y Libres al pandas DataFrame de salida
        DF_Salida.loc[IndCLib, 'PDem_pu'] = dataCLib  # demanda en [p.u.] ya que salen de valores DECIMALES en la etapa
        DF_Salida.loc[IndCReg, 'PDem_pu'] = dataCReg  # demanda en [p.u.] ya que salen de valores DECIMALES en la etapa
        yield (EtaNum,  DF_Salida)


def GeneradorDespacho( StageIndexesList=[], Dict_TiposGen = {}, DF_HistGenERNC = None,
                       DF_TSF = None, DF_PE_Hid = None, DesvEstDespCenEyS=1, DesvEstDespCenP=1, seed=None):
    """
        Genera un iterador de valores p.u. de las potencias de despacho en cada unidad de generación de las distintas tecnologías. Estos valores
        deben ser multiplicados por el valor nominal de potencia de generación de la unidad y limitados entre pmin y pmax). El iterador crea un
        arreglo por cada etapa para cada carga (largo variable), por lo que cada vez que se itere se generan los valores en cada etapa.

        Notar que se debe obtener previamente el número de cargas en cada etapa. Los parámetros ingresados deben tener largo del número de etapas.

        Dict_TiposGen  # Diccionario por etapa de lo tipos de generación en DataFrame (numero gen en Grid)
        DF_HistGenERNC  # tupla de dos pandas DataFrame (DF_Solar, DF_Eólico)
        Lista_TecGenSlack  # lista
        DF_TSF  # pandas DataFrame, para cada tecnología que recurra con falla se asigna
        DF_PE_Hid  # pandas DataFrame

        Para cada siguiente iteración de la función generadora, retorna una tupla de (EtaNum, DF_Pot_despchada_indiceGrid)

        Pasos:
            1.- Verifica que el largo de etapa en todas las entradas sea consistente, de lo contrario error.
            2.- Para cada Etapa (0-indexed),
                2.1.- Obtiene los indices de los tipos de centrales.
                2.2.- En caso de ser ERNC, identifica los nombres, genera despacho para cada uno y, les aplica TSF sobre despacho según entradas.
                2.3.- En caso de Embalse, genera despacho para cada uno y, les aplica TSF sobre despacho según entradas.
                2.4.- En caso de Serie, genera despacho para cada uno y, les aplica TSF sobre despacho según entradas.
                2.5.- En caso de Pasada, genera despacho para cada uno y, les aplica TSF sobre despacho según entradas.
                2.6.- En caso de Carbón, genera despacho para cada uno y, les aplica TSF sobre despacho según entradas.
                2.7.- En caso de Gas-Diésel, genera despacho para cada uno y, les aplica TSF sobre despacho según entradas.
                2.8.- En caso de Otras, genera despacho para cada uno y, les aplica TSF sobre despacho según entradas.
                2.9.- Retorna la tupla (EtaNum 1-indexed, pandas DataFrame)

        :param seed: Sets the random number to be tha same always
        :type seed: int, None
    """
    # Corrobora que el largo de los parámetros de entrada (teóricamente el Número de etapas), sea igual. De lo contrario retorna ValueError
    if len(Dict_TiposGen) != DF_HistGenERNC[0].shape[0] != DF_HistGenERNC[1].shape[0] != DF_TSF.shape[0] != DF_PE_Hid.shape[0]:
        msg = "El numero de etapa en los parámetros de entrada no coinciden."
        logger.error(msg)
        raise ValueError(msg)

    # para cada siguiente iteración de la función generadora, retorna una tupla de (EtaNum, DF_Pot_despchada_indiceGrid)
    for EtaNum in StageIndexesList:
        # Número de generadores
        # NGen = Dict_TiposGen[EtaNum].shape[0]
        # Inicializa DataFrame (nueva columna vacía) para potencias despachadas
        DF_IndGen_PDispatched = pd__concat([ Dict_TiposGen[EtaNum], pd__DataFrame(columns=['PGen_pu']) ], axis='columns')

        #
        # NUMERO DE TIPOS DE CENTRALES
        #
        # Obtiene nombres de los tipos ERNC (solar [0] y eólico [1])
        NombresERNCTipo = [GenTecNom[0] for RowNum, GenTecNom in Dict_TiposGen[EtaNum].iterrows() if ('EólicaZ' in GenTecNom[0]) | ('Solar' in GenTecNom[0])]
        # Calcula los indice de centrales de Embalse en la Grilla
        IndGenEmb = DF_IndGen_PDispatched[ DF_IndGen_PDispatched['type'] == 'Embalse' ].index.values
        # Calcula los indice de centrales de Serie en la Grilla
        IndGenSerie = DF_IndGen_PDispatched[ DF_IndGen_PDispatched['type'] == 'Serie' ].index.values
        # Calcula los indice de centrales de Pasada en la Grilla
        IndGenPasada = DF_IndGen_PDispatched[ DF_IndGen_PDispatched['type'] == 'Pasada'].index.values
        # Calcula los indice de centrales de Carbón en la Grilla
        IndGenTermoCarbon = DF_IndGen_PDispatched[ DF_IndGen_PDispatched['type'] == 'Carbón'].index.values
        # Calcula los indice de centrales de Gas-Diésel en la Grilla
        IndGenTermoGasDie = DF_IndGen_PDispatched[ DF_IndGen_PDispatched['type'] == 'Gas-Diésel'].index.values
        # Calcula los indice de centrales de Otras en la Grilla
        IndGenTermoOtras = DF_IndGen_PDispatched[ DF_IndGen_PDispatched['type'] == 'Otras'].index.values
        #
        #

        #
        # Encuentra el valor de despacho, aplicado con TSF
        #
        # Por cada nombre ERNC se obtiene e valor aleatorio, según sus medias y desviaciones estándar
        for NomERNC in NombresERNCTipo:
            # In case a seed is 'int', it's used to generate same numbers from seed. Otherwise makes it more random.
            np__random__seed(seed)

            if 'Solar' in NomERNC:
                Power_pu = np__random__normal( loc=float(DF_HistGenERNC[0].loc[EtaNum, NomERNC + '_mean']),
                                               scale=float(DF_HistGenERNC[0].loc[EtaNum, NomERNC + '_std']))
                # Asigna directamente TSF asignada (deja potencia en cero si está en falla)
                TasaFalla = DF_TSF[NomERNC][EtaNum]
                # La Tasa controla la probabilidad de falla
                Power_pu *= np__random__choice([0, 1], p=[TasaFalla, 1 - TasaFalla])
            elif 'EólicaZ' in NomERNC:
                Power_pu = np__random__normal( loc=float(DF_HistGenERNC[1].loc[EtaNum, NomERNC + '_mean']),
                                               scale=float(DF_HistGenERNC[1].loc[EtaNum, NomERNC + '_std']))
                # Asigna directamente TSF asignada (deja potencia en cero si está en falla)
                TasaFalla = DF_TSF[NomERNC][EtaNum]
                # La Tasa controla la probabilidad de falla
                Power_pu *= np__random__choice([0, 1], p=[TasaFalla, 1 - TasaFalla])
            else:
                msg = "NomERNC no es 'Solar' ni 'Eólico' de zona alguna!"
                logger.error(msg)
                raise ValueError(msg)

            # verifica que Power_pu sea positivo o cero, y limitado entre 0 y 1, inclusive
            Power_pu = 1.0 if Power_pu > 1 else 0 if Power_pu < 0 else Power_pu
            DF_IndGen_PDispatched.loc[DF_IndGen_PDispatched['type'] == NomERNC, 'PGen_pu'] = Power_pu
        # Para las tecnologías hidráulicas asigna promedio según PE y desv según parámetros 'DesvEstDespCenEyS' y 'DesvEstDespCenP'
        if len(IndGenEmb):  # EMBALSE
            # In case a seed is 'int', it's used to generate same numbers from seed. Otherwise makes it more random.
            np__random__seed(seed)

            # valor de pdf gaussiana/normal
            Power_pu = np__random__normal( loc=1 - DF_PE_Hid.loc[EtaNum, DF_PE_Hid.columns[0]],
                                           scale=DesvEstDespCenEyS,
                                           size=IndGenEmb.shape[0] )  # arrays
            # Asigna directamente TSF asignada (deja potencia en cero si está en falla)
            TasaFalla = DF_TSF['Embalse'][EtaNum]
            # multiplica las potencias despachadas para afectarlas por la tasa de falla. tasa controla la probabilidad de falla
            ModifyPower_Array = np__random__choice([0, 1], p=[TasaFalla, 1 - TasaFalla], size=IndGenEmb.shape[0])
            Power_pu *= ModifyPower_Array
            # Corrige limitando valores mayores que 1 y menores que 0
            Power_pu[ Power_pu < 0 ] = 0.0
            Power_pu[ 1 < Power_pu ] = 1.0
            # Asigna despacho al DataFrame
            DF_IndGen_PDispatched.loc[ IndGenEmb, 'PGen_pu'] = Power_pu

        if len(IndGenSerie):  # SERIE
            # In case a seed is 'int', it's used to generate same numbers from seed. Otherwise makes it more random.
            np__random__seed(seed)

            # valor de pdf gaussiana/normal
            Power_pu = np__random__normal( loc=1 - DF_PE_Hid.loc[EtaNum, DF_PE_Hid.columns[0]],
                                           scale=DesvEstDespCenEyS,
                                           size=IndGenSerie.shape[0] )  # arrays
            # Asigna directamente TSF asignada (deja potencia en cero si está en falla)
            TasaFalla = DF_TSF['Serie'][EtaNum]
            # multiplica las potencias despachadas para afectarlas por la tasa de falla. tasa controla la probabilidad de falla
            ModifyPower_Array = np__random__choice([0, 1], p=[TasaFalla, 1 - TasaFalla], size=IndGenSerie.shape[0])
            Power_pu *= ModifyPower_Array
            # Corrige limitando valores mayores que 1 y menores que 0
            Power_pu[ Power_pu < 0 ] = 0.0
            Power_pu[ 1 < Power_pu ] = 1.0
            # Asigna despacho al DataFrame
            DF_IndGen_PDispatched.loc[ IndGenSerie, 'PGen_pu'] = Power_pu

        if len(IndGenPasada):    # PASADA
            # In case a seed is 'int', it's used to generate same numbers from seed. Otherwise makes it more random.
            np__random__seed(seed)

            # valor de pdf gaussiana/normal
            Power_pu = np__random__normal( loc=1 - DF_PE_Hid.loc[EtaNum, DF_PE_Hid.columns[0]],
                                           scale=DesvEstDespCenP,
                                           size=IndGenPasada.shape[0] )  # arrays
            # Asigna directamente TSF asignada (deja potencia en cero si está en falla)
            TasaFalla = DF_TSF['Pasada'][EtaNum]
            # multiplica las potencias despachadas para afectarlas por la tasa de falla. tasa controla la probabilidad de falla
            ModifyPower_Array = np__random__choice([0, 1], p=[TasaFalla, 1 - TasaFalla], size=IndGenPasada.shape[0])
            Power_pu *= ModifyPower_Array
            # Corrige limitando valores mayores que 1 y menores que 0
            Power_pu[ Power_pu < 0 ] = 0.0
            Power_pu[ 1 < Power_pu ] = 1.0
            # Asigna despacho al DataFrame
            DF_IndGen_PDispatched.loc[ IndGenPasada, 'PGen_pu'] = Power_pu

        if len(IndGenTermoCarbon):    # CARBON
            # In case a seed is 'int', it's used to generate same numbers from seed. Otherwise makes it more random.
            np__random__seed(seed)

            # valor de pdf uniforme
            Power_pu = np_random__uniform( low=0.0,
                                           high=1.0,
                                           size=IndGenTermoCarbon.shape[0] )  # arrays
            # Asigna directamente TSF asignada (deja potencia en cero si está en falla)
            TasaFalla = DF_TSF['Carbón'][EtaNum]
            # multiplica las potencias despachadas para afectarlas por la tasa de falla. tasa controla la probabilidad de falla
            ModifyPower_Array = np__random__choice([0, 1], p=[TasaFalla, 1 - TasaFalla], size=IndGenTermoCarbon.shape[0])
            Power_pu *= ModifyPower_Array
            # Corrige limitando valores mayores que 1 y menores que 0
            Power_pu[ Power_pu < 0 ] = 0.0
            Power_pu[ 1 < Power_pu ] = 1.0
            # Asigna despacho al DataFrame
            DF_IndGen_PDispatched.loc[ IndGenTermoCarbon, 'PGen_pu'] = Power_pu

        if len(IndGenTermoGasDie):    # GAS-DIÉSEL
            # In case a seed is 'int', it's used to generate same numbers from seed. Otherwise makes it more random.
            np__random__seed(seed)

            # valor de pdf uniforme
            Power_pu = np_random__uniform( low=0.0,
                                           high=1.0,
                                           size=IndGenTermoGasDie.shape[0] )  # arrays
            # Asigna directamente TSF asignada (deja potencia en cero si está en falla)
            TasaFalla = DF_TSF['Gas-Diésel'][EtaNum]
            # multiplica las potencias despachadas para afectarlas por la tasa de falla. tasa controla la probabilidad de falla
            ModifyPower_Array = np__random__choice([0, 1], p=[TasaFalla, 1 - TasaFalla], size=IndGenTermoGasDie.shape[0])
            Power_pu *= ModifyPower_Array
            # Corrige limitando valores mayores que 1 y menores que 0
            Power_pu[ Power_pu < 0 ] = 0.0
            Power_pu[ 1 < Power_pu ] = 1.0
            # Asigna despacho al DataFrame
            DF_IndGen_PDispatched.loc[ IndGenTermoGasDie, 'PGen_pu'] = Power_pu

        if len(IndGenTermoOtras):    # OTRAS
            # In case a seed is 'int', it's used to generate same numbers from seed. Otherwise makes it more random.
            np__random__seed(seed)

            # valor de pdf uniforme
            Power_pu = np_random__uniform( low=0.0,
                                           high=1.0,
                                           size=IndGenTermoOtras.shape[0] )  # arrays
            # Asigna directamente TSF asignada (deja potencia en cero si está en falla)
            TasaFalla = DF_TSF['Otras'][EtaNum]
            # multiplica las potencias despachadas para afectarlas por la tasa de falla. tasa controla la probabilidad de falla
            ModifyPower_Array = np__random__choice([0, 1], p=[TasaFalla, 1 - TasaFalla], size=IndGenTermoOtras.shape[0])
            Power_pu *= ModifyPower_Array
            # Corrige limitando valores mayores que 1 y menores que 0
            Power_pu[ Power_pu < 0 ] = 0.0
            Power_pu[ 1 < Power_pu ] = 1.0
            # Asigna despacho al DataFrame
            DF_IndGen_PDispatched.loc[ IndGenTermoOtras, 'PGen_pu'] = Power_pu

        yield (EtaNum, DF_IndGen_PDispatched)
    pass


def overloaded_trafo2w(Grid, max_load=100):
    """
        Same as pandapower.overloaded_lines, but for two winding transformers
    """
    if Grid['res_trafo'].empty:
        return None
    else:
        return Grid['res_trafo'][ Grid['res_trafo']['loading_percent'] > max_load ].index


def overloaded_trafo3w(Grid, max_load=100):
    """
        Same as pandapower.overloaded_lines, but for three winding transformers
    """
    if Grid['res_trafo3w'].empty:
        return None
    else:
        return Grid['res_trafo3w'][ Grid['res_trafo3w']['loading_percent'] > max_load ].index


def TipoCong(Grid, max_load=100):
    """
        Identifica los grupos de elementos congestionados. El umbral para la cargabilidad
        del elemento de transmisión está dado en porcentaje por el parámetro 'max_load'.
        Para efectos prácticos los transformadores de tres devanados son considerados
        de potencia infinita, por lo que no se revisan por congestiones.
        Se hace de valer que los indices de las matrices de la red PandaPower
        son los mismo que los indices de los nodos en los grafos y subgrafos creados, siempre
        y cuando no exista un elemento perdido que las conecte, de lo contrario, se crea el nodo.

        Requiere que Grid posea valores del flujo de potencia ejecutado.

        Retorna una tupla con dos listas de grupos de elementos congestionados GCong serie.
        Ejemplo return:
            ( ListaCongInter, ListaCongIntra )
            donde:
                ListaCongIntra = [GCong1, GCong2, ...]
                ListaCongInter = [GCong1, GCong2, ...]
                GCong# = {  # diccionario
                            'line': [IndTableElmn, IndTableElmn, ...],
                            'trafo2w': [IndTableElmn, IndTableElmn, ...],
                            # 'trafo3w': [IndTableElmn, IndTableElmn, ...],
                        }
    """
    #
    # Verifica si existen congestiones (sobre 100% de carga), obtiene
    # los indices de cada DataFrame
    Saturated_lines = pp__overloaded_lines(Grid, max_load=max_load)
    Saturated_trafo2w = overloaded_trafo2w(Grid, max_load=max_load)
    # Saturated_trafo3w = overloaded_trafo3w(Grid, max_load=max_load)

    #
    # Inicializa set temporales de Congestiones tipo Inter e Intra
    InterCongestion = pd__DataFrame(columns=['TypeElmnt', 'IndTable', 'BarraA', 'BarraB', 'FluP_AB_kW'])
    IntraCongestion = pd__DataFrame(columns=['TypeElmnt', 'IndTable', 'BarraA', 'BarraB', 'FluP_AB_kW'])

    # IDENTIFICA LAS TIPOS DE CONGESTIONES en DataFrames
    #
    # Inicializa diccionario de grupo de elementos congestionados (solo para iterar for y reducir lineas código)
    Dict_GroupElmnts = {'line': Saturated_lines, 'trafo': Saturated_trafo2w}  # , 'trafo3w': Saturated_trafo3w}
    # Para cada congestión encontrada Identifica el tipo (Inter - Intra)
    for TypeElmnt, GroupElmnts in Dict_GroupElmnts.items():
        if GroupElmnts is None:
            # En caso de GroupElmnts ser None (No existen congestiones de este tipo),
            # se continúa con siguiente grupo, de lo contrario se revisa tipo cong
            continue
        for CongBrnch in GroupElmnts:
            # crea un grafo a partir de la Grilla
            Graph = pp__create_nxgraph( Grid, respect_switches=True, include_lines=True,
                                        include_trafos=True, include_impedances=False,
                                        nogobuses=None, notravbuses=None,
                                        # nogobuses=None, notravbuses=[CongBrnch],
                                        multi=True, calc_r_ohm=False, calc_z_ohm=False)
            # Obtiene números de nodos que conecta 'CongBrnch' (Da igual dirección)
            if TypeElmnt == 'line':
                Ind_NIni = Grid[TypeElmnt]['from_bus'][CongBrnch]
                Ind_NFin = Grid[TypeElmnt]['to_bus'][CongBrnch]
                # Elimina el Edge del grafo
                Graph.remove_edge(Ind_NIni, Ind_NFin)
                # Determina si aún sin el elemento existe conexión
                IsIntra = nx__has_path(Graph, Ind_NIni, Ind_NFin)
                name_from_BarraA = 'p_from_kw'
            elif TypeElmnt == 'trafo':
                Ind_NIni = Grid[TypeElmnt]['hv_bus'][CongBrnch]
                Ind_NFin = Grid[TypeElmnt]['lv_bus'][CongBrnch]
                # Elimina el Edge del grafo
                Graph.remove_edge(Ind_NIni, Ind_NFin)
                # Determina si aún sin el elemento existe conexión
                IsIntra = nx__has_path(Graph, Ind_NIni, Ind_NFin)
                name_from_BarraA = 'p_hv_kw'
            else:
                print("No es elemento identificado en congestión!")
                pass

            if IsIntra:
                # aún sin el elemento existe camino
                df_temp = pd__DataFrame(
                    { 'TypeElmnt': [ TypeElmnt ],
                      'IndTable': [ CongBrnch ],
                      'BarraA': [ Ind_NIni ],
                      'BarraB': [ Ind_NFin ],
                      'FluP_AB_kW': [ abs(Grid['res_' + TypeElmnt].at[CongBrnch, name_from_BarraA]) ],
                      })
                IntraCongestion = pd__concat([IntraCongestion, df_temp], axis='index', ignore_index=True)
                # APROXIMA valores del flujo al kW (posibles variaciones numéricas)
                IntraCongestion[['FluP_AB_kW']] = IntraCongestion['FluP_AB_kW'].round(decimals=0)
            else:
                # elemento separa el SEP
                df_temp = pd__DataFrame(
                    { 'TypeElmnt': [ TypeElmnt ],
                      'IndTable': [ CongBrnch ],
                      'BarraA': [ Ind_NIni ],
                      'BarraB': [ Ind_NFin ],
                      'FluP_AB_kW': [ abs(Grid['res_' + TypeElmnt].at[CongBrnch, name_from_BarraA]) ],
                      })
                InterCongestion = pd__concat([InterCongestion, df_temp], axis='index', ignore_index=True)
                # APROXIMA valores del flujo al kW (posibles variaciones numéricas)
                InterCongestion[['FluP_AB_kW']] = InterCongestion['FluP_AB_kW'].round(decimals=0)

    # Inicializa listas de salida
    ListaCongInter = []
    ListaCongIntra = []

    """
         ####                                ###           #
         #   #                                #            #
         #   #   ###   # ##    ###            #    # ##   ####    ###   # ##
         ####       #  ##  #      #           #    ##  #   #     #   #  ##  #
         #       ####  #       ####           #    #   #   #     #####  #
         #      #   #  #      #   #           #    #   #   #  #  #      #
         #       ####  #       ####          ###   #   #    ##    ###   #
    """
    # Por cada congestión del tipo Inter verifica las
    # potencias circulantes similares agrupándolas, además
    # estando los elementos adyacentes (una barra en común)
    #
    for FluP, DF_cong in InterCongestion.groupby(by=['FluP_AB_kW']):
        DF_cong.reset_index(drop=True, inplace=True)  # para trabajar con lo indices
        # En caso de ser un DF de una fila, se agrega como grupo de congestión independiente
        if DF_cong.shape[0] == 1:
            tipoElemento = DF_cong.iloc[0]['TypeElmnt']
            indiceTabla = DF_cong.iloc[0]['IndTable']
            if tipoElemento == 'line':
                ListaCongInter.append( {'line': [indiceTabla], 'trafo': []} )
            else:
                ListaCongInter.append( {'line': [], 'trafo': [indiceTabla]} )
            continue
        elif DF_cong.empty:
            # extraño caso que estuviera vacío (quizá no tanto)
            continue

        IndBarras = set()
        # Obtiene los indices de las barras por cada fila
        for row in DF_cong.iterrows():
            IndBarras.update( set(DF_cong['BarraA']) )
            IndBarras.update( set(DF_cong['BarraB']) )
        # crea sub-grilla con las barras de las filas
        SubGrilla = pp__select_subnet(Grid, IndBarras)
        # obtiene grafo de la grilla
        Grafo = pp__create_nxgraph(SubGrilla)
        # identifica los subgrafos existentes.
        # Aquí ya se sabe cuales son los grupos
        for SubGrafo in nx__connected_component_subgraphs(Grafo):
            # inicializa grupo de congestiones comunes
            GrupoCongCom = {'line': [], 'trafo': []}
            # Por cada arco del subgrafo, busca equivalente en Grid
            for Arco in SubGrafo.edges:
                # Obtiene los nodos/indice de barras del Grid del arco
                NIni = Arco[0]
                NFin = Arco[1]
                # Encuentra el elemento de la Grilla que se encuentra entre esos nodos.
                #
                # verifica si existen trafos conectados aquí
                Tw2 = Grid['trafo'][
                    ( (Grid['trafo']['hv_bus'] == NIni) & (Grid['trafo']['lv_bus'] == NFin) ) |  # un sentido
                    ( (Grid['trafo']['hv_bus'] == NFin) & (Grid['trafo']['lv_bus'] == NIni) )  # el otro sentido
                ]
                if not Tw2.empty:
                    # agrega elemento encontrado al grupo. Solo una coincidencia.
                    GrupoCongCom['trafo'].append(Tw2.index[0])
                #
                # verifica si existen lineas conectados aquí
                Ln = Grid['line'][
                    ( (Grid['line']['from_bus'] == NIni) & (Grid['line']['to_bus'] == NFin) ) |  # un sentido
                    ( (Grid['line']['from_bus'] == NFin) & (Grid['line']['to_bus'] == NIni) )  # el otro sentido
                ]
                if not Ln.empty:
                    # agrega elemento encontrado al grupo. Solo una coincidencia.
                    GrupoCongCom['line'].append(Ln.index[0])
            # Agrega grupo a lista
            ListaCongInter.append(GrupoCongCom)

    """
         ####                                ###           #
         #   #                                #            #
         #   #   ###   # ##    ###            #    # ##   ####   # ##    ###
         ####       #  ##  #      #           #    ##  #   #     ##  #      #
         #       ####  #       ####           #    #   #   #     #       ####
         #      #   #  #      #   #           #    #   #   #  #  #      #   #
         #       ####  #       ####          ###   #   #    ##   #       ####
    """
    # Por cada congestión del tipo Intra verifica las
    # potencias circulantes similares agrupándolas, además
    # estando los elementos adyacentes (una barra en común)
    #
    for FluP, DF_cong in IntraCongestion.groupby(by=['FluP_AB_kW']):
        DF_cong.reset_index(drop=True, inplace=True)  # para trabajar con lo indices
        # En caso de ser un DF de una fila, se agrega como grupo de congestión independiente
        if DF_cong.shape[0] == 1:
            tipoElemento = DF_cong.iloc[0]['TypeElmnt']
            indiceTabla = DF_cong.iloc[0]['IndTable']
            if tipoElemento == 'line':
                ListaCongIntra.append( {'line': [indiceTabla], 'trafo': []} )
            else:
                ListaCongIntra.append( {'line': [], 'trafo': [indiceTabla]} )
            continue
        elif DF_cong.empty:
            # extraño caso que estuviera vacío (quizá no tanto)
            continue

        IndBarras = set()
        # Obtiene los indices de las barras por cada fila
        for row in DF_cong.iterrows():
            IndBarras.update( set(DF_cong['BarraA']) )
            IndBarras.update( set(DF_cong['BarraB']) )
        # crea sub-grilla con las barras de las filas
        SubGrilla = pp__select_subnet(Grid, IndBarras)
        # obtiene grafo de la grilla
        Grafo = pp__create_nxgraph(SubGrilla)
        # identifica los subgrafos existentes.
        # Aquí ya se sabe cuales son los grupos
        for SubGrafo in nx__connected_component_subgraphs(Grafo):
            # inicializa grupo de congestiones comunes
            GrupoCongCom = {'line': [], 'trafo': []}
            # Por cada arco del subgrafo, busca equivalente en Grid
            for Arco in SubGrafo.edges:
                # Obtiene los nodos/indice de barras del Grid del arco
                NIni = Arco[0]
                NFin = Arco[1]
                # Encuentra el elemento de la Grilla que se encuentra entre esos nodos.
                #
                # verifica si existen trafos conectados aquí
                Tw2 = Grid['trafo'][
                    ( (Grid['trafo']['hv_bus'] == NIni) & (Grid['trafo']['lv_bus'] == NFin) ) |  # un sentido
                    ( (Grid['trafo']['hv_bus'] == NFin) & (Grid['trafo']['lv_bus'] == NIni) )  # el otro sentido
                ]
                if not Tw2.empty:
                    # agrega elemento encontrado al grupo. Solo una coincidencia.
                    GrupoCongCom['trafo'].append(Tw2.index[0])
                #
                # verifica si existen lineas conectados aquí
                Ln = Grid['line'][
                    ( (Grid['line']['from_bus'] == NIni) & (Grid['line']['to_bus'] == NFin) ) |  # un sentido
                    ( (Grid['line']['from_bus'] == NFin) & (Grid['line']['to_bus'] == NIni) )  # el otro sentido
                ]
                if not Ln.empty:
                    # agrega elemento encontrado al grupo. Solo una coincidencia.
                    GrupoCongCom['line'].append(Ln.index[0])
            # Agrega grupo a lista
            ListaCongIntra.append(GrupoCongCom)

    # Retorna valores
    return (ListaCongInter, ListaCongIntra)
