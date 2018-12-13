from datetime import timedelta as dt__timedelta
from pandas import DataFrame as pd__DataFrame, date_range as pd__date_range, Series as pd__Series
from pandas import datetime as pd__datetime, set_option as pd__set_option
from pandas import concat as pd__concat, Timedelta as pd__Timedelta
from numpy import mean as np__mean, nan as np__NaN, arange as np__arange, bool_ as np__bool_
from numpy.random import uniform as np_random__uniform
from dateutil.relativedelta import relativedelta as du__relativedelta
from collections import OrderedDict as collections__OrderedDict
from os.path import sep as os__path__sep
from pandapower import to_json as pp__to_json, to_pickle as pp__to_pickle
from json import dump as json__dump

import locale
import logging

locale.setlocale(locale.LC_ALL, 'es_ES.utf8')
logging.basicConfig(level=logging.DEBUG,
                    format="[%(levelname)s][%(asctime)s] - %(message)s")
logger = logging.getLogger()


def date_parser(x):
    return pd__datetime.strptime(x, '%Y-%m-%d %H:%M')


def print_full_df():
    # allow print all panda columns
    pd__set_option('precision', 4)
    pd__set_option('expand_frame_repr', False)


def Crea_Etapas_desde_Cambio_Mant(DF_CambioFechas, ref_fija=True):
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
    logger.debug("! entrando en función: 'Lista2DF_consecutivo' (aux_funcs.py) ...")
    LAux = []
    for IndElmn in range(len(Lista) - 1):
        if IndElmn == 0:
            LAux.append( [ Lista[IndElmn], Lista[IndElmn + 1] ] )
        else:
            LAux.append( [ Lista[IndElmn] + incremento, Lista[IndElmn + 1] ] )

    logger.debug("! saliendo en función: 'Lista2DF_consecutivo' (aux_funcs.py) ...")
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
        Desv_dec = np__mean( (Desv_dec['programado'] - Desv_dec['real']) / Desv_dec['programado'])   # Notar es muy difícil que demanda sistema programada sea cero.
        # print("Desv_dec:", Desv_dec)

        # Asigna valores al DataFrame de salida
        DF_Salida.loc[ Num, ['EtaNum', 'Desv_decimal'] ] = [Num, Desv_dec]
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
        Notar que para los meses que no sean informados, mantendrán la misma tasa que el último informado

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
    DF_TasaAcumulativa = DF_ProyDem + 1  # agrega constante a todos los valores del DataFrame
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


def Crea_hidrologias_futuras(DF_HistHid, DF_Etapas, PE_HidSeca, PE_HidMedia, PE_HidHumeda, FechaIniSim, FechaFinSim):
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


def Imprime_Data_Etapa(TempFolderName, EtaNum, BD_DemProy, BD_Hidrologias_futuras, BD_TSFProy, BD_RedesXEtapa):
    """
        Imprimir por cada etapa dos archivos, uno llamado '#.json' donde # es el número de la etapa y, otro llamado 'Grid_#.json' que contiene
        la red asociada a la etapa casi lista para simular lpf.

        Todo es creado dentro del directorio predefinido 'TempData'.
        La variable a imprimir es un diccionario escrito como json, que posee la información solicitada.
        No retorna valor alguno.
    """
    DictImprimir = {
        'BD_DemProy': BD_DemProy.loc[EtaNum, :].to_dict(),
        'BD_Hidrologias_futuras': BD_Hidrologias_futuras.loc[EtaNum, :].to_dict(),
        'BD_TSFProy': BD_TSFProy.loc[EtaNum, :].to_dict(),
        'BD_RedesXEtapa_ExtraData': BD_RedesXEtapa[EtaNum]['ExtraData']
    }

    # Guarda Datos de etapa en archivo JSON
    with open(TempFolderName + os__path__sep + "{}.json".format(EtaNum), 'w') as f:
        json__dump(DictImprimir, f)

    # Exporta la red a archivo pickle. Necesario para exportar tipos de lineas. Más pesado que JSON y levemente lento pero funcional... :c
    pp__to_pickle( BD_RedesXEtapa[EtaNum]['PandaPowerNet'], TempFolderName + os__path__sep + "Grid_{}.p".format(EtaNum) )

    return
