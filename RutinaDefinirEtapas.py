#!/usr/bin/env python3
"""
    Rutina DefineEtapas para programa MCFPL.py
    Autor: Gabriel Seguel G.

    Actualiza las base de datos proyectadas (BD_ProyDem) y de datos de mantención (BD_MantGen, BD_MantTx, BD_MantCargas, BD_MantBarras), pasandolas de valores temporales (hora/dia/mes/año) a su correspondiente índice de etapa.

    Las etapas son determinadas mediante la función DefineEtapas(), retornandose un diccionario con la relación (FechaIni, FechaFin): NumEta

    Contiene la función para identificar la cantidad y duración de etapas a partir de curvas de probabilidad.
    Contiene funciones para actualizar las base de datos para definir etapas.
"""
from datetime import datetime as dt__datetime
from datetime import timedelta as dt__timedelta
from pandas import DataFrame as pdDataFrame

def DefineEtapas(BD_ProyDem, BD_MantGen, BD_MantTx, BD_MantCargas, BD_MantBarras, BD_ProbGen, BD_ProbDem, BD_ParamSim):
    """
            ...
            Retorna diccionario con clave de tupla (FechaIni, FechaFin) de las fechas que componen la etapa, cuyo value es el número de la correspondiente etapa.
                        El diccionario tiene la forma:
                                {
                                        (FechaIni, FechaFin, NumEta),
                                        (FechaIni, FechaFin, NumEta),
                                }
            Recordar que la fechas en módulo datetime se escriben como dt__datetime(año, mes, día, hora, ...) hasta el microsegundo. Aquí importa solo hasta la hora. Así el rango horario es de 0 - 23. Los minutos son despreciados.

            Arguments:
                    BD_probgen {} --
                    BD_probgen {} --
                    BD_probgen {} --
                    BD_probgen {} --
    """

    # `MM(      )M' `MM'    `MM'
    #  `MM.     d'   MM      MM
    #   `MM.   d'    MM      MM    ___      ____     ____   ___  __
    #    `MM. d'     MM      MM  6MMMMb    6MMMMb.  6MMMMb  `MM 6MM
    #     `MMd       MMMMMMMMMM 8M'  `Mb  6M'   Mb 6M'  `Mb  MM69 "
    #      dMM.      MM      MM     ,oMM  MM    `' MM    MM  MM'
    #     d'`MM.     MM      MM ,6MM9'MM  MM       MMMMMMMM  MM
    #    d'  `MM.    MM      MM MM'   MM  MM       MM        MM
    #   d'    `MM.   MM      MM MM.  ,MM  YM.   d9 YM    d9  MM
    # _M(_    _)MM_ _MM_    _MM_`YMMM9'Yb. YMMMM9   YMMMM9  _MM_
    # Analizar a futuro que las fechas de las etapas estén separadas a almenos 1 hora y que no se solapen.

    # Identifica los cambios de topología
    # crea subdivisión horaria por día
    # lista de salida ordenada de temprano a tarde
    d = [
        (
            dt__datetime(2018, 6, 1, 0),
            dt__datetime(2018, 6, 14, 23),
            1
        ),
        (
            dt__datetime(2018, 6, 15, 0),
            dt__datetime(2018, 6, 30, 23),
            2
        ),
    ]
    return d


def BD_Fechas_a_Etapas(BD_Fechas, RelFechaEta):
    """
            Crea DataFrame con nueva columna de 'Eta' y con las etapas respectivas. No se consideran las columnas de 'FechaIni' ni 'FechaFin'.

            Retorna mismo DataFrame pero en función de etapas y no fechas.

            Arguments:
                    BD_Fechas   {pandas.DataFrame} -- Base de datos de proyección de demanda.
                    RelFechaEta {Dict}             -- Relación de etapa con sus fechas límite.
    """
    try:
        # Intenta ver si las fechas de cada fila cumplen con 'FechaIni' < 'FechaFin'. De lo contrario, escribe error personalizado
        if not BD_Fechas[ BD_Fechas['FechaIni'] >= BD_Fechas['FechaFin'] ].empty:
            raise ValueError("{0:¡^7}ERROR{1:!<7}: 'FechaIni' >= 'FechaFin' en archivo 'in_mcfpl_proydem.csv' o 'in_mcfpl_indisp???.csv':\n{2}".format(
                '¡',
                '!',
                BD_Fechas[ BD_Fechas['FechaIni'] >= BD_Fechas['FechaFin'] ][['FechaIni', 'FechaFin']]
            ))
    except ValueError as Verr:
        # Captura el error y termina la sesión del interprete.
        print(Verr)
        raise SystemExit  # request exit interpreter

    # print() # DEBUGGING
    # observa las fechas temporales y dependiendo de su ubicación con respecto a las etapa definidas, se les asigna en una considerando un valor promedio.
    DF = pdDataFrame()
    for etapa in RelFechaEta:
        # print('---> etapa {} -- Inicio {}  -- Final {}:'.format(etapa[2], etapa[0], etapa[1]))  # DEBUGGING
        # Identifica cuales son los rangos de cada elementos usado, que están completamente antes o después de las horas de la etapa
        RangoAntesDeEtapa = ( BD_Fechas['FechaIni'] < etapa[0]) & ( BD_Fechas['FechaFin'] < etapa[0])
        RangoDespuesDeEtapa = ( etapa[1] < BD_Fechas['FechaIni'] ) & ( etapa[1] < BD_Fechas['FechaFin'] )
        # El complemento de los que están afuera poseen alemnos una al menos una hora en la etapa.
        PoseeHoraEnEta = ~(RangoAntesDeEtapa | RangoDespuesDeEtapa)
        # print('RangoAntesDeEtapa:\n', RangoAntesDeEtapa)  # DEBUGGING
        # print('RangoDespuesDeEtapa:\n', RangoDespuesDeEtapa)  # DEBUGGING
        # print('--PoseeHoraEnEta:\n', PoseeHoraEnEta)  # DEBUGGING
        # crea DataFrame temporal para guardar datos de los rangos de cada elemento que pertenecen a la etapa
        DF_temp = BD_Fechas[ PoseeHoraEnEta ]
        # print('--DF_temp:\n', DF_temp)  # DEBUGGING
        # se calcula la diferencia de tiempo total que posee cada rango identificado y se agrega como columna 'DeltaTiempo'
        DF_temp = DF_temp.assign( DeltaTiempo=DF_temp['FechaFin'] - DF_temp['FechaIni'] )
        # print('--DF_temp:\n', DF_temp)  # DEBUGGING
        # agrega nueva columna lógica positiva a los rangos que exceden de la fecha de término de la etapa (etapa[1])
        DF_temp = DF_temp.assign( ExcedeEtaTerm=(DF_temp['FechaFin'] - etapa[1]) > dt__timedelta(0) )
        # print('--DF_temp:\n', DF_temp)  # DEBUGGING
        # agrega nueva columna con el excedente de los rangos que exceden de la fecha de término de la etapa (etapa[1])
        DF_temp = DF_temp.assign( ValorExcedente=DF_temp['FechaFin'] - etapa[1] )
        # print('--DF_temp:\n', DF_temp)  # DEBUGGING
        # actualiza columa de DeltaTiempo, restando el tiempo de aquellos mantenimientos que exceden la fecha de término de la etapa
        DF_temp['DeltaTiempo'] -= DF_temp['ExcedeEtaTerm'].astype(int) * DF_temp['ValorExcedente']
        # print('--DF_temp:\n', DF_temp)  # DEBUGGING

        # Selecciona el indice del rango de cada elemento con mayor timedelta, luego utiliza aquellos índices para filtrar los máximos (depende del DF de entrada)
        if 'GenNom' in DF_temp.index.name:  # sólo para BD_MantGen
            # print('--------GenNom--------')
            idx = DF_temp.groupby('GenNom', sort=False)['DeltaTiempo'].transform(max) == DF_temp['DeltaTiempo']
            DF_temp = DF_temp[idx]
        elif 'LinNom' in DF_temp.index.name:  # sólo para BD_MantTx
            # print('--------LinNom--------')
            idx = DF_temp.groupby('LinNom', sort=False)['DeltaTiempo'].transform(max) == DF_temp['DeltaTiempo']
            DF_temp = DF_temp[idx]
        elif 'LoadNom' in DF_temp.index.name:  # sólo para BD_ProyDem y BD_IndispCarga
            # print('--------LoadNom--------')
            idx = DF_temp.groupby('LoadNom', sort=False)['DeltaTiempo'].transform(max) == DF_temp['DeltaTiempo']
            DF_temp = DF_temp[idx]
        else:    # sólo para BD_MantBarras
            # print('--------BarNom--------')
            idx = DF_temp.groupby('BarNom', sort=False)['DeltaTiempo'].transform(max) == DF_temp['DeltaTiempo']
            DF_temp = DF_temp[idx]
        # print('--DF_temp(max):\n', DF_temp)  # DEBUGGING
        # Agrega a dataframe la columna extra 'Eta' con el número de la etapa
        DF_temp = DF_temp.assign( Eta=etapa[2] )
        # print('--DF_temp:\n', DF_temp)  # DEBUGGING
        # elimina las columnas innecesarias: 'FechaIni', 'FechaFin', 'ExcedeEtaTerm', 'DeltaTiempo', 'ValorExcedente'
        DF_temp = DF_temp.drop(['FechaIni', 'FechaFin', 'ExcedeEtaTerm', 'DeltaTiempo', 'ValorExcedente'], 1)
        # print('--DF_temp:\n', DF_temp)  # DEBUGGING
        # agrega datos al DF de salida
        DF = DF.append(DF_temp)

    # print('--DF:\n', DF)  # DEBUGGING

    return DF
