#!/usr/bin/env python3
"""
    Rutina LectorEntradas para programa MCFPL.py
    Autor: Gabriel Seguel G.

    Contiene funciones definidas para leer los archivos de entrada al programa y pasarlo a las base de datos.
"""
from pandas import read_csv as pd__read_csv
from xlrd import xldate_as_tuple as xlrd__xldate_as_tuple
from datetime import datetime as dt__datetime


def LeeDatosTecnicos(NombreArchivo):
    """
            Lee la información del Archivo 'NombreArchivo' para ser luego transformada a un diccionario de estilo
                    {'ElmNom':
                            {'Parámetro': Valor,
                            'Parámetro': Valor,
                            ...
                            },
                            ...
                    }
            Es útil para 4 archivos: 
                    Técnico Generadores: 'in_mcfpl_tecgen.csv'
                    Técnico Barras     : 'in_mcfpl_tecbarras.csv'
                    Técnico Lineas     : 'in_mcfpl_teclineas.csv'
                    Técnico Cargas     : 'in_mcfpl_teccargas.csv'

            Retorna diccionario estilo DataFrame con la información procesada según tipo. De estar vacío retorna Error.

            Syntax:
                    Dict = LeeDatosTecnicos(NombreArchivo)

            Arguments:
                            NombreArchivo {str} -- Nombre del archivo de entrada csv.

    """
    # Utiliza primera columna como los nombres (debiesen ingresarse así en los datos de entrada!).
    DF = pd__read_csv(NombreArchivo, index_col=0)
    # Lee dataFrame, lo transpone y transforma a diccionario.
    try:
        if DF.empty:
            raise ValueError("Error: {} No se pudo leer !! (DataFrame vacío)".format(NombreArchivo))
        else:
            return DF.T.to_dict()
    except ValueError as Verr:
        # Catch only error of empty dataframe
        print(Verr)
        raise SystemExit  # request exit interpreter


def LeeProyeccDem(NombreArchivoProyDem):
    """
            Lee la información existente de la proyección de la demanda para cada carga y la almacena como un dataFrame usando indice como nombre de la carga a la que le corresponde. Es útil solo para el archivo de demando proyectada: 'in_mcfpl_proydem.csv'.

            Retorna DataFrame con la información procesada según tipo. De estar vacío retorna Error.
            Es útil para 4 archivos: 
                    Proyección Media Demanda: 'in_mcfpl_proydem.csv'

            Syntax:
                    DF = LeeProyeccDem(NombreArchivoProyDem)

            Arguments:
                            NombreArchivoProyDem {str} -- Nombre del archivo de entrada csv.

    """
    def date_parser(x):
        """
            función que se aplica a cada 'parse_dates'. Se desempaca la tupla como argumentos (año, mes, día, hora, segundo) desde 'xlrd__xldate_as_tuple'. datemode=0 considera desfase de 1900.
        """
        return dt__datetime( *xlrd__xldate_as_tuple(float(x), datemode=0) )

    DF = pd__read_csv( NombreArchivoProyDem, index_col=['LoadNom'], parse_dates=['FechaIni', 'FechaFin'], date_parser=date_parser )

    # print(DF)
    # print( DF['FechaIni'].iloc[0].year )
    # print( DF['FechaIni'].iloc[0].month )
    # print( DF['FechaIni'].iloc[0].day )
    # print( DF['FechaIni'].iloc[0].hour )
    # print( DF['FechaIni'].iloc[0].second )

    try:
        if DF.empty:
            raise ValueError("Error: {} No se pudo leer !! (DataFrame vacío)".format(NombreArchivoProyDem))
        else:
            return DF
    except ValueError as Verr:
        print(Verr)
        raise SystemExit  # request exit interpreter


def LeeDatosMant(NombreArchivo):
    """
            Lee el archivo csv de mantención de algún elemento, como pueden los archivo de:
                    Generadores: 'in_mcfpl_mantpgen.csv'
                    Líneas: 'in_mcfpl_mantptx.csv'
                    Cargas: 'in_mcfpl_mantpcargas.csv'

            Retorna un DataFrame con la información procesada según tipo. De estar vacío retorna flag Falso.

            Syntax:
                    DF = LeeDatosMant(NombreArchivo)

            Arguments:
                    NombreArchivo {str} -- Nombre del archivo de entrada csv.
    """
    def date_parser(x):
        """
            función que se aplica a cada 'parse_dates'. Se desempaca la tupla como argumentos (año, mes, día, hora, segundo) desde 'xlrd__xldate_as_tuple'. datemode=0 considera desfase de 1900.
        """
        return dt__datetime( *xlrd__xldate_as_tuple(float(x), datemode=0) )

    DF = pd__read_csv( NombreArchivo, index_col=0, parse_dates=['FechaIni', 'FechaFin'], date_parser=date_parser, dtype={'Operativa': bool} )

    try:
        if DF.empty:
            raise ValueError("Error: {} No se pudo leer !! (DataFrame vacío)".format(NombreArchivo))
        else:
            return DF
    except ValueError as Verr:
        print(Verr)
        raise SystemExit  # request exit interpreter


def LeeDatosCurvProb(NombreArchivoCurvas):
    """
            Lee los archivos que describen las curvas de distribución de probabilidad de los grupos existentes de generación y demanda. Notar que cada parámetro (par 1, par 2, etc) va a definir uno correspondiente a su función de probabilidad asociada. Es utilizado para los archivos de entrada:
                    'in_mcfpl_probgen.csv'
                    'in_mcfpl_probdem.csv'

            Retorna un DataFrame con la información procesada según tipo. De estar vacío retorna flag Falso.

            Syntax:
                    DF = LeeDatosCurvProb(NombreArchivoCurvas)

            Arguments:
                    NombreArchivoCurvas {str} -- Nombre del archivo de probabilidad de generación o demanda.
    """
    # AUN NO LO VOY A HACER HASTA IDENTIFICAR LAS CURVAS A UTILIZAR.
    # ____      ___ ____    ____
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

    return {}


def LeeDatosSimulacion(ArchivoSimulacion):
    """
            Lee el archivo de simulación 'in_mcfpl_simulacion.csv' para obtener los parámetros necesarios y guardarlos en un diccionario. Este archivo debe existir con los parámetros y valores de lo contrario se tiene error.
            Los parámetros mínimos que deben estar son:
                                        Sbase         : Valor en MVA
                                        MaxItCongInter: Número máximo de iteraciones por congestion tipo 'inter'
                                        MaxItCongIntra: Número máximo de iteraciones por congestion tipo 'intra'
                                        FechaComienzo : Fecha de inicio de simulación del horizonte de estudio
                                        FechaTermino  : Fecha de inicio de simulación del horizonte de estudio
                                        NumVecesDem   : Número de veces que se generan puntos aleatorios a la demanda.
                                        NumVecesGen   : Número de veces que se generan puntos aleatorios a la generación.
                                        PerdCoseno    : Flag lógico que indica si se deben realizar cálculo de pérdidas por metodo coseno.

            Retorna Diccionario con parámetros de la simulación.

            Syntax:
                    DF = LeeDatosSimulacion(ArchivoSimulacion)

            Arguments:
                    ArchivoSimulacion {str} -- Nombre de Archivo de entrada para parámetros de 'in_mcfpl_simulacion.csv'.
    """
    DF = pd__read_csv(ArchivoSimulacion, index_col=0, header=None).transpose()

    # bloques try: indentifican existencia de parámetros criticos. Se asigna los TIPOS que no sean string.
    # Parámetro: Sbase
    try:
        # asigna tipo de dato.
        aux_valor = float(DF['Sbase'])
        DF['Sbase'] = aux_valor
        # corrobora que el valor asignado sea diferente de vacío
        if aux_valor <= 0:
            raise ValueError("Error: parámetro 'Sbase' debe ser mayor que cero !!")
    except ValueError as Verr:
        print(Verr)
        raise SystemExit  # request exit interpreter
    except:
        print("Error: Parámetro 'Sbase' no definido en 'in_mcfpl_simulacion.csv'!!!")
        raise SystemExit  # request exit interpreter

    # Parámetro: FechaComienzo
    try:
        # asigna tipo de dato.
        aux_valor = dt__datetime( *xlrd__xldate_as_tuple( float(DF['FechaComienzo']), datemode=0 ) )
        DF['FechaComienzo'] = aux_valor
        # corrobora que el valor asignado sea diferente de vacío
        if not aux_valor:
            raise ValueError("Error: valor del parámetro 'FechaComienzo' debe existir !!")
    except ValueError as Verr:
        print(Verr)
        raise SystemExit  # request exit interpreter
    except:
        print("Error: Parámetro 'FechaComienzo' no definido en 'in_mcfpl_simulacion.csv'!!!")
        raise SystemExit  # request exit interpreter

    # Parámetro: FechaTermino
    try:
        # asigna tipo de dato.
        aux_valor = dt__datetime( *xlrd__xldate_as_tuple( float(DF['FechaTermino']), datemode=0 ) )
        DF['FechaTermino'] = aux_valor
        # corrobora que el valor asignado sea diferente de vacío
        if not aux_valor:
            raise ValueError("Error: valor del parámetro 'FechaTermino' debe existir !!")
        if aux_valor <= DF['FechaComienzo'].iloc[0]:
            raise ValueError("Error: valor del parámetro 'FechaTermino' debe ser mayor que 'FechaComienzo' !!")
    except ValueError as Verr:
        print(Verr)
        raise SystemExit  # request exit interpreter
    except:
        print("Error: Parámetro 'FechaComienzo' no definido en 'in_mcfpl_simulacion.csv'!!!")
        raise SystemExit  # request exit interpreter

    # Parámetro: MaxItCongInter
    try:
        # asigna tipo de dato.
        aux_valor = int(float(DF['MaxItCongInter']))
        DF['MaxItCongInter'] = aux_valor
        # corrobora valor del valor asignado
        if aux_valor < 0:
            raise ValueError("Error: parámetro 'MaxItCongInter' debe ser mayor o igual que cero !!")
    except ValueError as Verr:
        print(Verr)
        raise SystemExit  # request exit interpreter
    except:
        print("Error: Parámetro 'MaxItCongInter' no definido en 'in_mcfpl_simulacion.csv'!!!")
        raise SystemExit  # request exit interpreter

    # Parámetro: MaxItCongIntra
    try:
        # asigna tipo de dato.
        aux_valor = int(float(DF['MaxItCongIntra']))
        DF['MaxItCongIntra'] = aux_valor
        # corrobora valor del valor asignado
        if aux_valor < 0:
            raise ValueError("Error: parámetro 'MaxItCongIntra' debe ser mayor o igual que cero !!")
    except ValueError as Verr:
        print(Verr)
        raise SystemExit  # request exit interpreter
    except:
        print("Error: Parámetro 'MaxItCongIntra' no definido en 'in_mcfpl_simulacion.csv'!!!")
        raise SystemExit  # request exit interpreter

    # Parámetro: NumVecesDem
    try:
        # asigna tipo de dato.
        aux_valor = int(float(DF['NumVecesDem']))
        DF['NumVecesDem'] = aux_valor
        # corrobora valor del valor asignado
        if aux_valor < 0:
            raise ValueError("Error: parámetro 'NumVecesDem' debe ser mayor o igual que cero !!")
    except ValueError as Verr:
        print(Verr)
        raise SystemExit  # request exit interpreter
    except:
        print("Error: Parámetro 'NumVecesDem' no definido en 'in_mcfpl_simulacion.csv'!!!")
        raise SystemExit  # request exit interpreter

    # Parámetro: NumVecesGen
    try:
        # asigna tipo de dato.
        aux_valor = int(float(DF['NumVecesGen']))
        DF['NumVecesGen'] = aux_valor
        # corrobora valor del valor asignado
        if aux_valor < 0:
            raise ValueError("Error: parámetro 'NumVecesGen' debe ser mayor o igual que cero !!")
    except ValueError as Verr:
        print(Verr)
        raise SystemExit  # request exit interpreter
    except:
        print("Error: Parámetro 'NumVecesGen' no definido en 'in_mcfpl_simulacion.csv'!!!")
        raise SystemExit  # request exit interpreter

    # Parámetro: PerdCoseno
    try:
        # asigna tipo de dato.
        aux_valor = bool(float(DF['PerdCoseno']))
        DF['PerdCoseno'] = aux_valor
        # corrobora valor del valor asignado
        if aux_valor < 0:
            raise ValueError("Error: parámetro 'PerdCoseno' debe ser mayor o igual que cero !!")
    except ValueError as Verr:
        print(Verr)
        raise SystemExit  # request exit interpreter
    except:
        print("Error: Parámetro 'PerdCoseno' no definido en 'in_mcfpl_simulacion.csv'!!!")
        raise SystemExit  # request exit interpreter

    # # Parámetro: SimTyp
    # try:
    #     if Diccionario['SimTyp'] != 'GenDet' and Diccionario['SimTyp'] != 'GenAleat':
    #         # verifica valor del parámetro 'SimTyp'
    #         raise ValueError("Error: parámetro 'SimTyp' debe ser 'GenDet' o 'GenAleat' !!")
    # except ValueError as Verr:
    #     print(Verr)
    #     return False
    # except:
    #     # imprime error si no se puede acceder al parámetro 'SimTyp' porque no existe.
    #     print("Parámetro 'SimTyp' no definido correctamente en 'in_mcfpl_simulacion.csv'!!!")
    #     return False

    return DF.to_dict(orient='records')[0]
