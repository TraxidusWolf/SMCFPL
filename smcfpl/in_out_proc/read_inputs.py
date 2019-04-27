"""
    Nombre de hojas (entradas necesarias). Todas en archivo de entrada.
"""
import logging
from datetime import timedelta as dt__timedelta
from os import sep as os__sep
from xlrd import open_workbook as xlrd__open_workbook
from pandas import read_excel as pd__read_excel, to_datetime as pd__to_datetime
from multiprocessing import cpu_count as mu__cpu_count, Pool as mu__Pool
from numpy import bool_ as np__bool_
import smcfpl.aux_funcs as aux_smcfpl
from smcfpl.smcfpl_exceptions import *


aux_smcfpl.print_full_df()
logging.basicConfig(level=logging.DEBUG,
                    format="[%(levelname)s][%(asctime)s][%(filename)s:%(funcName)s] - %(message)s")
logger = logging.getLogger()


def read_sheets_to_dataframes(ruta, NombreLibro, NumParallelCPU):
    """
        Lee las hojas necesarias para completar la información requerida para los archivos de entrada.
        Realiza una verificación de las hojas que son como mínimo requeridas, éstas en el diccionario 'HojasNecesarias'
    """
    logger.debug("! entrando en función: 'read_sheets_to_dataframes' ...")
    HojasNecesarias = {
        # Nombre de hoja necesaria: (varable1, varable2, ...). Notar que da lo mismo el orden, pero no deben existir otras columnas entremedio o error.
        'in_smcfpl_tecbarras'    : ( 'BarNom'        , 'Vnom')        ,
        'in_smcfpl_teclineas'    : ( 'LinNom'        , 'BarraA'       , 'BarraB'       , 'Parallel'    , 'Largo_km'    , 'TipoNom'       , 'Pmax_AB_MW'   , 'Pmax_BA_MW')  ,
        'in_smcfpl_tectrafos2w'  : ( 'Trafo2wNom'    , 'BarraA_HV'    , 'BarraB_LV'    , 'Parallel'    , 'TipoNom'     , 'Pmax_AB_MW'  , 'Pmax_BA_MW')   ,
        'in_smcfpl_tectrafos3w'  : ( 'Trafo3wNom'    , 'BarraA_HV'    , 'BarraB_MV'    , 'BarraC_LV'   , 'TipoNom'     , 'Pmax_inA_MW'   , 'Pmax_outA_MW' , 'Pmax_inB_MW'  , 'Pmax_outB_MW'  , 'Pmax_inC_MW' , 'Pmax_outC_MW')  ,
        'in_smcfpl_tipolineas'   : ( 'TipoNom'       , 'r_ohm_per_km' , 'x_ohm_per_km' , 'c_nf_per_km' , 'max_i_ka')   ,
        'in_smcfpl_tipotrafos2w' : ( 'TipoNom'       , 'vn_hv_kv'     , 'vn_lv_kv'     , 'sn_kva'      , 'pfe_kw'      , 'i0_percent'    , 'vsc_percent'  , 'vscr_percent' , 'shift_degree') ,
        'in_smcfpl_tipotrafos3w' : ( 'TipoNom'       , 'vn_hv_kv'     , 'vn_mv_kv'     , 'vn_lv_kv'    , 'sn_hv_kva'   , 'sn_mv_kva'     , 'sn_lv_kva'    , 'vn_hv_kv'     , 'vn_mv_kv'      , 'vn_lv_kv'    , 'vsc_hv_percent' , 'vsc_mv_percent' , 'vsc_lv_percent' , 'vscr_hv_percent' , 'vscr_mv_percent' , 'vscr_lv_percent' , 'pfe_kw' , 'i0_percent' , 'shift_mv_degree' , 'shift_lv_degree') ,
        'in_smcfpl_tecgen'       : ( 'GenNom'        , 'PmaxMW'       , 'PminMW'       , 'NomBarConn'    , 'GenTec'       , 'CVar'       , 'EsSlack')      ,
        'in_smcfpl_teccargas'    : ( 'LoadNom'       , 'NomBarConn'   , 'LoadTyp'      , 'DemNom_MW')       ,
        'in_smcfpl_proydem'      : ( 'Fecha'     , 'TasaCliLib'   , 'TasaCliReg') ,
        'in_scmfpl_histdemsist'  : ( 'Fecha'         , 'real'         , 'programado')  ,
        'in_smcfpl_mantbarras'   : ( 'BarNom'        , 'FechaIni'     , 'FechaFin')    ,
        'in_smcfpl_mantgen'      : ( 'GenNom'        , 'FechaIni'     , 'FechaFin'     , 'PmaxMW'      , 'PminMW'      , 'CVar'          , 'NomBarConn'   , 'Operativa'    , 'EsSlack')   ,
        'in_smcfpl_manttx'       : ( 'ElmTxNom', 'TipoElmn', 'FechaIni', 'FechaFin', 'Parallel', 'Largo_km', 'Pmax_AB_MW', 'Pmax_BA_MW', 'Operativa', 'BarraA', 'BarraB', 'BarraA_HV', 'BarraB_MV', 'BarraC_LV', 'BarraB_LV', 'Pmax_inA_MW', 'Pmax_outA_MW', 'Pmax_inB_MW', 'Pmax_outB_MW', 'Pmax_inC_MW', 'Pmax_outC_MW', 'TipoNom'),
        'in_smcfpl_mantcargas'   : ( 'LoadNom'       , 'FechaIni'     , 'FechaFin'     , 'DemNom_MW'   , 'NomBarConn'  ,  'LoadTyp'      , 'Operativa')   ,
        'in_smcfpl_histsolar'    : ( 'Fecha'         , 'EgenMWh')     ,
        'in_smcfpl_histeolicas'  : ( 'Fecha'         , 'EgenMWhZ1'    , 'EgenMWhZ2'    , 'EgenMWhZ3'   , 'EgenMWhZ4')  ,
        'in_smcfpl_tsfproy'      : ( 'Fecha'         , 'Carbón'       , 'Gas-Diésel'   , 'Otras'       , 'Solar'       , 'Embalse'       , 'Pasada'       , 'Serie'        , 'EólicaZ1'      , 'EólicaZ2'    , 'EólicaZ3'       , 'EólicaZ4')      ,
        'in_smcfpl_histhid'      : ( 'Año'           , 'abril'        , 'mayo'         , 'junio'       , 'julio'       , 'agosto'        , 'septiembre'   , 'octubre'      , 'noviembre'     , 'diciembre'   , 'enero'          , 'febrero'        , 'marzo'          , 'TOTAL')          ,
        'in_smcfpl_ParamHidEmb'  : ( ('Humeda', 'CVmin') , ('Media', 'CVmin') , ('Humeda', 'CVmax') , ('Media', 'CVmax') , ('Humeda', 'CotaMax') , ('Seca', 'b') , ('Seca', 'CotaMin') , ('Media', 'CotaMax') , ('Seca', 'CVmin') , ('Seca', 'CotaMax') , ('Media', 'b') , ('Humeda', 'b') , ('Humeda', 'CotaMin') , ('Media', 'CotaMin') , ('Seca', 'CVmax') ) ,
        'in_smcfpl_seriesconf'   : ( 'NombreEmbalse'   , 'CenNom'       , 'FuncCosto')
    }
    # Ruta completa (relativa) donde se encuentra el archivo
    RutaCompleta = ruta + os__sep + NombreLibro
    # Abre el libro
    logger.info("Reading Input file: {}...".format(RutaCompleta))
    Libro = xlrd__open_workbook(RutaCompleta)
    # verifica si falta alguna hoja en el libro excel
    HojasFaltantes = []
    for hoja in HojasNecesarias.keys():
        if hoja not in Libro.sheet_names():
            HojasFaltantes.append(hoja)
    if HojasFaltantes:  # de faltar alguna se muestra logging de error
        msg = "Sheets missing within input Spreadsheet: " + ", ".join(HojasFaltantes)
        logger.error(msg)
        raise ValueError(msg)

    # Crea diccionario con dataframe de todos los datos de entrada
    DFs_entrada = {}

    """Lee solo las hojas que son necesarias, al igual que las columnas que lo son"""
    if not NumParallelCPU:   # Lee archivos en Serie
        for Hoja, variables in HojasNecesarias.items():
            if Hoja == 'in_smcfpl_ParamHidEmb':
                # caso particular ya que debe ser un multindex de columnas variables (en función del nombre de embalses). Notar que queda transpuesto
                DFs_entrada['df_' + Hoja] = Lee_Hoja_planilla(RutaCompleta, Hoja, True, *variables)
            else:
                DFs_entrada['df_' + Hoja] = Lee_Hoja_planilla(RutaCompleta, Hoja, False, *variables)
    else:  # Lee archivos en Paralelo
        # Parámetros de paralelismo
        if isinstance(NumParallelCPU, int):
            Ncpu = NumParallelCPU
        elif NumParallelCPU == 'Max':
            Ncpu = mu__cpu_count()
        logger.info("Reading input sheets in parallel. Using maximun {} simultaneous processes.".format(Ncpu))
        Pool = mu__Pool(Ncpu)
        Results = []
        # Por cada hoja rellena el Pool
        for Hoja, variables in HojasNecesarias.items():
            if Hoja == 'in_smcfpl_ParamHidEmb':
                # caso particular ya que debe ser un multindex de columnas variables (en función del nombre de embalses). Notar que queda transpuesto
                Results.append( [Pool.apply_async(Lee_Hoja_planilla, (RutaCompleta, Hoja, True, *variables)), Hoja] )
            else:
                Results.append( [Pool.apply_async(Lee_Hoja_planilla, (RutaCompleta, Hoja, False, *variables)), Hoja] )
        # Obtiene los resultados del paralelismo
        for result, Hoja in Results:
            if Hoja == 'in_smcfpl_ParamHidEmb':
                DFs_entrada['df_' + Hoja] = result.get()
            else:
                DFs_entrada['df_' + Hoja] = result.get()

    logger.debug("! saliendo de función: 'read_sheets_to_dataframes' ...")
    return DFs_entrada


def Lee_Hoja_planilla(RutaCompleta, NombreHoja, EsParamHid, *args):
    """
        Notar que la primera fila de los datos en las hojas de la planilla de datos de entrada corresponden a los encabezados,
        y la tabla comenzar desde la celda A1 (primer encabezado) consecutivamente hacia la derecha.

        TODO: Desde los args, buscar el indice de columna de cada uno de ellos y leer dichas columnas con pandas.read_excel().
    """
    logger.debug("! entrando en función: 'Lee_Hoja_planilla' (read_inputs.py) - '{}'...".format(NombreHoja))
    logger.info("Extracting data from sheet: {} ...".format(NombreHoja))

    if not len(args):
        msg = "No variable names for sheet '{}' were declared.".format(NombreHoja)
        logger.warn(msg)
        raise ValueError(msg)

    if EsParamHid:
        # notar que header = [...] retorna un multicolumns dataframe que tiene varios títulos (pudiendo repetirse el nombre). Set de columnas son set de tuplas.
        df = pd__read_excel( RutaCompleta, sheet_name=NombreHoja, header=[0, 1], usecols=range(len(args) + 1) )  # uno más para leer columns 'Hidrologías' (columns.names)
    else:
        df = pd__read_excel( RutaCompleta, sheet_name=NombreHoja, header=0, usecols=range(len(args)) )

    if df.empty:
        # verifica si hoja está vacía
        msg = "Sheet: '{}' has no values.".format(NombreHoja)
        logger.warn(msg)
        # raise ValueError(msg)  # no levantar error, ya que pueden no existir determinados elementos de las planillas.

    columnas_df = set(df.columns)
    # encuentra los elementos del primer set que no están en el segundo
    VariablesFaltantes = set(args) - columnas_df
    if VariablesFaltantes:
        msg = "Dentro de hoja '{}', No se encontraron las variables requeridas: ".format(NombreHoja) + str(VariablesFaltantes)
        logger.error(msg)
        raise InsuficientInputData("Insufficient input variables.")

    # Warning en caso de ingresarse un mismo elemento (barra, linea, trafo, gen, o carga) en el mismo archivo técnico
    EntradasTecnicas = ['in_smcfpl_tecbarras', 'in_smcfpl_teclineas', 'in_smcfpl_teclineas', 'in_smcfpl_tectrafos2w', 'in_smcfpl_tectrafos3w',
                        'in_smcfpl_tecgen', 'in_smcfpl_teccargas',
                        'in_smcfpl_tipolineas', 'in_smcfpl_tipotrafos2w', 'in_smcfpl_tipotrafos3w']
    if NombreHoja in EntradasTecnicas:
        # Fija primera fila como índice del pandas DataFrame
        df.set_index(df.columns[0], inplace=True)
        Nrow, Ncol = df.shape
        dup = df.index.duplicated(keep='first')
        if dup.any():
            # De existir algún nombre de elemento duplicado dentro de estas entradas, se alerta al usuario, dejando válida la primera coincidencia
            dup = df[dup].index.tolist()
            msg = "El archivo de entrada {} presenta las siguientes filas duplicadas: '{}'.".format(NombreHoja, ', '.join(dup))
            logger.warning(msg)

    #
    # Formato de Celdas (FECHAS Y BOOLEANS)
    #
    # Da formato a las columnas con fechas (Solo en caso que posean columnas mostradas)
    PoseeFechaIni = 'FechaIni' in columnas_df
    PoseeFechaFin = 'FechaFin' in columnas_df
    PoseeColFecha = 'Fecha' in columnas_df
    PoseeOperativa = 'Operativa' in columnas_df
    PoseeEsSllack = 'EsSlack' in columnas_df
    if PoseeFechaIni:
        df['FechaIni'] = pd__to_datetime(df['FechaIni'], format="%Y-%m-%d %H:%M")
    if PoseeFechaFin:
        df['FechaFin'] = pd__to_datetime(df['FechaFin'], format="%Y-%m-%d %H:%M")
    if PoseeColFecha:
        df['Fecha'] = pd__to_datetime(df['Fecha'], format="%Y-%m-%d %H:%M")
    # Da formato a los archivos de mantenimiento que posean columna 'Operativa' (A veces no realiza buen casting)
    if PoseeOperativa:
        df['Operativa'] = df['Operativa'].astype(np__bool_)
    # Da formato a los archivos de mantenimiento que posean columna 'EsSlack' (A veces no realiza buen casting)
    if PoseeEsSllack:
        df['EsSlack'] = df['EsSlack'].astype(np__bool_)

    # Cambia los NaN por 0
    df = df.fillna(0)
    # Asigna nombre de la hoja como nombre del DataFrame
    df.name = 'df_' + NombreHoja

    # revisa que para las entradas que posean columnas 'FechaIni' y 'FechaFin' posean diferencia de al menos un día y estén en orden
    if PoseeFechaIni & PoseeFechaFin:
        NoCumplen = df['FechaIni'] + dt__timedelta(days=1) <= df['FechaFin']  # aquellos que no cumplen requisitos mencionados
        NoCumplen = NoCumplen[ ~ NoCumplen ]    # filtra dejando aquellos que son falsos
        NoCumplen.index += 2    # incrementa los indices en 1 para similitud con planilla excel (filas secuenciales)
        # En caso de no cumplir requisitos arroja error.
        if not NoCumplen.empty:
            logger.error("'FechaFin' en hoja '{}' debe(n) ser al menos un día MAYOR que 'FechaIni' para las fila(s): {} ".format(NombreHoja,
                                                                                                                                 NoCumplen.index.tolist()
                                                                                                                                 ))
            raise ValueError("'FechaFin' y 'FechaIni' no cumplen requisitos en hoja '{}'.".format(NombreHoja))

    if (NombreHoja == 'in_smcfpl_histsolar') | (NombreHoja == 'in_smcfpl_histeolicas') | (NombreHoja == 'in_scmfpl_histdemsist'):
        # Agrupa los dataframe de generación histórica solar, eólicas, y demanda sistémica en un objeto groupby con Mes/Dia/Hora. Genera error en caso de ser menor
        df = Agrupa_data_ERNC_Y_HistDem(df)

    if NombreHoja == 'in_smcfpl_ParamHidEmb':
        # transpone el dataframe para que sea multindex, con columnas de embalses (más fácil de utilizar)
        df = df.T
    logger.debug("! saliendo de función: 'Lee_Hoja_planilla' (read_inputs.py) - '{}'...".format(NombreHoja))

    return df


def Agrupa_data_ERNC_Y_HistDem(DF):
    """ Toma el promedio de los meses a lo largo de los años ingresados.
        Arroja error en caso de poseer data menor a un año.
    """
    logger.debug("! entrando en función: 'Agrupa_data_ERNC_Y_HistDem' (read_inputs.py) ...")
    # verifica la duración máxima de los datos del DataFrame
    RTiempoData = DF['Fecha'].iloc[-1] - DF['Fecha'].iloc[0]

    # Manejo de errores en datos mínimos
    if RTiempoData < dt__timedelta(days=364, hours=23):
        msg = "Hoja '{}' debe poseer como mínimo un rango de tiempo total de 364 días y 23 horas.".format(DF.name[3:])
        logger.error(msg)
        raise ValueError(msg)

    # Reduce el DataFrame al promedio de años (detalle horario)
    # groupby
    DF = DF.groupby(by=[DF['Fecha'].dt.month, DF['Fecha'].dt.day, DF['Fecha'].dt.hour]).mean()  # .reset_index()
    DF.index.names = ['Mes', 'Dia', 'Hora']  # elimina la componente año
    # Usec = timeit.timeit("DF[ DF.index.get_level_values('Mes') == 1 ]", globals=locals(), number=NumeroVeces)

    # ¿Cómo solicitar datos (filtrado)?
    # MesPedido = 1
    # DiaPedido = 1
    # HoraPedido = 0
    # Cond1 = DF.index.get_level_values('Mes') == MesPedido
    # Cond2 = DF.index.get_level_values('Dia') == DiaPedido
    # Cond3 = DF.index.get_level_values('Hora') == HoraPedido
    # print( DF[ Cond1 & Cond2 & Cond3 ] )
    logger.debug("! saliendo en función: 'Agrupa_data_ERNC_Y_HistDem' (read_inputs.py) ...")
    return DF
