"""
    Nombre de hojas (entradas necesarias). Todas en archivo de entrada.
"""
import logging
from datetime import timedelta as dt__timedelta
from os import sep as os__sep
from xlrd import open_workbook as xlrd__open_workbook
from pandas import read_excel as pd__read_excel, to_datetime as pd__to_datetime
from datetime import timedelta as dt__timedelta
from smcfpl.aux_funcs import print_full_df


print_full_df()
logging.basicConfig(level=logging.DEBUG,
                    format="[%(levelname)s][%(asctime)s] - %(message)s")
logger = logging.getLogger()


def read_sheets_to_dataframes(ruta, NombreLibro):
    """
        Lee las hojas necesarias para completar la información requerida para los archivos de entrada.
    """
    logger.debug("! entrando en función: 'read_sheets_to_dataframes' ...")
    HojasNecesarias = {
        # Nombre de hoja necesaria: (varable1, varable2, ...)
        'in_smcfpl_tecbarras'    : ( 'BarNom'      , 'Vnom')   ,
        'in_smcfpl_teclineas'    : ( 'LinNom'      , 'BarraA'       , 'BarraB'       , 'Parallel'    , 'Largo_km'    , 'TipoNom'       , 'Pmax_AB_MW'   , 'Pmax_BA_MW')   ,
        'in_smcfpl_tectrafos2w'  : ( 'Trafo2wNom'  , 'BarraA_HV'    , 'BarraB_LV'    , 'TipoNom'     , 'Pmax_AB_MW'  , 'Pmax_BA_MW')    ,
        'in_smcfpl_tectrafos3w'  : ( 'Trafo3wNom'  , 'BarraA_HV'    , 'BarraB_MV'    , 'BarraC_LV'   , 'TipoNom'     , 'Pmax_inA_MW'   , 'Pmax_outA_MW' , 'Pmax_inB_MW'  , 'Pmax_outB_MW'  , 'Pmax_inC_MW' , 'Pmax_outC_MW')      ,
        'in_smcfpl_tipolineas'   : ( 'TipoNom'     , 'r_ohm_per_km' , 'x_ohm_per_km' , 'c_nf_per_km' , 'max_i_ka')   ,
        'in_smcfpl_tipotrafos2w' : ( 'TipoNom'     , 'vn_hv_kv'     , 'vn_lv_kv'     , 'sn_kva'      , 'pfe_kw'      , 'i0_percent'    , 'vsc_percent'  , 'vscr_percent' , 'shift_degree') ,
        'in_smcfpl_tipotrafos3w' : ( 'TipoNom'     , 'vn_hv_kv'     , 'vn_mv_kv'     , 'vn_lv_kv'    , 'sn_hv_kva'   , 'sn_mv_kva'     , 'sn_lv_kva'    , 'vn_hv_kv'     , 'vn_mv_kv'      , 'vn_lv_kv'    , 'vsc_hv_percent' , 'vsc_mv_percent' , 'vsc_lv_percent' , 'vscr_hv_percent' , 'vscr_mv_percent' , 'vscr_lv_percent' , 'pfe_kw' , 'i0_percent' , 'shift_mv_degree' , 'shift_lv_degree') ,
        'in_smcfpl_tecgen'       : ( 'GenNom'      , 'PmaxMW'       , 'PminMW'       , 'Sn_kva'      , 'Vm_pu_ideal' , 'NomBarConn'    , 'GenTec'       , 'CVar'         , 'EsSlack')      ,
        'in_smcfpl_teccargas'    : ( 'LoadNom'     , 'NomBarConn'   , 'LoadTyp'      , 'DemMax_MW'   , 'DemMax_MVAr' , 'DemIni')    ,
        'in_smcfpl_proydem'      : ( 'FechaIni'    , 'FechaFin'     , 'TasaCliLib'   , 'TasaCliReg') ,
        'in_scmfpl_histdemsist'  : ( 'fecha'       , 'hora'         , 'real'         , 'programado') ,
        'in_smcfpl_mantbarras'   : ( 'BarNom'      , 'FechaIni'     , 'FechaFin')    ,
        'in_smcfpl_mantgen'      : ( 'GenNom'      , 'FechaIni'     , 'FechaFin'     , 'PmaxMW'      , 'PminMW'      , 'CVar'          , 'NomBarConn'   , 'Operativa')   ,
        'in_smcfpl_manttx'       : ( 'LinNom'      , 'TipoElmn'     , 'FechaIni'     , 'FechaFin'     , 'Parallel'    , 'Largo_km'    , 'Pmax_AB_MW'    , 'Pmax_BA_MW'   , 'Operativa'    , 'BarraA'        , 'BarraB'      , 'r_ohm_per_km'   , 'x_ohm_per_km'   , 'c_nf_per_km'    , 'max_i_ka')       ,
        'in_smcfpl_mantcargas'   : ( 'LoadNom'     , 'FechaIni'     , 'FechaFin'     , 'DemMax_MW'   , 'DemMax_MVAr' , 'NomBarConn'    , 'Operativa')   ,
        'in_smcfpl_histsolar'    : ( 'fecha'       , 'EgenMWh')     ,
        'in_smcfpl_histeolicas'  : ( 'fecha'       , 'EgenMWhZ1'    , 'EgenMWhZ2'    , 'EgenMWhZ3'   , 'EgenMWhZ4')  ,
        'in_smcfpl_tsfproy'      : ( 'Fecha'       , 'Carbón'       , 'Gas-Diésel'   , 'Otros'       , 'Solar'       , 'Embalse'       , 'Pasada'       , 'Serie'        , 'EólicaZ1'      , 'EólicaZ2'    , 'EólicaZ3'       , 'EólicaZ4')      ,
        'in_smcfpl_histhid'      : ( 'Año'         , 'abril'        , 'mayo'         , 'junio'       , 'julio'       , 'agosto'        , 'septiembre'   , 'octubre'      , 'noviembre'     , 'diciembre'   , 'enero'          , 'febrero'        , 'marzo'          , 'TOTAL')            ,
        'in_smcfpl_relcvarcota'  : ( 'NombreSerie' , 'Mes'          , 'Día'          , 'Hora'        , 'Cota_msnm'   , 'Cvar_USD_Mwh') ,
        'in_smcfpl_seriesconf'   : ( 'NombreSerie' , 'CenNom'       , 'FuncCosto')
    }
    # Ruta completa (relativa) donde se encuentra el archivo
    RutaCompleta = ruta + os__sep + NombreLibro
    # Abre el libro
    logger.info("Leyendo archivo de entrada: {}...".format(RutaCompleta))
    Libro = xlrd__open_workbook(RutaCompleta)
    # verifica si falta alguna hoja en el libro excel
    HojasFaltantes = []
    for hoja in HojasNecesarias.keys():
        if hoja not in Libro.sheet_names():
            HojasFaltantes.append(hoja)
    if HojasFaltantes:  # de faltar alguna se muestra logging de error
        logger.error("Faltan las hojas: " + ", ".join(HojasFaltantes))
        raise ValueError("Faltan hojas necesarias en planilla de entrada.")

    # Crea diccionario con dataframe de todos los datos de entrada
    DFs_entrada = {}
    #
    # PARA PROPOSITO DE DEPURACION
    HojasNecesarias = {k: HojasNecesarias[k] for k in ['in_smcfpl_mantbarras',
                                                       'in_smcfpl_mantgen',
                                                       'in_smcfpl_manttx',
                                                       'in_smcfpl_mantcargas',
                                                       'in_smcfpl_histsolar',
                                                       'in_smcfpl_histeolicas']}
    #
    for Hoja, variables in HojasNecesarias.items():
        #
        # PARA PROPOSITO DE DEPURACION
        # Hoja = 'in_smcfpl_mantbarras'; variables = ( 'BarNom'      , 'FechaIni'     , 'FechaFin')
        # Nombre de cada DataFrame es 'df_' + NombreHoja
        #
        DFs_entrada['df_' + Hoja] = Lee_Hoja_planilla(RutaCompleta, Hoja, *variables)
        # print(DFs_entrada['df_' + Hoja])
        # break

    logger.debug("! saliendo de función: 'read_sheets_to_dataframes' ...")
    return DFs_entrada


def Lee_Hoja_planilla(RutaCompleta, NombreHoja, *args):
    """
        Notar que la primera fila de los datos en las hojas de la planilla de datos de entrada corresponden a los encabezados,
        y la tabla comenzar desde la celda A1 (primer encabezado) consecutivamente hacia la derecha.

        TODO: Desde los args, buscar el indice de columna de cada uno de ellos y leer dichas columnas con pandas.read_excel().
    """
    logger.debug("! entrando en función: 'Lee_Hoja_planilla' ...")
    logger.info("Extrayendo datos desde hoja: {} ...".format(NombreHoja))

    if not len(args):
        logger.warn("No se ingresaron nombre de variables a leer para hoja '{}'".format(NombreHoja))
        raise ValueError("Ninguna columna para leer ha sido ingresada.")

    df = pd__read_excel( RutaCompleta, sheet_name=NombreHoja, header=0, usecols=range(len(args)) )

    if df.empty:
        logger.warn("Hoja: '{}' No posee valores.".format(NombreHoja))
    columnas_df = set(df.columns)
    # encuentra los elementos del primer set que no están en el segundo
    VariablesFaltantes = set(args) - columnas_df
    if VariablesFaltantes:
        logger.error("Dentro de hoja '{}', No se encontraron las variables requeridas: ".format(NombreHoja) + ", ".join(VariablesFaltantes) )
        raise ValueError("Variables de entrada insuficientes.")

    # Da formato a las columnas con fechas (Solo en caso que posean columnas mostradas)
    PoseeFechaIni = 'FechaIni' in columnas_df
    PoseeFechaFin = 'FechaFin' in columnas_df
    if PoseeFechaIni:
        df['FechaIni'] = pd__to_datetime(df['FechaIni'], format="%Y-%m-%d %H:%M")
    if PoseeFechaFin:
        df['FechaFin'] = pd__to_datetime(df['FechaFin'], format="%Y-%m-%d %H:%M")
    if 'fecha' in columnas_df:
        df['fecha'] = pd__to_datetime(df['fecha'], format="%Y-%m-%d %H:%M")
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

    # En caso de ser las entradas de datos historicos solares o eólicos, agrupa la info por meses (promedio de los años). Error en caso de ser menor
    if (NombreHoja == 'in_smcfpl_histsolar') | (NombreHoja == 'in_smcfpl_histeolicas'):
        df = Agrupa_data_ERNC(df)

    logger.debug("! saliendo de función: 'Lee_Hoja_planilla' ...")
    return df


def Agrupa_data_ERNC(DF):
    """ Toma el promedio de los meses a lo largo de los años ingresados.
        Arroja error en caso de poseer data menor a un año.
    """
    print(DF.name)
    # verifica la duración máxima de los datos del DataFrame
    RTiempoData = DF['fecha'].iloc[-1] - DF['fecha'].iloc[0]
    print( RTiempoData )

    # Manejo de errores en datos mínimos
    if RTiempoData < dt__timedelta(days=364, hours=23):
        msg = "Hoja '{}' debe tener poseer como mínimo un rango de tiempo total de 364 días y 23 horas.".format(DF.name[3:])
        logger.error(msg)
        raise ValueError(msg)

    # Reduce el DataFrame al promedio de años (detalle horario)
    # groupby
    DF = DF.groupby(by=[DF['fecha'].dt.month, DF['fecha'].dt.day, DF['fecha'].dt.hour]).mean()  # .reset_index()
    DF.index.names = ['Mes', 'Dia', 'Hora']  # elimina la componente año
    # print(DF.index)
    # Usec = timeit.timeit("DF[ DF.index.get_level_values('Mes') == 1 ]", globals=locals(), number=NumeroVeces)

    # ¿Cómo solicitar datos (filtrado)?
    # MesPedido = 1
    # DiaPedido = 1
    # HoraPedido = 0
    # Cond1 = DF.index.get_level_values('Mes') == MesPedido
    # Cond2 = DF.index.get_level_values('Dia') == DiaPedido
    # Cond3 = DF.index.get_level_values('Hora') == HoraPedido
    # print( DF[ Cond1 & Cond2 & Cond3 ] )
    return DF
