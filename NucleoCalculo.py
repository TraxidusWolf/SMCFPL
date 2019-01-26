"""
Este script esta diseñado para que pueda ser llamado como función 'Calcular' a partir del nombre ,i.e.,
    import smcfpl.NucleoCalculo as NucleoCalculo  # En módulo
    from NucleoCalculo import Calcular  # Directamente por interprete por ejemplo
    NucleoCalculo.Calcular()
"""
from pandapower import rundcpp as pp__rundcpp
import time
from os import sep as os__sep
from pandapower import from_pickle as pp__from_pickle
from pandapower import select_subnet as pp__select_subnet
from pandapower.topology import connected_components as pp__topology__connected_components
from pandas import DataFrame as pd__DataFrame
from pandas import concat as pd__concat
# from smcfpl.aux_funcs import overloaded_trafo2w as aux_smcfpl__overloaded_trafo2w
# from smcfpl.aux_funcs import overloaded_trafo3w as aux_smcfpl__overloaded_trafo3w
from smcfpl.aux_funcs import TipoCong as aux_funcs__TipoCong


import logging
logging.basicConfig(level=logging.DEBUG,
                    format="[%(levelname)s][%(asctime)s][%(filename)s:%(funcName)s] - %(message)s")
logger = logging.getLogger()


def Calcular(CasoNum, Hidrology, Grillas, StageIndexesList, DF_ParamHidEmb_hid,
             DF_seriesconf, MaxItCongInter, MaxItCongIntra, abs_OutFilePath='',
             File_Caso='.', in_node=False, DemGenerator_Dict=dict(),
             DispatchGenerator_Dict=dict() ):
    """
        Función base de cálculo para la resolución del modelo SMCFPL. Por cada etapa obtiene los valores
        de los generadores de demanda y generación para los elementos de la Grilla de la etapa ocurrente.

        :param CasoNum: Numero del caso corriendo.
        :type CasoNum: int

        :param Hidrology: Nombre de la hidrología actual
        :type Hidrology: string

        :param Grillas: Dict of pandaNetworks and extradata if 'in_node' == False. Dict of pandaNetworks if 'in_node' == True.
                        Indexed by Stage numbers.
        :type Grillas: Diccionario

        :param StageIndexesList: Indices de las etapas
        :type StageIndexesList: list

        :param DF_ParamHidEmb_hid: Los indices son: 'Hidrología', ya sea 'Humeda', 'Media', o 'Seca'.
                                   'Parámetros', definen la función de costo ('b','CVmin','CVmax','CotaMin','CotaMax').
                                   Las columnas son el correspondiente nombre de los embalses.

        :type DF_ParamHidEmb_hid: nombre del pandas Multindex DataFrame

        :param DF_seriesconf: Relación entre nombres de series hidráulicas y centrales.
                              Contiene indices del nombre ('NombreSeries'), Nombre de
                              la central de hidraúlica ('CenNom'); y nombre de la función
                              de costo predefinida para relacionar cota - cvar ('FuncCosto').
        :type DF_seriesconf: nombre del pandas DataFrame

        :param in_node: Verdadero indica que es ejecutado en nodo. Valores son retornados en archivo.
        :type in_node: bool

        :param DemGenerator_Dict: Diccionario constituido por { EtaNum, pd.DataFrame.loc[EtaNum, 'PDem_pu'] }
        :type DemGenerator_Dict: Diccionario

        :param DispatchGenerator_Dict: Diccionario constituido por { EtaNum, pd.DataFrame.loc[EtaNum, ['type', PGen_pu']] }
        :type DispatchGenerator_Dict: Diccionario

    """
    print("Hidrology:", Hidrology)
    # for each stage in the case
    for StageNum in StageIndexesList:
        # Load Data from every stage when 'in_node' is True
        print("StageNum:", StageNum, "CasoNum:", CasoNum)
        if in_node:
            """Carga la grilla con los valores actualizados"""
            Grid = Grillas[StageNum]
            # # load Grid writen for the Power System
            # FileName = "Grid_Eta{}.p".format(StageNum)
            # # Grid = pp__from_pickle(FileName)
            # print("FileName:", FileName)
            # # lee grilla correspondiente
            # Grid = pp__from_pickle(File_Caso + os__sep + FileName)
            # print("Grid['load']['name']:", Grid['load']['name'])

            # # load ExtraData writen for the Power System
            # FileName = "{}.json".format(StageNum)
            # # DictExtraData = json__load(FileName)
            # print("FileName:", FileName)
            # # lee archivo correspondiente
            # # with open(+FileName, 'r') as f:

        else:
            """Carga la grilla para actualizar los valores"""
            Grid = Grillas[StageNum]['PandaPowerNet']
            Dict_ExtraData = Grillas[StageNum]['ExtraData']
            D = DemGenerator_Dict[StageNum]  # pandas DataFrame
            print("D:", D)
            G = DispatchGenerator_Dict[StageNum]  # pandas DataFrame
            print("G:", G)
            # D['PDem_pu'].values: puede ser negativo, positivo o cero, pero siempre en [p.u.]
            Grid['load']['p_kw'] = Grid['load']['p_kw'] * D['PDem_pu'].values
            # G['PGen_pu'].values: siempre va a estar entre 0 y 1 inclusive
            Grid['gen']['p_kw'] = Grid['gen']['max_p_kw'] * G['PGen_pu'].values

        # print("Grid:\n", Grid)
        # print("Grid['gen']:\n", Grid['gen'])
        # print("Grid['load']:\n", Grid['load'])
        # print("Grid['trafo']:\n", Grid['trafo'])

        #
        # Revisa las grillas en cada etapa para verificar que el balance de potencia es posible (Gen-Dem)
        PGenMaxSist = Grid['gen']['max_p_kw'].sum()
        PGenSlackMax = Grid['ext_grid']['max_p_kw'].sum()
        PDemSist = Grid['load']['p_kw'].sum()
        # Cuantifica diferencia entre generación y demanda máxima. Negativo implica sobra potencia
        DeltaP_Uninodal = PGenMaxSist + PGenSlackMax + PDemSist  # De ser positivo indica PNS!
        # Corrobora factibilidad del despacho uninodal
        if DeltaP_Uninodal > 0:
            msg = "No es posible abastecer la demanda en etapa {}/{} del caso {}!.".format(
                StageNum, len(StageIndexesList), CasoNum)
            logger.warn(msg)
            # ¿Eliminar etapa?¿?
            # DF_Etapas.drop(labels=[StageNum], axis='index', inplace=True)
            # ¿Escribir PNS?
            #
            continue  # Continua con siguiente etapa

        #
        # Calcula el Flujo de Potencia Linealizado
        pp__rundcpp(Grid, trafo_model='pi', trafo_loading='power',
                    check_connectivity=True, r_switch=0.0,
                    trafo3w_losses='hv')

        #
        # Verifica Potencia para el GenSlack sea menor que su máximo (Negativo generación)
        PotSobrante = Grid['ext_grid'].loc[0, 'max_p_kw'] - Grid['res_ext_grid'].loc[0, 'p_kw']
        # Verifica Potencia para el GenSlack sea mayor que su mínimo (Negativo generación)
        PotFaltante = Grid['res_ext_grid'].loc[0, 'p_kw'] - Grid['ext_grid'].loc[0, 'min_p_kw']
        # Acota rangos factibles de potencia de generación de GenSlack
        if PotSobrante > 0:  # PGenSlack es más Negativo que Pmax
            msg = "El generador de referencia está sobrecargado en etapa {}/{} del caso {}!.".format(
                StageNum, len(StageIndexesList), CasoNum)
            logger.info(msg)
            continue
        if PotFaltante > 0:  # PGenSlack es más Positivo que Pmin
            msg = "El generador de referencia está absorbiendo potencia en etapa {}/{} del caso {}!.".format(
                StageNum, len(StageIndexesList), CasoNum)
            logger.info(msg)
            continue

        #
        # Calcula CMg previo congestión [$US/MWh]  # CVar en [$US/MWh]
        # Descarga las unidades que posean generación nula

        # Identifica el índice del generador con mayor costo variable
        IndMarginGen = Dict_ExtraData['CVarGenNoRef'].idxmax(axis='index')[0]  # primer GenNoRef
        CVarMarginGen = Dict_ExtraData['CVarGenNoRef']['CVar'][IndMarginGen]
        # Compara CVar de IndMarginGen con el de GenSlack
        # MarginUnit: (tipo de elemento, indice en grilla, CVar)
        if Dict_ExtraData['CVarGenRef'] > CVarMarginGen:
            MarginUnit = ('ext_grid', Grid['ext_grid'].index[0], Dict_ExtraData['CVarGenRef'])
            # Notar que debido a que se asume unico GenSlack, índice es siempre '0'.
        elif Dict_ExtraData['CVarGenRef'] < CVarMarginGen:
            MarginUnit = ('gen', IndMarginGen, CVarMarginGen)
        else:  # cuando son iguales, se escoge una unidad (No Slack)
            MarginUnit = ('gen', IndMarginGen, CVarMarginGen)

        # Modifica loading_percent para identificar congestiones
        """ Debido a que el loading_percent de los tranfos se calcula c/r a 'sn_kva' y,
        el fpl es calculado con loading_percent='power', se identifica la
        dirección de los flujos de los tranfos para multiplicar por
        'sn_kva' (del tipo) / 'Pmax_AB_MW' o respectivo.

        Este proceso se realiza para evitar modificar la potencia de los
        tranfos por tipo, ya que éstos pueden pertenecer a más de uno.

        Todo, en caso de existir transfos.
        """
        if not Grid['trafo'].empty:
            # Para Trafo (P positivo hacia adentro)
            #
            # Identifica si el flujo de P va de B a A
            pdSeries = Grid.res_trafo['p_hv_kw'] < Grid.res_trafo['p_lv_kw']
            # Asume BarraA como HV y BarraB como LV
            pdSeries_MaxP = Dict_ExtraData['PmaxMW_trafo2w']['Pmax_AB_MW'].values * pdSeries
            pdSeries_MaxP += Dict_ExtraData['PmaxMW_trafo2w']['Pmax_BA_MW'].values * ~pdSeries
            # Convierte la base de potencia del elemento a lo nueva límite
            Grid.res_trafo['loading_percent'] *= Grid.trafo['sn_kva'] / (pdSeries_MaxP * 1e3)
        if not Grid['trafo3w'].empty:
            # Para Trafo3w (P positivo hacia adentro)
            #
            # Identifica flujo entrante al terminal (Asume A -> HV)
            pdSeriesA = Grid['res_trafo3w']['p_hv_kw'].abs()
            # Identifica flujo entrante al terminal (Asume B -> MV)
            pdSeriesB = Grid['res_trafo3w']['p_mv_kw'].abs()
            # Identifica flujo entrante al terminal (Asume C -> LV)
            pdSeriesC = Grid['res_trafo3w']['p_lv_kw'].abs()
            # Obtiene el nombre del lado del trafo3w que posee mayor potencia: 'p_hv_kw', 'p_mv_kw', 'p_lv_kw'
            pdSeries_SideMaxP = pd__concat([pdSeriesA, pdSeriesB, pdSeriesC], axis='columns').idxmax(axis='columns')
            pdSeries_SideMaxP = 'sn_' + pdSeries_SideMaxP.str[2:4] + '_kva'  # convierte a nombre requerido por potencia
            # Obtiene el valor de potencia del lado correspondiente}
            List_ValMaxP = [Grid.trafo3w.loc[i, SideNom] for i, SideNom in zip(range(Grid.trafo3w.shape[0]), pdSeries_SideMaxP)]
            #
            # Obtiene la potencia máxima entrante/saliente del nodo con mayor flujo
            Dict_ExtraData['PmaxMW_trafo3w']['Pmax_AB_MW'].values
            List_NewNomValP = []
            for i, V in zip(range(Grid.trafo3w.shape[0]), pdSeries_SideMaxP.str[2:4]):
                if V == 'hv':  # potencia máxima es del nodo A/HV
                    if pdSeriesA[i] > 0:  # Potencia entra al transfo
                        List_NewNomValP.append( Dict_ExtraData['PmaxMW_trafo3w'].loc[i, 'Pmax_inA_MW'] * 1e3 )
                    else:  # cero no es posible que sea
                        List_NewNomValP.append( Dict_ExtraData['PmaxMW_trafo3w'].loc[i, 'Pmax_outA_MW'] * 1e3 )
                elif V == 'mv':  # potencia máxima es del nodo B/MV
                    if pdSeriesA[i] > 0:  # Potencia entra al transfo
                        List_NewNomValP.append( Dict_ExtraData['PmaxMW_trafo3w'].loc[i, 'Pmax_inB_MW'] * 1e3 )
                    else:  # cero no es posible que sea
                        List_NewNomValP.append( Dict_ExtraData['PmaxMW_trafo3w'].loc[i, 'Pmax_outB_MW'] * 1e3 )
                elif V == 'lv':  # potencia máxima es del nodo C/LV
                    if pdSeriesA[i] > 0:  # Potencia entra al transfo
                        List_NewNomValP.append( Dict_ExtraData['PmaxMW_trafo3w'].loc[i, 'Pmax_inC_MW'] * 1e3 )
                    else:  # cero no es posible que sea
                        List_NewNomValP.append( Dict_ExtraData['PmaxMW_trafo3w'].loc[i, 'Pmax_outC_MW'] * 1e3 )
            # Cambia de potencia base
            Grid['res_trafo3w']['loading_percent'] *= List_ValMaxP / List_NewNomValP

        #
        # Obtiene lista del tipo de congestiones (TypeElmnt, IndGrilla)
        ListaCongInter, ListaCongIntra = aux_funcs__TipoCong(Grid, max_load=100)

        import pdb; pdb.set_trace()  # breakpoint 49a4976c //
        #
        # Inicializa contadores de Congestiones (int)
        ContadorInter = ContadorIntra = 0

        #
        while ContadorIntra <= MaxItCongIntra:
            ContadorIntra += 1
        while ContadorInter <= MaxItCongInter:
            ContadorInter += 1

        # (Multiplica dos pandas Series) Indices
        # son creados secuencialmente, por lo que no necesita ser DataFrame
        pdSeries_CostDispatch = Dict_ExtraData['CVarGenNoRef'].squeeze() * -Grid['gen']['p_kw']  # en [$US]

        #
        # Calcula costo País
        TotalPowerCost = pdSeries_CostDispatch.sum()

        #
        # Calcula pérdidas por flujo de líneas

    print("----------------------------")
    print("Este fue CasoNum:", CasoNum)
    print("----------------------------")

    if in_node:
        pass
        # print("------------------\n Escribiendo archivo ficticio \n------------------")
        # with open(File_Caso + os__sep + '..' + os__sep + str(CasoNum), 'w') as f:
        #     f.write( "Este es caso " + str(CasoNum) )
    else:
        return (CasoNum, {})

    # # Notar que el número de etapas es siempre el mismo, por lo que nunca se retorna de StopIteration cuando se acaban.
    # for CustsDem, GensDispatch in zip(GeneratorDemand, GeneratorDispatch):
    #     StageNum = CustsDem[0]
    #     # StageNum = GensDispatch[0]
    #     # print( "StageNum:", StageNum )
    #     CustomersDemands = CustsDem[1]  # Free and Regulated Customers
    #     GensDispatched = GensDispatch[1]

    #     #
    #     #
    #     # Grid.bus.loc[ pp.topology.unsupplied_buses(Grid), : ]  # Nombre de barras no suministradas
    #     # pp.drop_inactive_elements(Grid)  # remueve elementos que están sin actividad o estado fuera de servicio
    #     #
    #     #

    #     # Calcula excedente disponible de potencia sistema

    #     # Según excedente determina si es factible el sistema, de lo contrario imprime caso infactible

    #     # Inicializa contadores de congestiones

    #     # Calcula el CMg del sistema

    #     # Corre el flujo de potencia linealizado en la grilla de la etapa
    #     pp__rundcpp(DictRawSyst_per_Stage[StageNum])
    #     # Identifica lineas con sobrecarga para ver congestiones

    #     # De las congestiones verifica las Inter e Intra

    #     """ De existir Intra,
    #         repite lpf mientras condición de contadores o No existen congestiones. aumenta contador Intra
    #     """

    #     """ De existir Inter,
    #         separa el sep dejando como referencia ???... . Repite lpf mientras condición de contadores
    #         o No existen congestiones. aumenta contador Inter
    #     """

    #     #

    #     print(DictRawSyst_per_Stage[StageNum])

    #     # print()
    # Numero de cargas (Depende de etapa)
    # Escoger lista demandas. Desde generator
    # Numero de Unidades - 1 (Depende de etapa)
    # Escoger lista despachos. Desde generator
