"""
Este script esta diseñado para que pueda ser llamado como función 'Calc' a partir del nombre ,i.e.,
    import smcfpl.NucleoCalculo as NucleoCalculo  # En módulo
    from NucleoCalculo import Calc  # Directamente por interprete por ejemplo
    NucleoCalculo.Calc()
"""
from pandapower import rundcpp as pp__rundcpp
import time
from os import sep as os__sep
from pandapower import from_pickle as pp__from_pickle
from pandapower import select_subnet as pp__select_subnet
from pandapower.topology import connected_components as pp__topology__connected_components
from pandapower.idx_brch import BR_R, BR_X, PF
from pandapower.idx_bus import VA
from pandas import DataFrame as pd__DataFrame
from pandas import concat as pd__concat
from numpy import cos as np__cos, real as np__real
# from smcfpl.aux_funcs import overloaded_trafo2w as aux_smcfpl__overloaded_trafo2w
# from smcfpl.aux_funcs import overloaded_trafo3w as aux_smcfpl__overloaded_trafo3w
from smcfpl.aux_funcs import TipoCong as aux_funcs__TipoCong
from smcfpl.redispatch import redispatch as redispatch__redispatch, make_Bbus_Bpr_A as redispatch__make_Bbus_Bpr_A
from smcfpl.smcfpl_exceptions import *


import logging
logging.basicConfig(level=logging.DEBUG,
                    format="[%(levelname)s][%(asctime)s][%(filename)s:%(funcName)s] - %(message)s")
logger = logging.getLogger()


def calc(CasoNum, Hidrology, Grillas, StageIndexesList, DF_ParamHidEmb_hid,
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
            # Carga la grilla y extradata con valores actualizados
            Grid, Dict_ExtraData = UpdateGridPowers(Grillas, StageNum,
                                                    DemGenerator_Dict,
                                                    DispatchGenerator_Dict)

        #
        # Verifica que el balance de potencia es posible en la etapa del caso (Gen-Dem)
        DeltaP_Uninodal, msg = Power_available_after_dispatch(Grid)
        # Corrobora factibilidad del despacho uninodal
        if DeltaP_Uninodal > 0:  # De ser positivo indica PNS!
            msg = msg.format(StageNum, len(StageIndexesList), CasoNum)
            logger.warn(msg)
            #
            # ¿Eliminar etapa?¿?
            # DF_Etapas.drop(labels=[StageNum], axis='index', inplace=True)
            # ¿Escribir PNS?
            continue  # Continua con siguiente etapa

        #
        # Calcula el Flujo de Potencia Linealizado
        pp__rundcpp(Grid, trafo_model='pi', trafo_loading='power',
                    check_connectivity=True, r_switch=0.0, trafo3w_losses='hv')

        #
        # Verifica que LA red externa se encuentre dentro de sus límites
        GenRefExceeded, msg = check_limits_GenRef(Grid)
        if GenRefExceeded == 1:  # PGenSlack es más Negativo que Pmax (sobrecargado)
            msg = msg.format(StageNum, len(StageIndexesList), CasoNum)
            logger.info(msg)
            continue
        elif GenRefExceeded == -1:  # PGenSlack es más Positivo que Pmin (comporta como carga)
            msg = msg.format(StageNum, len(StageIndexesList), CasoNum)
            logger.info(msg)
            continue

        ################################################################################
        ################################################################################
        ################################################################################
        ################################################################################
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
        ################################################################################
        ################################################################################
        ################################################################################
        ################################################################################

        # Modifica loading_percent para identificar congestiones. Copy by reference.
        Adjust_transfo_power_limit_2_max_allowed(Grid, Dict_ExtraData)

        #
        # Obtiene lista del tipo de congestiones (TypeElmnt, IndGrilla)
        ListaCongInter, ListaCongIntra = aux_funcs__TipoCong(Grid, max_load=100)

        #
        # Inicializa contadores de Congestiones (int)
        ContadorInter = ContadorIntra = 0

        # Revisa la congestiones intra hasta que se llega a algún límite
        for GCongDict in ListaCongIntra:
            for TypeElmnt, ListIndTable in GCongDict.items():
                for IndTable in ListIndTable:
                    try:
                        redispatch__redispatch(Grid, TypeElmnt, IndTable, max_load_percent=100, decs=30)
                    except (FalseCongestion, CapacityOverloaded):
                        # Redispatch has no meaning. Congestion wan't real. Or
                        # Generator limits can't hadle congestion
                        continue
                    except Exception as e:  # cath other different errors
                        print("Exepciones!!!")
                        print(e)
                        import pdb; pdb.set_trace()  # breakpoint cb41466d //  !import code; code.interact(local=vars())
                        continue

                    # keeps track of number of times done
                    ContadorIntra += 1
                    if ContadorIntra <= MaxItCongIntra:
                        # limit of IntraCongestion is reached
                        msg = "Límite de MaxItCongIntra alcanzado en etapa {}/{} del caso {}!.".format(
                            StageNum, len(StageIndexesList), CasoNum)
                        logger.warn(msg)
                        break
                else:  # if for not break then:
                    continue
                break
            else:  # if for not break then:
                continue
            break

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
        PLoss = estimates_power_losses(Grid, method='linear')
        PLoss = estimates_power_losses(Grid, method='cosine')
        print("PLoss:\n", PLoss)

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


"""

     #####                         #                                                      #
     #                                                                                    #
     #      #   #  # ##    ###    ##     ###   # ##    ###    ###           ###   #   #  ####   # ##    ###
     ####   #   #  ##  #  #   #    #    #   #  ##  #  #   #  #             #   #   # #    #     ##  #      #
     #      #   #  #   #  #        #    #   #  #   #  #####   ###          #####    #     #     #       ####
     #      #  ##  #   #  #   #    #    #   #  #   #  #          #         #       # #    #  #  #      #   #
     #       ## #  #   #   ###    ###    ###   #   #   ###   ####           ###   #   #    ##   #       ####


"""


def UpdateGridPowers(Grillas, StageNum, DemGenerator_Dict, DispatchGenerator_Dict):
    Grid = Grillas[StageNum]['PandaPowerNet']
    Dict_ExtraData = Grillas[StageNum]['ExtraData']
    D = DemGenerator_Dict[StageNum]  # pandas DataFrame
    # print("D:", D)
    G = DispatchGenerator_Dict[StageNum]  # pandas DataFrame
    # print("G:", G)
    # D['PDem_pu'].values: puede ser negativo, positivo o cero, pero siempre en [p.u.]
    Grid['load']['p_kw'] = Grid['load']['p_kw'] * D['PDem_pu'].values
    # G['PGen_pu'].values: siempre va a estar entre 0 y 1 inclusive
    Grid['gen']['p_kw'] = Grid['gen']['max_p_kw'] * G['PGen_pu'].values
    return Grid, Dict_ExtraData


def Power_available_after_dispatch(Grid):
    # Revisa las grillas en cada etapa para verificar que el balance de potencia es posible (Gen-Dem)
    PGenMaxSist = Grid['gen']['max_p_kw'].sum()
    PGenSlackMax = Grid['ext_grid']['max_p_kw'].sum()
    PDemSist = Grid['load']['p_kw'].sum()
    # Cuantifica diferencia entre generación y demanda máxima. Negativo implica sobra potencia
    DeltaP_Uninodal = PGenMaxSist + PGenSlackMax + PDemSist  # De ser positivo indica PNS!
    msg = "No es posible abastecer la demanda en etapa {}/{} del caso {}!."
    return DeltaP_Uninodal, msg


def Adjust_transfo_power_limit_2_max_allowed(Grid, Dict_ExtraData):
    """
        Debido a que el loading_percent de los tranfos se calcula c/r a 'Grid.sn_kva' y,
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
        # Dict_ExtraData['PmaxMW_trafo3w']['Pmax_AB_MW'].values
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


def check_limits_GenRef(Grid):
    # Verifica Potencia para el GenSlack sea menor que su máximo (Negativo generación)
    PotSobrante = Grid['ext_grid'].loc[0, 'max_p_kw'] - Grid['res_ext_grid'].loc[0, 'p_kw']
    # Verifica Potencia para el GenSlack sea mayor que su mínimo (Negativo generación)
    PotFaltante = Grid['res_ext_grid'].loc[0, 'p_kw'] - Grid['ext_grid'].loc[0, 'min_p_kw']
    # Acota rangos factibles de potencia de generación de GenSlack
    if PotSobrante > 0:  # PGenSlack es más Negativo que Pmax
        msg = "El generador de referencia está sobrecargado en etapa {}/{} del caso {}!."
        GenRefExceeded = 1
    elif PotFaltante > 0:  # PGenSlack es más Positivo que Pmin
        msg = "El generador de referencia está absorbiendo potencia en etapa {}/{} del caso {}!."
        GenRefExceeded = -1
    else:
        GenRefExceeded = 0
        msg = ""
    return GenRefExceeded, msg


def estimates_power_losses(net, method='linear'):
    """
    Computes de approximate power losses of each line
    method can be,
        - 'linear': Makes an estimate of linear power losses of the operation point according to p.u. resistance and power flow of each branch.
        - 'cosine': Makes a cosinoidal estimattion of power losses (non-linear).
    Returns Power Losses of dimensions: Rx1
    """
    Data = np__real( net._ppc['branch'][:, [PF, BR_R, BR_X]] )
    Z_branches = Data[:, 1] + 1j * Data[:, 2]
    F_ik = Data[:, [0]]  # power [MW] inyected from 'from_bus' to branch
    if method == 'linear':
        """ Low error is achived when R < 0.25 X per branch.
                PLos = r * F_ik**2
        Arguments:
            F_ik (2D array of R branches by 1 column)
            Z_branches (2D array of 1 row by R branches)
        """
        R_vector = np__real(Z_branches).T
        # [R_vector]{1xR} * [F_ik]{Rx1} ^ 2
        return (R_vector * (F_ik ** 2)[:, 0]).T
    elif method == 'cosine':
        """ Computes cosine aproximation.
        PLoss = 2 * G * (1 - cos(delta_i-delta_j))
        Arguments:
            F_ik (2D array of R branches by one column)
            Z_branches (2D array of one row by R branches)
            IncidenceMat (2D array of R rows by N nodes)
            DeltaBarra (2D array of N nodes by 1 column). Results of power flow calculation
        """
        Bbus, Bpr, IncidenceMat = redispatch__make_Bbus_Bpr_A(net)  # from linear power flow (no Resistance in B)
        G_vector = np__real(1 / Z_branches).T  # inverse of each value. Requieres resitance not to be Susceptance.
        DeltaBarra = net._ppc['bus'][:, [VA]]  # these should real values
        # PLoss = 2 * [G_vector]{1xR} * (1 - cos( [IncidenceMat]{RxN}* [DeltaBarra]{Nx1}))
        return 2 * G_vector.T * np__cos( IncidenceMat * DeltaBarra ).T
    else:
        msg = "method '{}' is no available yet.".format(method)
        logger.error(msg)
        raise ValueError(msg)
