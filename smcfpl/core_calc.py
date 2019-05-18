"""
Este script esta diseñado para que pueda ser llamado como función 'Calc' a partir del nombre ,i.e.,
    import smcfpl.NucleoCalculo as NucleoCalculo  # En módulo
    from NucleoCalculo import Calc  # Directamente por interprete por ejemplo
    NucleoCalculo.Calc()
"""
from pandapower import rundcpp as pp__rundcpp
import time
import warnings
from os import sep as os__sep
from itertools import chain as itertools__chain
from pandapower import select_subnet as pp__select_subnet
from pandapower import create_ext_grid as pp__create_ext_grid
from pandapower.topology import connected_components as pp__topology__connected_components
from pandapower.topology import create_nxgraph as pp__topology__create_nxgraph
from pandapower.idx_brch import BR_R, BR_X, PF
from pandapower.idx_bus import VA
from pandapower.auxiliary import ppException
from pandapower.powerflow import LoadflowNotConverged
from pandas import DataFrame as pd__DataFrame
from pandas import concat as pd__concat
from numpy import cos as np__cos, real as np__real, sign as np__sign
from numpy import isnan as np__isnan, seterr as np__seterr
from scipy.sparse import linalg
from pickle import load as pickle__load
from multiprocessing import cpu_count as mu__cpu_count, Pool as mu__Pool
# from smcfpl.aux_funcs import overloaded_trafo2w as aux_smcfpl__overloaded_trafo2w
# from smcfpl.aux_funcs import overloaded_trafo3w as aux_smcfpl__overloaded_trafo3w
from smcfpl.in_out_proc import write_output_case as smcfpl__in_out_proc__write_output_case
from smcfpl.aux_funcs import TipoCong as aux_funcs__TipoCong
from smcfpl.aux_funcs import setup_logger as aux_funcs__setup_logger
from smcfpl.redispatch import redispatch as redispatch__redispatch, make_Bbus_Bpr_A as redispatch__make_Bbus_Bpr_A
from smcfpl.redispatch import power_over_congestion as redispatch__power_over_congestion
from smcfpl.smcfpl_exceptions import *

import logging

# get loggers created
logger = logging.getLogger('stdout_only')
logger_IntraCong = logging.getLogger('Intra_congestion')


def in_node_manager(group_info, base_BD_names, gral_params):
    """
        this must run within a single node.
        Take input from send_work.send() and use it
        to run parallel case as much cores are available,
        that is, run core_calc.calc() in parallel.
    """
    # get the group_info (tuple)
    nth_group = group_info[0]
    cases_per_group = group_info[1]
    n_groups = group_info[2]
    n_cases = group_info[3]
    group_details = group_info[4]
    nth_G_start = group_info[5]  # 0-indexed
    nth_D_start = group_info[6]  # 0-indexed
    # get some simulation parameters. Useful for every simulation. (came from self)
    random_seed = gral_params[0]
    DesvEstDespCenEyS = gral_params[1]
    DesvEstDespCenP = gral_params[2]
    abs_OutFilePath = gral_params[3]
    abs_path_temp = gral_params[4]
    NumVecesDem = gral_params[5]
    NumVecesGen = gral_params[6]
    # read BD files on head node
    base_BDs = dict.fromkeys(base_BD_names, None)
    for fname in base_BD_names:  # pickle assumed
        with open(abs_path_temp + os__sep + fname, 'rb') as f:
            base_BDs[fname] = pickle__load(f)
    ################################################
    ################################################
    ################################################
    ################################################
    ################################################
    # get general databases from file to each variable (came form file)
    StageIndexesList = base_BDs['BD_Etapas.p'].index.tolist()
    DF_TasaCLib = base_BDs['BD_DemProy.p'][ ['TasaCliLib'] ]
    DF_TasaCReg = base_BDs['BD_DemProy.p'][ ['TasaCliReg'] ]
    DF_DesvDec = base_BDs['BD_DemProy.p'][ ['Desv_decimal'] ]
    DF_HistGenERNC = base_BDs['BD_HistGenRenovable.p']
    DF_TSF = base_BDs['BD_TSFProy.p']
    grillas = base_BDs['BD_RedesXEtapa.p']
    DictTypoCargasEta = {k: v['PandaPowerNet']['load'][['type']] for k, v in grillas.items()}
    DF_GenType_per_unit = {k: d['ExtraData']['Tipos'] for k, d in grillas.items()}
    ################################################
    ################################################
    ################################################
    ################################################
    ################################################
    # parallel parameters
    n_cpu = min(mu__cpu_count(), cases_per_group)  # optimize core request
    Pool = mu__Pool(n_cpu)
    results = []
    # cases processing
    nth_case = (nth_group - 1) * cases_per_group + 1  # case associated with group
    for case_hid, cases_per_hid in group_details.items():
        nth_G = nth_G_start[case_hid] + 1
        nth_D = nth_D_start[case_hid] + 1
        # Note: if n_cases_per_hid == 0, this for loop is skipped
        for sub_nth_case in range(cases_per_hid):
            case_identifier = (case_hid, nth_D, nth_G)
            print("nth_case: {} == case_identifier: {}".format(nth_case, case_identifier))
            # filter database dependent on hydrology
            DF_PE_Hid = base_BDs['BD_Hydro'][HidNom]['DF_PEsXEtapa']
            # Creates an iterator (class type with __next__ dunder) for each loop (different values)
            instance_IterDem = aux_funcs.IteratorDemand(StageIndexesList=StageIndexesList,
                                                        DF_TasaCLib=DF_TasaCLib,  # pandas DataFrame
                                                        DF_TasaCReg=DF_TasaCReg,  # pandas DataFrame
                                                        DF_DesvDec=DF_DesvDec,  # pandas DataFrame
                                                        DictTypoCargasEta=DictTypoCargasEta,  # diccionario
                                                        seed=random_seed)  # int, None
            instance_IterDispatched = aux_funcs.IteratorDespatch(StageIndexesList=StageIndexesList,
                                                                 DF_GenType_per_unit=DF_GenType_per_unit,  # dict of numpy array
                                                                 DF_HistGenERNC=DF_HistGenERNC,  # tupla de dos pandas DataFrame
                                                                 DF_TSF=DF_TSF,  # para cada tecnología que recurra con falla se asigna
                                                                 DF_PE_Hid=DF_PE_Hid,  # pandas DataFrame
                                                                 DesvEstDespCenEyS=DesvEstDespCenEyS,  # float
                                                                 DesvEstDespCenP=DesvEstDespCenP,  # float
                                                                 seed=random_seed)  # int, None

            results.append(
                Pool.apply_async(
                    calc,
                    (
                        nth_case, case_hid, grillas, StageIndexesList, DF_ParamHidEmb_hid,
                        DF_seriesconf, DF_CVarReservoir_hid, MaxNumVecesSubRedes, MaxItCongIntra,
                        # n_cases,
                        # nth_group,
                        # n_groups,
                    ),
                    {
                        'abs_OutFilePath': abs_OutFilePath,
                        'DemGenerator': instance_IterDem,
                        'DispatchGenerator': instance_IterDispatched,
                        'CaseID': case_identifier,
                        'in_node': True,
                    }
                )
            )
            nth_case += 1
            # increments each G per case. If maxed increment D once and reset G counter
            if nth_G < NumVecesGen:
                nth_G += 1
                if nth_D < NumVecesDem:
                    nth_D += 1
                    nth_G = 0

    # fetch parallel status info
    for result in results:
        res = result.get()
        msg = res[2]
        print(msg)
    msg = "Finished manage() for group {}/{}.".format(nth_group, n_groups)
    logger.info(msg)

    return


def calc(CaseNum, Hidrology, Grillas, StageIndexesList, DF_ParamHidEmb_hid,
         DF_seriesconf, DF_CVarReservoir_hid, MaxNumVecesSubRedes, MaxItCongIntra, abs_OutFilePath='',
         DemGenerator=iter(()), DispatchGenerator=iter(()), CaseID=('hid', 0, 0),
         in_node=False):
    """
        Función base de cálculo para la resolución del modelo SMCFPL. Por cada etapa obtiene los valores
        de los generadores de demanda y generación para los elementos de la Grilla de la etapa ocurrente.

        :param CaseNum: Numero del caso corriendo.
        :type CaseNum: int

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

        :param DemGenerator: Iterador con valores tipo: (EtaNum, pd.DataFrame.loc[EtaNum, 'PDem_pu'])
        :type DemGenerator: iterator

        :param DispatchGenerator: Iterador con valores tipo: (EtaNum, pd.DataFrame.loc[EtaNum, ['type', PGen_pu']])
        :type DispatchGenerator: iterator

    """
    SuccededStages = 0
    RelevantData = {}
    print("Hidrology:", Hidrology)
    # for each stage in the case
    for (StageNum, DF_Dem), (StageNum, DF_Gen) in zip(DemGenerator, DispatchGenerator):
        print("StageNum:", StageNum, "CaseNum:", CaseNum)
        # Load Data from every stage when 'in_node' is True
        Grid, Dict_ExtraData = UpdateGridPowers( Grillas, StageNum,
                                                 DF_Dem, DF_Gen)

        #
        # Verifica que el balance de potencia es posible en la etapa del caso (Gen-Dem)
        try:
            # Corrobora factibilidad del despacho uninodal
            Power_available_after_dispatch(Grid)
        except DemandGreaterThanInstalledCapacity as msg:  # indica PNS!
            msg = str(msg).format(StageNum, len(StageIndexesList), CaseNum)
            msg += ' Skipping stage.'
            logger.warn(msg)
            # ¿Eliminar etapa?¿?
            # DF_Etapas.drop(labels=[StageNum], axis='index', inplace=True)
            # ¿Escribir PNS?
            continue  # Continua con siguiente etapa

        #
        # use to convert specific warnings with message to error.
        warnings.filterwarnings('error', message='Matrix is exactly singular')

        try:
            #
            # Calcula el Flujo de Potencia Linealizado
            pp__rundcpp(Grid, trafo_model='pi', trafo_loading='power',
                        check_connectivity=True, r_switch=0.0, trafo3w_losses='hv')
        except linalg.linsolve.MatrixRankWarning:
            msg = "Could not solve Linear powerflow for stage {}/{} in CaseNum {}. Skipping stage."
            msg = msg.format(StageNum, len(StageIndexesList), CaseNum)
            logger.warn(msg)
            continue  # jumps to next stage.
        except Exception as e:
            print("Something else happened during rundcpp() ...")
            print(e)
            import pdb; pdb.set_trace()  # breakpoint 4c98318c //
            continue
        else:  # excecute when not exception
            msg = "LPF ran on Grid for stage {}/{} in case {}."
            msg = msg.format(StageNum, len(StageIndexesList), CaseNum)
            logger.debug(msg)
        finally:  # execute alter all cases (even exceptions)
            pass

        #
        # Verifica que LA red externa se encuentre dentro de sus límites
        try:
            check_limits_GenRef(Grid)
        except GeneratorReferenceOverloaded as msg:  # PGenSlack es más Negativo que Pmax (sobrecargado)
            msg = str(msg).format(StageNum, len(StageIndexesList), CaseNum)
            msg += ' Skipping stage.'
            logger.warn(msg)
            continue
        except GeneratorReferenceUnderloaded as msg:  # PGenSlack es más Positivo que Pmin (comporta como carga)
            msg = str(msg).format(StageNum, len(StageIndexesList), CaseNum)
            msg += ' Skipping stage.'
            logger.warn(msg)
            continue
        except LoadFlowError as msg:
            msg = str(msg) + " Skipping stage."
            logger.error(msg)
            continue

        ################################################################################
        ################################################################################
        ################################################################################
        ################################################################################
        #
        # Calcula CMg previo congestión [$US/MWh]  # CVar en [$US/MWh]
        # Descarga las unidades que posean generación nula

        ################################################################################
        ################################################################################
        ################################################################################
        ################################################################################

        #
        # Finds merit list for current grid increasingly ordered and the marginal unit from same list
        marginal_unit, merit_list = find_merit_list(Grid, Dict_ExtraData)
        print("merit_list:", merit_list)
        print("marginal_unit", marginal_unit)
        #
        # Modifica loading_percent para identificar congestiones. Copy by reference.
        Adjust_transfo_power_limit_2_max_allowed(Grid, Dict_ExtraData)
        #
        # Obtiene lista del tipo de congestiones (TypeElmnt, IndGrilla)
        ListaCongInter, ListaCongIntra = aux_funcs__TipoCong(Grid, max_load=100)
        print("ListaCongIntra:", ListaCongIntra)
        print("ListaCongInter:", ListaCongInter)

        """
        #####                        ###           #                    ###                                       #       #
        #                             #            #                   #   #                                      #
        #       ###   # ##            #    # ##   ####   # ##    ###   #       ###   # ##    ## #   ###    ###   ####    ##     ###   # ##
        ####   #   #  ##  #           #    ##  #   #     ##  #      #  #      #   #  ##  #  #  #   #   #  #       #       #    #   #  ##  #
        #      #   #  #               #    #   #   #     #       ####  #      #   #  #   #   ##    #####   ###    #       #    #   #  #   #
        #      #   #  #               #    #   #   #  #  #      #   #  #   #  #   #  #   #  #      #          #   #  #    #    #   #  #   #
        #       ###   #              ###   #   #    ##   #       ####   ###    ###   #   #   ###    ###   ####     ##    ###    ###   #   #
                                                                                            #   #
                                                                                             ###
        """
        # Revisa la congestiones intra hasta que se llega a algún límite
        if ListaCongIntra:  # Is there any IntraCongestion?
            try:
                do_intra_congestion( Grid, Dict_ExtraData, ListaCongIntra,
                                     MaxItCongIntra, in_node,
                                     StageNum, StageIndexesList,
                                     CaseNum)
            except IntraCongestionIterationExceeded as e:
                print("**************************************")
                print(" You got an Exception: {}! Moving to next stage.".format(e))
                print("**************************************")
                continue  # nothing valueble to return, jump to next stage
            except CapacityOverloaded as e:
                print("**************************************")
                print(" You got an Exception: {}! Moving to next stage.".format(e))
                print("**************************************")
                continue  # nothing valueble to return, jump to next stage
            except (GeneratorReferenceOverloaded, GeneratorReferenceUnderloaded) as e:
                print("**************************************")
                print(" You got an Exception: {}! Moving to next stage.".format(e))
                print("**************************************")
                continue  # nothing valueble to return, jump to next stage
            # except Exception as e:
            #     print("Exception was:", e)
            #     continue

        """
              ###           #                    ###                                       #       #
               #            #                   #   #                                      #
               #    # ##   ####    ###   # ##   #       ###   # ##    ## #   ###    ###   ####    ##     ###   # ##
               #    ##  #   #     #   #  ##  #  #      #   #  ##  #  #  #   #   #  #       #       #    #   #  ##  #
               #    #   #   #     #####  #      #      #   #  #   #   ##    #####   ###    #       #    #   #  #   #
               #    #   #   #  #  #      #      #   #  #   #  #   #  #      #          #   #  #    #    #   #  #   #
              ###   #   #    ##    ###   #       ###    ###   #   #   ###    ###   ####     ##    ###    ###   #   #
                                                                     #   #
                                                                      ###
        """
        SubGrids = []
        if ListaCongInter:  # Is there any InterCongestion?
            # Divide Grid in N SubGrids if there is inter-congestion
            SubGrids = divide_grid_by_intercongestion(Grid, ListaCongInter)
            # for subgrid in SubGrids:
            #     pass

        # (Multiplica dos pandas Series) Indices
        # son creados secuencialmente, por lo que no necesita ser DataFrame
        # Falta incorporar costo de hidraulica DF_CVarReservoir y red externa
        pdSeries_CostDispatch = Dict_ExtraData['CVarGenNoRef'].squeeze() * -Grid['gen']['p_kw']  # en [$US]

        #
        # Calcula costo País
        TotalPowerCost = pdSeries_CostDispatch.sum()

        #
        # Calcula pérdidas por flujo de líneas
        PLoss = estimates_power_losses(Grid, method='linear')
        # PLoss = estimates_power_losses(Grid, method='cosine')
        print("PLoss:\n", PLoss)
        # Before finishing correctly, saves the relevant data.
        RelevantData[(StageNum, CaseNum)] = {
            'Grid': Grid,
            'merit_list': merit_list,
            'marginal_unit': marginal_unit,
            'ListaCongInter': ListaCongInter,
            'SubGrids': SubGrids,
            'TotalPowerCost': TotalPowerCost,
            'PLoss': PLoss,
        }
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("StageNum:", StageNum, "Finished successfully")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        SuccededStages += 1

    print("----------------------------")
    print("Este fue CaseNum:", CaseNum)
    print("----------------------------")
    if RelevantData:  # if it's not empty
        write_values_and_finish(in_node, RelevantData, CaseNum, CaseID, outputDir=abs_OutFilePath)
    return SuccededStages


"""

     #####                         #                                                      #
     #                                                                                    #
     #      #   #  # ##    ###    ##     ###   # ##    ###    ###           ###   #   #  ####   # ##    ###
     ####   #   #  ##  #  #   #    #    #   #  ##  #  #   #  #             #   #   # #    #     ##  #      #
     #      #   #  #   #  #        #    #   #  #   #  #####   ###          #####    #     #     #       ####
     #      #  ##  #   #  #   #    #    #   #  #   #  #          #         #       # #    #  #  #      #   #
     #       ## #  #   #   ###    ###    ###   #   #   ###   ####           ###   #   #    ##   #       ####


"""


def UpdateGridPowers(Grillas, StageNum, DF_Dem, DF_Gen):
    """
        :type StageNum: int
        :type DF_Dem: pandas DataFrame
        :type DF_Gen: pandas DataFrame
    """
    Grid = Grillas[StageNum]['PandaPowerNet']
    Dict_ExtraData = Grillas[StageNum]['ExtraData']
    # D['PDem_pu'].values: puede ser negativo, positivo o cero, pero siempre en [p.u.]
    Grid['load']['p_kw'] = Grid['load']['p_kw'] * DF_Dem['PDem_pu'].values
    # G['PGen_pu'].values: siempre va a estar entre 0 y 1 inclusive
    Grid['gen']['p_kw'] = Grid['gen']['max_p_kw'] * DF_Gen['PGen_pu'].values
    return Grid, Dict_ExtraData


def Power_available_after_dispatch(Grid):
    # Revisa las grillas en cada etapa para verificar que el balance de potencia es posible (Gen-Dem)
    PGenMaxSist = Grid['gen']['max_p_kw'].sum()
    PGenSlackMax = Grid['ext_grid']['max_p_kw'].sum()
    PDemSist = Grid['load']['p_kw'].sum()
    # Cuantifica diferencia entre generación y demanda máxima. Negativo implica sobra potencia
    DeltaP_Uninodal = PGenMaxSist + PGenSlackMax + PDemSist  # De ser positivo indica PNS!
    if DeltaP_Uninodal > 0:
        msg = "No es posible abastecer la demanda en etapa {}/{} del caso {}!."
        raise DemandGreaterThanInstalledCapacity(msg)


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
        # Obtiene el valor de potencia del lado correspondiente
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
    """ Allows to catch every unwishable outcome"""
    # Verifica Potencia para el GenSlack sea menor que su máximo (Negativo generación)
    PotSobrante = Grid['ext_grid'].at[0, 'max_p_kw'] - Grid['res_ext_grid'].at[0, 'p_kw']
    # Acota rangos factibles de potencia de generación de GenSlack
    if PotSobrante > 0:  # PGenSlack es más Negativo que Pmax
        msg = "Reference generator is overloaded in stage {}/{} from case {}!."
        raise GeneratorReferenceOverloaded(msg)

    # Verifica Potencia para el GenSlack sea mayor que su mínimo (Negativo generación)
    PotFaltante = Grid['res_ext_grid'].at[0, 'p_kw'] - Grid['ext_grid'].at[0, 'min_p_kw']
    if PotFaltante > 0:  # PGenSlack es más Positivo que Pmin
        msg = "Reference generator is underloaded in stage {}/{} from case {}!."
        raise GeneratorReferenceUnderloaded(msg)

    if np__isnan(PotFaltante) | np__isnan(PotSobrante):
        msg = "External grid returns NaN value."
        raise LoadFlowError(msg)


def find_merit_list(Grid, Dict_ExtraData):
    """
        Finds within the current Grid and the ExtraData an ordered list
        of dispached (avaiable) units according to their costs in
        increasing order. It requires previuos dispatched. Do not consider
        (filters) unit with power 0, that might mean they are active with
        no inyection, or they are in failure state.

        Returns two DataFrames: The first is the marginal unit as a single row
        pandas dataframe. The second element contains the merit list (last unit is marginal)
        in increasing order.
        Columns are:
            - index within corresponding Grid (gen|ext_grid).
            - variable cost of unit.
            - boolean flag meaning if it correspond a a reference unit (ext_grid).

        Steps: (Grid and ExtraData share same element indices)
          1.- filters out every Grid['res_gen'] con p=0 and get indices.
          2.- filters out every Grid['res_ext_grid'] con p=0 and get indices.
          3.- filter from ExtraData all rows filtered for Grid (GenRef and GenNoRefs)

    """
    # 1.- filters out every Grid['res_gen'] con p=0 and get indices.
    gen_no_ref_indxs = Grid['res_gen'][ Grid['res_gen']['p_kw'] == 0 ].index
    # 2.- filters out every Grid['res_ext_grid'] con p=0 and get indices.
    gen_ref_indxs = Grid['res_ext_grid'][ Grid['res_ext_grid']['p_kw'] == 0 ].index
    # 3.- filter from ExtraData all rows filtered for Grid (GenRef and GenNoRefs)
    cvar_no_ref = Dict_ExtraData['CVarGenNoRef'].loc[ gen_no_ref_indxs, : ]
    cvar_ref = Dict_ExtraData['CVarGenRef'].loc[ gen_ref_indxs, : ]
    # .- Adds new column to both DF for after identification of index (GenRef=True, GenNoRef=False)
    cvar_no_ref = cvar_no_ref.assign(Genref=False)
    cvar_ref = cvar_ref.assign(Genref=True)
    # .- concatenates (index-wise) the dataframes for all columns and sort by 'CVar'. Increasing order.
    cvar_df = pd__concat([cvar_ref, cvar_no_ref], axis='index').sort_values('CVar', ascending=True)
    # .- Last row is the marginal unit
    marginal_unit = cvar_df.tail(1)  # DF
    return (marginal_unit, cvar_df)


def do_intra_congestion(Grid, Dict_ExtraData, ListaCongIntra, MaxItCongIntra, in_node, StageNum, StageIndexesList, CaseNum):
    """
        Do the algorithm of intercongestion.
        Returns values if something goes wrong. Otherwise it's None.
    """
    IntraCounter = 0
    ListaCongIntra = iter(ListaCongIntra)
    while IntraCounter <= MaxItCongIntra:
        try:
            GCongDict = next(ListaCongIntra)
        except StopIteration:
            print("No more Intra congestions are allowed.")
            break

        for TypeElmnt, ListIndTable in GCongDict.items():
            # Trafos3w are not being consider
            if TypeElmnt == 'line':
                ColNames4Type = ['Pmax_AB_MW', 'Pmax_BA_MW']
                res_FlowFromNameCol = 'p_from_kw'
                TypeElmnt_aux = TypeElmnt
            elif TypeElmnt == 'trafo':
                ColNames4Type = ['Pmax_AB_MW', 'Pmax_BA_MW']
                res_FlowFromNameCol = 'p_hv_kw'
                TypeElmnt_aux = TypeElmnt + '2w'
            else:
                msg = "Type element on IntraCongestion element is no 'line' nor 'trafo'."
                logger.error(msg)
                raise ValueError(msg)

            FirstTime = True  # to get first value
            # iterates over all elements in same congestion element series to get minimum capacity
            for IndTable in ListIndTable:
                FlowDir = np__sign(Grid['res_' + TypeElmnt].at[IndTable, res_FlowFromNameCol])  # line flow sign from A to B is 1
                if FlowDir == 1:  # choose column (name) according to flow direction of overload element within Grid
                    ColName4Type_flow = ColNames4Type[0]
                else:
                    ColName4Type_flow = ColNames4Type[1]
                if FirstTime:  # works as initialization if ListIndTable is not empty
                    PrevCap = Dict_ExtraData['PmaxMW_' + TypeElmnt_aux].reset_index().at[IndTable, ColName4Type_flow]
                    FirstTime = False
                    MinCapElmtType = (TypeElmnt, IndTable)
                # Get the TypeElmnt and Index of lowest Tx capacity
                ActualCap = Dict_ExtraData['PmaxMW_' + TypeElmnt_aux].reset_index().at[IndTable, ColName4Type_flow]
                if ActualCap < PrevCap:  # overwrites every time it finds a smaller
                    MinCapElmtType = (TypeElmnt, IndTable)
                else:  # updates previous capacity
                    PrevCap = ActualCap

        TypeElmnt, IndTable = MinCapElmtType
        # keeps track of number of times done. Must stay before redispath because of continue
        IntraCounter += 1
        print("TypeElmnt:", TypeElmnt)
        print("IndTable:", IndTable)
        print("******")
        print("IntraCounter:", IntraCounter)
        print("******")
        loading_percent = Grid['res_' + TypeElmnt].at[IndTable, 'loading_percent']
        msg = ','.join( ['', str(StageNum), str(CaseNum), TypeElmnt, str(IndTable), str(loading_percent)])
        logger_IntraCong.info(msg)

        if IntraCounter >= MaxItCongIntra:
            # limit of IntraCongestion is reached
            msg = "Límite de MaxItCongIntra alcanzado en etapa {}/{} del caso {}!.".format(
                StageNum, len(StageIndexesList), CaseNum)
            logger.warn(msg)
            # Enough time!! Go home.
            raise IntraCongestionIterationExceeded(msg)

        try:
            msg = "Congestion redispatch started..."
            logger.debug(msg)
            redispatch__redispatch(Grid, TypeElmnt, IndTable, max_load_percent=100, decs=30)
        except FalseCongestion:
            # Redispatch has no meaning. Congestion wan't real. Jump to next one
            continue
        except CapacityOverloaded as e:
            # Generators limits can't handle congestion. No much meaning to keep going.
            raise e
        except (LoadflowNotConverged, ppException):
            # If power flow did not converged, then skip to next stage
            # (This should now happend unless base Grid is someway wrong).
            continue

        # Checks for results of redispatch problems and values

        # calculates load flow
        pp__rundcpp(Grid)
        # checks for reference generator limits
        RE_MaxGen_kW = abs(Grid.ext_grid.at[0, 'max_p_kw'])
        RE_MinGen_kW = abs(Grid.ext_grid.at[0, 'min_p_kw'])
        RE_Gen_kW = abs(Grid.res_ext_grid.at[0, 'p_kw'])
        if RE_Gen_kW > RE_MaxGen_kW:
            msg = "...Limite máximo generador de referencia superados. {}>={}".format(RE_MaxGen_kW, RE_Gen_kW)
            # finish function when outside limits
            print(msg)
            raise GeneratorReferenceOverloaded(msg)
        if RE_Gen_kW < RE_MinGen_kW:
            msg = "...Limite mínimo generador de referencia superados. {}>={}".format(RE_Gen_kW, RE_MinGen_kW)
            # finish function when outside limits
            print(msg)
            raise GeneratorReferenceUnderloaded(msg)
        msg = "Re-checking congestion existance..."
        logger.debug(msg)
        # checks for congestión again
        ListaCongInter, ListaCongIntra = aux_funcs__TipoCong(Grid, max_load=100)
        print("ListaCongIntra:", ListaCongIntra)
        print("ListaCongInter:", ListaCongInter)
        print("--- fin iteración while ---")
        ListaCongIntra = iter(ListaCongIntra)
    return None


def divide_grid_by_intercongestion(Grid, ListaCongInter):
    """
        Divides input Grid in N SubGrids in function of Number of congestions.
        Returns a list of Grids which was divides, each with one ext_grid.

        Considers:
            - Calculate delta power congestion over MaxP of branch.
            - Assure only one ext_grid in each subgrid
            - "Virtual" load and gen at corresponding bus of subgrid, value of Pmax branch. Import/export power
            - Sharing delta power between generators and ext_grid on both subgrids (not considering virtual elements)
              starting with the most expensive generator.
                1. Reduce power of exporting subgrid (reducing CMg of subgrid)
                2. Increment power of importing subgrid (increment CMg of subgrid)
    """
    SubGrids = []
    Buses2RemoveEdge = []  # list of tuples (Nini, Nfin)
    # iterate over structure oredered congestions to get bus indices
    for GCongDict in ListaCongInter:
        for TypeElmnt, ListIndTable in GCongDict.items():
            for IndTable in ListIndTable:
                if TypeElmnt == 'line':
                    Fbus = 'from_bus'
                    Tbus = 'to_bus'
                elif TypeElmnt == 'trafo':
                    Fbus = 'hv_bus'
                    Tbus = 'lv_bus'
                elif TypeElmnt == 'trafo3w':
                    msg = "Trafo3w should never be used for inter-congestions"
                    logger.error(msg)
                    raise ValueError(msg)
                # get propper bus indices
                IndxBusIni = Grid[TypeElmnt].at[IndTable, Fbus]
                IndxBusFin = Grid[TypeElmnt].at[IndTable, Tbus]
                Buses2RemoveEdge.append((IndxBusIni, IndxBusFin))
    # creates a graph to remove the congested branches
    G = pp__topology__create_nxgraph(Grid)
    # remove edges/branch asosiated
    for BusIni, BusFin in Buses2RemoveEdge:
        G.remove_edge(BusIni, BusFin)
    # get the set (iterator) of isolated buses
    SetBuses = pp__topology__connected_components(G)  # Table indices are not reseted
    for SetBus in SetBuses:  # for every set, there is a network
        net_i = pp__select_subnet(Grid, SetBus)  # equivalent network
        # checks for ext_grid within current network, in order to add it if doesnt.
        if net_i.ext_grid.shape[0] < 1:  # if shape = (0, 9) there is no ext_grid
            # look over all bus indices involved in the congestions, to find one where to put ext_grid
            for busindx in itertools__chain.from_iterable(Buses2RemoveEdge):
                if busindx in SetBus:
                    ConectingBus = busindx
                    break
            else:  # if for not break
                msg = "There is no 'busindx' in the bar set of bus indices."
                logger.error(msg)
                raise ValueError(msg)
            # creates an external grid with 0 power
            pp__create_ext_grid(net_i, bus=ConectingBus, vm_pu=1.0, va_degree=0.0, name='VirtualRef', max_p_kw=-9999999, min_p_kw=0)

            # CALC |PCong - Pmax|
            # OverloadP_kW, loading_percent = redispatch__power_over_congestion(net_i, TypeElmnt, BranchIndTable, max_load_percent)
            # ADD LOAD AND GEN POWER Pmax
            # share |PCong - Pmax| among generators (first, the most costly!!!)

        SubGrids.append(net_i)
    return SubGrids


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


def write_values_and_finish(in_node, RelevantData, CaseNum, CaseID, outputDir='.'):
    print("------------------\n Escribiendo archivo salida \n------------------")
    if in_node:
        smcfpl__in_out_proc__write_output_case(RelevantData, CaseNum, CaseID, pathto='.')  # re-check
    else:
        smcfpl__in_out_proc__write_output_case(RelevantData, CaseNum, CaseID, pathto=outputDir)
