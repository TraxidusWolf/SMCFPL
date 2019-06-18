from pandapower.idx_brch import F_BUS, T_BUS, BR_X, BR_STATUS, TAP, PF
from pandapower.idx_gen import PG, GEN_BUS, QMAX, QMIN
from pandapower import rundcpp as pp__rundcpp

from scipy.sparse.linalg import inv as sps__linalg__inv
from scipy.sparse import csr_matrix as sps__csr_matrix
from scipy.sparse import dok_matrix as sps__dok_matrix
from scipy.sparse import spdiags as sps__spdiags
from scipy.sparse import vstack as sps__vstack, hstack as sps__hstack

from numpy.linalg import cond
from numpy import round as np__round, ones as np__ones, real as np__real
from numpy import flatnonzero as np__flatnonzero, r_ as np__r_
from numpy import arange as np__arange, array as np__array, sign as np__sign
from numpy import uint8, uint32, uint64, int8, int32, int64
from numpy import repeat as np__repeat, unique as np__unique
from numpy import nonzero as np__nonzero, isnan as np__isnan
from numpy import argwhere as np__argwhere, seterr as np__seterr
from numpy import copy as np__copy, where as np__where
from numpy import set_printoptions as np__set_printoptions
from pandas import concat as pd__concat
import logging


# custom module
try:    # try importing locally when run independently
    from smcfpl_exceptions import *
except Exception:
    from smcfpl.smcfpl_exceptions import *

np__seterr(divide='ignore', invalid='ignore')
logging.basicConfig(level=logging.DEBUG,
                    format="[%(levelname)s][%(asctime)s][%(filename)s:%(funcName)s] - %(message)s")
logger = logging.getLogger()


def redispatch(Grid, Dict_ExtraData, TypeElmnt, BranchIndTable, max_load_percent=100, decs=14):
    """
        Elimina la congestión en TypeElmnt: BranchIndTable, en caso de poseer carga de
        potencia > max_load_percent.

        Modifica la generación de las unidades conectadas a sus barras que principalmente
        influyen a la rama congestionada 'BranchIndTable' (0-indexed). Se calculan los factores
        de distribución GSDF, GGDF y de utilización ponderada FUPTG con el fin de transferir potencia
        desde las unidades que más aportan y más disminuyen (en este momento) al flujo de potencia de la línea
        desde su barra 'from_bus' hasta 'to_bus', definidas en un comienzo. La referencia de los GSDF
        es LA red externa de Grid.

        Notar que Grid, debe venir con el flujo de potencia linealizado previamente calculado.
        Aquí se realiza un 'copy by reference', por lo que los cambios hechos aquí afectan al
        objeto en si!

        El delta de potencia sobrante en la línea congestionada se transfiere (incrementado en
        un 12% (default) del flujo permitido - por linealización y aproximación del punto de operación. Notar que
        es un valor arbitrario, pero otorga buenos resultados) de la unidad que más aporta al flujo
        según los FUPTG a la que menos aporta (o en su defecto realiza más flujo contrario).

        En caso de no poder transferir generación a dicha unidad, la potencia se va repartiendo
        consecutivamente a aquellas que aportan a la descongestión.

        Para la situación en la que la unidad de destino 'G_to' sea la referencia (Red externa),
        entonces solo se disminuye la potencia del generador 'G_from' ya que la referencia aceptará
        el cambio. En caso contrario, se aumenta la generación del 'G_from' para lograr mismo efecto.

        Realiza un proceso iterativo hasta que la POTENCIA de la rama 'BranchIndTable' es igual o
        menor al límite.

        Return None (inplace power Grid generation modification)

        Syntax:
            redispatch(Grid, Dict_ExtraData, TypeElmnt, BranchIndTable, max_load_percent=100)

        Example:
            redispatch(Grid, Dict_ExtraData, 'line', 0, max_load_percent = 2.8)
    """
    # verifies argument types
    if not isinstance(BranchIndTable, (int, uint8, uint32, uint64, int8, int32, int64)):
        msg = "BranchIndTable bus be an integer."
        logger.error(msg)
        raise ValueError(msg)

    # checks if power flow was executed in Grid. Otherwise runs LPF
    if not Grid.converged:
        msg = "Network had no power flow calculations... Calculating within 'redispatch()'."
        logger.warn(msg)
        pp__rundcpp(Grid)  # it doesn't matter which actually
    # Calculates Susceptance matrix of lpf (NxN), B Primitive (LxL) y Incidence matrix (LxN)
    Bbus, Bpr, Amat = make_Bbus_Bpr_A(Grid)  # Grid must had previos power flow calculation
    # Get index of reference busbar (external grid)
    RefBus = Grid.ext_grid.at[0, 'bus']  # only first index allowed. For now. TODO: convert others to gen. choose one.
    RefBusMatIndx = np__nonzero(Grid.bus.index == RefBus)[0][0]  # index of RefBus within Grid.bus.index. First value match
    # Get the dict of list of indices (for relationship with Bpr and Amat)
    IndxBrnchElmnts = ind_of_branch_elmnt(Grid)
    # Calculate GSDF matrix factors (uses aproximation for really small numbers)
    GSDFmat = calc_GSDF_factor(Bbus, Bpr, Amat, RefBusMatIndx, CondMat=True, decs=decs)
    # Calculate GGDF and Generation Utilizatión matrix factors (uses aproximation for really small numbers)
    GGDFmat, FUPTG, IndGenRef_GBusMat, GenBus_mat = calc_GGDF_Factor( Grid, GSDFmat,
                                                                      RefBus, decs=decs,
                                                                      FUTP_cf=False)
    # Find the NON-ERNC unit, so they are the ONLY TO CHANGE POWER DISPATCHED
    ERNC_type_list_names = ['Solar', 'EólicaZ1', 'EólicaZ2', 'EólicaZ3', 'EólicaZ4', 'Pasada']  # hard coded, MUST CHANGE!

    non_fixed_units_pd2ppc = dict()
    non_fixed_gen_net_idx = Grid.gen[ ~Grid.gen['type'].isin(ERNC_type_list_names) ].index
    non_fixed_units_pd2ppc['gen'] = Grid._pd2ppc_lookups['gen'][non_fixed_gen_net_idx]
    aux_filter_DF = Dict_ExtraData['TecGenSlack']['GenTec'].isin(ERNC_type_list_names).values
    non_fixed_ext_grid_net_idx = Dict_ExtraData['TecGenSlack'][ ~aux_filter_DF ]['pp_ext_grid_idx'].values
    non_fixed_units_pd2ppc['ext_grid'] = Grid._pd2ppc_lookups['ext_grid'][non_fixed_ext_grid_net_idx]

    # Get the row of the branch of interest from matrices (according to IndxBrnchElmnts and input argument)
    TheRowBrnchIndMat = IndxBrnchElmnts[TypeElmnt][BranchIndTable]  # asume elementos de tabla ordenados secuencialmente según tablas Grid
    # Uses FUPTG to determine the most and least participation generators on line flow. Uses ordered
    RowValues = FUPTG[TheRowBrnchIndMat, :]
    # calcula la potencia sobrante
    power2distribute_kW, loading_percent = power_over_congestion( Grid, TypeElmnt,
                                                                  BranchIndTable,
                                                                  max_load_percent)
    if (power2distribute_kW <= 0) :  # & (loading_percent < max_load_percent) ?
        # here catch negative o zero value "DiffPower"
        msg = "Branch '{}' wasn't really congested ({:.4f} %)".format(TypeElmnt, loading_percent)
        logger.error(msg)
        raise FalseCongestion(msg)

    modify_dispatch_by_FUPTG( Grid, RowValues, power2distribute_kW, non_fixed_units_pd2ppc)
    return None


def modify_dispatch_by_FUPTG(net, branch_FUPTG, power2distribute_kW, non_fixed_units_pd2ppc, method=1):
    """
        Algorithm to modify distribute some excess power by dispatching more influence units.
        Modifies PandaPower network 'net' as it is copied by reference.
        Return None.
    """
    print("branch_FUPTG:", branch_FUPTG)
    print("power2distribute_kW:", power2distribute_kW)
    # reconstruct order from FUPTG columns (first all ext_grids, then gens)
    #
    if method == 1:
        ###############################################################################################################
        ###############################################################################################################
        ###############################################################################################################
        #  Method 1: Technically more efficient - iterative (FUPTG positive and negative variation)
        #  Unit pondered factor by positive and negative FUPTG to increase and decrease respectively power2distribute_kW
        #  Some units are considered 'fixed2increase' generation, specially those that can only reduce (if implemented
        #  technology) their power for redispatch. THAT IS, ALL POSITIVE FACTORS CAM BE USE TO REDUCE power.
        # Steps:
        #   1.- Split FUPTG for branch into positive and negative (zero value are not considered).
        #   2.- Filter selection leaving available Non NCRE units to increase (conventional generation).
        #       and the ability of every unit with enough technology to reduce it's generation by specific amount.
        #   3.- Every unit able to increase is associated with 'neg' suffix. Every unit able to decrease is asociated with 'pos' suffix.
        #   4.- Ponder the available FUPTG as weighs like a group ('pos' and 'neg'). Normalization.
        #   5.- Find the power available to change (is it possible to redispatch).
        #   6.- Create other weighs according to the delta power for each changeable unit.
        #   7.- Average a pondered sum of the two weighs (factorN from FUPTGs and factors from delta power available)
        #   8.- Multiply power2distribute_kW by the factors, so total sum is twice as big, given 'pos' and 'neg' factors.
        #   9.- Generation corresponding to slack machine is split uniformly across generator that have opposite factor sign.
        #       That is if current ext_grid belong to the group of 'neg', it's power is sent uniformly to all 'pos'
        #       generator, and viceversa.
        #   ¿LIMITES DE UNIDADES EN FORMA INDIVIDUAL? ¿COMO SE ASEGURA CORRECTO FUNCIONAMIENTO?
        #
        # work with dataframes (copies of pp tables). None means not applied.
        generators = net.gen.drop(['scaling', 'in_service', 'min_q_kvar', 'sn_kva', 'max_q_kvar', 'vm_pu', 'type', 'bus'], axis='columns')
        generators = generators.assign(**{
            'pd2ppc': net._pd2ppc_lookups['gen'],
            'FUPTG_cong': branch_FUPTG[net._pd2ppc_lookups['gen']],
            'fixed2increase': True,
            'factorsN': None,
            'dP_kW': None,  # dp: delta power ...
            'dP_weigh': None,  # dp: delta power ...
            'final_weigh': None,  # weigh factor as sum of the one from deltaP and FUPTG
        })
        ext_grids = net.ext_grid.drop(['vm_pu', 'va_degree', 'in_service', 'bus'], axis='columns')
        ext_grids = ext_grids.assign(**{
            'pd2ppc': net._pd2ppc_lookups['ext_grid'],
            'FUPTG_cong': branch_FUPTG[net._pd2ppc_lookups['ext_grid']],
            'fixed2increase': True,
            'factorsN': None,
            'dP_kW': None,  # dp: delta power ...
            'dP_weigh': None,  # dp: delta power ...
            'final_weigh': None,  # weigh factor as sum of the one from deltaP and FUPTG
        })
        # update 'fixed' values for generators and ext_grids
        non_fixed_net_gen_idx = generators['pd2ppc'].isin( non_fixed_units_pd2ppc['gen'] )
        generators.loc[ non_fixed_net_gen_idx, 'fixed2increase' ] = False
        non_fixed_net_gen_idx = ext_grids['pd2ppc'].isin( non_fixed_units_pd2ppc['ext_grid'] )
        ext_grids.loc[ non_fixed_net_gen_idx, 'fixed2increase' ] = False
        # find out the rows that can change (non fixed and positive of negative values)
        pos_gen_vals_idx = generators['FUPTG_cong'] > 0  # non is considered limited to reduce power
        neg_gen_vals_idx = (generators['FUPTG_cong'] < 0) & (~generators['fixed2increase'])
        pos_ext_grid_vals_idx = ext_grids['FUPTG_cong'] > 0  # non is considered limited to reduce power
        neg_ext_grid_vals_idx = (ext_grids['FUPTG_cong'] < 0) & (~ext_grids['fixed2increase'])
        # indices that will have new generation
        # new_Pgen_idx = neg_gen_vals_idx | pos_gen_vals_idx
        # new_Pext_grid_idx = pos_ext_grid_vals_idx | neg_ext_grid_vals_idx
        # get sum of positives and negatives
        sum_pos_factors = generators[pos_gen_vals_idx]['FUPTG_cong'].sum() + ext_grids[pos_ext_grid_vals_idx]['FUPTG_cong'].sum()
        sum_neg_factors = generators[neg_gen_vals_idx]['FUPTG_cong'].sum() + ext_grids[neg_ext_grid_vals_idx]['FUPTG_cong'].sum()
        # update positive and negative factor values for generators and ext_grids
        generators.loc[ pos_gen_vals_idx, 'factorsN'] = generators['FUPTG_cong'] / sum_pos_factors
        generators.loc[ neg_gen_vals_idx, 'factorsN'] = generators['FUPTG_cong'] / sum_neg_factors
        ext_grids.loc[ pos_ext_grid_vals_idx, 'factorsN'] = ext_grids['FUPTG_cong'] / sum_pos_factors
        ext_grids.loc[ neg_ext_grid_vals_idx, 'factorsN'] = ext_grids['FUPTG_cong'] / sum_neg_factors
        # find delta power available corresponding factor, NaN if not. (positive to decrease or negative to increase powers)
        generators.loc[pos_gen_vals_idx, 'dP_kW'] = generators['p_kw'].abs() - generators['min_p_kw'].abs()
        generators.loc[neg_gen_vals_idx, 'dP_kW'] = generators['max_p_kw'].abs() - generators['p_kw'].abs()
        ext_grids.loc[pos_ext_grid_vals_idx, 'dP_kW'] = net.res_ext_grid['p_kw'].abs() - ext_grids['min_p_kw'].abs()
        ext_grids.loc[neg_ext_grid_vals_idx, 'dP_kW'] = ext_grids['max_p_kw'].abs() - net.res_ext_grid['p_kw'].abs()
        # get power available
        delta_Ppos_avail = pd__concat(
            [
                generators[pos_gen_vals_idx],
                ext_grids[pos_ext_grid_vals_idx],
            ], axis='index')['dP_kW'].fillna(0).sum()
        delta_Pneg_avail = pd__concat(
            [
                generators[neg_gen_vals_idx],
                ext_grids[neg_ext_grid_vals_idx],
            ], axis='index')['dP_kW'].fillna(0).sum()
        if power2distribute_kW > delta_Pneg_avail:
            msg = "Not enough installed generation capacity for dispatch and solving by redispatch."
            logger.warn(msg)
            raise CapacityOverloaded(msg)
        if power2distribute_kW > delta_Ppos_avail:
            msg = "Not enough capacity to decrease for solving redispatch without curtailment."
            logger.warn(msg)
            raise CapacityOverloaded(msg)

        # weigh factors given deltaP power
        generators.loc[ pos_gen_vals_idx, 'dP_weigh'] = generators['dP_kW'] / delta_Ppos_avail
        generators.loc[ neg_gen_vals_idx, 'dP_weigh'] = generators['dP_kW'] / delta_Pneg_avail
        ext_grids.loc[pos_ext_grid_vals_idx, 'dP_weigh'] = ext_grids['dP_kW'] / delta_Ppos_avail
        ext_grids.loc[neg_ext_grid_vals_idx, 'dP_weigh'] = ext_grids['dP_kW'] / delta_Pneg_avail
        # re-weigh final factor from power and FUTPGs. JUST ADD THEM!!
        generators['final_weigh'] = generators['dP_weigh'] + generators['factorsN']
        ext_grids['final_weigh'] = ext_grids['dP_weigh'] + ext_grids['factorsN']
        # calculate sum for ponderation
        pos_total_weigh = pd__concat(
            [
                generators[pos_gen_vals_idx],
                ext_grids[pos_ext_grid_vals_idx],
            ], axis='index')['final_weigh'].sum()
        neg_total_weigh = pd__concat(
            [
                generators[neg_gen_vals_idx],
                ext_grids[neg_ext_grid_vals_idx],
            ], axis='index')['final_weigh'].sum()
        # update pondered final_weighs
        generators.loc[pos_gen_vals_idx, 'final_weigh'] /= pos_total_weigh
        generators.loc[neg_gen_vals_idx, 'final_weigh'] /= neg_total_weigh
        ext_grids.loc[pos_ext_grid_vals_idx, 'final_weigh'] /= pos_total_weigh
        ext_grids.loc[neg_ext_grid_vals_idx, 'final_weigh'] /= neg_total_weigh

        # Updates power difference (abs values)
        newP_kW_gen = generators['final_weigh'].fillna(0) * power2distribute_kW

        # Power corresponding to slack bus has to be transfer to other generators, per slack unit (unique)
        # If slack belongs to positive, then all other units have to equally increase generation (those negative)
        for unit_name, unit_data in ext_grids[pos_ext_grid_vals_idx].iterrows():
            # get value of total power to share
            # import pdb; pdb.set_trace()  # breakpoint 75caa2ff //
            P_ext_grid2others = unit_data['final_weigh'] * power2distribute_kW
            # here absolut values
            newP_kW_gen += P_ext_grid2others / newP_kW_gen[neg_gen_vals_idx].shape[0]
        # else, (belongs to negative) all other units have to equally decrease generation (those positive)
        for unit_name, unit_data in ext_grids[neg_ext_grid_vals_idx].iterrows():
            # get value of total power to share
            P_ext_grid2others = unit_data['final_weigh'] * power2distribute_kW
            # here absolut values
            newP_kW_gen -= P_ext_grid2others / newP_kW_gen[pos_gen_vals_idx].shape[0]

        # Finally assign the new power to Grid generators. Respect signs
        #
        # decrement the ones with positive FPUTG (and possible)
        net.gen.loc[pos_gen_vals_idx, 'p_kw'] -= -newP_kW_gen[pos_gen_vals_idx].values
        # increment the ones with negative FPUTG (and possible)
        net.gen.loc[neg_gen_vals_idx, 'p_kw'] += -newP_kW_gen[neg_gen_vals_idx].values

    elif method == 2:
        ###############################################################################################################
        ###############################################################################################################
        ###############################################################################################################
        #  Method 2: Technically more efficient - single step (FUPTG postive or negative and near zero values)
        #  Verifica potencia disponible a unidades que menos afectan a la congestion (dado umbral de entrada), así como
        #  aquellas que más afectan (positiva o negativamente) a la congestión.
        #  De no existir unidades suficiente potencia (o en su defencto sin unidades), este método ya no es factible pués
        #  se convierte en una problemática no lineal al modificar dos veces las variables.
        #
        # Steps:
        #   1.-

        ###############################################################################################################
        ###############################################################################################################
        ###############################################################################################################

        net.gen
    return None


def make_Bbus_Bpr_A(Grid):
    """Taken from padapower/pf/makeBdc.py: makeBdc() and adapted-

    Function not adapted for multiple bus representing one.

    Builds the Bbus, primitive susceptance and Incidence matrix from
    _ppc['bus'] and _ppc['branch'] of Grid.

    Grid must have '_ppc' key, either by power flow or converter.to_ppc

    returns Bbus, Bpr, IncidenceMatrix  # (-1: enter. +1: leave.)
    """
    ## check that power flow was run before
    if not Grid.converged:
        msg = "Network had no power flow calculations... Calculating within 'make_Bbus_Bpr_A()'."
        logger.warn(msg)
        pp__rundcpp(Grid)  # it doesn't matter which actually
    bus = Grid._ppc['bus']
    branch = Grid._ppc['branch']
    nb = bus.shape[0]  # number of buses
    nl = branch.shape[0]  # number of lines

    stat = branch[:, BR_STATUS]  # ones at in-service branches
    b = np__real(stat / branch[:, BR_X])  # series susceptance
    tap = np__ones(nl)  # default tap ratio = 1
    i = np__flatnonzero(np__real(branch[:, TAP]))  # indices of non-zero tap ratios
    tap[i] = np__real(branch[i, TAP])  # assign non-zero tap ratios
    b = b / tap

    ## build connection matrix Cft = Cf - Ct for line and from - to buses
    f = np__real(branch[:, F_BUS]).astype(int)  # list of "from" buses
    t = np__real(branch[:, T_BUS]).astype(int)  # list of "to" buses
    i = np__r_[range(nl), range(nl)]  # double set of row indices
    ## connection matrix (brnach x nodes)
    data = np__r_[np__ones(nl), -np__ones(nl)]
    row_ind = i
    col_ind = np__r_[f, t]
    Cft = sps__csr_matrix((data, (row_ind, col_ind)), (nl, nb))  # -1: entra. +1: sale.

    ## build Bf such that Bf * Va is the vector of real branch powers injected
    ## at each branch's "from" bus
    Bf = sps__csr_matrix((np__r_[b, -b], (i, np__r_[f, t])), (nl, nb))

    ## build Bbus
    Bbus = Cft.T * Bf

    # con los mismo indices de ramas construye Bpr (Diagonal de susceptancias)
    Bpr = sps__spdiags(b, 0, len(b), len(b))
    return Bbus, Bpr, Cft.astype(int)


def ind_of_branch_elmnt(Grid):
    """
        Obtiene un diccionario con los indices relativos a la matriz Bbus
        de los distintos 'branch' que existen (line, trafo, or trafo3w).
        Retorna diccionario con lista de ítemes de indices correspondientes en valores.
    """
    # inicializa diccionario de salida
    Dict_salida = {}
    try:
        IndxLine = Grid._pd2ppc_lookups['branch']['line']
        Dict_salida['line'] = np__arange(IndxLine[0], IndxLine[1])
    except Exception:
        pass
    try:
        IndxTrafo2w = Grid._pd2ppc_lookups['branch']['trafo']
        Dict_salida['trafo'] = np__arange(IndxTrafo2w[0], IndxTrafo2w[1])
    except Exception:
        pass
    try:
        IndxTrafo3w = Grid._pd2ppc_lookups['branch']['trafo3w']
        Dict_salida['trafo3w'] = np__arange(IndxTrafo3w[0], IndxTrafo3w[1])
    except Exception:
        pass

    return Dict_salida


def calc_GSDF_factor(Bbus, Bpr, IncidenceMat, RefBusMatIndx, CondMat=False, decs=10):
    """
        Calcula la matriz GSDF a partir de:

        Bbus: Susceptance csr_sparse matrix. Dimensions nodes x nodes.

        Bpr: Diagonal sparse matrix of primitive susceptance. Dimensiones branches x branches.

        Incidence matrix (IncidenceMat) has to represent (-1 entering node
        and +1 for exiting), with dimensions branches x nodes. Csr_sparse matrix

        RefBusMatIndx: Integer index of the pandapower net bus index (external grid bus.). By previous definition can only be one.

        CondMat: Flag booleano para detectar condicionamiento de matriz Bbus.toarray()

        decs: número entero de decimales a aproximar valores de matriz retornada.

    """
    if not isinstance(RefBusMatIndx, (int, uint8, uint32, uint64, int8, int32, int64)):
        msg = "RefBusMatIndx bus be an integer representing the index of the RefBus index of a pandapower net bus."
        logger.error(msg)
        raise ValueError(msg)
    NNodes = Bbus.shape[0]  # Bbus must be squared
    # Assures correct format of sparse matrices
    Bpr = Bpr.tocsr()
    IncidenceMat = IncidenceMat.tocsr()
    Bbus = Bbus.tocsr()
    ###########################
    FrstStackNRef = np__array(range(0, RefBusMatIndx))
    # get indices of second stack of non reference buses
    ScndStackNRef = np__array(range(RefBusMatIndx + 1, NNodes))
    # list of non ref buses
    NonRefBuses = np__r_[FrstStackNRef, ScndStackNRef]
    # Remove row and column from Bbus to invert
    Bbus = Bbus[NonRefBuses, :].tocsc()[:, NonRefBuses]

    # Detecta si Bbus recoratada se encuentra mal condicionada. Se ofrece
    # opcional ya que Bbus debe transformarse a densa (numpy.array)
    if CondMat:
        NumCond = cond(Bbus.toarray())
        if NumCond > 1e15:
            msg = "Matrix Bbus without reference bus is bad conditioned (>1e15). Xbus values may not be representative."
            logger.warn(msg)

    Xbus = sps__linalg__inv(Bbus)
    # Second stack of Non reference buses must shift down one unit
    ScndStackNRef = ScndStackNRef - 1
    # adds reference column bus as zero vectors
    Mzeros = sps__dok_matrix((NNodes - 1, 1))
    Xbus = sps__hstack((Xbus[:, FrstStackNRef], Mzeros, Xbus[:, ScndStackNRef])).tocsc()
    # adds reference row bus as zero vectors
    Mzeros = sps__dok_matrix((1, NNodes))
    Xbus = sps__vstack((Xbus[FrstStackNRef, :], Mzeros, Xbus[ScndStackNRef, :]))
    # calculates GSDF Factor Matrix
    GSDFmat = Bpr @ IncidenceMat @ Xbus  # dot multiplication
    # GSDFmat.data = np__round(GSDFmat.data, decs)  # approximate values to avoid numerical errors
    return GSDFmat


def calc_GGDF_Factor(Grid, GSDF, RefBus, decs=10, FUTP_cf=True):
    """
        Calculates Generalized Generation Distribution Factors (GGDFs).

        There must exists only one ext_grid within Grid. If more, user should
        convert them to conventional generators.

        FUPTG convention: First group of columns ares for reference (all ext_grid). Other for generators (all gen)

        Arguments:
        Grid (pandapower net) with at least one external grid, a generator and one bus. (able to run power flow)

        GSDF: sparse csr_matrixwith GSDF factors.

        It should consider same reference bus as for GSDF matrix.
        RefBus: Integer index of bus reference (external grid bus.)

        decs: integer number to approximate values of returned matrix.

        FUTP_cf: booblean flag to correct the flow given by GGDF.

    """
    _ppc = Grid._ppc
    _pd2ppc_lookups = Grid._pd2ppc_lookups
    np__set_printoptions(linewidth=500)
    # Number of nodes, branches and generator
    NNodes = _ppc['bus'].shape[0]
    Nbranches = _ppc['branch'].shape[0]
    Ngen = _ppc['gen'][:, PG].shape[0]
    # Creates vector of power flows (Branch x 1)
    F_ik = np__real(_ppc['branch'][:, [PF]])  # power [MW] inyected from 'from_bus'
    F_ik = sps__csr_matrix(F_ik)

    # get gen ref index of _ppc matrix (unique reference considered)
    IndGenRef_GBusMat = _pd2ppc_lookups['ext_grid'][0]
    ref_units_GbusMat = _pd2ppc_lookups['ext_grid']  # matrix _ppc index ref_units
    non_ref_units_GbusMat = _pd2ppc_lookups['gen']  # matrix _ppc index non_ref_units
    # FUPTG convention: Use same indexing as _ppc matrices.
    ppc_gen_idx_from_pd = np__r_[ref_units_GbusMat, non_ref_units_GbusMat]
    # initialize data and matrix indices for Generator power per bus matrix (GenBus_mat)
    data, idx, jdx = [], [], []
    for NumGen, (BusInd, PGen_MW) in enumerate(_ppc['gen'][:, [GEN_BUS, PG]]):
        # iterate over rows from numpy array (matrix). 'gen' includes ext_grids and generators.
        BusInd = int(BusInd)  # bus id. Supposed in ascending order because of previous filtering.
        PGen_MW = float(PGen_MW)  # power dispatched on unit.

        # table_idx = np__where( NumGen == ppc_gen_idx_from_pd )[0][0]
        table_idx = ppc_gen_idx_from_pd[NumGen]
        idx.append(BusInd)  # must start from zero (0-indexed)
        jdx.append(table_idx)  # keeps dataframe index correlation for further use
        data.append(PGen_MW)
    GenBus_mat = sps__csr_matrix( (data, (idx, jdx)), (NNodes, Ngen) )  # Nodes x NumGen

    # Creates generation matrix. Equivalent Power injected to each node (nodos x 1).
    Gmat = sum_rows(GenBus_mat)  # potencias de generación por barra. Suma los generadores por barra

    # Total generation on the system
    Gsist = Gmat.toarray().sum()  # _ppc['gen'][:, PG].sum()

    # Creates the GGDF factors for reference bus (only one). Branch x 1
    GGDF_bref = (F_ik - GSDF @ Gmat) / Gsist

    # initialize return matrix
    GGDF = sps__dok_matrix((Nbranches, NNodes))
    # for each node, populates the return matrix
    for NumNodo in range(NNodes):
        if NumNodo == RefBus:
            # GGDF = GGDF_bref
            GGDF[:, NumNodo] = GGDF_bref
        else:
            # GGDF = GSDF + GGDF_bref
            GGDF[:, NumNodo] = GGDF_bref + GSDF[:, NumNodo]
    GGDF = GGDF.tocsr()
    # GGDF.data = np__round(GGDF.data, decs)  # approximate values to avoid numerical errors

    # Calculates "Factores de Utilización por Tramo de Generación"
    FUPTG = calc_FUTP_gen(GGDF, GenBus_mat, Gmat, F_ik, FUTP_cf, decs = decs)
    return GGDF, FUPTG, IndGenRef_GBusMat, GenBus_mat


def calc_FUTP_gen(GGDFmat, GenBus_mat, Gmat, F_ik, cf = True, decs = 10):
    """
        Un generador nunca va a estar conectado a dos barras simultáneamente (GenBus_mat).
        if cf=True:
            Se aplica corrección de flujo a los factores GGDF, tal que se mantiene valor si (GGDF_ik,i * F_ik) >= 0
            o 0 en otro caso.
        else:
            NO se modifica GGDF previamente

        arguments:
            GGDFmat: sparse csr_matrix with GGDF factors. Dimensions branches x nodes.
            GenBus_mat: sparse dok_matrix of incidence generator matrix. Dimensions nodes x NumGen.
            Gmat: sparse csr_matrix / vector of power inyected to nodes. Dimensions nodes x 1.
            F_ik: sparse csr_matrix / vector of active power on branches. Dimensions branches x 1.
            cf=True: corrects values accordingly to flow and GGDF sign multiplication.
            decs: integer number to approximate values of returned matrix.

        Returns:
            FUPT: numpy.ndarray of participation values
    """
    if cf:
        GGDFmat = GGDFmat.tocoo()
        GGDFmat_cf = GGDFmat.copy().tolil()
        for row, col, data in zip(GGDFmat.row, GGDFmat.col, GGDFmat.data):
            # Get the sign of F_ik row
            sFik = np__sign(F_ik[row, 0])
            # Get the sign of GGDFmat data
            sData = np__sign(data)
            if sFik * sData < 0:
                # only if decreases set to zero, otherwise keeps same value
                GGDFmat_cf[row, col] = 0
            else:
                GGDFmat_cf[row, col] = data
        GGDFmat = GGDFmat_cf.tocsr()

    # En caso de haberse modificado GGDFmat con cf=True,
    # entonces PTG es PTG_cf
    # Calcula la "Participación Total de Generacion" por rama
    PTG = (GGDFmat @ Gmat).toarray()  # matriz ya es bastante densa para usarla como ndarray
    # Amplia vector columna PTG al número de columnas de FUPT (Ngen)
    PTG_mat = np__repeat(PTG, GenBus_mat.shape[1], axis = 1)  # recordar GenBus_mat (branch x Ngen)
    # Calcula los "Factores de Utilización por Tramo de generación"
    FUPT = (GGDFmat @ GenBus_mat).toarray() / PTG_mat
    # FUPT = np__round(FUPT, decs)  # approximate values to avoid numerical errors
    return FUPT


def sum_rows(A, rows = 'all'):
    """ sumation along the rows (add all elements of the columns for each row independently).
        Similar to A.sum(axis=1) but return csr_matrix. Used to avoid pytest deprecated warning.

        A must be csr_matrix to be efficient.

        rows = 'all':
            Adds all rows resulting in a 1D matrix (repsented as 2D sparse matrix).
        rows = list()
            Indices of rows to sum.
    """
    # about A
    NRows_A, NCols_A = A.shape
    indxrow_A, indxcol_A = A.nonzero()
    # about Aux
    NCols_aux = 1
    NRows_aux = NCols_A
    # drop duplicates of indxcol_A to set ones on Aux (to not multiply > 1)
    row_indx_aux = np__unique(indxcol_A)
    col_indx_aux = [0] * len(row_indx_aux)
    data = [1] * len(indxcol_A)
    Aux = sps__csr_matrix( (data, (row_indx_aux, col_indx_aux)), (NRows_aux, NCols_aux) )
    return A * Aux


def power_over_congestion(Grid, TypeElmnt, BranchIndTable, max_load_percent):
    """
        Calculates delta power over from max power expected under the max_load_percent.
        Increases OverloadP_kW in 12% over max power branch limit.

        Returns (OverloadP_kW, loading_percent)
    """
    if TypeElmnt == 'line':
        ColName = 'p_from_kw'
    elif TypeElmnt == 'trafo':
        ColName = 'p_hv_kw'
    else:
        msg = "No TypeElmnt={} supported.".format(TypeElmnt)
        raise ValueError(msg)
    # FLUJO DE LINEA CAMBIA SIGNO. Absolute values required
    loading_percent = Grid['res_' + TypeElmnt].at[BranchIndTable, 'loading_percent']
    FluPAB_kW = abs(Grid['res_' + TypeElmnt].at[BranchIndTable, ColName])  # da igual dirección (signo)
    OverloadP_kW = FluPAB_kW * (1 - max_load_percent / loading_percent)  # >= 0
    return OverloadP_kW, loading_percent


def name_of_elmnt_row(Grid, idxNan):
    """
        Returns the name of row of from ppc elements of idxNan
        Here row means branch.
    """
    DictIndices = ind_of_branch_elmnt(Grid)
    for key, value in DictIndices.items():
        if idxNan in value:
            TypeElmnt, pdIndx = key, np__argwhere(value == idxNan)[0, 0]
            break
    else:
        raise NameError("No idxNan: '{}' found in Grid".format(idxNan))
    Name = Grid[TypeElmnt].at[pdIndx, 'name']
    return Name
