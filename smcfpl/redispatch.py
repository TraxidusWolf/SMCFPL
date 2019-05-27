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
from numpy import copy as np__copy
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


def redispatch(Grid, TypeElmnt, BranchIndTable, max_load_percent=100, decs=14):
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
            redispatch(Grid, TypeElmnt, BranchIndTable, max_load_percent=100)

        Example:
            redispatch(Grid, 'line', 0, max_load_percent = 2.8)
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
    GGDFmat, FUPTG, GenIndx2BusIndx, IndGenRef_GBusMat = calc_GGDF_Factor(Grid,
                                                                          GSDFmat,
                                                                          RefBus,
                                                                          decs=decs,
                                                                          FUTP_cf=False)  # no correction flow
    # Find the Non ERNC unit, so they can not change power dispatched
    ListaERNC = ['Solar', 'EólicaZ1', 'EólicaZ2', 'EólicaZ3', 'EólicaZ4']  # hard coded, MUST CHANGE!
    FUPTG_GenGridNonERNCIndx, NonERNCIndx_GenFUPTG = gen_grid_indx_Non_ERNC_from_FUPTG( Grid.gen,
                                                                                        ListaERNC,
                                                                                        IndGenRef_GBusMat,
                                                                                        FUPTG.shape[1])
    # FUPTG_GenGridIndices = gen_grid_indx_from_FUPTG(Grid.gen, IndGenRef_GBusMat, FUPTG.shape[1])
    # Get the row of the branch of interest from matrices (according to IndxBrnchElmnts and input argument)
    TheRowBrnchIndMat = IndxBrnchElmnts[TypeElmnt][BranchIndTable]  # asume elementos de tabla ordenados secuencialmente según tablas Grid
    ### (Por ahora) Uses FUPTG to determine the most and least participation generators on line flow
    # Obtiene el máximo y el mínimo de fila de la rama en cuestión
    RowValues = FUPTG[TheRowBrnchIndMat, :]
    # print("IndGenRef_GBusMat:", IndGenRef_GBusMat)
    # print("FUPTG_GenGridNonERNCIndx:", FUPTG_GenGridNonERNCIndx)
    # print("FUPTG_GenGridIndices:", FUPTG_GenGridIndices)
    print("RowValues:", RowValues)

    # calcula la potencia sobrante
    OverloadP_kW, loading_percent = power_over_congestion( Grid,
                                                           TypeElmnt,
                                                           BranchIndTable,
                                                           max_load_percent)
    print("OverloadP_kW:", OverloadP_kW)
    print("loading_percent:", loading_percent)
    if (OverloadP_kW <= 0) :  # & (loading_percent < max_load_percent) ?
        # here catch negative o zero value "DiffPower"
        msg = "Branch '{}' wasn't really congested ({:.4f} %)".format(TypeElmnt, loading_percent)
        logger.error(msg)
        raise FalseCongestion(msg)

    # normalization about sumation of absolute values
    RowValuesNorm = RowValues / abs(RowValues).sum()  # abs(RowValuesNorm).sum() == 1
    print("RowValuesNorm:", RowValuesNorm)

    # creates vector of new power value expected to generators
    Power2Change_kW = -RowValuesNorm[FUPTG_GenGridNonERNCIndx] * OverloadP_kW
    print("Power2Change_kW:", Power2Change_kW)

    # Finds the limits and actual power for units
    MaxGen_kW = abs(Grid['gen'].loc[NonERNCIndx_GenFUPTG, 'max_p_kw'].values)
    MinGen_kW = abs(Grid['gen'].loc[NonERNCIndx_GenFUPTG, 'min_p_kw'].values)
    ActualGen_kW = abs(Grid['res_gen'].loc[NonERNCIndx_GenFUPTG, 'p_kw'].values)
    print("MaxGen_kW:", MaxGen_kW)
    print("MinGen_kW:", MinGen_kW)
    print("ActualGen_kW:", ActualGen_kW)

    # Calculates new power to dispatch
    NewPower_kW = ActualGen_kW + Power2Change_kW

    # Do redispatch at least once
    PossibleRedispath = True
    # MoreGenerationPossible = LessGenerationPossible = True
    while PossibleRedispath:
        """ Loop required when maximum or minimum power of units are reached.
            Three possible exit conditions:
            1.- Not enough generation available to redispatch at maximum power.
            2.- Not enough generation available to redispatch at minimum power.
            3.- Redispatch worked successfully.

            In order to each, it's required:
                1.- AvailablePower2Increase.sum() < 0: NewPower expected to dispatch is greater than MaxPower
                2.- AvailablePower2Decrease.sum() < 0: NewPower expected to dispatch is smaller than MinPower
                3.- (not SomeGenOverloaded) and (not SomeGenUnderloaded): No generation limits were reached
            Note, while loop will continue as long there was a limit reached.
        """
        # CHECKS IF POWERS LIMITS ARE REACHED for each generator. Assumes same order from FUPTG columns
        # print("NewPower_kW:", NewPower_kW)
        SomeGenOverloaded = any( NewPower_kW > MaxGen_kW )
        SomeGenUnderloaded = any( MinGen_kW > NewPower_kW )
        if SomeGenOverloaded:
            msg = "Some generators got overloaded (SomeGenOverloaded: {}).".format(SomeGenOverloaded)
            logger.warn(msg)
            # finds if available power can be increased on generators, if not calc new powers
            AvailablePower2Increase = MaxGen_kW - NewPower_kW
            AvailablePower2Increase[AvailablePower2Increase < 0] = 0  # clean those over the maximum
            # if no more possible set LessGenerationPossible = False
            if AvailablePower2Increase.sum() < 0:
                msg = "No more generation can be increases."
                logger.warn(msg)
                # LessGenerationPossible = False
                PossibleRedispath = False
                break  # break while
            # calculate power left
            DeltaP_kW = NewPower_kW - MaxGen_kW
            DeltaP_kW[DeltaP_kW < 0] = 0
            DeltaP_kW = DeltaP_kW.sum()
            # get inidices of limited units
            IndxLimitedUnits = NewPower_kW > MaxGen_kW
            IndxNonLimitedUnits = NewPower_kW <= MaxGen_kW
            # fix to limit values
            NewPower_kW[IndxLimitedUnits] = MaxGen_kW[IndxLimitedUnits]
            # calculates weights for others generators to split the OverloadP_kW
            NewWeights = RowValuesNorm[FUPTG_GenGridNonERNCIndx][IndxNonLimitedUnits]
            # split power acorss other generators accordingly to FUPTG factors. Preserves the sign.
            NewPower_kW[IndxNonLimitedUnits] += NewWeights / abs(NewWeights).sum() * DeltaP_kW
        if SomeGenUnderloaded:
            msg = "Some generators got underloaded (SomeGenUnderloaded: {}).".format(SomeGenUnderloaded)
            logger.warn(msg)
            # finds if available power can be reduced on generators, if not calc new powers
            AvailablePower2Decrease = NewPower_kW - MinGen_kW
            AvailablePower2Decrease[AvailablePower2Decrease < 0] = 0  # clean those below the minimum
            # if no more possible set LessGenerationPossible = False
            if AvailablePower2Decrease.sum() < 0:
                msg = "No less generation can be decrease."
                logger.warn(msg)
                # MoreGenerationPossible = False
                PossibleRedispath = False
                break  # break while
            # calculate power left
            DeltaP_kW = MinGen_kW - NewPower_kW
            DeltaP_kW[DeltaP_kW < 0] = 0
            DeltaP_kW = DeltaP_kW.sum()
            # get inidices of limited units
            IndxLimitedUnits = MinGen_kW > NewPower_kW
            IndxNonLimitedUnits = MinGen_kW <= NewPower_kW
            # fix to limit values
            NewPower_kW[IndxLimitedUnits] = MinGen_kW[IndxLimitedUnits]
            # calculates weights for others generators to split the OverloadP_kW
            NewWeights = RowValuesNorm[FUPTG_GenGridNonERNCIndx][IndxNonLimitedUnits]
            # split power acorss other generators accordingly to FUPTG factors. Preserves the sign.
            NewPower_kW[IndxNonLimitedUnits] += NewWeights / abs(NewWeights).sum() * DeltaP_kW
        if (not SomeGenOverloaded) and (not SomeGenUnderloaded):
            # Exit condition for redispatch complete
            print("Exit condition for redispatch complete")
            break
        import pdb; pdb.set_trace()  # breakpoint 44f1fdce //

    if not PossibleRedispath:
        msg = "Not enough power available on generators to stop 'inter-congestion'."
        logger.error(msg)
        raise CapacityOverloaded(msg)

    # Finally assign the new power to Grid generators
    Grid['gen'].loc[NonERNCIndx_GenFUPTG, 'p_kw'] = -NewPower_kW


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
    GSDFmat.data = np__round(GSDFmat.data, decs)  # approximate values to avoid numerical errors
    return GSDFmat


def calc_GGDF_Factor(Grid, GSDF, RefBus, decs=10, FUTP_cf=True):
    """
        Calculates Generalized Generation Distribution Factors (GGDFs).

        There must exists only one ext_grid within Grid. If more, user should
        convert them to conventional generators.

        Arguments:
        Grid (pandapower net) with at least one external grid, a generator and one bus. (able to run power flow)

        GSDF: sparse csr_matrixwith GSDF factors.

        It should consider same reference bus as for GSDF matrix.
        RefBus: Integer index of bus reference (external grid bus.)

        decs: integer number to approximate values of returned matrix.

        FUTP_cf: booblean flag to correct the flow given by GGDF.

    """
    _ppc = Grid._ppc
    # Number of nodes, branches and generator
    NNodes = _ppc['bus'].shape[0]
    Nbranches = _ppc['branch'].shape[0]
    Ngen = _ppc['gen'][:, PG].shape[0]
    # Creates vector of power flows (Branch x 1)
    F_ik = np__real(_ppc['branch'][:, [PF]])  # power [MW] inyected from 'from_bus'
    F_ik = sps__csr_matrix(F_ik)

    # initialize list to relation generator index to bus index
    GenIndx2BusIndx = []  # each value represents the bus index (Grid table) of the generator as list index (for GenBus_mat)
    IndGenRef_GBusMat = -1  # represents the index of GenIndx2BusIndx, to know which generator is the reference one on GenBus_mat.
    NumGen = 0
    # initialize data and matrix indices for Generator power per bus matrix (GenBus_mat)
    data, idx, jdx = [], [], []
    for BusInd, PGen_MW, Qmax, Qmin in _ppc['gen'][:, [GEN_BUS, PG, QMAX, QMIN]]:  # GEN_BUS indices of Grid.bus (same as BUS_I)
        BusInd = int(BusInd)
        PGen_MW = float(PGen_MW)

        idx.append(BusInd)  # must start from zero (0-indexed)
        jdx.append(NumGen)
        data.append(PGen_MW)
        GenIndx2BusIndx.append(BusInd)
        if (IndGenRef_GBusMat == -1) & (Qmax == 0) & (Qmin == 0):
            # external grid do no have parameter to reactive power limits
            IndGenRef_GBusMat = NumGen
        NumGen += 1  # Orden creciente de generadores
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
    GGDF.data = np__round(GGDF.data, decs)  # approximate values to avoid numerical errors

    # Calculates "Factores de Utilización por Tramo de Generación"
    FUPTG = calc_FUTP_gen(GGDF, GenBus_mat, Gmat, F_ik, FUTP_cf, decs = decs)
    # if np__isnan(FUPTG).any():
    #     import numpy as np
    #     np.set_printoptions(linewidth=20000, threshold=2000)
    #     # get the row with nan row
    #     idxNan = np.unique( np.where( np.isnan(FUPTG) )[0] )[0]
    #     print("idxNan:\n", idxNan)
    #     # get to know name of element asociated to row
    #     NameRowElmnt = name_of_elmnt_row(Grid, idxNan)
    #     print("NameRowElmnt:\n", NameRowElmnt)
    #     # find the column index of only one value = -1
    #     NameColElmntNegative = ''
    #     print("NameColElmntNegative:\n", NameColElmntNegative)

    #     print("FUPTG:\n", FUPTG)
    #     print("FUPTG.shape:\n", FUPTG.shape)
    #     print("GGDF.A:\n", GGDF.A)
    #     print("GGDF.shape:\n", GGDF.shape)
    #     print("GenBus_mat.A:\n", GenBus_mat.A)
    #     print("GenBus_mat.shape:\n", GenBus_mat.shape)
    #     print("Gmat.A:\n", Gmat.A)
    #     print("Gmat.shape:\n", Gmat.shape)
    #     print("Grid._pd2ppc_lookups:", Grid._pd2ppc_lookups)

    #     print("Grid:\n", Grid)
    return GGDF, FUPTG, GenIndx2BusIndx, IndGenRef_GBusMat


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
    FUPT = np__round(FUPT, decs)  # approximate values to avoid numerical errors
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


def gen_grid_indx_from_FUPTG(Grid_gen_tbl, IndGenRef_GBusMat, NumColumnsFUPTG):
    """
        Makes the assumtion to have all Grid_gen_tbl 0-indexed ordered. Note taht 'IndGenRef_GBusMat'
        is zero-idexed as well as 'range(NumColumnsFUPTG)'.

        Inputs:
            **Grid_gen_tbl**: pandas table of grid
            **IndGenRef_GBusMat**: (0-indexed)
            **NumColumnsFUPTG**: FUPTG.shape[1]

        Returns a list of the grid generator indices.
    """
    # verifies that Grid_gen_tbl index length is the same as NumColumnsFUPTG
    if len(Grid_gen_tbl.index) != (NumColumnsFUPTG - 1):
        msg = "Grid_gen_tbl is not the same as NumColumnsFUPTG"
        logger.error(msg)
        raise ValueError(msg)
    lista = []
    counter = 0
    for i in range(NumColumnsFUPTG):
        if i != IndGenRef_GBusMat:
            lista.append(counter)
        counter += 1
    return lista


def gen_grid_indx_Non_ERNC_from_FUPTG(Grid_gen_tbl, ListaERNC, IndGenRef_GBusMat, NumColumnsFUPTG):
    """
        Makes the assumtion to have all Grid_gen_tbl 0-indexed ordered.

        Returns a list of the grid generator indices and the pandas index dataframe NonERNC_GridGen.
    """
    if Grid_gen_tbl.shape[0] != (NumColumnsFUPTG - 1):
        msg = "Grid_gen_tbl num rows is not the same as NumColumnsFUPTG"
        logger.error(msg)
        raise ValueError(msg)
    NonERNC_GridGen = Grid_gen_tbl[ ~Grid_gen_tbl['type'].isin(ListaERNC) ].index
    ListReturned = []
    Counter = 0
    for i in range(NumColumnsFUPTG):
        if i in NonERNC_GridGen:
            ListReturned.append(i)
            Counter += 1
    return (ListReturned, NonERNC_GridGen)


if __name__ == '__main__':
    # directly import conftest.py file (test only)
    import pandapower as pp
    from sys import path as sys__path
    sys__path.insert(0, "test/")
    sys__path.insert(0, '..')
    from smcfpl.aux_funcs import TipoCong

    LoadPercent = 1.73  # 1.5
    # import test case
    from conftest import Sist5Bus
    Grid = Sist5Bus()
    pp.rundcpp(Grid)
    Inter, Intra = TipoCong(Grid, max_load=LoadPercent)
    print("Inter:", Inter); print("Intra:", Intra)
    print(Grid.res_line)
    print(Grid.res_gen)
    print(Grid.res_ext_grid)
    print()

    print("--- 1st Redispatch to line: 0")
    redispatch(Grid, 'line', 0, max_load_percent=LoadPercent, decs=30)  # hard codede
    pp.rundcpp(Grid)
    Inter, Intra = TipoCong(Grid, max_load=LoadPercent)
    print("Inter:", Inter); print("Intra:", Intra)
    print(Grid.res_line)
    print(Grid.res_gen)
    print(Grid.res_ext_grid)
    # print()

    # print("--- 2st Redispatch to line: 4")
    # redispatch(Grid, 'line', 4, max_load_percent=LoadPercent, decs=30)  # hard codede
    # pp.rundcpp(Grid)
    # Inter, Intra = TipoCong(Grid, max_load=LoadPercent)
    # print("Inter:", Inter); print("Intra:", Intra)
    # print(Grid.res_line)
    # print()

    # print("--- 3st Redispatch to line: 4")
    # redispatch(Grid, 'line', 4, max_load_percent=LoadPercent, decs=30)  # hard codede
    # pp.rundcpp(Grid)
    # Inter, Intra = TipoCong(Grid, max_load=LoadPercent)
    # print("Inter:", Inter); print("Intra:", Intra)
    # print(Grid.res_line)
    # print()
