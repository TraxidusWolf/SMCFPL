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

# custom module
try:    # try importing locally when run independently
    from smcfpl_exceptions import *
except Exception:
    from smcfpl.smcfpl_exceptions import *

import logging
logging.basicConfig(level=logging.DEBUG,
                    format="[%(levelname)s][%(asctime)s][%(filename)s:%(funcName)s] - %(message)s")
logger = logging.getLogger()


def redispatch(Grid, TypeElmnt, BranchIndTable, max_load_percent=100):
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
        un 5% del flujo permitido - por aproximación de FPL) de la unidad que más aporta al flujo
        según los FUPTG a la que menos aporta (o en su defecto realiza más flujo contrario).

        En caso de no poder transferir generación a dicha unidad, la potencia se va repartiendo
        consecutivamente a aquellas que aportan a la descongestión.

        Para la situación en la que la unidad de destino 'G_to' sea la referencia (Red externa),
        entonces solo se disminuye la potencia del generador 'G_from' ya que la referencia aceptará
        el cambio. En caso contrario, se aumenta la generación del 'G_from' para lograr mismo efecto.

        Realiza un proceso iterativo hasta que la POTENCIA de la rama 'BranchIndTable' es igual o
        menor al límite.

        Return None (inplace modification)

        Syntax:
            redispatch(Grid, TypeElmnt, BranchIndTable, max_load_percent=100)

        Example:
            redispatch(Grid, 'line', 0, max_load_percent = 2.8)
    """
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
    GSDFmat = calc_GSDF_factor(Bbus, Bpr, Amat, RefBusMatIndx, CondMat=True, decs=14)
    # Calculate GGDF and Generation Utilizatión matrix factors (uses aproximation for really small numbers)
    GGDFmat, FUPTG, GenIndx2BusIndx, IndGenRef_GBusMat = calc_GGDF_Factor(Grid,
                                                                          GSDFmat,
                                                                          RefBus,
                                                                          decs=14,
                                                                          FUTP_cf=False)  # no correction flow

    # Get the row of the branch of interest from matrices (according to IndxBrnchElmnts and input argument)
    TheRowBrnchIndMat = IndxBrnchElmnts[TypeElmnt][BranchIndTable]  # asume elementos de tabla ordenados secuencialmente según tablas Grid
    ### (Por ahora) Uses FUPTG to determine the most and least participation generators on line flow
    # Obtiene el máximo y el mínimo de fila de la rama en cuestión
    RowValues = FUPTG[TheRowBrnchIndMat, :]

    # identifica valores del más al menos influyente en valor absoluto
    IndxSorted = abs(RowValues).argsort()[::-1]  # [::-1] is reversed

    # calcula la potencia sobrante
    OverloadP_kW, loading_percent = power_over_congestion(Grid,
                                                          TypeElmnt,
                                                          BranchIndTable,
                                                          max_load_percent)
    if (OverloadP_kW <= 0) :  # & (loading_percent < max_load_percent) ?
        # here catch negative o zero value "DiffPower"
        msg = "Branch '{}' wasn't really congested ({:.4f} %)".format(TypeElmnt, loading_percent)
        logger.error(msg)
        raise FalseCongestion(msg)

    # Get all generator indices from Grid.gen to get their power limits
    Indices_GridGen = Grid['gen'].index
    count = 0  # requiere to call generators and skip the reference one
    for ind in IndxSorted:
        # iterates over decreasing (absolute) ordered FUPTG of branch
        ValInd = RowValues[ind]  # value of FUPTG associated
        # jumps if reaches reference generators
        if ind == IndGenRef_GBusMat:
            # nothing to modify here (Reference moves alone)
            continue

        IndxGenGrid = Indices_GridGen[count]  # generator index of pandas table
        # determina si debe disminuir la potencia o aumentar en el generador
        if ValInd >= 0:  # This generator should reduce it's power (FUPTG >= 0)
            LimitekW = abs(Grid['gen'].at[IndxGenGrid, 'min_p_kw'])
            CurrentPowerkW = abs(Grid['gen'].at[IndxGenGrid, 'p_kw'])
            # calculates delta power available to reduce
            DeltaPAvail_kW = CurrentPowerkW - LimitekW
            # calculate new power to give to generator
            NewPowerkW = DeltaPAvail_kW - OverloadP_kW

        else:   # This generator should increase it's power (FUPTG < 0)
            LimitekW = abs(Grid['gen'].at[IndxGenGrid, 'max_p_kw'])
            CurrentPowerkW = abs(Grid['gen'].at[IndxGenGrid, 'p_kw'])
            # calculates delta power available to increase
            DeltaPAvail_kW = LimitekW - CurrentPowerkW
            # calculate new power to give to generator
            NewPowerkW = DeltaPAvail_kW - OverloadP_kW

        if DeltaPAvail_kW <= 0 :  # is it possible?
            # no power available to reduce in this generator
            continue
        elif NewPowerkW < 0:
            # generator should reduce more than available
            OverloadP_kW -= DeltaPAvail_kW
            NewPowerkW = LimitekW
        else:  # NewPowerkW >= 0
            OverloadP_kW -= DeltaPAvail_kW
        Grid['gen'].at[IndxGenGrid, 'p_kw'] = -NewPowerkW

        # determina condición de salida. O que se acaben los generadores
        if OverloadP_kW <= 0:
            break
        count += 1  # helps keep track of Indices_GridGen
    else:  # if not break for loop
        if OverloadP_kW > 0:
            msg = "Not enought power available on generators to stop congestion."
            logger.error(msg)
            raise CapacityOverloaded(msg)


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
    if np__isnan(FUPTG).any():
        import numpy as np
        np.set_printoptions(linewidth=20000, threshold=2000)
        print("GGDF.A:\n", GGDF.A)
        print("GGDF.shape:\n", GGDF.shape)
        print("Gmat.A:\n", Gmat.A)
        print("Gmat.shape:\n", Gmat.shape)
        print("Grid._pd2ppc_lookups:", Grid._pd2ppc_lookups)

        print("Grid:\n", Grid)
        print("Grid.bus:\n", Grid.bus)
        # print("Grid:\n", Grid)
        import pdb; pdb.set_trace()  # breakpoint 81a312e5 //
    return GGDF, FUPTG, GenIndx2BusIndx, IndGenRef_GBusMat


def calc_FUTP_gen(GGDFmat, GenBus_mat, Gmat, F_ik, cf=True, decs = 10):
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
    PTG_mat = np__repeat(PTG, GenBus_mat.shape[1], axis= 1)  # recordar GenBus_mat (branch x Ngen)
    # Calcula los "Factores de Utilización por Tramo de generación"
    print("PTG_mat:\n", PTG_mat)
    FUPT = (GGDFmat @ GenBus_mat).toarray() / PTG_mat
    FUPT = np__round(FUPT, decs)  # approximate values to avoid numerical errors
    print("FUPT:\n", FUPT)
    return FUPT


def sum_rows(A, rows='all'):
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
    """ Calculates delta power over from max power expected under the max_load_percent.
        Increases OverloadP_kW in 5% over max power branch limit.
    Returns (OverloadP_kW, loading_percent)
    """
    # FLUJO DE LINEA CAMBIA SIGNO. Absolute values requiered
    loading_percent = Grid['res_' + TypeElmnt].at[BranchIndTable, 'loading_percent']
    FluPAB_kW = abs(Grid['res_' + TypeElmnt].at[BranchIndTable, 'p_from_kw'])  # da igual dirección (signo)
    MaxFluP_kW = FluPAB_kW * max_load_percent / loading_percent  # >= 0
    OverloadP_kW = FluPAB_kW - MaxFluP_kW
    OverloadP_kW += MaxFluP_kW * (0.05)  # increase 5% to compensate fpl error
    return OverloadP_kW, loading_percent


if __name__ == '__main__':
    # directly import conftest.py file (test only)
    import pandapower as pp
    from sys import path as sys__path
    sys__path.insert(0, "test/")

    # import test case
    from conftest import Sist5Bus
    Grid = Sist5Bus()
    redispatch(Grid, 'line', 0, max_load_percent = 0.08)
    # redispatch(Grid, 'line', 0, max_load_percent = 2.8)
    pp.rundcpp(Grid)
