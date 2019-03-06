import pytest
from scipy.sparse import csr_matrix
from pandapower import rundcpp as pp__rundcpp
from numpy import array as np__array
####### temporal #####
from sys import path as sys__path
sys__path.insert(0, "../..")
######################
from smcfpl.Redespacho import *


def run_all_tests():
    """Executes all test within this file.
    Thought to be used when called directly.
    """
    pytest.main( [] )


def test_make_Bbus_Bpr_A(Sist5Bus, Mat_Sist5bus):
    """ test normal output values """
    net = Sist5Bus
    Bbus, Bpr, Cft, GenBus_mat, Gmat, F_ik = Mat_Sist5bus
    pp__rundcpp(net)
    Bbus_calc, Bpr_calc, Cft_calc = make_Bbus_Bpr_A(net)
    # tests for matrix inequality size (!= is faster ans makes simpler to compare sizes)
    assert (Bbus != Bbus_calc).nnz == 0
    assert (Bpr != Bpr_calc).nnz == 0
    assert (Cft != Cft_calc).nnz == 0


def test_IndOfBranchElmnt(Sist5Bus):
    """ tests for _pd2pcc_lookups according to elements of grid"""
    DictElmnts = IndOfBranchElmnt(Sist5Bus)
    assert 'line' in DictElmnts
    assert 'trafo' not in DictElmnts
    assert 'trafo3w' not in DictElmnts
    assert ( DictElmnts['line'] == np__array([0, 1, 2, 3, 4, 5, 6]) ).all()


def test_Calc_Factor_GSDF(Mat_Sist5bus, Factores_Distribucion_Sist5Bus):
    """ tests for normal expected output """
    Bbus, Bpr, Cft, GenBus_mat, Gmat, F_ik = Mat_Sist5bus
    GSDF, GGDF, FUPTG = Factores_Distribucion_Sist5Bus
    assert (Calc_Factor_GSDF(Bbus, Bpr, Cft, 1, CondMat=False, decs=14) != GSDF).nnz == 0
    assert (Calc_Factor_GSDF(Bbus, Bpr, Cft, 1, CondMat=True, decs=14) != GSDF).nnz == 0
    with pytest.raises(ValueError):
        assert Calc_Factor_GSDF(Bbus, Bpr, Cft, (1,), decs=14)
        assert Calc_Factor_GSDF(Bbus, Bpr, Cft, [1, ], decs=14)


def test_Calc_Factor_GGDF(Sist5Bus, Mat_Sist5bus, Factores_Distribucion_Sist5Bus):
    """ tests for normal expected output """
    GSDF, GGDF, FUPTG = Factores_Distribucion_Sist5Bus
    GGDF_calc, FUPTG_calc, GenIndx2BusIndx_calc, IndGenRef_GBusMat_calc = Calc_Factor_GGDF(Sist5Bus,
                                                                                           GSDF,
                                                                                           1,
                                                                                           decs=14,
                                                                                           FUTP_cf=False)
    assert (GGDF != GGDF_calc).nnz == 0
    assert (FUPTG == FUPTG_calc).all()
    assert GenIndx2BusIndx_calc == [0, 1, 4]  # G1 @ Bus=0, G2 @ Bus=1, G3 @ Bus=4
    assert IndGenRef_GBusMat_calc == 1  # G2 represents external grid


def test_Calc_FUTP_gen(Mat_Sist5bus, Factores_Distribucion_Sist5Bus):
    """ tests for normal expected output """
    Bbus, Bpr, Cft, GenBus_mat, Gmat, F_ik = Mat_Sist5bus
    GSDF, GGDF, FUPTG = Factores_Distribucion_Sist5Bus
    FUPTG_calc = Calc_FUTP_gen(GGDF, GenBus_mat, Gmat, F_ik, cf=False, decs=14)
    print(FUPTG_calc)
    print(FUPTG)
    assert (FUPTG == FUPTG_calc).all()


if __name__ == '__main__':
    run_all_tests()
    # net = Sist5Bus()
    # pp__rundcpp(net)
