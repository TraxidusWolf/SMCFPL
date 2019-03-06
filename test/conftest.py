import pytest
import pandapower as pp
from scipy.sparse import csr_matrix
from numpy import array


@pytest.fixture(scope="session")
def Factores_Distribucion_Sist5Bus():
    GSDF = csr_matrix([
        [ 0.84285714285714,  0.,                0.21428571428571,  0.17142857142857,   0.05714285714286],
        [ 0.15714285714286,  0.,               -0.21428571428571, -0.17142857142857,  -0.05714285714286],
        [-0.07142857142857,  0.,               -0.35714285714286, -0.28571428571429,  -0.0952380952381 ],
        [-0.05714285714286,  0.,               -0.28571428571429, -0.36190476190476,  -0.12063492063492],
        [-0.02857142857143,  0.,               -0.14285714285714, -0.18095238095238,  -0.72698412698413],
        [ 0.08571428571429,  0.,                0.42857142857143, -0.45714285714286,  -0.15238095238095],
        [ 0.02857142857143,  0.,                0.14285714285714,  0.18095238095238,  -0.27301587301587],
    ])
    GGDF = csr_matrix([
        [ 0.71772805507745, -0.12512908777969,  0.08915662650602,  0.04629948364888,  -0.06798623063683],
        [ 0.27624784853701,  0.11910499139415, -0.09518072289156, -0.05232358003442,   0.06196213425129],
        [ 0.12908777969019,  0.20051635111876, -0.1566265060241,  -0.08519793459553,   0.10527825588066],
        [ 0.15146299483649,  0.20860585197935, -0.07710843373494, -0.15329890992541,   0.08797093134443],
        [ 0.31669535283993,  0.34526678141136,  0.20240963855422,  0.16431440045898,  -0.38171734557277],
        [ 0.1342512908778,   0.04853700516351,  0.47710843373494, -0.40860585197935,  -0.10384394721744],
        [ 0.0447504302926,   0.01617900172117,  0.15903614457831,  0.19713138267355,  -0.2568368712947 ],
    ])
    FUPTG = array([
        [ 1.18061247457224, -0.1322448765807,  -0.04836759799154],
        [ 0.72778883694356,  0.20160884410808,  0.07060231894835],
        [ 0.42539715551103,  0.42455345115259,  0.15004939333638],
        [ 0.46814348597845,  0.41425910677987,  0.11759740724169],
        [ 0.84805675425226,  0.59403379782706, -0.44209055207932],
        [ 1.11389850194932,  0.25874576932779, -0.37264427127711],
        [-0.80002188094038, -0.18583585192074,  1.98585773286112],
    ])
    return GSDF, GGDF, FUPTG


@pytest.fixture(scope="session")
def Mat_Sist5bus():
    Bbus = csr_matrix([
        [1 / 0.06 + 1 / 0.24 , -1 / 0.06                                 , -1 / 0.24                      , 0                              , 0                   ] ,
        [-1 / 0.06           , 1 / 0.06 + 1 / 0.18 + 1 / 0.18 + 1 / 0.12 , -1 / 0.18                      , -1 / 0.18                      , -1 / 0.12           ] ,
        [-1 / 0.24          , -1 / 0.18                                 , 1 / 0.24 + 1 / 0.18 + 1 / 0.03 , -1 / 0.03                      , 0                   ] ,
        [0                   , -1 / 0.18                                 , -1 / 0.03                      , 1 / 0.18 + 1 / 0.03 + 1 / 0.24 , -1 / 0.24           ] ,
        [0                   , -1 / 0.12                                 , 0                              , -1 / 0.24                      , 1 / 0.12 + 1 / 0.24 ] ,
    ] )
    Bpr = csr_matrix([
        [1 / 0.06 , 0        , 0        , 0        , 0        , 0        , 0        ] ,
        [0        , 1 / 0.24 , 0        , 0        , 0        , 0        , 0        ] ,
        [0        , 0        , 1 / 0.18 , 0        , 0        , 0        , 0        ] ,
        [0        , 0        , 0        , 1 / 0.18 , 0        , 0        , 0        ] ,
        [0        , 0        , 0        , 0        , 1 / 0.12 , 0        , 0        ] ,
        [0        , 0        , 0        , 0        , 0        , 1 / 0.03 , 0        ] ,
        [0        , 0        , 0        , 0        , 0        , 0        , 1 / 0.24 ] ,
    ] )
    Cft = csr_matrix([
        [1 , -1 , 0  , 0  , 0  ] ,
        [1 , 0  , -1 , 0  , 0  ] ,
        [0 , 1  , -1 , 0  , 0  ] ,
        [0 , 1  , 0  , -1 , 0  ] ,
        [0 , 1  , 0  , 0  , -1 ] ,
        [0 , 0  , 1  , -1 , 0  ] ,
        [0 , 0  , 0  , 1  , -1 ] ,
    ] )
    GenBus_mat = csr_matrix([
        [80.,   0. ,  0. ],
        [ 0.,  51.40000000000003,  0. ],
        [ 0.,   0. ,  0. ],
        [ 0.,   0. ,  0. ],
        [ 0.,   0. , 34.6],
    ])
    Gmat = csr_matrix([
        [80. ],
        [51.40000000000003],
        [ 0. ],
        [ 0. ],
        [34.6],
    ])
    F_ik = csr_matrix([
        [48.6342857142857 ],
        [30.36571428571429],
        [24.27619047619049],
        [25.88317460317461],
        [29.87492063492065],
        [ 9.64190476190475],
        [-4.47492063492064],
    ])
    return Bbus, Bpr, Cft, GenBus_mat, Gmat, F_ik


@pytest.fixture(scope="session")
def Sist5Bus():
    # Sbase = 1 [MW] to match p.u. values
    net = pp.create_empty_network(name="5bus", f_hz=50, sn_kva=1e3)
    # barras
    pp.create_buses(net, nr_buses=5, vn_kv=1)
    # lineas
    pp.create_line_from_parameters(net, from_bus=0, to_bus=1, length_km=1, r_ohm_per_km=0.02, x_ohm_per_km=0.06, c_nf_per_km=0, max_i_ka=1000)
    pp.create_line_from_parameters(net, from_bus=0, to_bus=2, length_km=1, r_ohm_per_km=0.08, x_ohm_per_km=0.24, c_nf_per_km=0, max_i_ka=1000)
    pp.create_line_from_parameters(net, from_bus=1, to_bus=2, length_km=1, r_ohm_per_km=0.06, x_ohm_per_km=0.18, c_nf_per_km=0, max_i_ka=1000)
    pp.create_line_from_parameters(net, from_bus=1, to_bus=3, length_km=1, r_ohm_per_km=0.06, x_ohm_per_km=0.18, c_nf_per_km=0, max_i_ka=1000)
    pp.create_line_from_parameters(net, from_bus=1, to_bus=4, length_km=1, r_ohm_per_km=0.04, x_ohm_per_km=0.12, c_nf_per_km=0, max_i_ka=1000)
    pp.create_line_from_parameters(net, from_bus=2, to_bus=3, length_km=1, r_ohm_per_km=0.01, x_ohm_per_km=0.03, c_nf_per_km=0, max_i_ka=1000)
    pp.create_line_from_parameters(net, from_bus=3, to_bus=4, length_km=1, r_ohm_per_km=0.08, x_ohm_per_km=0.24, c_nf_per_km=0, max_i_ka=1000)
    # cargas
    pp.create_load(net, bus=0, p_kw=1e3)
    pp.create_load(net, bus=1, p_kw=20e3)
    pp.create_load(net, bus=2, p_kw=45e3)
    pp.create_load(net, bus=3, p_kw=40e3)
    pp.create_load(net, bus=4, p_kw=60e3)
    # gen
    pp.create_gen(net, bus=0, p_kw=-80e3, max_p_kw = -100e3, min_p_kw = 0)
    pp.create_ext_grid(net, bus=1, max_q_kvar=1e3, max_p_kw = -100e3, min_p_kw = 0)
    pp.create_gen(net, bus=4, p_kw=-34.6e3, max_p_kw = -100e3, min_p_kw = 0)
    return net


if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    np.set_printoptions(linewidth=2000)
    pd.set_option('expand_frame_repr', False)
    net = Sist5Bus()
    pp.rundcpp(net)
    # print(net._ppc['internal']['ref_gens'][0])
    print(net.res_gen.columns)
    # print(net._pd2ppc_lookups)
