import pandapower as pp
import pandas as pd
import json
pd.set_option('precision', 4)
pd.set_option('expand_frame_repr', False)   # allow print all panda columns

# IMPORTA TIPO LINEAS
with open('tipo_lineas.json', 'r') as f:
    tipo_lineas = json.load(f)
# IMPORTA TIPO TR2
with open('tipo_trafos2w.json', 'r') as f:
    tipo_trafos2w = json.load(f)
# IMPORTA TIPO TR3
with open('tipo_trafos3w.json', 'r') as f:
    tipo_trafos3w = json.load(f)


Grid = pp.create_empty_network(name='39 Busbar system', f_hz=50, sn_kva=100e3)
pp.create_std_types(Grid, data=tipo_lineas, element='line')
pp.create_std_types(Grid, data=tipo_trafos2w, element='trafo')
pp.create_std_types(Grid, data=tipo_trafos3w, element='trafo3w')
# print(pp.available_std_types(Grid, element='trafo3w'))


# IMPORTA BARRAS Y CREA BARRAS
with open('terminales.json', 'r') as f:
    Barras = json.load(f)
    DF_aux = pd.DataFrame.from_dict(Barras, orient='index')
pp.create_buses(net=Grid, nr_buses=len(Barras), vn_kv=DF_aux['vn_kv'].tolist(), name=DF_aux['name'].tolist())


# IMPORTA LINEAS Y CREA LINEAS
with open('lineas.json', 'r') as f:
    lineas = json.load(f)
    DF_aux = pd.DataFrame.from_dict(lineas, orient='index')
for fila in DF_aux.index:
    from_busNom = DF_aux.loc[fila, 'from_busNom']
    to_busNom = DF_aux.loc[fila, 'to_busNom']
    pp.create_line(Grid, from_bus=Grid.bus[ Grid.bus['name'] == from_busNom ].index[0],
                   to_bus=Grid.bus[ Grid.bus['name'] == to_busNom ].index[0], length_km=DF_aux.loc[fila, 'length_km'],
                   std_type=DF_aux.loc[fila, 'std_type'], name=fila)


# IMPORTA GENERADORES
with open('unidades.json', 'r') as f:
    unidades = json.load(f)
    DF_aux = pd.DataFrame.from_dict(unidades, orient='index')
# CREA LOS GENERADORES (un solo ref)
NomGenSlack = DF_aux[ DF_aux['EsSlack'] == 1 ].index[0]  # el primero encontrado es considerado slack
# crea el GENERADOR DE REFERENCIA
GenSlack_BarNom = DF_aux.loc[NomGenSlack, 'BusNom']
pp.create_ext_grid(net=Grid, bus=Grid.bus[ Grid.bus['name'] == GenSlack_BarNom].index[0], vm_pu=DF_aux.loc[NomGenSlack, 'vm_pu'], name=NomGenSlack)
# crea el RESTO DE UNIDADES (eliminando del DataFrame el generador de referencia)
DF_aux = DF_aux.drop(index=NomGenSlack)
for gen in DF_aux.index:
    NomBarrConn = DF_aux.loc[gen, 'BusNom']
    pp.create_gen(net=Grid, bus=Grid.bus[ Grid.bus['name'] == NomBarrConn].index[0],
                  p_kw=DF_aux.loc[gen, 'p_kw'], vm_pu=DF_aux.loc[gen, 'vm_pu'],
                  sn_kva=DF_aux.loc[gen, 'sn_kva'], name=gen)

# IMPORTA CARGAS Y CREA CARGAS
with open('cargas.json', 'r') as f:
    cargas = json.load(f)
    DF_aux = pd.DataFrame.from_dict(cargas, orient='index')
for load in DF_aux.index:
    NomBarrConn = DF_aux.loc[load, 'BusNom']
    pp.create_load(net=Grid, bus=Grid.bus[ Grid.bus['name'] == NomBarrConn ].index[0],
                   p_kw=DF_aux.loc[load, 'p_kw'], q_kvar=DF_aux.loc[load, 'q_kvar'],
                   sn_kva=DF_aux.loc[load, 'sn_kva'], name=load)


# IMPORTA TRAFOS2W Y CREA TRAFOS2W
with open('trafos2w.json', 'r') as f:
    Trafos = json.load(f)
    DF_aux = pd.DataFrame.from_dict(Trafos, orient='index')
for tr2 in DF_aux.index:
    NomBarrConnHV = DF_aux.loc[tr2, 'hv_busNom']
    NomBarrConnLV = DF_aux.loc[tr2, 'lv_busNom']
    pp.create_transformer(net=Grid, hv_bus=Grid.bus[ Grid.bus['name'] == NomBarrConnHV ].index[0],
                          lv_bus=Grid.bus[ Grid.bus['name'] == NomBarrConnLV ].index[0], std_type=DF_aux.loc[tr2, 'std_type'],
                          name=tr2)
# print(Grid.trafo)

# pp.rundcpp(Grid)
pp.runpp(Grid, algorithm='nr', calculate_voltage_angles=True, trafo_model='pi', numba=True)
# print(Grid.res_bus)
# print(Grid.res_gen)
# print(Grid.res_ext_grid)
print(Grid.load)
print(Grid.res_load)
# print(Grid)
# pp.diagnostic(Grid, report_style='detailed')
# print(Grid.bus)
