__version__ = 1.0
"""
    Archivo principal para ejecutar programa.
    Creado por:
        Gabriel Seguel G.
        Ingeniero Civil Eléctrico
"""
# notar que la ruta de trabajo es mantenida desde el archivo que hace la llamada al módulo
from smcfpl.CrearElementos import *
from smcfpl.in_out_files import *
from smcfpl.aux_funcs import *


# import RutinaLectorEntradas as LectorEntradas
# import RutinaDefinirEtapas as DefinirEtapas
# import RutinaBDsXEtapa as BDsXEtapa
# import RutinaCreaGrafo as CreaGrafo


# #

# #

# # Ruta de trabajo
# RutaTrabajo = '.'
# # Ruta archivos de entrada
# DirectorioEntradas = RutaTrabajo + '/' + 'DatosEntrada'
# # Ruta archivos de salida
# DirectorioSalidas = RutaTrabajo + '/' + 'DatosSalida'

# # Crea Bases de datos por tiempo    (PARALELIZABLE)
# # Datos Técnicos
# BD_TecGen = LectorEntradas.LeeDatosTecnicos( DirectorioEntradas + '/' + 'in_mcfpl_tecgen.csv' )  # Dict
# BD_TecBarras = LectorEntradas.LeeDatosTecnicos( DirectorioEntradas + '/' + 'in_mcfpl_tecbarras.csv' )  # Dict
# BD_TecLineas = LectorEntradas.LeeDatosTecnicos( DirectorioEntradas + '/' + 'in_mcfpl_teclineas.csv' )  # Dict
# BD_TecCargas = LectorEntradas.LeeDatosTecnicos( DirectorioEntradas + '/' + 'in_mcfpl_teccargas.csv' )  # Dict
# # Datos Proyectados
# BD_ProyDem = LectorEntradas.LeeProyeccDem( DirectorioEntradas + '/' + 'in_mcfpl_proydem.csv' )  # DataFame
# # Datos Mantenimiento
# BD_MantBarras = LectorEntradas.LeeDatosMant( DirectorioEntradas + '/' + 'in_mcfpl_mantbarras.csv' )  # DataFame
# BD_MantGen = LectorEntradas.LeeDatosMant( DirectorioEntradas + '/' + 'in_mcfpl_mantgen.csv' )  # DataFame
# BD_MantTx = LectorEntradas.LeeDatosMant( DirectorioEntradas + '/' + 'in_mcfpl_manttx.csv' )  # DataFame
# BD_MantCargas = LectorEntradas.LeeDatosMant( DirectorioEntradas + '/' + 'in_mcfpl_mantcargas.csv' )  # DataFame
# # Curvas Probabilidad
# BD_ProbGen = LectorEntradas.LeeDatosCurvProb( DirectorioEntradas + '/' + 'in_mcfpl_probgen.csv' )  # Dict
# BD_ProbDem = LectorEntradas.LeeDatosCurvProb( DirectorioEntradas + '/' + 'in_mcfpl_probdem.csv' )  # Dict
# # Datos Simulación
# BD_ParamSim = LectorEntradas.LeeDatosSimulacion( DirectorioEntradas + '/' + 'in_mcfpl_simulacion.csv' )  # Dict


# print('\n--->BD_TecBarras:\n', BD_TecBarras)
# print('\n--->BD_TecGen:\n', BD_TecGen)
# print('\n--->BD_TecCargas:\n', BD_TecCargas)
# print('\n--->BD_TecLineas:\n', BD_TecLineas)


# print('\n--->BD_ProyDem:\n', BD_ProyDem)
# print('\n--->BD_MantGen:\n', BD_MantGen)
# print('\n--->BD_MantTx:\n', BD_MantTx)
# print('\n--->BD_MantCargas:\n', BD_MantCargas)
# print('\n--->BD_MantBarras:\n', BD_MantBarras)

# print('\n--->BD_ParamSim:\n', BD_ParamSim)

# # Crea lista de tuplas de relación Fecha-Etapas (PARALELIZABLE)
# RelFechaEta = DefinirEtapas.DefineEtapas(BD_ProyDem,
#                                          BD_MantGen,
#                                          BD_MantTx,
#                                          BD_MantCargas,
#                                          BD_MantBarras,
#                                          BD_ProbGen,
#                                          BD_ProbDem,
#                                          BD_ParamSim)
# print('\n--->RelFechaEta:\n', RelFechaEta)

# # Transforma los DataFrame de fecha a etapas    (PARALELIZABLE)
# BD_ProyDem = DefinirEtapas.BD_Fechas_a_Etapas(BD_ProyDem, RelFechaEta)  # sobreescribe DataFrame de fechas a uno de etapas
# BD_MantBarras = DefinirEtapas.BD_Fechas_a_Etapas(BD_MantBarras, RelFechaEta)
# BD_MantGen = DefinirEtapas.BD_Fechas_a_Etapas(BD_MantGen, RelFechaEta)
# BD_MantTx = DefinirEtapas.BD_Fechas_a_Etapas(BD_MantTx, RelFechaEta)
# BD_MantCargas = DefinirEtapas.BD_Fechas_a_Etapas(BD_MantCargas, RelFechaEta)

# print('\n--->BD_ProyDem:\n', BD_ProyDem)
# print('\n--->BD_MantGen:\n', BD_MantGen)
# print('\n--->BD_MantTx:\n', BD_MantTx)
# print('\n--->BD_MantCargas:\n', BD_MantCargas)
# print('\n--->BD_MantBarras:\n', BD_MantBarras)


# # Analiza para cada etapa (PARALELIZABLE)
# for etapa in RelFechaEta:
#     EtaNum = etapa[2]
#     print('--->EtaNum:', EtaNum)   # DEBUGGING

#     # Crea la base de datos temporal (BD_TempDispEta)
#     BD_TempDispEta = BDsXEtapa.CreaDB_Etapa(BD_TecBarras,
#                                             BD_TecGen,
#                                             BD_TecLineas,
#                                             BD_TecCargas,
#                                             BD_MantBarras,
#                                             BD_MantGen,
#                                             BD_MantTx,
#                                             BD_MantCargas,
#                                             BD_ProyDem,
#                                             BD_ParamSim,
#                                             EtaNum)
#     print('\n--->BD_TempDispEta (EtaNum={}):\n'.format(EtaNum), BD_TempDispEta)

#     GrafoSEP = CreaGrafo.CreaGrafoSEP(BD_TempDispEta, BD_TecBarras, BD_TecLineas, BD_TecGen, BD_TecCargas)
#     print(GrafoSEP.nodes(data=True))
#     print(GrafoSEP.edges(data=True))

#     import matplotlib.pyplot as plt
#     import networkx as nx

#     # pos = nx.random_layout(GrafoSEP)    # layout aleatorio
#     pos = nx.kamada_kawai_layout(GrafoSEP, weight='PmaxBA')    # layout en función del costo del peso 'weight'
#     shells = [GrafoSEP.nodes]
#     # pos = nx.shell_layout(GrafoSEP, shells=shells)    # layout con niveles circulares
#     # pos = nx.spring_layout(GrafoSEP, weight='X')    # layout using Fruchterman-Reingold force-directed algorithm
#     # pos = nx.spectral_layout(GrafoSEP, weight='PmaxAB')    # layout using the eigenvectors of the graph Laplacian
#     nx.draw_networkx_nodes(GrafoSEP, pos)  # grafica nodos
#     nx.draw_networkx_edges(GrafoSEP, pos)  # grafica edges
#     nx.draw_networkx_labels(GrafoSEP, pos)    # grafica nodos labels
#     # nx.draw_networkx_edge_labels(GrafoSEP, pos, { (A, B): d['LinNom'] for A, B, d in GrafoSEP.edges(data=True) })  # grafica edges labels
#     # nx.draw_networkx(GrafoSEP)
#     plt.show()

#     break
