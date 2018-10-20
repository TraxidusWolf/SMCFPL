#!/usr/bin/env python3
"""
    Rutina CreaGrafo para programa MCFPL.py
    Autor: Gabriel Seguel G.
"""
import networkx as nx

def CreaGrafoSEP(BD_TempDispEta, BD_TecBarras, BD_TecLineas, BD_TecGen, BD_TecCargas):
    """
        Crea el grafo del sistema para la barra en la que 

        Syntax:
        GrafoSEP = CreaGrafoSEP(BD_TempDispEta, BD_TecBarras, BD_TecLineas, BD_TecGen, BD_TecCargas)

        Arguments:
            BD_TempDispEta {Diccionario} -- Obtiene casi toda la información de la etapa
            BD_TecBarras {Diccionario} -- Obtiene nombres y atributo 'EsSlack' de los nodos que existen.
            BD_TecGen {Diccionario} -- Obtiene el nodo al que se conecta cada generadora.
            BD_TecCargas {Diccionario} -- Obtiene el nodo al que se conecta cada carga.
    """

    GrafoSEP = nx.DiGraph()
    # Agrega ramas desde BD_TempDispEta, usando la información de conexión de BD_TecBarras
    n = []
    for Bar in BD_TempDispEta['Barras']:
        n.append( (Bar,
                   { 'EsSlack': bool(BD_TecBarras[Bar]['EsSlack']),
                     'Pprog': 0}
                   ) )
    # for Bar, Atrib in BD_TecBarras.items():
        # if Bar in BD_TempDispEta['Barras']:
        # n.append( (Bar, {'EsSlack': bool(Atrib['EsSlack'])}) )
    GrafoSEP.add_nodes_from( n )

    # Agrega ramas desde BD_TempDispEta, usando la información de conexión de BD_TecLineas
    r = []
    for Lin, Valores in BD_TempDispEta['Tx'].items():
        # obtiene barras de conexión
        Nini, Nfin = BD_TecLineas[Lin]['BarraA'], BD_TecLineas[Lin]['BarraB']
        # crea diccionario de atributos de lineas
        Atrib = {'LinNom': Lin, 'R': Valores[0], 'X': Valores[1], 'PmaxAB': Valores[2], 'PmaxBA': Valores[3], 'FluPAB': 0, 'FluPBA': 0}
        r.append( (Nini, Nfin, Atrib) )
    GrafoSEP.add_edges_from( r )

    return GrafoSEP
