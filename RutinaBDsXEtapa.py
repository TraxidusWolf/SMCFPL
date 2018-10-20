#!/usr/bin/env python3
"""
    Rutina CreaBDxEtapa para programa MCFPL.py
    Autor: Gabriel Seguel G.

    Contiene la función para crear una base de datos temporal con valores en por unidad para ser utilizada en la creación del grafo e información durante cada simulación. Notar que los elementos (Gx, Tx, o Dem) siempre van a estar presentes a no ser que se definan no operativos (mediante mantenimientos) o que el elemento de conexión (barra) no se encuentra habilitado en la etapa.
"""

def CreaDB_Etapa(BD_TecBarras, BD_TecGen, BD_TecLineas, BD_TecCargas, BD_MantBarras, BD_MantGen, BD_MantTx, BD_MantCargas, BD_ProyDem, BD_ParamSim, EtaNum):
    """
        Crea una base de datos unificada (con datos en por unidad)

        Syntax:
        BD_TempDispEta = CreaDB_Etapa(BD_TecGen, BD_TecLineas, BD_TecCargas, BD_MantBarras, BD_MantGen, BD_MantTx, BD_MantCargas, BD_ParamSim, EtaNum)

        Arguments:
            BD_TecBarras, {Dict} -- Barras existentes durante toda la simulación con datos técnicos.
            BD_TecGen {Dict} -- Unidades existentes durante toda la simulación con datos técnicos.
            BD_TecLineas {Dict} -- Lineas/Transformadores existentes durante toda la simulación con datos técnicos.
            BD_TecCargas {Dict} -- Cargas existentes durante toda la simulación con datos técnicos.
            BD_MantBarras {pandas.DataFrame} -- DataFrame de etapas cuando una Barra no se encuentra disponible.
            BD_MantGen {pandas.DataFrame.groupby} -- DataFrame de etapas cuando una Unidad se encuentra con datos modificados.
            BD_MantTx {pandas.DataFrame.groupby} -- DataFrame de etapas cuando una Linea/Transformador se encuentra con datos modificados.
            BD_MantCargas {pandas.DataFrame.groupby} -- DataFrame de etapas cuando una Carga se encuentra con datos modificados.
            BD_ProyDem {pandas.DataFrame.groupby} -- DataFrame de etapas cuando la demanda media de cada Carga.
            BD_ParamSim {Dict} -- Datos de configuración de la simulación.
            EtaNum {Int} -- Número de la etapa de interés.
    """
    # Inicializa diccionario de salida
    BD_TempDispEta = { 'Gen': {}, 'Dem': {}, 'Tx': {} }

    # Identifica la potencia base para la simulación [MVA]
    Sbase = BD_ParamSim['Sbase']

    # inicializa set de barras en etapa
    BarrasEnEta = set()
    # Identifica las barras disponibles. Todo elemento conectado a una barra fuera de servicio no es considerado. Crea set de barras.
    for Barra, Valores in BD_TecBarras.items():
        # print(Barra)   # DEBUGGING
        # si barra existe en BD_MantBarras y se no encuentra operativa, entonces no existe en etapa y se continua con siguiente barra.
        if Barra in BD_MantBarras[ (BD_MantBarras['Eta'] == EtaNum) & (BD_MantBarras['Operativa'] == False) ].index:
            print('Barra:', Barra, 'No considerada.')   # DEBUGGING
            continue
        # si barra existe en BD_MantBarras y se encuentra operativa, entonces se considera operativa.
        elif Barra in BD_MantBarras[ (BD_MantBarras['Eta'] == EtaNum) & (BD_MantBarras['Operativa'] == True) ].index:
            BarrasEnEta.update([Barra])
        # en caso de no estar en BD_MantBarras, entonces está operativa.
        else:
            BarrasEnEta.update([Barra])
    # print('BarrasEnEta:\n', BarrasEnEta)   # DEBUGGING
    # agrega el set de barras a BD_TempDispEta
    BD_TempDispEta['Barras'] = BarrasEnEta

    # Asigna a la BD de salida los datos de Unidades
    for Unidad, Valores in BD_TecGen.items():
        # print(Unidad)   # DEBUGGING
        # Si la unidad no está conectada a alguna de las barras disponibles, pasa a siguiente Unidad y no es agregada a BD_TempDispEta
        if not Valores['NomBarConn'] in BarrasEnEta:
            print('Unidad:', Unidad, 'No considerada.')   # DEBUGGING
            continue
        # si Unidad existe en BD_MantGen y no se encuentra operativa, entonces se continua con siguiente Unidad.
        if Unidad in BD_MantGen[ (BD_MantGen['Eta'] == EtaNum) & (BD_MantGen['Operativa'] == False) ].index:
            print('Unidad:', Unidad, 'No considerada.')   # DEBUGGING
            continue
        # si Unidad existe en BD_MantGen y se encuentra operativa, entonces se considera operativa con dichos valores.
        elif Barra in BD_MantGen[ (BD_MantGen['Eta'] == EtaNum) & (BD_MantGen['Operativa'] == True) ].index:
            BD_TempDispEta['Gen'][Unidad] = [ BD_MantGen[ BD_MantGen['Eta'] == EtaNum ]['PmaxMW'][Unidad] / Sbase,
                                              BD_MantGen[ BD_MantGen['Eta'] == EtaNum ]['PminMW'][Unidad] / Sbase,
                                              BD_MantGen[ BD_MantGen['Eta'] == EtaNum ]['CVar'][Unidad],
                                              0 ]
        # si Unidad no existe en BD_MantGen usa valores predeterminados.
        else:
            BD_TempDispEta['Gen'][Unidad] = [ Valores['PmaxMW'] / Sbase,
                                              Valores['PminMW'] / Sbase,
                                              Valores['CVar'],
                                              0 ]

    # Asigna a la BD de salida los datos de Cargas (Dem)
    for Carga, Valores in BD_TecCargas.items():
        # print(Carga)   # DEBUGGING
        # Si la Carga no está conectada a alguna de las barras disponibles, pasa a siguiente Carga y no es agregada a BD_TempDispEta
        if not Valores['NomBarConn'] in BarrasEnEta:
            print('Carga:', Carga, 'No considerada.')   # DEBUGGING
            continue
        # si Carga existe en BD_MantCargas y no se encuentra operativa, entonces se continua con siguiente Carga.
        if Carga in BD_MantCargas[ (BD_MantCargas['Eta'] == EtaNum) & (BD_MantCargas['Operativa'] == False) ].index:
            print('Carga:', Carga, 'No considerada.')   # DEBUGGING
            continue
        # si Carga existe en BD_MantCargas y se encuentra operativa, entonces se considera operativa con dichos valores.
        elif Carga in BD_MantCargas[ (BD_MantCargas['Eta'] == EtaNum) & (BD_MantCargas['Operativa'] == True) ].index:
            # si la carga posee datos en BD_ProyDem se le asigna, de lo contrario proyecta con 0
            if Carga in BD_ProyDem[ BD_ProyDem['Eta'] == EtaNum ].index:
                BD_TempDispEta['Dem'][Carga] = [ BD_MantCargas[ BD_MantCargas['Eta'] == EtaNum ]['DemMax'][Carga] / Sbase,
                                                 BD_ProyDem['DemMed'][Carga] / Sbase,
                                                 0 ]
            else:
                BD_TempDispEta['Dem'][Carga] = [ BD_MantCargas[ BD_MantCargas['Eta'] == EtaNum ]['DemMax'][Carga] / Sbase,
                                                 0,
                                                 0 ]
        # si Carga no existe en BD_MantCargas usa valores predeterminados.
        else:
            BD_TempDispEta['Dem'][Carga] = [ Valores['DemMax'] / Sbase,
                                             0,
                                             0 ]

    # Asigna a la BD de salida los datos de Elementos de transmisión (Tx)
    for Linea, Valores in BD_TecLineas.items():
        # print(Linea)   # DEBUGGING
        # Identifica las barras a las que se conecta la línea
        Nini, Nfin = Valores['BarraA'], Valores['BarraB']
        # print(Nini, Nfin)   # DEBUGGING
        # Si la Linea no posee barra Nini de conexión dentro de las barras disponibles, pasa a siguiente Linea y no es agregada a BD_TempDispEta
        if not Nini in BarrasEnEta:
            print('Linea:', Linea, 'No considerada por barra', Nini, 'no disponible.')   # DEBUGGING
            continue
        # Si la Linea no posee barra Nini de conexión dentro de las barras disponibles, pasa a siguiente Linea y no es agregada a BD_TempDispEta
        if not Nfin in BarrasEnEta:
            print('Linea:', Linea, 'No considerada por barra', Nfin, 'no disponible.')   # DEBUGGING
            continue
        # obtiene la tensión de las barra de conexión
        VNini, VNfin = BD_TecBarras[Nini]['VnomkV'], BD_TecBarras[Nfin]['VnomkV']
        # print(VNini, VNfin)   # DEBUGGING
        # Utiliza la mayor tensión como Vbase para la línea
        Vbase = max(VNini, VNfin)
        # print(Vbase)   # DEBUGGING
        # calcula el Zbase
        Zbase = Vbase**2 / Sbase
        # print(Zbase,'\n')   # DEBUGGING
        # si Linea existe en BD_MantTx y se encuentra NO operativa, entonces se continua con siguiente Linea.
        if Linea in BD_MantTx[ (BD_MantTx['Eta'] == EtaNum) & (BD_MantTx['Operativa'] == False) ].index:
            print('Linea:', Linea, 'No considerada.')   # DEBUGGING
            continue
        # si Linea existe en BD_MantTx y se encuentra operativa, entonces se considera operativa con dichos valores.
        elif Barra in BD_MantTx[ (BD_MantTx['Eta'] == EtaNum) & (BD_MantTx['Operativa'] == True) ].index:
            BD_TempDispEta['Tx'][Linea] = [ BD_MantTx[ BD_MantTx['Eta'] == EtaNum ]['Rohm'][Linea] / Zbase,
                                            BD_MantTx[ BD_MantTx['Eta'] == EtaNum ]['Xohm'][Linea] / Zbase,
                                            BD_MantTx[ BD_MantTx['Eta'] == EtaNum ]['PmaxABMW'][Linea] / Sbase,
                                            BD_MantTx[ BD_MantTx['Eta'] == EtaNum ]['PmaxBAMW'][Linea] / Sbase,
                                            0,
                                            0 ]
        # si Linea no existe en BD_MantTx usa valores predeterminados.
        else:
            BD_TempDispEta['Tx'][Linea] = [ Valores['Rohm'] / Zbase,
                                            Valores['Xohm'] / Zbase,
                                            Valores['PmaxABMW'] / Sbase,
                                            Valores['PmaxBAMW'] / Sbase,
                                            0,
                                            0 ]

    return BD_TempDispEta
