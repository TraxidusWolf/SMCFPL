"""
Este script esta diseñado para que pueda ser llamado como función 'Calcular' a partir del nombre ,i.e.,
    import smcfpl.NucleoCalculo as NucleoCalculo  # En módulo
    from NucleoCalculo import Calcular  # Directamente por interprete por ejemplo
    NucleoCalculo.Calcular()
"""
from pandapower import rundcpp as pp__rundcpp
import time
from os import sep as os__sep


def Calcular(CasoNum, Hidrology, StageIndexesList, DF_ParamHidEmb_hid,
             DF_seriesconf, File_Caso='.', in_node=False, DemGenerator_Dict=dict(),
             DispatchGenerator_Dict=dict() ):
    """
        Función base de cálculo para la resolución del modelo SMCFPL. Por cada etapa obtiene los valores
        de los generadores de demanda y generación para los elementos de la Grilla de la etapa ocurrente.

        :param File_Caso: Ruta completa del directorio donde se encuentra el caso.
        :type File_Caso: string

        :param CasoNum: Numero del caso corriendo.
        :type CasoNum: int

        :param StageIndexesList: Indices de las etapas
        :type StageIndexesList: list

        :param Hidrology: Nombre de la hidrología actual
        :type Hidrology: string

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

        :param DemGenerator_Dict: Diccionario constituido por { EtaNum, pd.DataFrame.loc[EtaNum, 'PDem_pu'] }
        :type DemGenerator_Dict: dictionary

        :param DispatchGenerator_Dict: Diccionario constituido por { EtaNum, pd.DataFrame.loc[EtaNum, ['type', PGen_pu']] }
        :type DispatchGenerator_Dict: dictionary

    """
    # for each stage in the case
    for StageNum in StageIndexesList:
        # Load Data from every stage when 'in_node' is True
        print("StageNum:", StageNum)
        if in_node:
            # load Grid writen for the Power System
            FileName = "Grid_Eta{}.p".format(StageNum)
            # Grid = pp__from_pickle(FileName)
            print("FileName:", FileName)

            # load ExtraData writen for the Power System
            FileName = "{}.json".format(StageNum)
            # DictExtraData = json__load(FileName)
            print("FileName:", FileName)

        else:
            D = DemGenerator_Dict[StageNum]  # pandas DataFrame
            print("D:", D)
            G = DispatchGenerator_Dict[StageNum]  # pandas DataFrame
            print("G:", G)

    # time.sleep(3)

    if in_node:
        print("------------------\n Escribiendo archivo ficticio \n------------------")
        with open(File_Caso + os__sep + '..' + os__sep + str(CasoNum), 'w') as f:
            f.write( "Este es caso " + str(CasoNum) )
    else:
        print("----------------------------")
        print("Este fue CasoNum:", CasoNum)
        print("----------------------------")
        return (CasoNum,)

    # # Notar que el número de etapas es siempre el mismo, por lo que nunca se retorna de StopIteration cuando se acaban.
    # for CustsDem, GensDispatch in zip(GeneratorDemand, GeneratorDispatch):
    #     StageNum = CustsDem[0]
    #     # StageNum = GensDispatch[0]
    #     # print( "StageNum:", StageNum )
    #     CustomersDemands = CustsDem[1]  # Free and Regulated Customers
    #     GensDispatched = GensDispatch[1]

    #     #
    #     #
    #     # Grid.bus.loc[ pp.topology.unsupplied_buses(Grid), : ]  # Nombre de barras no suministradas
    #     # pp.drop_inactive_elements(Grid)  # remueve elementos que están sin actividad o estado fuera de servicio
    #     #
    #     #

    #     # Calcula excedente disponible de potencia sistema

    #     # Según excedente determina si es factible el sistema, de lo contrario imprime caso infactible

    #     # Inicializa contadores de congestiones

    #     # Calcula el CMg del sistema

    #     # Corre el flujo de potencia linealizado en la grilla de la etapa
    #     pp__rundcpp(DictRawSyst_per_Stage[StageNum])
    #     # Identifica lineas con sobrecarga para ver congestiones

    #     # De las congestiones verifica las Inter e Intra

    #     """ De existir Intra,
    #         repite lpf mientras condición de contadores o No existen congestiones. aumenta contador Intra
    #     """

    #     """ De existir Inter,
    #         separa el sep dejando como referencia ???... . Repite lpf mientras condición de contadores
    #         o No existen congestiones. aumenta contador Inter
    #     """

    #     #

    #     print(DictRawSyst_per_Stage[StageNum])

    #     # print()
    # Numero de cargas (Depende de etapa)
    # Escoger lista demandas. Desde generator
    # Numero de Unidades - 1 (Depende de etapa)
    # Escoger lista despachos. Desde generator
