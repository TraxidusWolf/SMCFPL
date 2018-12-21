"""
Este script esta programado a modo que pueda se llamado como función 'Calcular' a partir del nombre ,i.e.,
    import smcfpl.NucleoCalculo as NucleoCalculo  # En módulo
    from NucleoCalculo import Calcular  # Directamente por interprete por ejemplo
    NucleoCalculo.Calcular()
o utilizado directamente desde la terminal mediante: $ python3 NucleoCalculo.py Arg1 Arg2 ...
"""
from sys import argv as sys__argv
from pandapower import rundcpp as pp__rundcpp


def Calcular(StageIndexes, Hidrology, GeneratorDemand, GeneratorDispatch, DF_ParamHidEmb_hid, DF_seriesconf,
             DictRawSyst_per_Stage):
    """
        Función base de cálculo para la resolución del modelo SMCFPL. Por cada etapa obtiene los valores
        de los generadores de demanda y generación para los elementos de la Grilla de la etapa ocurrente.

        :param StageIndexes: Contiene los índices de la base de datos de etapas creadas al comienzo del
                             modelo.
        :type StageIndexes: pandas indexes

        :param Hydrology: Nombre de la hidrología en la que se sitúa el problema.
        :type Hydrology: string

        :param GeneratorDemand: Este generador entrega valores de tasa de crecimiento a todas las cargas
                                según su tipo ('L' o 'R') respecto del valor inicial nominal al comienzo
                                de la simulación. Notar que este valor ya se encuentra asignado en la
                                tabla de demandas de la Grilla ('PandaPowerNet').
        :type GeneratorDemand: tupla (EtaNum, pandas DataFrame)

        :param GeneratorDispatch: Este generador entrega las potencias aleatorias en por unidad ([0,1])
                                  de las unidades dispuestas en el sistema. Ya viene incorporada la TSF
                                  en el despacho al considerar una potencia 0.
        :type GeneratorDispatch: tupla (EtaNum, pandas DataFrame)

        :param DF_ParamHidEmb_hid: Parámetros de la función logística S que define la relación del costo de
                                   generación variable respecto al valor de la cota en la hidrología actual.
        :type DF_ParamHidEmb_hid: pandas Multindex DataFrame

        :param DF_seriesconf: Configuración de las series hidráulicas, y conexiones hidrológicas.
                              Principalmente para obtención de costo de generación.
        :type DF_seriesconf: pandas DataFrame

        :param DictRawSyst_per_Stage: Contiene como clave el número de etapa y como valor la red PandaPower
                                   con valores brutos, que se requieren modificar por Montecarlo.
        :type DictRawSyst_per_Stage: Diccionario

        :param ?????: 
        :type ?????: 


    """
    print("Hidrology:", Hidrology)
    # Notar que el número de etapas es siempre el mismo, por lo que nunca se retorna de StopIteration cuando se acaban.
    for CustsDem, GensDispatch in zip(GeneratorDemand, GeneratorDispatch):
        StageNum = CustsDem[0]
        # StageNum = GensDispatch[0]
        # print( "StageNum:", StageNum )
        CustomersDemands = CustsDem[1]  # Free and Regulated Customers
        GensDispatched = GensDispatch[1]
        # print( "GensDispatched:\n", GensDispatched )
        # print( "CustomersDemands:\n", CustomersDemands )
        # Actualiza Potencias de Demanda
        DictRawSyst_per_Stage[StageNum]['load']['p_kw'] *= CustomersDemands['PDem_pu'].values
        # Actualiza Potencias de Generación
        DictRawSyst_per_Stage[StageNum]['gen']['p_kw'] *= GensDispatched['PGen_pu'].values
        #
        #
        # Grid.bus.loc[ pp.topology.unsupplied_buses(Grid), : ]  # Nombre de barras no suministradas
        # pp.drop_inactive_elements(Grid)  # remueve elementos que están sin actividad o estado fuera de servicio
        #
        #

        # Calcula excedente disponible de potencia sistema

        # Según excedente determina si es factible el sistema, de lo contrario imprime caso infactible

        # Inicializa contadores de congestiones

        # Calcula el CMg del sistema

        # Corre el flujo de potencia linealizado en la grilla de la etapa
        pp__rundcpp(DictRawSyst_per_Stage[StageNum])
        # Identifica lineas con sobrecarga para ver congestiones

        # De las congestiones verifica las Inter e Intra

        """ De existir Intra,
            repite lpf mientras condición de contadores o No existen congestiones. aumenta contador Intra
        """

        """ De existir Inter,
            separa el sep dejando como referencia ???... . Repite lpf mientras condición de contadores
            o No existen congestiones. aumenta contador Inter
        """

        # 

        print(DictRawSyst_per_Stage[StageNum])

        # print()
    # Numero de cargas (Depende de etapa)
    # Escoger lista demandas. Desde generator
    # Numero de Unidades - 1 (Depende de etapa)
    # Escoger lista despachos. Desde generator


if __name__ == '__main__':
    Calcular(*sys__argv[1:])
    print("sys__argv:", sys__argv)
