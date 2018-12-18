"""
Este script esta programado a modo que pueda se llamado como función 'Calcular' a partir del nombre ,i.e.,
    import smcfpl.NucleoCalculo as NucleoCalculo  # En módulo
    from NucleoCalculo import Calcular  # Directamente por interprete por ejemplo
    NucleoCalculo.Calcular()
o utilizado directamente desde la terminal mediante: $ python3 NucleoCalculo.py Arg1 Arg2 ...
"""
from sys import argv as sys__argv


def Calcular(StageNumbers, Hidrology, GeneratorDemand_FreeCust, GeneratorDemand_RegCust, GeneratorDispatch, StdDevDispatchCenEyS,
             StdDevDispatchCenP, DF_ParamHidEmb, DF_seriesconf, BD_Etapas):
    print("Hidrology:", Hidrology)
    # Notar que el número de etapas es siempre el mismo, por lo que nunca se retorna de StopIteration cuando se acaban.
    for FreeCusts, RegCusts, GensDispath in zip(GeneratorDemand_FreeCust, GeneratorDemand_RegCust, GeneratorDispatch):
        # StageNum = FreeCust[0]
        StageNum = RegCusts[0]
        print( "StageNum:", StageNum )
        print( "BD_Etapas.loc[[StageNum],:]:\n", BD_Etapas.loc[[StageNum], :] )
        # FreeCustDemands = FreeCusts[1]
        # RegCustDemands = RegCusts[1]
        GensDispatched = GensDispath[1]
        print( "GensDispatched:\n", GensDispatched )
        # print( "FreeCustDemands:", FreeCustDemands )
        # print( "RegCustDemands:", RegCustDemands )

        print()
        import pdb; pdb.set_trace()  # breakpoint 4108548b //
    # Numero de cargas (Depende de etapa)
    # Escoger lista demandas. Desde generator
    # Numero de Unidades - 1 (Depende de etapa)
    # Escoger lista despachos. Desde generator


if __name__ == '__main__':
    Calcular('Humeda')
    print("sys__argv:", sys__argv)
