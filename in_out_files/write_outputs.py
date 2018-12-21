"""
"""
from os.path import sep as os__path__sep
from json import dump as json__dump
from pandapower import to_pickle as pp__to_pickle


import logging
logging.basicConfig(level=logging.DEBUG,
                    format="[%(levelname)s][%(asctime)s] - %(message)s")
logger = logging.getLogger()


def ImprimeBDs(instance):
    """
        Imprime en el directorio temporal definido 'instance.TempFolderName', las siguientes base de datos.
    """
    logger.info("Exportando a archivos temporales ...")
    # Guarda una copia base de datos de etapas en los archivos
    instance.BD_Etapas.to_csv(instance.TempFolderName + os__path__sep + 'BD_Etapas.csv')
    # Guarda una copia base de datos de Parámetros de hidrologías de los embalses en los archivos
    instance.DFs_Entradas['df_in_smcfpl_ParamHidEmb'].to_csv(instance.TempFolderName + os__path__sep + 'ParamHidEmb.csv')
    # Guarda una copia base de datos de configuración hidráulica en los archivos
    instance.DFs_Entradas['df_in_smcfpl_seriesconf'].to_csv(instance.TempFolderName + os__path__sep + 'seriesconf.csv')
    # Guarda una copia base de datos de Proyección de la Demanda Sistema
    instance.BD_DemProy.to_csv(instance.TempFolderName + os__path__sep + 'BD_DemProy.csv')
    # Guarda una copia base de datos de la Probabilidad de Excedencia (PE) por etapa
    instance.BD_Hidrologias_futuras.to_csv(instance.TempFolderName + os__path__sep + 'BD_Hidrologias_futuras.csv')
    # Guarda una copia base de datos de la Tasa de Falla/Salida Forzada en el directorio temporal
    instance.BD_TSFProy.to_csv(instance.TempFolderName + os__path__sep + 'BD_TSFProy.csv')

    # Imprime las los archivos de Redes/Grids de cada etapa, para luego ser leídos por los nodos.
    for EtaNum in instance.BD_Etapas.index:
        """ Por cada etapa imprime dos archivos, uno llamado '#.json' (donde # es el número de la etapa) con info extra de la etapa y, otro
        llamado 'Grid_#.json' que contiene la red asociada a la etapa casi lista para simular lpf.
        """
        BD_RedesXEtapa_ExtraData = instance.BD_RedesXEtapa[EtaNum]['ExtraData']

        # Guarda Datos de etapa en archivo JSON
        with open(instance.TempFolderName + os__path__sep + "{}.json".format(EtaNum), 'w') as f:
            json__dump(BD_RedesXEtapa_ExtraData, f)
            # Exporta la red a archivo pickle. Necesario para exportar tipos de lineas. Más pesado que JSON y levemente más lento pero funcional... :c
            pp__to_pickle( instance.BD_RedesXEtapa[EtaNum]['PandaPowerNet'], instance.TempFolderName + os__path__sep + "Grid_Eta{}.p".format(EtaNum) )
    logger.info("Exportando completado.")
