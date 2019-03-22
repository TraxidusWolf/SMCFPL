"""
"""
from os import makedirs as os__makedirs
from os import sep as os__sep
from os.path import exists as os__path__exists
from json import dump as json__dump
from pandapower import to_pickle as pp__to_pickle
from pandas import DataFrame as pd__DataFrame
from pickle import HIGHEST_PROTOCOL as pickle__HIGHEST_PROTOCOL, dump as pickle__dump
import smcfpl.aux_funcs as aux_smcfpl


import logging
logging.basicConfig(level=logging.DEBUG,
                    format="[%(levelname)s][%(asctime)s][%(filename)s:%(funcName)s] - %(message)s")
logger = logging.getLogger()


def dump_BDs_to_pickle(Names_Variables, pathto='.', FileFormat='pickle'):
    """
        Names_Variables: dict like argument with names of variables as key and variable to print as values.
            {   'BD_Etapas': self.BD_Etapas,
                'BD_DemProy': self.BD_DemProy,
                'BD_Hidrologias_futuras': self.BD_Hidrologias_futuras,
                'BD_TSFProy': self.BD_TSFProy,
                'BD_MantEnEta': self.BD_MantEnEta,
                'BD_RedesXEtapa': self.BD_RedesXEtapa,
                'BD_HistGenRenovable': self.BD_HistGenRenovable,
                'BD_ParamHidEmb': self.BD_ParamHidEmb,
                'BD_seriesconf': self.BD_seriesconf,
            }
        FileFormat='pickle': format to be printed each variable.
        pathto: string with absolute path.
    """
    if FileFormat == 'pickle':
        postfix = 'p'
    else:
        raise IOError("'{}' format not implemented yet or des not exists.". format(FileFormat))

    for name, var in Names_Variables.items():
        with open(pathto + os__sep + "{}.{}".format(name, postfix), 'wb') as f:
            pickle__dump(var, f, pickle__HIGHEST_PROTOCOL)


def ImprimeBDsGrales(instance):
    """
        Imprime en el directorio temporal definido 'instance.abs_path_temp', las siguientes base de datos.
        Estas son escritas, a no ser que ya existan.
    """
    logger.info("Exportando a archivos genéricos temporales ...")
    Nom_BD = instance.abs_path_temp + os__sep + 'BD_Etapas.csv'
    if not os__path__exists(Nom_BD):
        # Guarda una copia base de datos de etapas en los archivos de no existir
        instance.BD_Etapas.to_csv(Nom_BD)
    Nom_BD = instance.abs_path_temp + os__sep + 'ParamHidEmb.csv'
    if not os__path__exists(Nom_BD):
        # Guarda una copia base de datos de Parámetros de hidrologías de los embalses en los archivos
        instance.DFs_Entradas['df_in_smcfpl_ParamHidEmb'].to_csv(Nom_BD)
    Nom_BD = instance.abs_path_temp + os__sep + 'seriesconf.csv'
    if not os__path__exists(Nom_BD):
        # Guarda una copia base de datos de configuración hidráulica en los archivos
        instance.DFs_Entradas['df_in_smcfpl_seriesconf'].to_csv(Nom_BD)
    Nom_BD = instance.abs_path_temp + os__sep + 'BD_DemProy.csv'
    if not os__path__exists(Nom_BD):
        # Guarda una copia base de datos de Proyección de la Demanda Sistema
        instance.BD_DemProy.to_csv(Nom_BD)
    Nom_BD = instance.abs_path_temp + os__sep + 'BD_Hidrologias_futuras.csv'
    if not os__path__exists(Nom_BD):
        # Guarda una copia base de datos de la Probabilidad de Excedencia (PE) por etapa
        instance.BD_Hidrologias_futuras.to_csv(Nom_BD)
    Nom_BD = instance.abs_path_temp + os__sep + 'BD_TSFProy.csv'
    if not os__path__exists(Nom_BD):
        # Guarda una copia base de datos de la Tasa de Falla/Salida Forzada en el directorio temporal
        instance.BD_TSFProy.to_csv(Nom_BD)

    logger.info("Exportado de archivos genéricos temporales completado.")


def write_BDs_input_case(instance, IdentificadorCaso, InputList):
    """
        Escribe en disco en directorio de trabajo, las base de datos dependientes de cada caso,
        como son las grillas (pickle) y, la ExtraData (JSON) que no puede ser almacenada directamente
        en la grilla.

        Is it really nedeed?
    """
    # Crea el nombre de la carpeta del caso, en función del Identificador
    CasoNom = "{0}_D{1}_G{2}".format( *IdentificadorCaso )

    logger.info("Exportando a archivos temporales del caso {} ...".format(CasoNom))

    # Nombre de la carpeta correspondiente al caso 'CasoNom-ésimo'
    FolderName = instance.abs_path_temp + os__sep + CasoNom

    # verifica que exista directorio, de lo contrario lo crea.
    if not os__path__exists(FolderName):
        os__makedirs(FolderName)
    else:
        logger.info("Caso: {} ya se encuentra creado.".format(CasoNom))
        return

    # Crea los generadores según los datos del caso
    PyGeneratorDemand = aux_smcfpl.GeneradorDemanda(DF_TasaCLib=instance.BD_DemProy[['TasaCliLib']],  # pandas DataFrame
                                                    DF_TasaCReg=instance.BD_DemProy[['TasaCliReg']],  # pandas DataFrame
                                                    DF_DesvDec=instance.BD_DemProy[['Desv_decimal']],  # pandas DataFrame
                                                    ListTypoCargasEta=instance.ListTypoCargasEta,  # lista
                                                    seed=instance.UseRandomSeed)  # int
    PyGeneratorDispatched = aux_smcfpl.GeneradorDespacho(Lista_TiposGen=instance.ListTiposGenNoSlack,  # lista
                                                         DF_HistGenERNC=instance.BD_HistGenRenovable,  # tupla de dos pandas DataFrame
                                                         DF_TSF=instance.BD_TSFProy,  # para cada tecnología que recurra con falla se asigna
                                                         DF_PE_Hid=InputList['DF_PEsXEtapa'],  # pandas DataFrame
                                                         DesvEstDespCenEyS=instance.DesvEstDespCenEyS,  # float
                                                         DesvEstDespCenP=instance.DesvEstDespCenP,  # float
                                                         seed=instance.UseRandomSeed)  # int

    # Imprime las los archivos de Redes/Grids de cada etapa, para luego ser leídos por los nodos.
    for GenDisp, GenDem in zip(PyGeneratorDispatched, PyGeneratorDemand):
        """ Por cada etapa imprime dos archivos, uno llamado '#.json' (donde # es el número de la etapa) con info extra de la etapa y, otro
        llamado 'Grid_#.json' que contiene la red asociada a la etapa casi lista para simular lpf.
        """
        StageNum = GenDisp[0]  # StageNum = GenDem[0]
        # Obtiene ExtraData para cada grilla de etapa
        BD_RedesXEtapa_ExtraData = instance.BD_RedesXEtapa[StageNum]['ExtraData']
        #
        # Modifica potencias en la grilla para ajustar los valores de los casos, previamente a ser escritos
        # Actualiza Potencias de Demanda
        instance.BD_RedesXEtapa[StageNum]['PandaPowerNet']['load']['p_kw'] *= GenDem[1]['PDem_pu'].values
        # Actualiza Potencias de Generación
        instance.BD_RedesXEtapa[StageNum]['PandaPowerNet']['gen']['p_kw'] *= GenDisp[1]['PGen_pu'].values

        # Transforma los DataFrame existentes en el diccionario a diccionarios.
        # Evita error y permite escribir a JSON file de etapa
        for k, v in BD_RedesXEtapa_ExtraData.items():
            if isinstance(v, pd__DataFrame):
                BD_RedesXEtapa_ExtraData[k] = v.to_dict()

        # Guarda Datos de etapa en archivo JSON
        with open(FolderName + os__sep + "{}.json".format(StageNum), 'w') as f:
            json__dump(BD_RedesXEtapa_ExtraData, f)

        # Exporta la red a archivo pickle. Necesario para exportar tipos de lineas. Más pesado que JSON y levemente más lento pero funcional... :c
        pp__to_pickle( instance.BD_RedesXEtapa[StageNum]['PandaPowerNet'], FolderName + os__sep + "Grid_Eta{}.p".format(StageNum) )
    logger.info("Exportado del caso {} completado.".format(CasoNom))


def write_output_case(RelevantData, CaseNum, CaseID):
    """ Write to output directory one pickled file with available results.


    """
    CaseNom = "{0}_D{1}_G{2}".format( *CaseID )
    dump_BDs_to_pickle({CaseNom: RelevantData}, pathto='.', format='pickle')  # try write to output folder
