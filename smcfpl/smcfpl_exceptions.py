"""
    Custom exceptions for SMCFPL module
"""


class InsuficientInputData(Exception):
    """ Used when reading data from input sheadsheet. Not enough data. """
    pass


class InvalidOption(Exception):
    """ Generally used in if-elif-ELSE statements to notice one case was missing to capture. """
    pass


class CapacityOverloaded(Exception):
    """ Used for capacity messages of any object. """
    pass


class FalseCongestion(Exception):
    """ Used to detect congestion that were not congestions. """
    pass


class FolderDoesNotExist(Exception):
    """ Used to alert of existance of folder. """
    pass


class IntraCongestionIterationExceeded(Exception):
    """ Used to alert of iteration counter of intracongestions redipatch exceeded MaxItCongIntra. """
    pass


class GeneratorReferenceOverloaded(Exception):
    """ Used to alert of Generator of reference should dispatch more power than available. """
    pass


class GeneratorReferenceUnderloaded(Exception):
    """ Used to alert of Generator of reference should dispatch less power than possible. """
    pass


class LoadFlowError(Exception):
    """ Used to any possible load flow problem. """
    pass


class DemandGreaterThanInstalledCapacity(Exception):
    """ Used to alert of installed capacity problems. """
    pass


class SlurmCallError(Exception):
    """ Used to alert general Slurm calling errors. """
    pass


class sinfoNotAvailable(Exception):
    """ Used to alert general sinfo program errors. """
    pass


class NoNodesAvailable(Exception):
    """ Used when no cluster nodes are available. """
    pass
