"""
    Custom exceptions for SMCFPL module
"""


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
