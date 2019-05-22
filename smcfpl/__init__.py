__version__ = '0.1.4.1'
"""
    Main init file for SMCFPL model.
    Version control: 'mayor'.'minor'.'build number'.'revision'
        'mayor':    major release (usually many new features or changes to the UI or underlying OS).
        'minor':    minor release (perhaps some new features) on a previous major release. Reset with mayor change.
        'build number': usually a fix for a previous minor release (no new functionality). Reset with minor number change.
        'revision':  incremented for each latest build of a revision. Reset with build number change.
    By Gabriel Seguel G.
"""
# notar que la ruta de trabajo es mantenida desde el archivo que hace la llamada al módulo
from smcfpl.create_elements import *
from smcfpl.in_out_proc import *
from smcfpl.aux_funcs import *
