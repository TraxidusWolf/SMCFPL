from pandas import datetime as pd__datetime, set_option as pd__set_option
import locale

locale.setlocale(locale.LC_ALL, 'es_ES.utf8')


def date_parser(x):
    return pd__datetime.strptime(x, '%Y-%m-%d %H:%M')


def print_full_df():
    # allow print all panda columns
    pd__set_option('precision', 4)
    pd__set_option('expand_frame_repr', False)
