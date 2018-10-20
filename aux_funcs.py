from datetime import timedelta as dt__timedelta
from pandas import DataFrame as pd__DataFrame
from pandas import datetime as pd__datetime, set_option as pd__set_option

import locale

locale.setlocale(locale.LC_ALL, 'es_ES.utf8')


def date_parser(x):
    return pd__datetime.strptime(x, '%Y-%m-%d %H:%M')


def print_full_df():
    # allow print all panda columns
    pd__set_option('precision', 4)
    pd__set_option('expand_frame_repr', False)


def Crea_Etapas_desde_Cambio_Mant(DF_CambioFechas, ref=True):
    fila = 0
    NumFilas = len(DF_CambioFechas.index) - 1
    # Inicializa la primera fecha (inicio simulación, ya que está ordenada)
    ListaFechasFinales = [ DF_CambioFechas.loc[fila, 0] ]
    while fila < NumFilas:
        # recorre las filas, tal que la fila actual depende de fila + i
        i = 0   # indicador de cuantas filas hay que desplazarse desde 'fila' para siguiente valor no cercano.
        Continuar = True
        while Continuar:    # parecido a un 'do-while'
            if ref is True:
                # calcula la diferencia temporal entre el de referencia y el siguiente
                NextHorasDiff = DF_CambioFechas.loc[fila + i + 1, 0] - DF_CambioFechas.loc[fila, 0]
            else:
                # calcula la diferencia temporal entre el instante de tiempo actual y el siguiente
                NextHorasDiff = DF_CambioFechas.loc[fila + i + 1, 0] - DF_CambioFechas.loc[fila + i, 0]
            Condicion1 = NextHorasDiff >= dt__timedelta(days=1)  # condición: diferencia de tiempo sea mayor o igual a 1 día (type: timedelta)
            Condicion2 = fila + i > NumFilas    # condición: supere el número de filas DataFrame
            i += 1
            if Condicion1 | Condicion2:
                Continuar = False
        """
        Agrega a una lista la primera coincidencia de las fechas cercanas. Notar que el largo de 'ListaFechasFinales'
        es el Número de etapas resultantes + 1
        """
        ListaFechasFinales.append( DF_CambioFechas.loc[fila + i, 0] )
        fila += i

    # Convierte la lista de fechas finales en una lista para ser ingresada al data del DataFrame de salida.
    LAux = []
    for IndFecha in range(len(ListaFechasFinales) - 1):
        LAux.append( [IndFecha + 1, ListaFechasFinales[IndFecha], ListaFechasFinales[IndFecha + 1]] )
    # print('DF_Etapas:\n', pd__DataFrame(data=LAux, columns=['EtaNum', 'FechaIni', 'FechaFin']).set_index('EtaNum'))

    return pd__DataFrame(data=LAux, columns=['EtaNum', 'FechaIni', 'FechaFin']).set_index('EtaNum')
