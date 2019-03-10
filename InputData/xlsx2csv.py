#!/usr/bin/env python3
"""
	CREA ARCHIVOS CSV A PARTIR DE PLANILLA 'DatosEntrada.xlsx' por cada una de las hojas.
"""
import xlrd, csv

NombreArchivoExcel = 'DatosEntrada.xlsx'

Libro = xlrd.open_workbook(NombreArchivoExcel)
for hoja in Libro.sheets():
	# para cada hoja del libreo obtiene el nombre y lo usa para crear archivos csv independientes
	NombreHoja = hoja.name
	with open(NombreHoja+'.csv','w') as f:
		writer = csv.writer(f)
		# writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC, quotechar="'")
		for NumeroFila in range(hoja.nrows):
			writer.writerow( hoja.row_values(NumeroFila) )
