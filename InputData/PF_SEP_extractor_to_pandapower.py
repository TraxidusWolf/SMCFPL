import json, os, operator
import powerfactory as pf
app = pf.GetApplication()
project = app.GetActiveProject()
app.ClearOutputWindow()	# limpiando la casa
app.PrintInfo("Utilizando proyecto: {}".format(project.loc_name))

def Extrae_tipo(PF_element_extens, Parametros, Ruta, json_out_name, verbose_level=1):
	"""
		Lee los elementos con la extensión 'PF_element_extens' desde DIgSILENT y los escribe en un archivo json llamado 'json_out_name'.json

		PF_element_extens (str) : 	Nombre de la extensión que se quiere sacar información, e.g., para *.TypLne' usar 'TypLne'.
		RelParam (list)         : 	Lista de nombres de parámetros pandapower a extraer. La conversión de unidades se hace internamente.
		Ruta (str)              : 	Ruta completa (temrinada en \\) del directorio donde se crea el archivo de salida.	E.g. '\\\\VBOXSVR\\to_w7\\DIgSILENT\\'
		json_out_name (str)     : 	Nombre del archivo tipo *.json . E.g. 'output'
		vebose (int)            : 	Si 0 no imprime valores en output window. Si 1 imprime solo pasos generales (INFO). Si 2 imprime todo (DEBUG). Otro valor asume 2

		ConversionParametros={ 'ElmPF' : {'parametro PandaPower': (Multiplicador, atributo completo PF)} }	# utiliza attrgetter para extraer
	"""
	# Revisa valores posibles verbose_levels
	if verbose_level not in [0, 1, 2]:
		verbose_level = 2
	# Revisa que los tipos de variables ingresados sean corretos
	if type(PF_element_extens) != str:
		app.PrintError("Variable 'PF_element_extens' debe ser string representando extensión de objetos DIgSILENT.")
		raise ValueError("Variable 'PF_element_extens' debe ser string representando extensión de objetos DIgSILENT.")

	ConversionParametros={
		# completar a necesidad
		'TypLne' : {
			'name'         : 	(None,	'loc_name') ,
			'max_i_ka'     : 	(None,	'sline'   ) ,
			'c_nf_per_km'  : 	(1e3,	'cline'   ) ,
			'r_ohm_per_km' : 	(None,	'rline'   ) ,
			'x_ohm_per_km' : 	(None,	'xline'   )
		},
		'TypTr2' : {
			'name'         : 	(None,	'loc_name') ,
			'sn_kva'       : 	(1e3,	'strn'    ) ,
			'vn_hv_kv'     : 	(None,	'utrn_h'  ) ,
			'vn_lv_kv'     : 	(None,	'utrn_l'  ) ,
			'vsc_percent'  : 	(None,	'uktr'    ) ,
			'vscr_percent' : 	(None,	'uktrr'   )	,
			'pfe_kw'       : 	(None,	'pfe'     ) ,
			'i0_percent'   : 	(None,	'curmg'   ) ,
			'shift_degree' : 	(30,	'nt2ag'   )
		},
		'TypTr3' : {
			'name'             : 	(None,	'loc_name' ) ,
			'sn_hv_kva'	       : 	(1e3,	'strn3_h'  ) ,
			'sn_mv_kva'	       : 	(1e3,	'strn3_m'  ) ,
			'sn_lv_kva'	       : 	(1e3,	'strn3_l'  ) ,
			'vn_hv_kv'	       : 	(None,	'utrn3_h'  ) ,
			'vn_mv_kv'	       : 	(None,	'utrn3_m'  ) ,
			'vn_lv_kv'	       : 	(None,	'utrn3_l'  ) ,
			'vsc_hv_percent'   : 	(None,	'uktr3_h'  ) ,
			'vsc_mv_percent'   : 	(None,	'uktr3_m'  ) ,
			'vsc_lv_percent'   : 	(None,	'uktr3_l'  ) ,
			'vscr_hv_percent'  : 	(None,	'uktrr3_h' ) ,
			'vscr_mv_percent'  : 	(None,	'uktrr3_m' ) ,
			'vscr_lv_percent'  : 	(None,	'uktrr3_l' ) ,
			'pfe_kw'	       : 	(None,	'pfe'      ) ,
			'i0_percent'	   : 	(None,	'curmg'    ) ,
			'shift_mv_degree'  : 	(30,	'nt3ag_m'  ) ,
			'shift_lv_degree'  : 	(30,	'nt3ag_l'  )
		},
		'ElmLne' : {
			'name'       : (None , 'loc_name')            ,
			'std_type'   : (None , 'typ_id.loc_name')     ,
			'parallel'   : (None , 'nlnum')               ,
			'length_km'  : (None , 'dline')               ,
			'in_service' : (None , 'outserv')             ,	# negado se hace internamente
			'from_busNom': (None , 'bus1.cterm.loc_name') ,
			'to_busNom'  : (None , 'bus2.cterm.loc_name') ,
		},
		'ElmTerm' : {
			'name'       : (None , 'loc_name') ,
			'vn_kv'      : (None , 'uknom')    ,
			'in_service' : (None , 'outserv')  ,	# negado se hace internamente
		},
		'ElmSym' : {
			'name'       : (None , 'loc_name') ,
			'BusNom'     : (None , 'bus1.cterm.loc_name') ,
			'p_kw'       : (-1   , 'pgini') ,
			'vm_pu'      : (None , 'usetp') ,
			'sn_kva'     : (None , 'typ_id.sgn') ,
			'in_service' : (None , 'outserv') ,	# negado se hace internamente
			'EsSlack'    : (None , 'ip_ctrl') ,
		},
		'ElmLod' : {
			'name'       : (None , 'loc_name')            ,
			'BusNom'     : (None , 'bus1.cterm.loc_name') ,
			'p_kw'       : (None , 'plini')               ,
			'q_kvar'     : (None , 'qlini')               ,
			'in_service' : (None , 'outserv')             , 	# negado se hace internamente
			'sn_kva'     : (None , 'slini')               ,
		},
		'ElmTr2' : {
			'name'       : (None , 'loc_name')             ,
			'std_type'   : (None , 'typ_id.loc_name')      ,
			'parallel'   : (None , 'ntnum')                ,
			'in_service' : (None , 'outserv')              , 	# negado se hace internamente
			'hv_busNom'  : (None , 'bushv.cterm.loc_name') ,
			'lv_busNom'  : (None , 'buslv.cterm.loc_name') ,
		},
		'ElmTr3' : {
			'name'       : (None , 'loc_name')             ,
			'std_type'   : (None , 'typ_id.loc_name')      ,
			'in_service' : (None , 'outserv')              , 	# negado se hace internamente
			'hv_busNom'  : (None , 'bushv.cterm.loc_name') ,
			'mv_busNom'  : (None , 'busmv.cterm.loc_name') ,
			'lv_busNom'  : (None , 'buslv.cterm.loc_name') ,
		},
	}
	if verbose_level == 1:	app.PrintInfo("Leyendo elementos {} ...".format(PF_element_extens))
	DictSalida = {}
	# Busca las coincidencias del tipo en el SEP
	Elementos = app.GetCalcRelevantObjects(PF_element_extens)
	if not Elementos:
		app.PrintPlain("No hay elementos '{}' encontrados.".format(PF_element_extens))
	for elmn in Elementos:
		if verbose_level==2:	app.PrintInfo( elmn )	# # escribe en output window ssi verbose_level es 2
		ElmnNom = elmn.loc_name
		DictSalida[ElmnNom] = {}
		# Itera sobre los parámetros ingresados a la función
		for param in Parametros:
			Multiplicador = ConversionParametros[PF_element_extens][param][0]
			# app.PrintInfo( 'Multiplicador: {}'.format(Multiplicador) )
			Atributo = ConversionParametros[PF_element_extens][param][1]
			if param == 'in_service':
				valorAttr = not operator.attrgetter(Atributo)(elmn)	# permite obtener nested attributes. No como getattr(class, attr)
			elif Multiplicador:
				valorAttr = operator.attrgetter(Atributo)(elmn)*Multiplicador	# permite obtener nested attributes. No como getattr(class, attr)
			else:
				valorAttr = operator.attrgetter(Atributo)(elmn)	# permite obtener nested attributes. No como getattr(class, attr)
			# Actualiza valores
			DictSalida[ElmnNom][param] = valorAttr
			if verbose_level==2:	app.PrintInfo( '{} <==> {}: {}'.format(param, Atributo, valorAttr) )	# escribe en output window ssi verbose_level es 2
		if verbose_level==2:	app.PrintPlain("")	# escribe en output window ssi verbose_level==2
	if verbose_level in [1,2]:	app.PrintInfo("Elementos obtenidos!\n")	# escribe en output window ssi verbose_level es 1 o 2
	
	if verbose_level in [1,2]:	app.PrintInfo("Escribiendo elementos '*.{}'' en '{}.json' ...".format(PF_element_extens, json_out_name))	# escribe en output window ssi verbose_level es 1 o 2
	# Guarda los tipo sen archivo json
	with open(RutaDestino+json_out_name+'.json', 'w', newline='') as f:
		json.dump(DictSalida, f)
	app.PrintInfo("Archivo escrito!\n\n")










# PARÁMETROS DE CONFIGURACIÓN
RutaDestino = "\\\\VBOXSVR\\to_w7\\DIgSILENT\\DPL\\Python\\extrae_json_data_39bus\\"
# VarName = (¿Extrae?, verbose_level)
TipoLin  = (True, 1)
TipoTr2  = (True, 1)
TipoTr3  = (True, 1)
ElmnLin  = (True, 1)
ElmnTr2  = (True, 1)
ElmnTr3  = (True, 1)
ElmnGen  = (True, 1)
ElmnLoad = (True, 1)
ElmnBar  = (True, 1)

# 	######## ##    ## ########  ##       #### ##    ##
# 	   ##     ##  ##  ##     ## ##        ##  ###   ##
# 	   ##      ####   ##     ## ##        ##  ####  ##
# 	   ##       ##    ########  ##        ##  ## ## ##
# 	   ##       ##    ##        ##        ##  ##  ####
# 	   ##       ##    ##        ##        ##  ##   ###
# 	   ##       ##    ##        ######## #### ##    ##
if TipoLin[0]:
	Parametros = ['name', 'max_i_ka', 'c_nf_per_km', 'r_ohm_per_km', 'x_ohm_per_km']
	Extrae_tipo(PF_element_extens='TypLne', Parametros=Parametros, Ruta=RutaDestino, json_out_name='tipo_lineas', verbose_level=TipoLin[1])
# 	######## ##    ## ########  ######## ########   #######
# 	   ##     ##  ##  ##     ##    ##    ##     ## ##     ##
# 	   ##      ####   ##     ##    ##    ##     ##        ##
# 	   ##       ##    ########     ##    ########   #######
# 	   ##       ##    ##           ##    ##   ##   ##
# 	   ##       ##    ##           ##    ##    ##  ##
# 	   ##       ##    ##           ##    ##     ## #########
if TipoTr2[0]:
	Parametros = ['name', 'sn_kva', 'vn_hv_kv', 'vn_lv_kv', 'vsc_percent', 'vscr_percent', 'pfe_kw', 'i0_percent', 'shift_degree']
	Extrae_tipo(PF_element_extens='TypTr2', Parametros=Parametros, Ruta=RutaDestino, json_out_name='tipo_trafos2w', verbose_level=TipoTr2[1])
# 	######## ##    ## ########  ######## ########   #######
# 	   ##     ##  ##  ##     ##    ##    ##     ## ##     ##
# 	   ##      ####   ##     ##    ##    ##     ##        ##
# 	   ##       ##    ########     ##    ########   #######
# 	   ##       ##    ##           ##    ##   ##          ##
# 	   ##       ##    ##           ##    ##    ##  ##     ##
# 	   ##       ##    ##           ##    ##     ##  #######
if TipoTr3[0]:
	Parametros = ['name', 'sn_hv_kva', 'sn_mv_kva', 'sn_lv_kva', 'vn_hv_kv', 'vn_mv_kv', 'vn_lv_kv', 'vsc_hv_percent', 'vsc_mv_percent', 'vsc_lv_percent', 'vscr_hv_percent', 'vscr_mv_percent', 'vscr_lv_percent', 'pfe_kw', 'i0_percent', 'shift_mv_degree', 'shift_lv_degree']
	Extrae_tipo(PF_element_extens='TypTr3', Parametros=Parametros, Ruta=RutaDestino, json_out_name='tipo_trafos3w', verbose_level=TipoTr3[1])
# 	##       #### ##    ## ########    ###     ######
# 	##        ##  ###   ## ##         ## ##   ##    ##
# 	##        ##  ####  ## ##        ##   ##  ##
# 	##        ##  ## ## ## ######   ##     ##  ######
# 	##        ##  ##  #### ##       #########       ##
# 	##        ##  ##   ### ##       ##     ## ##    ##
# 	######## #### ##    ## ######## ##     ##  ######
if ElmnLin[0]:
	Parametros = ['name','std_type','parallel','length_km','in_service','from_busNom','to_busNom']
	Extrae_tipo(PF_element_extens='ElmLne', Parametros=Parametros, Ruta=RutaDestino, json_out_name='lineas', verbose_level=ElmnLin[1])
# 	########     ###    ########  ########     ###     ######
# 	##     ##   ## ##   ##     ## ##     ##   ## ##   ##    ##
# 	##     ##  ##   ##  ##     ## ##     ##  ##   ##  ##
# 	########  ##     ## ########  ########  ##     ##  ######
# 	##     ## ######### ##   ##   ##   ##   #########       ##
# 	##     ## ##     ## ##    ##  ##    ##  ##     ## ##    ##
# 	########  ##     ## ##     ## ##     ## ##     ##  ######
if ElmnBar[0]:
	Parametros = ['name', 'vn_kv', 'in_service']
	Extrae_tipo(PF_element_extens='ElmTerm', Parametros=Parametros, Ruta=RutaDestino, json_out_name='terminales', verbose_level=ElmnBar[1])
# 	 ######   ######## ##    ##  ######
# 	##    ##  ##       ###   ## ##    ##
# 	##        ##       ####  ## ##
# 	##   #### ######   ## ## ##  ######
# 	##    ##  ##       ##  ####       ##
# 	##    ##  ##       ##   ### ##    ##
# 	 ######   ######## ##    ##  ######
if ElmnGen[0]:
	Parametros = ['name', 'BusNom', 'p_kw', 'vm_pu', 'sn_kva', 'in_service', 'EsSlack']
	Extrae_tipo(PF_element_extens='ElmSym', Parametros=Parametros, Ruta=RutaDestino, json_out_name='unidades', verbose_level=ElmnGen[1])
# 	##        #######     ###    ########   ######
# 	##       ##     ##   ## ##   ##     ## ##    ##
# 	##       ##     ##  ##   ##  ##     ## ##
# 	##       ##     ## ##     ## ##     ##  ######
# 	##       ##     ## ######### ##     ##       ##
# 	##       ##     ## ##     ## ##     ## ##    ##
# 	########  #######  ##     ## ########   ######
if ElmnLoad[0]:
	Parametros = ['name', 'BusNom', 'p_kw', 'q_kvar', 'in_service', 'sn_kva']
	Extrae_tipo(PF_element_extens='ElmLod', Parametros=Parametros, Ruta=RutaDestino, json_out_name='cargas', verbose_level=ElmnLoad[1])
# 	######## ########     ###    ########  #######   #######  ##      ##
# 	   ##    ##     ##   ## ##   ##       ##     ## ##     ## ##  ##  ##
# 	   ##    ##     ##  ##   ##  ##       ##     ##        ## ##  ##  ##
# 	   ##    ########  ##     ## ######   ##     ##  #######  ##  ##  ##
# 	   ##    ##   ##   ######### ##       ##     ## ##        ##  ##  ##
# 	   ##    ##    ##  ##     ## ##       ##     ## ##        ##  ##  ##
# 	   ##    ##     ## ##     ## ##        #######  #########  ###  ###
if ElmnTr2[0]:
	Parametros = ['name', 'std_type', 'parallel', 'in_service', 'hv_busNom', 'lv_busNom']
	Extrae_tipo(PF_element_extens='ElmTr2', Parametros=Parametros, Ruta=RutaDestino, json_out_name='trafos2w', verbose_level=ElmnTr2[1])
# 	######## ########     ###    ########  #######   #######  ##      ##
# 	   ##    ##     ##   ## ##   ##       ##     ## ##     ## ##  ##  ##
# 	   ##    ##     ##  ##   ##  ##       ##     ##        ## ##  ##  ##
# 	   ##    ########  ##     ## ######   ##     ##  #######  ##  ##  ##
# 	   ##    ##   ##   ######### ##       ##     ##        ## ##  ##  ##
# 	   ##    ##    ##  ##     ## ##       ##     ## ##     ## ##  ##  ##
# 	   ##    ##     ## ##     ## ##        #######   #######   ###  ###
if ElmnTr3[0]:
	Parametros = ['name', 'std_type', 'in_service', 'hv_busNom', 'mv_busNom', 'lv_busNom']
	Extrae_tipo(PF_element_extens='ElmTr3', Parametros=Parametros, Ruta=RutaDestino, json_out_name='trafos3w', verbose_level=ElmnTr3[1])








