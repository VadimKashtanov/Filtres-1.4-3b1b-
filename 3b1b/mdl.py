import struct as st

from bitget import __sources

def lire_uint(I, _bin):
	l = st.unpack('I'*I, _bin[:st.calcsize('I')*I])
	return l, _bin[st.calcsize('I')*I:]

def lire_flotants(I, _bin):
	l = st.unpack('f'*I, _bin[:st.calcsize('f')*I])
	return l, _bin[st.calcsize('f')*I:]

def norme(arr):
	_min = min(arr)
	_max = max(arr)
	return [(e-_min)/(_max-_min) for e in arr]

def e_norme(arr):
	_min = min(arr)
	_max = max(arr)
	return [2*(e-_min)/(_max-_min)-1 for e in arr]

def ema(arr, K):
	e = [arr[0]]
	for p in arr[1:]:
		e += [e[-1]*(1-1/(1+K)) + p*1/(1+K)]
	return e

with open("structure_generale.bin", "rb") as co:
	bins = co.read()

	constantes, bins = lire_uint(18, bins)

	exec("""P,
		P_INTERV,
		N,
		MAX_INTERVALLES,
		MAX_DECALES,
		SOURCES,
		MAX_PARAMS, NATURES,
		MAX_EMA, MAX_PLUS, MAX_COEF_MACD,
		NORMER_LES_FILTRES, BORNER_LES_FILTRES,
		C, MAX_Y, BLOQUES, F_PAR_BLOQUES,
		INSTS
		""".replace('\n', '') + " = constantes")

	min_param, bins = lire_uint(NATURES*MAX_PARAMS, bins)
	max_param, bins = lire_uint(NATURES*MAX_PARAMS, bins)

	NATURE_PARAMS, bins = lire_uint(NATURES, bins)

MIN_EMA = 1
MIN_NATURES = 0
MIN_NATURES = NATURES-1
MIN_INTERVALLES = 1

########################

def filtre_prixs__poids(Y,X):
	return Y*N

def dot1d__poids(Y,X):
	return (X+1)*N

########################

def filtre_prixs__f(
	ligne, intervalle, decale,
	t, f):
	#
	for b in range(BLOQUES):
		for _f in range(F_PAR_BLOQUES):
			x = norme([ligne[t-(decale+i)*_intervalle] for i in range(N)])#[::-1] j'avais pas inverser dans le C mais pas gravce, ca change rien si j'oublie pas
			#
			s = (sum((1+abs(x[i]-f[b*F_PAR_BLOQUES*N+_f*N+i]))**.5 for i in range(N))) / N - 1
			d = (sum((1+abs(x[i+1]-x[i]-f[b*F_PAR_BLOQUES*N+_f*N+i+1]+f[b*F_PAR_BLOQUES*N+_f*N+i]))**2 for i in range(N-1))) / (N-1) - 1
			#
			y[b*F_PAR_BLOQUES+_f] = 2*exp(-s*s -d*d)-1

def dot1d__f(x, p, y):
	X = len(x)
	for i in range(len(y)):
		y[i] = tanh(sum(p[(X+1)*i + j]*x[j] for j in range(X)) + p[(X+1)*i + (X+1-1)])

########################

inst_poids = [
	filtre_prixs__poids,
	dot1d__poids,
	0,
	0
]

inst_f = [
	filtre_prixs__f,
	dot1d__f,
	0,
	0
]

########################

### Natures ###

def __nature__ema(source, params):
	return source

def __nature_macd(source, params):
	coef, = params
	#
	assert coef > 0.0
	ema12 = ema(source, 12*coef);
	ema26 = ema(source, 26*coef);
	_macd = [ema12[i]-ema26[i] for i in range(len(source))]
	ema9  = ema(_macd,  12*coef);
	return [_macd[i] - ema9[i] for i in range(len(source))]

def __nature_chiffre(source, params):
	D, = params
	return [2*(D-min([abs(x-D*round((x+0)/D)), abs(x-D*round((x+D)/D))]))/D-1 for x in source]

__natures = [
	__nature__ema,
	__nature__macd,
	__nature__chiffre
]

########################

class Mdl:
	def __init__(self, fichier):
		with open(fichier, "rb") as co:
			bins = co.read()

		self.Y,     bins = lire_uint(C, bins)
		self.insts, bins = lire_uint(C, bins)

		for inst in self.insts:
			assert inst <= 1

		self.ema_int = []
		self.lignes  = []
		for _ in range(BLOQUES):
			(source, nature, K_ema, intervalle, decale), bins = lire_uint(5, bins)
			params, bins = lire_uint(MAX_PARAMS, bins)

			self.ema_int += [{
				'source'     : source,
				'nature'     : nature,
				'K_ema'      : K_ema,
				'intervalle' : intervalle,
				'decale'     : decale,
				'params'     : params
			}]
			
			assert nature <= max([0,1,2])

			self.lignes += [
				__natures[nature]( ema(__sources[source], K_ema), params)
			]

		self.p = []
		self.poids = []
		for i in range(C):
			Y, X = (self.Y[i-1] if i!=0 else 0), self.Y[i]
			self.poids += [inst_poids[self.insts[i]](Y,X)]
			poids, bins = lire_flotants(self.poids[i], bins)
			self.p += [poids]

	def ecrire(self, fichier):
		with open(fichier, "wb") as co:
			co.write(st.pack('I'*C, *self.Y))
			co.write(st.pack('I'*C, *self.insts))
			#
			for ema_int in self.ema_int:
				co.write(st.pack(
					'I'*5,
					ema_int['source'],
					ema_int['nature'],
					ema_int['K_ema'],
					ema_int['intervalle'],
					ema_int['decale']
				))
				co.write(st.pack('I'*MAX_PARAMS, *ema_int['params']))
			#
			for i in range(len(self.p)):
				co.write(st.pack('f'*self.poids[i], *self.p[i]))

	def __call__(self, sources, t):
		assert len(__sources[0]) != 0
		assert len(__sources[0]) == len(__sources[1]) == len(__sources[2]) == len(__sources[3])
		#
		y = [[0 for i in range(Y)] for Y in self.Y]
		for b in range(self.bloques):
			ligne, intervalle, decale = self.lignes[b], self.ema_int['intervalle'], self.ema_int['decale']
			for f in range(self.f_par_bloque):
				y[0][b*self.f_par_bloque + f] = filtre_prixs__f(
					ligne, intervalle, decale,
					t, self.f[b*self.f_par_bloque*N + f*N:b*self.f_par_bloque*N + f*N+N]
				)

		for c in range(1, C):
			inst_f[self.insts[c]](y[c-1], self.p[c], y[c])

		return y[-1]