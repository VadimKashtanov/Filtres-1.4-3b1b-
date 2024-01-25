#! /usr/bin/python3

from mdl import *

import matplotlib.pyplot as plt

signe = lambda x: (1 if x >= 0 else -1)

plusde50 = lambda x: ((x) if abs(x) > 0.01 else 0)

prixs = I__sources[0]

if __name__ == "__main__":
	mdl = Mdl("mdl.bin")

	#	Lignes
	#print(mdl.lignes)

	#	I dernieres Prediction
	I = 1*7*24
	prixs = list(norme(prixs[-I:]))
	PRIXS = len(mdl.lignes[0])

	print("Calcule ...")
	pred = mdl()
	print("Fin Calcule")

	plt.plot(norme(prixs[DEPART:])); plt.plot(norme(pred)); plt.show()

	#	Prixs && predictions
	'''plt.plot([2*x-1 for x in prixs], label='prixs')
	for i in range(I):
		s = 0
		a = [s:=(s + 0.1*signe(pred[i][j])) for j in range(len(pred[i]))]
		b = [i+j for j in range(len(pred[i]))]
		plt.plot(a, b)

	#plt.plot(pred, label='pred')

	#	Horizontale et verticales
	plt.plot([0 for _ in pred], label='-')
	for i in range(len(pred)): plt.plot([i for _ in pred], e_norme(list(range(len(pred)))), '--')'''

	#	plt
	plt.legend()
	plt.show()

	##	================ Gain ===============
	u = 50
	usd = []
	T = 2*7*24
	#
	decale = 0
	LEVIER = 25
	#
	for i in range(T-1):
		u += u * signe(pred[I_PRIXS-1-T+i])*(prixs[I_PRIXS-1-T+i+1]/prixs[I_PRIXS-1-T+i]-1)*LEVIER
		if (u <= 0): u = 0
		print(f"usd = {u}")
		usd += [u]
	plt.plot(usd); plt.show()