import time
import datetime

import requests

requette_bitget = lambda de, a: eval(
	requests.get(
		f"https://api.bitget.com/api/mix/v1/market/history-candles?symbol=BTCUSDT_UMCBL&granularity=1H&startTime={de*1000}&endTime={a*1000}"
	).text
)

donnees = []
H = (8+0)*256  #200
la = int(time.time())
for i in range(int(1000*8/H + 1)):
	derniere = requette_bitget(la-(i+1)*H*60*60, la-i*H*60*60)[::-1]
	donnees += derniere
	if i%1 == 0: print(f"%% = {i/int(1000*8/H + 1)*100},   len(derniere)={len(derniere)}")
donnees = donnees[::-1]

prixs   = [float(c)                       for _,o,h,l,c,vB,vU in donnees]
hight   = [float(h)                       for _,o,h,l,c,vB,vU in donnees]
low     = [float(l)                       for _,o,h,l,c,vB,vU in donnees]
volumes = [float(c)*float(vB) - float(vU) for _,o,h,l,c,vB,vU in donnees]

__sources = [
	prixs,
	volumes,
	hight,
	low
]