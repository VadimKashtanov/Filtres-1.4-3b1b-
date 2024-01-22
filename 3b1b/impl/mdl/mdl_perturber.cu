#include "mdl.cuh"

static uint couche_aleatoire(Mdl_t * mdl) {
	uint a = mdl->inst_POIDS[0] + (rand() % (mdl->total_POIDS-mdl->inst_POIDS[0]));
	
}

static void perturber_fois_zero  (Mdl_t * mdl) {
	uint c = 1 + rand() % (C-1);
	if (mdl->insts[c] == DOT1D) {
		uint X=mdl->Y[c-1], Y=mdl->Y[c];
		mdl->p[c][(X+1)*(rand()%Y) + (rand()%X)] = 0.01;
	}
};

static void perturber_echanger   (Mdl_t * mdl) {
	uint c = 1 + rand() % (C-1);
	if (mdl->insts[c] == DOT1D) {
		uint X=mdl->Y[c-1], Y=mdl->Y[c];
		uint p0 = (X+1)*(rand()%Y) + (rand()%X);
		uint p1 = (X+1)*(rand()%Y) + (rand()%X);
		float vp0 = mdl->p[c][p0], vp1 = mdl->p[c][p1];
		mdl->p[c][p0] = vp1;
		mdl->p[c][p1] = vp0;
	}
};

static void perturber_egale_rnd  (Mdl_t * mdl) {
	uint c = 1 + rand() % (C-1);
	if (mdl->insts[c] == DOT1D) {
		uint X=mdl->Y[c-1], Y=mdl->Y[c];
		mdl->p[c][(X+1)*(rand()%Y) + (rand()%X)] = 2*(rnd()-0.5);
	}
};

//	======================================

typedef void (*perturber_f)(Mdl_t * mdl);

static perturber_f arr_perturber_f[3] = {
	perturber_fois_zero,
	perturber_echanger,
	perturber_egale_rnd
};

void perturber(Mdl_t * mdl, uint L) {
	mdl_gpu_vers_cpu(mdl);
	FOR(0, i, L) arr_perturber_f[rand() % 3](mdl);
	mdl_cpu_vers_gpu(mdl);
};