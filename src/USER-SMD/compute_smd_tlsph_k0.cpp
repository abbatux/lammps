/* ----------------------------------------------------------------------
 LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
 http://lammps.sandia.gov, Sandia National Laboratories
 Steve Plimpton, sjplimp@sandia.gov

 Copyright (2003) Sandia Corporation.  Under the terms of Contract
 DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
 certain rights in this software.  This software is distributed under
 the GNU General Public License.

 See the README file in the top-level LAMMPS directory.
 ------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: A. de Vaucorbeil, alban.devaucorbeil@monash.edu
                        Copyright (C) 2018
------------------------------------------------------------------------- */

#include <string.h>
#include "compute_smd_tlsph_k0.h"
#include "fix_smd_tlsph_reference_configuration.h"
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "error.h"
#include "pair.h"
#include <Eigen/Eigen>
using namespace Eigen;
using namespace LAMMPS_NS;


/* ---------------------------------------------------------------------- */

ComputeSMDTLSPHK0::ComputeSMDTLSPHK0(LAMMPS *lmp, int narg, char **arg) :
		Compute(lmp, narg, arg) {
	if (narg != 3)
		error->all(FLERR, "Illegal compute smd/tlsph_k0 command");

	peratom_flag = 1;
	size_peratom_cols = 6;

	nmax = 0;
	K_array = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeSMDTLSPHK0::~ComputeSMDTLSPHK0() {
	memory->sfree(K_array);
}

/* ---------------------------------------------------------------------- */

void ComputeSMDTLSPHK0::init() {

	int count = 0;
	for (int i = 0; i < modify->ncompute; i++)
		if (strcmp(modify->compute[i]->style, "smd/tlsph_k0") == 0)
			count++;
	if (count > 1 && comm->me == 0)
		error->warning(FLERR, "More than one compute smd/tlsph_k0");
}

/* ---------------------------------------------------------------------- */

void ComputeSMDTLSPHK0::compute_peratom() {
	invoked_peratom = update->ntimestep;

	// grow vector array if necessary

	if (atom->nmax > nmax) {
		memory->destroy(K_array);
		nmax = atom->nmax;
		memory->create(K_array, nmax, size_peratom_cols, "k0tensorVector");
		array_atom = K_array;
	}

	int itmp = 0;
	int ifix_tlsph;
	for (int i = 0; i < modify->nfix; i++)
		if (strcmp(modify->fix[i]->style, "SMD_TLSPH_NEIGHBORS") == 0)
			ifix_tlsph = i;
	Matrix3d *T = ((FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[ifix_tlsph])->K0;
	if (T == NULL) {
		error->all(FLERR, "compute smd/tlsph_k0 could not access K0 tensor. Are the matching pair styles present?");
	}
	int nlocal = atom->nlocal;
	int *mask = atom->mask;

	for (int i = 0; i < nlocal; i++) {
		if (mask[i] & groupbit) {
			K_array[i][0] = T[i](0, 0); // xx
			K_array[i][1] = T[i](1, 1); // yy
			K_array[i][2] = T[i](2, 2); // zz
			K_array[i][3] = T[i](0, 1); // xy
			K_array[i][4] = T[i](0, 2); // xz
			K_array[i][5] = T[i](1, 2); // yz
		} else {
			for (int j = 0; j < size_peratom_cols; j++) {
				K_array[i][j] = 0.0;
			}
		}
	}
}

/* ----------------------------------------------------------------------
 memory usage of local atom-based array
 ------------------------------------------------------------------------- */

double ComputeSMDTLSPHK0::memory_usage() {
	double bytes = size_peratom_cols * nmax * sizeof(double);
	return bytes;
}
