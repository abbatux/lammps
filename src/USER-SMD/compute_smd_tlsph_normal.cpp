/* ----------------------------------------------------------------------
 *
 *                    *** Smooth Mach Dynamics ***
 *
 * This file is part of the USER-SMD package for LAMMPS.
 * Copyright (2014) Georg C. Ganzenmueller, georg.ganzenmueller@emi.fhg.de
 * Fraunhofer Ernst-Mach Institute for High-Speed Dynamics, EMI,
 * Eckerstrasse 4, D-79104 Freiburg i.Br, Germany.
 *
 * ----------------------------------------------------------------------- */

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

#include <string.h>
#include "compute_smd_tlsph_normal.h"
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

ComputeSMDTLSPHnormal::ComputeSMDTLSPHnormal(LAMMPS *lmp, int narg, char **arg) :
		Compute(lmp, narg, arg) {
	if (narg != 3)
		error->all(FLERR, "Illegal compute smd/tlsph_normal command");

	peratom_flag = 1;
	size_peratom_cols = 3;

	nmax = 0;
	n_array = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeSMDTLSPHnormal::~ComputeSMDTLSPHnormal() {
	memory->sfree(n_array);
}

/* ---------------------------------------------------------------------- */

void ComputeSMDTLSPHnormal::init() {

	int count = 0;
	for (int i = 0; i < modify->ncompute; i++)
		if (strcmp(modify->compute[i]->style, "smd/tlsph_normal") == 0)
			count++;
	if (count > 1 && comm->me == 0)
		error->warning(FLERR, "More than one compute smd/tlsph_normal");
}

/* ---------------------------------------------------------------------- */

void ComputeSMDTLSPHnormal::compute_peratom() {
	invoked_peratom = update->ntimestep;

	// grow vector array if necessary

	if (atom->nmax > nmax) {
		memory->destroy(n_array);
		nmax = atom->nmax;
		memory->create(n_array, nmax, size_peratom_cols, "normaltensorVector");
		array_atom = n_array;
	}

	int itmp = 0;
	int ifix_tlsph;
	for (int i = 0; i < modify->nfix; i++)
		if (strcmp(modify->fix[i]->style, "SMD_TLSPH_NEIGHBORS") == 0)
			ifix_tlsph = i;
	Vector3d *T = ((FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[ifix_tlsph])->normal;
	if (T == NULL) {
		error->all(FLERR, "compute smd/tlsph_normal could not access normal tensor. Are the matching pair styles present?");
	}
	int nlocal = atom->nlocal;
	int *mask = atom->mask;

	for (int i = 0; i < nlocal; i++) {
		if (mask[i] & groupbit) {
			n_array[i][0] = T[i](0); // xx
			n_array[i][1] = T[i](1); // yy
			n_array[i][2] = T[i](2); // zz
		} else {
			for (int j = 0; j < size_peratom_cols; j++) {
				n_array[i][j] = 0.0;
			}
		}
	}
}

/* ----------------------------------------------------------------------
 memory usage of local atom-based array
 ------------------------------------------------------------------------- */

double ComputeSMDTLSPHnormal::memory_usage() {
	double bytes = size_peratom_cols * nmax * sizeof(double);
	return bytes;
}
