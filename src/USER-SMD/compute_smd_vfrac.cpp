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
#include "compute_smd_vfrac.h"
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeSMDVFrac::ComputeSMDVFrac(LAMMPS *lmp, int narg, char **arg) :
		Compute(lmp, narg, arg) {
	if (narg != 3)
		error->all(FLERR, "Illegal compute smd/vfrac command");
	if (atom->vfrac_flag != 1)
		error->all(FLERR, "compute smd/vfrac command requires atom_style with volume (e.g. smd)");

	peratom_flag = 1;
	size_peratom_cols = 0;

	nmax = 0;
	vfracVector = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeSMDVFrac::~ComputeSMDVFrac() {
	memory->sfree(vfracVector);
}

/* ---------------------------------------------------------------------- */

void ComputeSMDVFrac::init() {

	int count = 0;
	for (int i = 0; i < modify->ncompute; i++)
		if (strcmp(modify->compute[i]->style, "smd/vfrac") == 0)
			count++;
	if (count > 1 && comm->me == 0)
		error->warning(FLERR, "More than one compute smd/vfrac");
}

/* ---------------------------------------------------------------------- */

void ComputeSMDVFrac::compute_peratom() {
	invoked_peratom = update->ntimestep;

	// grow vfracVector array if necessary

	if (atom->nmax > nmax) {
		memory->sfree(vfracVector);
		nmax = atom->nmax;
		vfracVector = (double *) memory->smalloc(nmax * sizeof(double), "atom:vfracVector");
		vector_atom = vfracVector;
	}

	double *vfrac = atom->vfrac;
	int *mask = atom->mask;
	int nlocal = atom->nlocal;

	for (int i = 0; i < nlocal; i++) {
		if (mask[i] & groupbit) {
			vfracVector[i] = vfrac[i];
		} else {
			vfracVector[i] = 0.0;
		}
	}

}

/* ----------------------------------------------------------------------
 memory usage of local atom-based array
 ------------------------------------------------------------------------- */

double ComputeSMDVFrac::memory_usage() {
	double bytes = nmax * sizeof(double);
	return bytes;
}
