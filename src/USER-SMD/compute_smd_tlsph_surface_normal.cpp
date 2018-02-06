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
#include "compute_smd_tlsph_surface_normal.h"
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

ComputeSMDTLSPHSurfaceNormal::ComputeSMDTLSPHSurfaceNormal(LAMMPS *lmp, int narg, char **arg) :
		Compute(lmp, narg, arg) {
	if (narg != 3)
		error->all(FLERR, "Illegal compute smd/tlsph_surface_normal command");

	peratom_flag = 1;
	size_peratom_cols = 3;

	nmax = 0;
	surface_normal_vector = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeSMDTLSPHSurfaceNormal::~ComputeSMDTLSPHSurfaceNormal() {
	memory->sfree(surface_normal_vector);
}

/* ---------------------------------------------------------------------- */

void ComputeSMDTLSPHSurfaceNormal::init() {

	int count = 0;
	for (int i = 0; i < modify->ncompute; i++)
		if (strcmp(modify->compute[i]->style, "smd/tlsph_surface_normal") == 0)
			count++;
	if (count > 1 && comm->me == 0)
		error->warning(FLERR, "More than one compute smd/tlsph_surface_normal");
}

/* ---------------------------------------------------------------------- */

void ComputeSMDTLSPHSurfaceNormal::compute_peratom() {
	invoked_peratom = update->ntimestep;

	// grow vector array if necessary

	if (atom->nmax > nmax) {
		memory->destroy(surface_normal_vector);
		nmax = atom->nmax;
		memory->create(surface_normal_vector, nmax, size_peratom_cols, "surfaceNormalVector");
		array_atom = surface_normal_vector;
	}

	int itmp = 0;
	Vector3d *N = (Vector3d *) force->pair->extract("smd/tlsph/surfaceNormal_ptr", itmp);
	if (N == NULL) {
		error->all(FLERR,
				"compute smd/tlsph_surface_normal could not access surface normal. Are the matching pair styles present?");
	}

	int nlocal = atom->nlocal;

	for (int i = 0; i < nlocal; i++) {

		surface_normal_vector[i][0] = N[i](0); // x
		surface_normal_vector[i][1] = N[i](1); // y
		surface_normal_vector[i][2] = N[i](2); // z
	}
}

/* ----------------------------------------------------------------------
 memory usage of local atom-based array
 ------------------------------------------------------------------------- */

double ComputeSMDTLSPHSurfaceNormal::memory_usage() {
	double bytes = size_peratom_cols * nmax * sizeof(double);
	return bytes;
}
