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
#include <stdlib.h>
#include "fix_smd_setangularvel.h"
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "domain.h"
#include "region.h"
#include "input.h"
#include "variable.h"
#include "memory.h"
#include "error.h"
#include "force.h"

using namespace LAMMPS_NS;
using namespace FixConst;

enum {
        NONE, CONSTANT, EQUAL, ATOM
};

/* ---------------------------------------------------------------------- */

FixSMDSetAngVel::FixSMDSetAngVel(LAMMPS *lmp, int narg, char **arg) :
                Fix(lmp, narg, arg) {
        if (narg < 5)
                error->all(FLERR, "Illegal fix setangularvelocity command");

        dynamic_group_allow = 1;
        vector_flag = 1;
        size_vector = 3;
        global_freq = 1;
        extvector = 1;

	axis = arg[3][0];

	omegastr = NULL;

        if (strstr(arg[4], "v_") == arg[4]) {
	  int n = strlen(&arg[4][2]) + 1;
	  omegastr = new char[n];
	  strcpy(omegastr, &arg[4][2]);
        } else {
	  omegavalue = force->numeric(FLERR, arg[4]);
	  omegastyle = CONSTANT;
        }

        // optional args

        iregion = -1;
        idregion = NULL;

        int iarg = 6;
        while (iarg < narg) {
                if (strcmp(arg[iarg], "region") == 0) {
                        if (iarg + 2 > narg)
                                error->all(FLERR, "Illegal fix setangvelocity command");
                        iregion = domain->find_region(arg[iarg + 1]);
                        if (iregion == -1)
                                error->all(FLERR, "Region ID for fix setangvelocity does not exist");
                        int n = strlen(arg[iarg + 1]) + 1;
                        idregion = new char[n];
                        strcpy(idregion, arg[iarg + 1]);
                        iarg += 2;
                } else
                        error->all(FLERR, "Illegal fix setangvelocity command");
        }

        force_flag = 0;
        foriginal[0] = foriginal[1] = foriginal[2] = 0.0;

        maxatom = atom->nmax;
        memory->create(sforce, maxatom, "setvelocity:sforce");
}

/* ---------------------------------------------------------------------- */

FixSMDSetAngVel::~FixSMDSetAngVel() {
        delete[] omegastr;
        delete[] idregion;
        memory->destroy(sforce);
}

/* ---------------------------------------------------------------------- */

int FixSMDSetAngVel::setmask() {
        int mask = 0;
        //mask |= INITIAL_INTEGRATE;
        mask |= POST_FORCE;
        return mask;
}

/* ---------------------------------------------------------------------- */

void FixSMDSetAngVel::init() {
        // check variables

        if (omegastr) {
                omegavar = input->variable->find(omegastr);
                if (omegavar < 0)
                        error->all(FLERR, "Variable name for fix setangvelocity does not exist");
                if (input->variable->equalstyle(omegavar))
                        omegastyle = EQUAL;
                else if (input->variable->atomstyle(omegavar))
                        omegastyle = ATOM;
                else
                        error->all(FLERR, "Variable for fix setangvelocity is invalid style");
        }

        // set index and check validity of region

        if (iregion >= 0) {
                iregion = domain->find_region(idregion);
                if (iregion == -1)
                        error->all(FLERR, "Region ID for fix setangvelocity does not exist");
        }

        if (omegastyle == ATOM)
	  varflag = ATOM;
        else if (omegastyle == EQUAL)
	  varflag = EQUAL;
        else
	  varflag = CONSTANT;

        // cannot use non-zero forces for a minimization since no energy is integrated
        // use fix addforce instead

        int flag = 0;
        if (update->whichflag == 2) {
                if (omegastyle == EQUAL || omegastyle == ATOM)
                        flag = 1;
                if (omegastyle == CONSTANT && omegavalue != 0.0)
                        flag = 1;
        }
        if (flag)
                error->all(FLERR, "Cannot use non-zero forces in an energy minimization");
}

/* ---------------------------------------------------------------------- */

void FixSMDSetAngVel::setup(int vflag) {
        if (strstr(update->integrate_style, "verlet"))
                post_force(vflag);
        else
      error->all(FLERR,"Fix smd/setangvel does not support RESPA");
}

/* ---------------------------------------------------------------------- */

void FixSMDSetAngVel::min_setup(int vflag) {
        post_force(vflag);
}

/* ---------------------------------------------------------------------- */

//void FixSMDSetAngVel::initial_integrate(int vflag) {
void FixSMDSetAngVel::post_force(int vflag) {
        double **x = atom->x;
        double **f = atom->f;
        double **v = atom->v;
        double **vest = atom->vest;
        int *mask = atom->mask;
        int nlocal = atom->nlocal;

        // update region if necessary

        Region *region = NULL;
        if (iregion >= 0) {
                region = domain->regions[iregion];
                region->prematch();
        }

        // reallocate sforce array if necessary

        if (varflag == ATOM && atom->nmax > maxatom) {
                maxatom = atom->nmax;
                memory->destroy(sforce);
                memory->create(sforce, maxatom, "setangvelocity:sforce");
        }

        foriginal[0] = foriginal[1] = foriginal[2] = 0.0;
        force_flag = 0;

        if (varflag == CONSTANT) {
                for (int i = 0; i < nlocal; i++)
                        if (mask[i] & groupbit) {
                                if (region && !region->match(x[i][0], x[i][1], x[i][2]))
                                        continue;
                                foriginal[0] += f[i][0];
                                foriginal[1] += f[i][1];
                                foriginal[2] += f[i][2];
                                if (omegastyle) {
				  if (axis == 'x') {
				    v[i][0] = 0;
				    v[i][1] = -omegavalue*x[i][2];
				    v[i][2] = omegavalue*x[i][1];
				    vest[i][0] = 0;
				    vest[i][1] = -omegavalue*x[i][2];
				    vest[i][2] = omegavalue*x[i][1];
				  }
				  if (axis == 'y') {
				    v[i][0] = omegavalue*x[i][2];
				    v[i][1] = 0;
				    v[i][2] = -omegavalue*x[i][0];
				    vest[i][0] = omegavalue*x[i][2];
				    vest[i][1] = 0;
				    vest[i][2] = -omegavalue*x[i][0];
				  }
				  if (axis == 'z') {
				    v[i][0] = -omegavalue*x[i][1];
				    v[i][1] = omegavalue*x[i][0];
				    v[i][2] = 0;
				    vest[i][0] = -omegavalue*x[i][1];
				    vest[i][1] = omegavalue*x[i][0];
				    vest[i][2] = 0;
				  }
				  f[i][0] = 0.0;
				  f[i][1] = 0.0;
				  f[i][2] = 0.0;
                                }
                        }

                // variable force, wrap with clear/add

        } else {

                modify->clearstep_compute();

                if (omegastyle == EQUAL)
                        omegavalue = input->variable->compute_equal(omegavar);
                else if (omegastyle == ATOM)
		  input->variable->compute_atom(omegavar, igroup, &sforce[0], 1, 0);

                modify->addstep_compute(update->ntimestep + 1);

                //printf("setting velocity at timestep %d\n", update->ntimestep);

                for (int i = 0; i < nlocal; i++)
                        if (mask[i] & groupbit) {
                                if (region && !region->match(x[i][0], x[i][1], x[i][2]))
                                        continue;
                                foriginal[0] += f[i][0];
                                foriginal[1] += f[i][1];
                                foriginal[2] += f[i][2];
                                if (omegastyle == ATOM) {
				  if (axis == 'x') {
				    v[i][0] = 0;
				    v[i][1] = -sforce[i]*x[i][2];
				    v[i][2] = sforce[i]*x[i][1];
				    vest[i][0] = 0;
				    vest[i][1] = -sforce[i]*x[i][2];
				    vest[i][2] = sforce[i]*x[i][1];
				  }
				  if (axis == 'y') {
				    v[i][0] = sforce[i]*x[i][2];
				    v[i][1] = 0;
				    v[i][2] = -sforce[i]*x[i][0];
				    vest[i][0] = sforce[i]*x[i][2];
				    vest[i][1] = 0;
				    vest[i][2] = -sforce[i]*x[i][0];
				  }
				  if (axis == 'z') {
				    v[i][0] = -sforce[i]*x[i][1];
				    v[i][1] = sforce[i]*x[i][0];
				    v[i][2] = 0;
				    vest[i][0] = -sforce[i]*x[i][1];
				    vest[i][1] = sforce[i]*x[i][0];
				    vest[i][2] = 0;
				  }
				  f[i][0] = 0.0;
				  f[i][1] = 0.0;
				  f[i][2] = 0.0;
                                } else if (omegastyle) {
				  if (axis == 'x') {
				    v[i][0] = 0;
				    v[i][1] = -omegavalue*x[i][2];
				    v[i][2] = omegavalue*x[i][1];
				    vest[i][0] = 0;
				    vest[i][1] = -omegavalue*x[i][2];
				    vest[i][2] = omegavalue*x[i][1];
				  }
				  if (axis == 'y') {
				    v[i][0] = omegavalue*x[i][2];
				    v[i][1] = 0;
				    v[i][2] = -omegavalue*x[i][0];
				    vest[i][0] = omegavalue*x[i][2];
				    vest[i][1] = 0;
				    vest[i][2] = -omegavalue*x[i][0];
				  }
				  if (axis == 'z') {
				    v[i][0] = -omegavalue*x[i][1];
				    v[i][1] = omegavalue*x[i][0];
				    v[i][2] = 0;
				    vest[i][0] = -omegavalue*x[i][1];
				    vest[i][1] = omegavalue*x[i][0];
				    vest[i][2] = 0;
				  }
				  f[i][0] = 0.0;
				  f[i][1] = 0.0;
				  f[i][2] = 0.0;
                                }
                        }
        }
}

/* ----------------------------------------------------------------------
 return components of total force on fix group before force was changed
 ------------------------------------------------------------------------- */

double FixSMDSetAngVel::compute_vector(int n) {
// only sum across procs one time

        if (force_flag == 0) {
                MPI_Allreduce(foriginal, foriginal_all, 3, MPI_DOUBLE, MPI_SUM, world);
                force_flag = 1;
        }
        return foriginal_all[n];
}

/* ----------------------------------------------------------------------
 memory usage of local atom-based array
 ------------------------------------------------------------------------- */

double FixSMDSetAngVel::memory_usage() {
        double bytes = 0.0;
        if (varflag == ATOM)
                bytes = atom->nmax * 3 * sizeof(double);
        return bytes;
}
