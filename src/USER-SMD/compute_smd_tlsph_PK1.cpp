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
#include "compute_smd_tlsph_PK1.h"
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

ComputeSMDTLSPHPK1::ComputeSMDTLSPHPK1(LAMMPS *lmp, int narg, char **arg) :
                Compute(lmp, narg, arg) {
        if (narg != 3)
                error->all(FLERR, "Illegal compute smd/ulsph_PK1 command");

        peratom_flag = 1;
        size_peratom_cols = 9;

        nmax = 0;
        PK1_array = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeSMDTLSPHPK1::~ComputeSMDTLSPHPK1() {
        memory->sfree(PK1_array);
}

/* ---------------------------------------------------------------------- */

void ComputeSMDTLSPHPK1::init() {

        int count = 0;
        for (int i = 0; i < modify->ncompute; i++)
                if (strcmp(modify->compute[i]->style, "smd/ulsph_PK1") == 0)
                        count++;
        if (count > 1 && comm->me == 0)
                error->warning(FLERR, "More than one compute smd/ulsph_PK1");
}

/* ---------------------------------------------------------------------- */

void ComputeSMDTLSPHPK1::compute_peratom() {
        invoked_peratom = update->ntimestep;

        // grow vector array if necessary

        if (atom->nmax > nmax) {
                memory->destroy(PK1_array);
                nmax = atom->nmax;
                memory->create(PK1_array, nmax, size_peratom_cols, "stresstensorVector");
                array_atom = PK1_array;
        }

        int itmp = 0;
        Matrix3d *P = (Matrix3d *) force->pair->extract("smd/tlsph/PK1_ptr", itmp);
        if (P == NULL) {
                error->all(FLERR,
                                "compute smd/tlsph_PK1 could not access strain rate. Are the matching pair styles present?");
        }

        int nlocal = atom->nlocal;

        for (int i = 0; i < nlocal; i++) {

                PK1_array[i][0] = P[i](0, 0); // xx
                PK1_array[i][1] = P[i](0, 1); // xy
                PK1_array[i][2] = P[i](0, 2); // xz
                PK1_array[i][3] = P[i](1, 0); // yx
                PK1_array[i][4] = P[i](1, 1); // yy
                PK1_array[i][5] = P[i](1, 2); // yz
                PK1_array[i][3] = P[i](2, 0); // zx
                PK1_array[i][4] = P[i](2, 1); // zy
                PK1_array[i][5] = P[i](2, 2); // zz
        }
}

/* ----------------------------------------------------------------------
 memory usage of local atom-based array
 ------------------------------------------------------------------------- */

double ComputeSMDTLSPHPK1::memory_usage() {
        double bytes = size_peratom_cols * nmax * sizeof(double);
        return bytes;
}
