/* ----------------------------------------------------------------------
 *
 *                    *** Smooth Mach Dynamics ***
 *
 * This file is part of the USER-SMD package for LAMMPS.
 * Copyright (2014) Georg C. Ganzenmueller, georg.ganzenmueller@emi.fhg.de
 * Fraunhofer Ernst-Mach Institute for High-Speed Dynamics, EMI,
 * Eckerstrasse 4, D-79104 Freiburg i.Br, Germany.
 *
 * This file is based on the FixShearHistory class.
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

#include "lattice.h"
#include <mpi.h>
#include <string.h>
#include <stdio.h>
#include "fix_smd_tlsph_reference_configuration.h"
#include "atom.h"
#include "atom_vec.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "force.h"
#include "pair.h"
#include "update.h"
#include "modify.h"
#include "memory.h"
#include "error.h"
#include "domain.h"
#include <Eigen/Eigen>
#include "smd_kernels.h"
#include "smd_math.h"

using namespace Eigen;
using namespace LAMMPS_NS;
using namespace FixConst;
using namespace SMD_Kernels;
using namespace std;
using namespace SMD_Math;

#define DELTA 16384

#define BUFEXTRA 1000

#define INSERT_PREDEFINED_CRACKS false

/* ---------------------------------------------------------------------- */

FixSMD_TLSPH_ReferenceConfiguration::FixSMD_TLSPH_ReferenceConfiguration(LAMMPS *lmp, int narg, char **arg) :
		Fix(lmp, narg, arg) {

	if (atom->map_style == 0)
		error->all(FLERR, "Pair tlsph with partner list requires an atom map, see atom_modify");

	maxpartner = 1;
	npartner = NULL;
	partner = NULL;
	wfd_list = NULL;
	wf_list = NULL;
	energy_per_bond = NULL;
	degradation_ij = NULL;
	partnerdx = NULL;
	r0 = NULL;
	g_list = NULL;
	K = NULL;
	K_g_dot_dx0_normalized = NULL;
	grow_arrays(atom->nmax);
	atom->add_callback(0);
	atom->add_callback(1); // To add the fix to atom->extra_restart

	// initialize npartner to 0 so neighbor list creation is OK the 1st time
	int nlocal = atom->nlocal;
	for (int i = 0; i < nlocal; i++) {
		npartner[i] = 0;
	}

	comm_forward = 14;
	updateFlag = 1;
	restart_global = 1; // this fix stores global (i.e., not per-atom) info needed at restart: maxpartner
	restart_peratom = 1;

	nrecv = NULL;
	sendlist = NULL;
	maxsendlist = NULL;
	buf_send = NULL;
	buf_recv = NULL;
	maxsend = 0;
	maxrecv = 0;
	n_missing_all = 0;
	need_forward_comm = true;
	nprocs = comm->nprocs;
}

/* ---------------------------------------------------------------------- */

FixSMD_TLSPH_ReferenceConfiguration::~FixSMD_TLSPH_ReferenceConfiguration() {
	// unregister this fix so atom class doesn't invoke it any more

	atom->delete_callback(id, 0);
	atom->delete_callback(id, 1);
// delete locally stored arrays

	memory->destroy(npartner);
	memory->destroy(partner);
	memory->destroy(wfd_list);
	memory->destroy(wf_list);
	memory->destroy(degradation_ij);
	memory->destroy(energy_per_bond);
	memory->destroy(partnerdx);
	memory->destroy(r0);
	memory->destroy(g_list);
	memory->destroy(K);
	memory->destroy(K_g_dot_dx0_normalized);
	
	if (sendlist) for (int i = 0; i < nprocs; i++) memory->destroy(sendlist[i]);
	memory->sfree(sendlist);
	memory->destroy(nrecv);
	memory->destroy(nsendlist);
	memory->destroy(maxsendlist);
	memory->destroy(buf_send);
	memory->destroy(buf_recv);
}

/* ---------------------------------------------------------------------- */

int FixSMD_TLSPH_ReferenceConfiguration::setmask() {
	int mask = 0;
	mask |= PRE_EXCHANGE;
	mask |= POST_NEIGHBOR;
	return mask;
}

/* ---------------------------------------------------------------------- */

void FixSMD_TLSPH_ReferenceConfiguration::init() {
	if (atom->tag_enable == 0)
		error->all(FLERR, "Pair style tlsph requires atoms have IDs");

	sendlist = (int **) memory->smalloc(nprocs*sizeof(int *),"tlsph_refconfig_neigh:sendlist");
	memory->create(maxsendlist,nprocs,"tlsph_refconfig_neigh:maxsendlist");

	for (int i = 0; i < nprocs; i++) {
	  maxsendlist[i] = 1;
	  memory->create(sendlist[i],1,"tlsph_refconfig_neigh:sendlist[i]");
	}
	memory->create(nrecv,nprocs,"tlsph_refconfig_neigh:nrecv");
	memory->create(nsendlist,nprocs,"tlsph_refconfig_neigh:nsendlist");
	memory->create(buf_send,maxsend + BUFEXTRA,"tlsph_refconfig_neigh:buf_send");
	memory->create(buf_recv,maxrecv + BUFEXTRA,"tlsph_refconfig_neigh:buf_recv");
}

/* ---------------------------------------------------------------------- */

void FixSMD_TLSPH_ReferenceConfiguration::pre_exchange() {
	//return;
  // printf("in FixSMD_TLSPH_ReferenceConfiguration::pre_exchange() with updateFlag = %d \n", updateFlag);
	double **defgrad = atom->smd_data_9;
	double *radius = atom->radius;
	double *rho = atom->rho;
	double *vfrac = atom->vfrac;
	double **x = atom->x;
	double **x0 = atom->x0;
	double *rmass = atom->rmass;
	int nlocal = atom->nlocal;
	int i, itmp;
	int *mask = atom->mask;
	if (igroup == atom->firstgroup) {
		nlocal = atom->nfirst;
	}

	int *updateFlag_ptr = (int *) force->pair->extract("smd/tlsph/updateFlag_ptr", itmp);
	if (updateFlag_ptr == NULL) {
		error->one(FLERR,
				"fix FixSMD_TLSPH_ReferenceConfiguration failed to access updateFlag pointer. Check if a pair style exist which calculates this quantity.");
	}

	int *nn = (int *) force->pair->extract("smd/tlsph/numNeighsRefConfig_ptr", itmp);
	if (nn == NULL) {
		error->all(FLERR, "FixSMDIntegrateTlsph::updateReferenceConfiguration() failed to access numNeighsRefConfig_ptr array");
	}

	// sum all update flag across processors
	MPI_Allreduce(updateFlag_ptr, &updateFlag, 1, MPI_INT, MPI_MAX, world);

	if (updateFlag > 0) {
		if (comm->me == 0) {
			printf("**** updating ref config at step: %ld\n", update->ntimestep);
		}

		for (i = 0; i < nlocal; i++) {

			if (mask[i] & groupbit) {

				// re-set x0 coordinates
				x0[i][0] = x[i][0];
				x0[i][1] = x[i][1];
				x0[i][2] = x[i][2];

				// re-set deformation gradient
				defgrad[i][0] = 1.0;
				defgrad[i][1] = 0.0;
				defgrad[i][2] = 0.0;
				defgrad[i][3] = 0.0;
				defgrad[i][4] = 1.0;
				defgrad[i][5] = 0.0;
				defgrad[i][6] = 0.0;
				defgrad[i][7] = 0.0;
				defgrad[i][8] = 1.0;
				/*
				 * Adjust particle volume as the reference configuration is changed.
				 * We safeguard against excessive deformations by limiting the adjustment range
				 * to the intervale J \in [0.9..1.1]
				 */
				vfrac[i] = rmass[i] / rho[i];
//
				if (nn[i] < 15) {
					radius[i] *= 1.2;
				} // else //{
				  //	radius[i] *= pow(J, 1.0 / domain->dimension);
				  //}
			}
		}

		// update of reference config could have changed x0, vfrac, radius
		// communicate these quantities now to ghosts: x0, vfrac, radius
		comm->forward_comm_fix(this);

		setup(0);
	}
}

/* ----------------------------------------------------------------------
 copy partner info from neighbor lists to atom arrays
 so can be migrated or stored with atoms
 ------------------------------------------------------------------------- */

void FixSMD_TLSPH_ReferenceConfiguration::setup(int vflag) {
	int i, j, ii, jj, n, inum, jnum;
	int *ilist, *jlist, *numneigh, **firstneigh;
	double r, h, wf, wfd;
	Vector3d dx;
	int *n_lagrange_partner = atom->n_lagrange_partner;

	if (updateFlag == 0)
		return;

	int nlocal = atom->nlocal;
	nmax = atom->nmax;
	grow_arrays(nmax);

// 1st loop over neighbor list
// calculate npartner for each owned atom
// nlocal_neigh = nlocal when neigh list was built, may be smaller than nlocal

	double **x0 = atom->x;
	double *radius = atom->radius;
	int *mask = atom->mask;
	tagint *tag = atom->tag;
	NeighList *list = pair->list;
	double *vfrac = atom->vfrac;
	tagint *mol = atom->molecule;
	inum = list->inum;
	ilist = list->ilist;
	numneigh = list->numneigh;
	firstneigh = list->firstneigh;
	Vector3d x0i, x0j;

	// zero npartner for all current atoms
	for (i = 0; i < nlocal; i++){
	  npartner[i] = 0;
	  n_lagrange_partner[i] = 0;
	}

	for (ii = 0; ii < inum; ii++) {
		i = ilist[ii];
		jlist = firstneigh[i];
		jnum = numneigh[i];

		for (jj = 0; jj < jnum; jj++) {
			j = jlist[jj];
			j &= NEIGHMASK;

			if (INSERT_PREDEFINED_CRACKS) {
				if (!crack_exclude(i, j))
					continue;
			}

			dx(0) = x0[i][0] - x0[j][0];
			dx(1) = x0[i][1] - x0[j][1];
			dx(2) = x0[i][2] - x0[j][2];
			r = dx.norm();
			h = radius[i] + radius[j];

			if (r <= h) {
				npartner[i]++;
				n_lagrange_partner[i]++;
				if (j < nlocal) {
					npartner[j]++;
					n_lagrange_partner[j]++;
				}
			}
		}
	}

	maxpartner = 0;
	for (i = 0; i < nlocal; i++)
		maxpartner = MAX(maxpartner, npartner[i]);
	int maxall;
	MPI_Allreduce(&maxpartner, &maxall, 1, MPI_INT, MPI_MAX, world);
	maxpartner = maxall;

	grow_arrays(nmax);

	for (i = 0; i < nlocal; i++) {
		npartner[i] = 0;
		n_lagrange_partner[i] = 0;
		K[i].setZero();

		for (jj = 0; jj < maxpartner; jj++) {
			wfd_list[i][jj] = 0.0;
			wf_list[i][jj] = 0.0;
			degradation_ij[i][jj] = 0.0;
			energy_per_bond[i][jj] = 0.0;
			partnerdx[i][jj].setZero();
		}
	}

	for (ii = 0; ii < inum; ii++) {
		i = ilist[ii];
		jlist = firstneigh[i];
		jnum = numneigh[i];

		for (jj = 0; jj < jnum; jj++) {
			j = jlist[jj];
			j &= NEIGHMASK;

			x0i[0] = x0[i][0];
			x0i[1] = x0[i][1];
			x0i[2] = x0[i][2];

			x0j[0] = x0[j][0];
			x0j[1] = x0[j][1];
			x0j[2] = x0[j][2];

			dx = x0j - x0i;
			r = dx.norm();
			h = radius[i] + radius[j];

			if (r < h) {
				spiky_kernel_and_derivative(h, r, domain->dimension, wf, wfd);

				if (INSERT_PREDEFINED_CRACKS) {
				  if (!crack_exclude(i, j))
				    continue;
				}

				partner[i][npartner[i]] = tag[j];
				wfd_list[i][npartner[i]] = wfd;
				wf_list[i][npartner[i]] = wf;

				partnerdx[i][npartner[i]] = dx;
				r0[i][npartner[i]] = r;
				g_list[i][npartner[i]] = wfd * dx / r;
				K[i] -= vfrac[j] * g_list[i][npartner[i]] * dx.transpose();
				npartner[i]++;
				n_lagrange_partner[i]++;

				if (j < nlocal) {
					partner[j][npartner[j]] = tag[i];
					wfd_list[j][npartner[j]] = wfd;
					wf_list[j][npartner[j]] = wf;
					
					partnerdx[j][npartner[j]] = -dx;
					r0[j][npartner[j]] = r;
					g_list[j][npartner[j]] = -wfd * dx / r;
					K[j] += vfrac[i] * g_list[j][npartner[j]] * dx.transpose();
					npartner[j]++;
					n_lagrange_partner[j]++;
				}
			}
		}
	}

	for (i = 0; i < nlocal; i++) {
	  pseudo_inverse_SVD(K[i]);
	  npartner[i] = 0;
	  n_lagrange_partner[i] = 0;
	}
	
	Vector3d dx0_normalized;
	for (ii = 0; ii < inum; ii++) {
		i = ilist[ii];
		jlist = firstneigh[i];
		jnum = numneigh[i];

		for (jj = 0; jj < jnum; jj++) {
			j = jlist[jj];
			j &= NEIGHMASK;

			if (INSERT_PREDEFINED_CRACKS) {
				if (!crack_exclude(i, j))
					continue;
			}

			dx(0) = x0[j][0] - x0[i][0];
			dx(1) = x0[j][1] - x0[i][1];
			dx(2) = x0[j][2] - x0[i][2];
			r = dx.norm();
			h = radius[i] + radius[j];

			if (r < h) {
			  dx0_normalized = dx / r;
			  K_g_dot_dx0_normalized[i][npartner[i]] = dx0_normalized.dot(K[i] * g_list[i][npartner[i]]);
			  npartner[i]++;
			  n_lagrange_partner[i]++;

			  if (j < nlocal) {
			    K_g_dot_dx0_normalized[j][npartner[j]] = -dx0_normalized.dot(K[j] * g_list[j][npartner[j]]);
			    npartner[j]++;
			    n_lagrange_partner[j]++;
			  }
			}
		}
	}

	// count number of particles for which this group is active

	// bond statistics
	if (update->ntimestep > -1) {
		n = 0;
		int count = 0;
		for (i = 0; i < nlocal; i++) {
			if (mask[i] & groupbit) {
				n += npartner[i];
				count += 1;
			}
		}
		int nall, countall;
		MPI_Allreduce(&n, &nall, 1, MPI_INT, MPI_SUM, world);
		MPI_Allreduce(&count, &countall, 1, MPI_INT, MPI_SUM, world);
                if (countall < 1) countall = 1;

		if (comm->me == 0) {
			if (screen) {
				printf("\n>>========>>========>>========>>========>>========>>========>>========>>========\n");
				fprintf(screen, "TLSPH neighbors:\n");
				fprintf(screen, "  max # of neighbors for a single particle = %d\n", maxpartner);
				fprintf(screen, "  average # of neighbors/particle in group tlsph = %g\n", (double) nall / countall);
				printf(">>========>>========>>========>>========>>========>>========>>========>>========\n\n");
			}
			if (logfile) {
				fprintf(logfile, "\nTLSPH neighbors:\n");
				fprintf(logfile, "  max # of neighbors for a single particle = %d\n", maxpartner);
				fprintf(logfile, "  average # of neighbors/particle in group tlsph = %g\n", (double) nall / countall);
			}
		}
	}

	updateFlag = 0; // set update flag to zero after the update
}

/* ----------------------------------------------------------------------
 memory usage of local atom-based arrays
 ------------------------------------------------------------------------- */

double FixSMD_TLSPH_ReferenceConfiguration::memory_usage() {
	int nmax = atom->nmax;
	int bytes = nmax * sizeof(int);
	bytes += nmax * maxpartner * sizeof(tagint); // partner array
	bytes += nmax * maxpartner * sizeof(float); // wf_list
	bytes += nmax * maxpartner * sizeof(float); // wfd_list
	bytes += nmax * maxpartner * sizeof(float); // damage_per_interaction array
	bytes += nmax * maxpartner * sizeof(Vector3d); // partnerdx array
	bytes += nmax * sizeof(int); // npartner array
	bytes += nmax * maxpartner * sizeof(double); // r0 array
	bytes += nmax * maxpartner * sizeof(double); // K_g_dot_dx0_normalized array
	bytes += nmax * maxpartner * sizeof(Vector3d); // g_list array
	bytes += nmax * sizeof(Matrix3d); // K matrix
	return bytes;

}

/* ----------------------------------------------------------------------
 allocate local atom-based arrays
 ------------------------------------------------------------------------- */

void FixSMD_TLSPH_ReferenceConfiguration::grow_arrays(int nmax) {
	//printf("in FixSMD_TLSPH_ReferenceConfiguration::grow_arrays\n");
	memory->grow(npartner, nmax, "tlsph_refconfig_neigh:npartner");
	memory->grow(partner, nmax, maxpartner, "tlsph_refconfig_neigh:partner");
	memory->grow(wfd_list, nmax, maxpartner, "tlsph_refconfig_neigh:wfd");
	memory->grow(wf_list, nmax, maxpartner, "tlsph_refconfig_neigh:wf");
	memory->grow(degradation_ij, nmax, maxpartner, "tlsph_refconfig_neigh:degradation_ij");
	memory->grow(energy_per_bond, nmax, maxpartner, "tlsph_refconfig_neigh:damage_onset_strain");
	memory->grow(partnerdx, nmax, maxpartner, "tlsph_refconfig_neigh:partnerdx");
	memory->grow(r0, nmax, maxpartner, "tlsph_refconfig_neigh:r0");
	memory->grow(K_g_dot_dx0_normalized, nmax, maxpartner, "tlsph_refconfig_neigh:K_g_dot_dx0_normalized");
	memory->grow(g_list, nmax, maxpartner, "tlsph_refconfig_neigh:g_list");
	memory->grow(K, nmax, "tlsph_refconfig_neigh:K");
}

/* ----------------------------------------------------------------------
 copy values within local atom-based arrays
 ------------------------------------------------------------------------- */

void FixSMD_TLSPH_ReferenceConfiguration::copy_arrays(int i, int j, int delflag) {
	npartner[j] = npartner[i];
	K[j](0) = K[i](0);
	K[j](1) = K[i](1);
	K[j](2) = K[i](2);
	K[j](3) = K[i](3);
	K[j](4) = K[i](4);
	K[j](5) = K[i](5);
	K[j](6) = K[i](6);
	K[j](7) = K[i](7);
	K[j](8) = K[i](8);
	for (int m = 0; m < npartner[j]; m++) {
		partner[j][m] = partner[i][m];
		wfd_list[j][m] = wfd_list[i][m];
		wf_list[j][m] = wf_list[i][m];
		degradation_ij[j][m] = degradation_ij[i][m];
		energy_per_bond[j][m] = energy_per_bond[i][m];
		partnerdx[j][m] = partnerdx[i][m];
		r0[j][m] = r0[i][m];
		K_g_dot_dx0_normalized[j][m] = K_g_dot_dx0_normalized[i][m];
		g_list[j][m] = g_list[i][m];
	}
}

/* ----------------------------------------------------------------------
 pack values in local atom-based arrays for exchange with another proc
 ------------------------------------------------------------------------- */

int FixSMD_TLSPH_ReferenceConfiguration::pack_exchange(int i, double *buf) {
// NOTE: how do I know comm buf is big enough if extreme # of touching neighs
// Comm::BUFEXTRA may need to be increased

//printf("pack_exchange ...\n");

	int m = 0;
	buf[m++] = npartner[i];
	buf[m++] = K[i](0);
	buf[m++] = K[i](1);
	buf[m++] = K[i](2);
	buf[m++] = K[i](3);
	buf[m++] = K[i](4);
	buf[m++] = K[i](5);
	buf[m++] = K[i](6);
	buf[m++] = K[i](7);
	buf[m++] = K[i](8);
	for (int n = 0; n < npartner[i]; n++) {
		buf[m++] = partner[i][n];
		buf[m++] = wfd_list[i][n];
		buf[m++] = wf_list[i][n];
		buf[m++] = degradation_ij[i][n];
		buf[m++] = energy_per_bond[i][n];
		buf[m++] = partnerdx[i][n][0];
		buf[m++] = partnerdx[i][n][1];
		buf[m++] = partnerdx[i][n][2];
		buf[m++] = r0[i][n];
		buf[m++] = K_g_dot_dx0_normalized[i][n];
		buf[m++] = g_list[i][n][0];
		buf[m++] = g_list[i][n][1];
		buf[m++] = g_list[i][n][2];
	}
	return m;

}

/* ----------------------------------------------------------------------
 unpack values in local atom-based arrays from exchange with another proc
 ------------------------------------------------------------------------- */

int FixSMD_TLSPH_ReferenceConfiguration::unpack_exchange(int nlocal, double *buf) {
	if (nlocal == nmax) {
		nmax = nmax / DELTA * DELTA;
		nmax += DELTA;
		grow_arrays(nmax);

		error->message(FLERR,
				"in Fixtlsph_refconfigNeighGCG::unpack_exchange: local arrays too small for receiving partner information; growing arrays");
	}

	int m = 0;
	npartner[nlocal] = static_cast<int>(buf[m++]);
	K[nlocal](0) = static_cast<double>(buf[m++]);
	K[nlocal](1) = static_cast<double>(buf[m++]);
	K[nlocal](2) = static_cast<double>(buf[m++]);
	K[nlocal](3) = static_cast<double>(buf[m++]);
	K[nlocal](4) = static_cast<double>(buf[m++]);
	K[nlocal](5) = static_cast<double>(buf[m++]);
	K[nlocal](6) = static_cast<double>(buf[m++]);
	K[nlocal](7) = static_cast<double>(buf[m++]);
	K[nlocal](8) = static_cast<double>(buf[m++]);
	for (int n = 0; n < npartner[nlocal]; n++) {
		partner[nlocal][n] = static_cast<tagint>(buf[m++]);
		wfd_list[nlocal][n] = static_cast<float>(buf[m++]);
		wf_list[nlocal][n] = static_cast<float>(buf[m++]);
		degradation_ij[nlocal][n] = static_cast<float>(buf[m++]);
		energy_per_bond[nlocal][n] = static_cast<float>(buf[m++]);
		partnerdx[nlocal][n][0] = static_cast<double>(buf[m++]);
		partnerdx[nlocal][n][1] = static_cast<double>(buf[m++]);
		partnerdx[nlocal][n][2] = static_cast<double>(buf[m++]);
		r0[nlocal][n] = static_cast<double>(buf[m++]);
		K_g_dot_dx0_normalized[nlocal][n] = static_cast<double>(buf[m++]);
		g_list[nlocal][n][0] = static_cast<double>(buf[m++]);
		g_list[nlocal][n][1] = static_cast<double>(buf[m++]);
		g_list[nlocal][n][2] = static_cast<double>(buf[m++]);
	}
	return m;
}

/* ----------------------------------------------------------------------
 pack values in local atom-based arrays for restart file
 ------------------------------------------------------------------------- */

int FixSMD_TLSPH_ReferenceConfiguration::pack_restart(int i, double *buf) {

	int m = 0;
	buf[m++] = npartner[i];
	buf[m++] = K[i](0,0);
	buf[m++] = K[i](0,1);
	buf[m++] = K[i](0,2);
	buf[m++] = K[i](1,0);
	buf[m++] = K[i](1,1);
	buf[m++] = K[i](1,2);
	buf[m++] = K[i](2,0);
	buf[m++] = K[i](2,1);
	buf[m++] = K[i](2,2);

	for (int n = 0; n < npartner[i]; n++) {
		buf[m++] = partner[i][n];
		buf[m++] = wfd_list[i][n];
		buf[m++] = wf_list[i][n];
		buf[m++] = degradation_ij[i][n];
		buf[m++] = energy_per_bond[i][n];
		buf[m++] = partnerdx[i][n][0];
		buf[m++] = partnerdx[i][n][1];
		buf[m++] = partnerdx[i][n][2];
		buf[m++] = r0[i][n];
		buf[m++] = K_g_dot_dx0_normalized[i][n];
		buf[m++] = g_list[i][n][0];
		buf[m++] = g_list[i][n][1];
		buf[m++] = g_list[i][n][2];
	}
	return m;
}

/* ----------------------------------------------------------------------
 unpack values from atom->extra array to restart the fix
 ------------------------------------------------------------------------- */

void FixSMD_TLSPH_ReferenceConfiguration::unpack_restart(int nlocal, int nth) {

  // skip to Nth set of extra values

  double **extra = atom->extra;

  int m = 0;
  for (int i = 0; i < nth; i++)
    m += static_cast<int>(extra[nlocal][m]);

  npartner[nlocal] = static_cast<int>(extra[nlocal][m++]);

  K[nlocal](0,0) = extra[nlocal][m++];
  K[nlocal](0,1) = extra[nlocal][m++];
  K[nlocal](0,2) = extra[nlocal][m++];
  K[nlocal](1,0) = extra[nlocal][m++];
  K[nlocal](1,1) = extra[nlocal][m++];
  K[nlocal](1,2) = extra[nlocal][m++];
  K[nlocal](2,0) = extra[nlocal][m++];
  K[nlocal](2,1) = extra[nlocal][m++];
  K[nlocal](2,2) = extra[nlocal][m++];


  for (int n = 0; n < npartner[nlocal]; n++) {
    partner[nlocal][n] = static_cast<tagint>(extra[nlocal][m++]);
    wfd_list[nlocal][n] = extra[nlocal][m++];
    wf_list[nlocal][n] = extra[nlocal][m++];
    degradation_ij[nlocal][n] = extra[nlocal][m++];
    energy_per_bond[nlocal][n] = extra[nlocal][m++];
    partnerdx[nlocal][n][0] = extra[nlocal][m++];
    partnerdx[nlocal][n][1] = extra[nlocal][m++];
    partnerdx[nlocal][n][2] = extra[nlocal][m++];
    r0[nlocal][n] = extra[nlocal][m++];
    K_g_dot_dx0_normalized[nlocal][n] = extra[nlocal][m++];
    g_list[nlocal][n][0] = extra[nlocal][m++];
    g_list[nlocal][n][1] = extra[nlocal][m++];
    g_list[nlocal][n][2] = extra[nlocal][m++];
  }
  updateFlag = 0;
}

/* ----------------------------------------------------------------------
 maxsize of any atom's restart data
 ------------------------------------------------------------------------- */

int FixSMD_TLSPH_ReferenceConfiguration::maxsize_restart() {
// maxtouch_all = max # of touching partners across all procs

	int maxtouch_all;
	MPI_Allreduce(&maxpartner, &maxtouch_all, 1, MPI_INT, MPI_MAX, world);
	return 13 * maxtouch_all + 10;
}

/* ----------------------------------------------------------------------
 size of atom nlocal's restart data
 ------------------------------------------------------------------------- */

int FixSMD_TLSPH_ReferenceConfiguration::size_restart(int nlocal) {
	return 13 * npartner[nlocal] + 10;
}

/* ---------------------------------------------------------------------- */

int FixSMD_TLSPH_ReferenceConfiguration::pack_forward_comm(int n, int *list, double *buf, int pbc_flag, int *pbc) {
	int i, j, m;
	double *radius = atom->radius;
	double *vfrac = atom->vfrac;
	double **x0 = atom->x0;
	double **defgrad0 = atom->smd_data_9;

	//printf("FixSMD_TLSPH_ReferenceConfiguration:::pack_forward_comm\n");
	m = 0;
	for (i = 0; i < n; i++) {
		j = list[i];
		buf[m++] = x0[j][0];
		buf[m++] = x0[j][1];
		buf[m++] = x0[j][2];

		buf[m++] = vfrac[j];
		buf[m++] = radius[j];

		buf[m++] = defgrad0[i][0];
		buf[m++] = defgrad0[i][1];
		buf[m++] = defgrad0[i][2];
		buf[m++] = defgrad0[i][3];
		buf[m++] = defgrad0[i][4];
		buf[m++] = defgrad0[i][5];
		buf[m++] = defgrad0[i][6];
		buf[m++] = defgrad0[i][7];
		buf[m++] = defgrad0[i][8];

	}
	return m;
}

/* ---------------------------------------------------------------------- */

void FixSMD_TLSPH_ReferenceConfiguration::unpack_forward_comm(int n, int first, double *buf) {
	int i, m, last;
	double *radius = atom->radius;
	double *vfrac = atom->vfrac;
	double **x0 = atom->x0;
	double **defgrad0 = atom->smd_data_9;

	m = 0;
	last = first + n;
	for (i = first; i < last; i++) {
		x0[i][0] = buf[m++];
		x0[i][1] = buf[m++];
		x0[i][2] = buf[m++];

		vfrac[i] = buf[m++];
		radius[i] = buf[m++];

		defgrad0[i][0] = buf[m++];
		defgrad0[i][1] = buf[m++];
		defgrad0[i][2] = buf[m++];
		defgrad0[i][3] = buf[m++];
		defgrad0[i][4] = buf[m++];
		defgrad0[i][5] = buf[m++];
		defgrad0[i][6] = buf[m++];
		defgrad0[i][7] = buf[m++];
		defgrad0[i][8] = buf[m++];
	}
}

/* ----------------------------------------------------------------------
 routine for excluding bonds across a hardcoded slit crack
 Note that everything is scaled by lattice constant l0 to avoid
 numerical inaccuracies.
 ------------------------------------------------------------------------- */

bool FixSMD_TLSPH_ReferenceConfiguration::crack_exclude(int i, int j) {

	double **x = atom->x;
	double l0 = domain->lattice->xlattice;

	// line between pair of atoms i,j
	double x1 = x[i][0] / l0;
	double y1 = x[i][1] / l0;

	double x2 = x[j][0] / l0;
	double y2 = x[j][1] / l0;

	// hardcoded crack line
	double x3 = -0.1 / l0;
	double y3 = ((int) 1.0 / l0) + 0.5;
	//printf("y3 = %f\n", y3);
	double x4 = 0.1 / l0 - 1.0 + 0.1;
	double y4 = y3;

	bool retVal = DoLineSegmentsIntersect(x1, y1, x2, y2, x3, y3, x4, y4);

	return !retVal;
	//return 1;
}


/* ----------------------------------------------------------------------
 pack entire state of Fix into one write
 ------------------------------------------------------------------------- */

void FixSMD_TLSPH_ReferenceConfiguration::write_restart(FILE *fp) {
  // maxtouch_all = max # of touching partners across all procs

  int maxtouch_all;
  MPI_Allreduce(&maxpartner, &maxtouch_all, 1, MPI_INT, MPI_MAX, world);

  int n = 0;
  double list[1];
  list[n++] = maxtouch_all;

  if (comm->me == 0) {
    int size = n * sizeof(double);
    fwrite(&size, sizeof(int), 1, fp);
    fwrite(list, sizeof(double), n, fp);
  }
}

/* ----------------------------------------------------------------------
 use state info from restart file to restart the Fix
 ------------------------------------------------------------------------- */

void FixSMD_TLSPH_ReferenceConfiguration::restart(char *buf) {
  int n = 0;
  double *list = (double *) buf;
  maxpartner = list[n++]; // Set maxpartner as the # of touching partners across all procs prior to restart to grow_arrays.

  grow_arrays(atom->nmax);
}


void FixSMD_TLSPH_ReferenceConfiguration::post_neighbor() {
  // Check if Lagrangian connection between particles is lost:

  int i, jnum, jj, j;
  int *n_lagrange_partner = atom->n_lagrange_partner;

  // Create the list of missing particles
  
  int **missing = NULL;
  int n_missing = 0;
  int *max_missing = NULL;
  bool already_missing;
  int imissing;
  
  missing = (int **) memory->smalloc(nprocs*sizeof(int *),"tlsph_refconfig_neigh:missing");
  memory->create(max_missing, nprocs,"tlsph_refconfig_neigh:max_missing");
  memset(&max_missing[0], 0, nprocs*sizeof(int));

  for (i = 0; i < atom->nlocal; i++) {
    jnum = npartner[i];
    for (jj = 0; jj < jnum; jj++) {
      
      j = atom->map(partner[i][jj]);
      if (j < 0) {
	// printf("Connection lost between particle %d and %d in CPU %d\n", atom->tag[i], partner[i][jj], comm->me);

	// Check if the particle was already reported as missing:
	already_missing = false;

	if (n_missing > 0) {
	  for (imissing=0; imissing < n_missing; imissing++) {
	    // printf("missing[comm->me][%d] = %d, partner[i][jj] = %d\n", imissing, missing[comm->me][imissing], partner[i][jj]);
	    if (missing[comm->me][imissing] == partner[i][jj]) {
	      already_missing = true;
	      break;
	    }
	  }
	}

	// If the particle was not already reported missing, report it as such:
	if (!already_missing) {
	  n_missing++;

	  if (n_missing > max_missing[comm->me]) {
	    if (max_missing[comm->me] == 0) memory->create(missing[comm->me],n_missing,"tlsph_refconfig_neigh:missing[i]");
	    else memory->grow(missing[comm->me],n_missing,"tlsph_refconfig_neigh:missing[i]");
	    max_missing[comm->me] = n_missing;
	  }
	  missing[comm->me][n_missing - 1] = partner[i][jj];
	  // printf("Proc %d: # %d missing particle %d\n", comm->me, n_missing, missing[comm->me][n_missing - 1]);
	}
      }
    }
  }

  MPI_Allreduce(&n_missing, &n_missing_all, 1, MPI_INT, MPI_MAX, world);

  if (n_missing_all > 0){
    int count, nsend, n, nrecv_tot, nrecv_pos;   // counter and # of particle to send
    maxsend = 1;
    maxrecv = 1;

    nrecv_tot = 0;

    for (int proc_i = 0; proc_i < nprocs; proc_i++) {

      // Send the local list of missing particles to other CPUs:
      count = n_missing;

      MPI_Bcast(&count, 1, MPI_INT, proc_i, world);

      // printf("Proc %d is missing %d particles\n", proc_i, count);

      // If proc_i is not missing any particle, look at the next:
      if (count == 0) continue;

      if (count > max_missing[proc_i]) {
	if (max_missing[proc_i] == 0) memory->create(missing[proc_i], count,"tlsph_refconfig_neigh:missing[i]");
	else memory->grow(missing[proc_i], count,"tlsph_refconfig_neigh:missing[i]");
	max_missing[proc_i] = count;
      }

      MPI_Bcast(missing[proc_i], count, MPI_INT, proc_i, world);

      if (proc_i != comm->me) {
	nsend = 0;
	n = 0;

	// Check if the missing particles are located on this CPU:
	for (i=0; i<count; i++) {
	  j = atom->map(missing[proc_i][i]);
	  if ( j>=0 ) {
	    // printf("Particle %d asked by CPU %d is located on CPU %d\n", missing[proc_i][i], proc_i, comm->me);

	    // Send the particle over as ghost atom to the right CPU:
	    if (nsend == maxsendlist[proc_i]) {
	      maxsendlist[proc_i] = nsend + 1;
	      memory->grow(sendlist[proc_i],maxsendlist[proc_i],"tlsph_refconfig_neigh:sendlist[i]");
	    }
	    sendlist[proc_i][nsend++] = j;
	  }
	}

	if (nsend) {
	  // Now that the sendlist is created, pack attributes:
	  if (nsend*comm->size_border_() > maxsend) {
	    maxsend = nsend*comm->size_border_();
	    memory->grow(buf_send,maxsend + BUFEXTRA,"tlsph_refconfig_neigh:buf_send");
	  }

	  if (comm->ghost_velocity)
	    n = atom->avec->pack_border_vel(nsend,sendlist[proc_i],buf_send,0,NULL);
	  else
	    n = atom->avec->pack_border(nsend,sendlist[proc_i],buf_send,0,NULL);
	}

	// Send particles to the CPU that is missing them:
	// First send the number of particles found:

	// printf("I (%d) have %d particles to send to proc %d\n", comm->me, nsend, proc_i);
	MPI_Send(&nsend, 1, MPI_INT, proc_i, 0, world);
	if (nsend) {
	  // printf("I (%d) am sending %d particles to send to proc %d, n=%d\n", comm->me, nsend, proc_i, n);
	  MPI_Send(buf_send, n, MPI_DOUBLE, proc_i, 0, world);
	  // printf("I (%d) am done sending %d particles to send to proc %d\n", comm->me, nsend, proc_i);
	  
	}

	// Store the number of particles that I (comm->me) have to send to proc_i:
	nsendlist[proc_i] = nsend;

      } else {

	// If I am proc_i:

	nrecv_pos = 0;

	for (int proc_j = 0; proc_j < nprocs; proc_j++) {
	  if (proc_j != comm->me) {
	    MPI_Recv(&nrecv[proc_j], 1, MPI_INT, proc_j, 0, world, MPI_STATUS_IGNORE);

	    if (nrecv[proc_j] == 0) continue;

	    nrecv_tot += nrecv[proc_j];
	    // printf("Proc %d has %d particles to send to me (%d)\n", proc_j, nrecv[proc_j], comm->me);


	    if (nrecv_tot*comm->size_border_() > maxrecv) {
	      maxrecv = nrecv_tot*comm->size_border_();
	      memory->grow(buf_recv,maxrecv + BUFEXTRA,"tlsph_refconfig_neigh:buf_recv");
	    }

	    // printf("Proc %d receives from %d, n=%d\n", comm->me, proc_j, nrecv[proc_j]*comm->size_border_());

	    if (nrecv[proc_j]) {
	      MPI_Recv(&buf_recv[nrecv_pos], nrecv[proc_j]*comm->size_border_(),MPI_DOUBLE, proc_j, 0, world, MPI_STATUS_IGNORE);
	      nrecv_pos += nrecv[proc_j]*comm->size_border_();
	    }

	    // printf("Proc %d reception from %d done\n", comm->me, proc_j);
	  }
	}
      }
      MPI_Barrier(world);
    }

    // printf("Proc %d unpacks, nrecv_tot=%d\n", comm->me, nrecv_tot);

    firstrecv = atom->nlocal+atom->nghost;

    if (nrecv_tot) {
      // unpack buffer
      if (comm->ghost_velocity)
	atom->avec->unpack_border_vel(nrecv_tot,firstrecv,buf_recv);
      else
	atom->avec->unpack_border(nrecv_tot,firstrecv,buf_recv);

      atom->nghost += nrecv_tot;
      atom->map_set();
    }
  }
 
  if (missing) for (i = 0; i < nprocs; i++) if (max_missing[i]) memory->destroy(missing[i]);
  memory->sfree(missing);

  memory->destroy(max_missing);
  need_forward_comm = false;
}


void FixSMD_TLSPH_ReferenceConfiguration::forward_comm_tl() {
  if (!need_forward_comm) return;
  if (n_missing_all == 0) return;

  // printf("In FixSMD_TLSPH_ReferenceConfiguration::forward_comm_tl()\n");

  int n, nrecv_tot, nrecv_pos;

  nrecv_tot = 0;

  for (int proc_i = 0; proc_i < nprocs; proc_i++) {
    
    if (proc_i != comm->me) {
      if (nsendlist[proc_i]) {
	// Now that the sendlist is created, pack attributes:
	if (nsendlist[proc_i]*comm->size_border_() > maxsend) {
	  maxsend = nsendlist[proc_i]*comm->size_border_();
	  memory->grow(buf_send,maxsend + BUFEXTRA,"tlsph_refconfig_neigh:buf_send");
	}

	if (comm->ghost_velocity)
	  n = atom->avec->pack_border_vel(nsendlist[proc_i],sendlist[proc_i],buf_send,0,NULL);
	else
	  n = atom->avec->pack_border(nsendlist[proc_i],sendlist[proc_i],buf_send,0,NULL);
      }

      MPI_Send(&nsendlist[proc_i], 1, MPI_INT, proc_i, 0, world);

      if (nsendlist[proc_i]) {
	MPI_Send(buf_send, n, MPI_DOUBLE, proc_i, 0, world);
      }
    } else {

      // If I am proc_i:

      nrecv_pos = 0;

      for (int proc_j = 0; proc_j < nprocs; proc_j++) {
	if (proc_j != comm->me) {
	  MPI_Recv(&nrecv[proc_j], 1, MPI_INT, proc_j, 0, world, MPI_STATUS_IGNORE);

	  if (nrecv[proc_j] == 0) continue;

	  nrecv_tot += nrecv[proc_j];


	  if (nrecv_tot*comm->size_border_() > maxrecv) {
	    maxrecv = nrecv_tot*comm->size_border_();
	    memory->grow(buf_recv,maxrecv + BUFEXTRA,"tlsph_refconfig_neigh:buf_recv");
	  }

	  if (nrecv[proc_j]) {
	    MPI_Recv(&buf_recv[nrecv_pos], nrecv[proc_j]*comm->size_border_(),MPI_DOUBLE, proc_j, 0, world, MPI_STATUS_IGNORE);
	    nrecv_pos += nrecv[proc_j]*comm->size_border_();
	  }
	}
      }
    }
    MPI_Barrier(world);
  }

  if (nrecv_tot) {
    // unpack buffer
    if (comm->ghost_velocity)
      atom->avec->unpack_border_vel(nrecv_tot,firstrecv,buf_recv);
    else
      atom->avec->unpack_border(nrecv_tot,firstrecv,buf_recv);
  }
  need_forward_comm = false;
}
