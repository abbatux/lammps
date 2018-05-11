/* -*- c++ -*- ----------------------------------------------------------
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

#ifdef COMPUTE_CLASS

ComputeStyle(smd/tlsph/surface/normal, ComputeSMDTLSPHSurfaceNormal)

#else

#ifndef LMP_COMPUTE_SMD_TLSPH_SURFACE_NORMAL_H
#define LMP_COMPUTE_SMD_TLSPH_SURFACE_NORMAL_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeSMDTLSPHSurfaceNormal : public Compute {
 public:
  ComputeSMDTLSPHSurfaceNormal(class LAMMPS *, int, char **);
  ~ComputeSMDTLSPHSurfaceNormal();
  void init();
  void compute_peratom();
  double memory_usage();

 private:
  int nmax;
  double **n_array;
};

}

#endif
#endif
