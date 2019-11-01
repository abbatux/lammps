/* -*- c++ -*- ----------------------------------------------------------
 *
 *                    *** Smooth Mach Dynamics ***
 *
 * This file is part of the USER-SMD package for LAMMPS.
 * Copyright (2014) Georg C. Ganzenmueller, georg.ganzenmueller@emi.fhg.de
 * Fraunhofer Ernst-Mach Institute for High-Speed Dynamics, EMI,
 * Eckerstrasse 4, D-79104 Freiburg i.Br, Germany.
 *
 * ----------------------------------------------------------------------- */

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

#ifdef FIX_CLASS

FixStyle(smd/setangularvelocity,FixSMDSetAngVel)

#else

#ifndef LMP_FIX_SMD_SET_ANGULAR_VELOCITY_H
#define LMP_FIX_SMD_SET_ANGULAR_VELOCITY_H

#include "fix.h"

namespace LAMMPS_NS {

class FixSMDSetAngVel : public Fix {
 public:
  FixSMDSetAngVel(class LAMMPS *, int, char **);
  ~FixSMDSetAngVel();
  int setmask();
  void init();
  void setup(int);
  void min_setup(int);
  //void initial_integrate(int);
  void post_force(int);
  double compute_vector(int);
  double memory_usage();

 private:
  char axis;
  double omegavalue;
  int varflag,iregion;
  char *omegastr;
  char *idregion;
  int omegavar, omegastyle;
  double foriginal[3],foriginal_all[3];
  int force_flag;
  int nlevels_respa;

  int maxatom;
  double *sforce;
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Region ID for fix setforce does not exist

Self-explanatory.

E: Variable name for fix setforce does not exist

Self-explanatory.

E: Variable for fix setforce is invalid style

Only equal-style variables can be used.

E: Cannot use non-zero forces in an energy minimization

Fix setforce cannot be used in this manner.  Use fix addforce
instead.

*/
