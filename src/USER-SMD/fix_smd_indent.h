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

FixStyle(smd/indent,FixSMDIndent)

#else

#ifndef LMP_FIX_SMD_INDENT_H
#define LMP_FIX_SMD_INDENT_H

#include "fix.h"

namespace LAMMPS_NS {

class FixSMDIndent : public Fix {
 public:
  FixSMDIndent(class LAMMPS *, int, char **);
  ~FixSMDIndent();
  int setmask();
  void init();
  void setup(int);
  void min_setup(int);
  void post_force(int);
  void post_force_respa(int, int, int);
  void min_post_force(int);
  double compute_scalar();
  double compute_vector(int);
  double dtCFL;

 private:
  int istyle,scaleflag,side;
  double k,k3;
  char *xstr,*ystr,*zstr,*rstr,*pstr,*vxstr,*vystr,*vzstr;
  int xvar,yvar,zvar,rvar,pvar,vxvar,vyvar,vzvar;
  double xvalue,yvalue,zvalue,rvalue,pvalue,vxvalue,vyvalue,vzvalue;
  int indenter_flag,planeside;
  double indenter[4],indenter_all[4];
  int cdim,varflag;
  int ilevel_respa;

  void options(int, char **);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Variable name for fix indent does not exist

Self-explanatory.

E: Variable for fix indent is invalid style

Only equal-style variables can be used.

E: Variable for fix indent is not equal style

Only equal-style variables can be used.

*/
