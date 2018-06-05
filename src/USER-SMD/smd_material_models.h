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

#ifndef SMD_MATERIAL_MODELS_H_
#define SMD_MATERIAL_MODELS_H_

using namespace Eigen;

#define MAX(A,B) ((A) > (B) ? (A) : (B))

class FlowStress {
 public:
  void LH(double a, double b, double n) { // LUDWICK_HOLLOMON
    C[0] = a;
    C[1] = b;
    C[2] = n;
    type = 0;
  }

  void VOCE(double A, double Q1, double n1, double Q2, double n2, double c, double epsdot0) {
    C[0] = A;
    C[1] = Q1;
    C[2] = n1;
    C[3] = Q2;
    C[4] = n2;
    C[5] = c;
    C[6] = epsdot0;
    type = 1;
  }

  void SWIFT(double a, double b, double n, double eps0) {
    C[0] = a;
    C[1] = b;
    C[2] = n;
    C[3] = eps0;
    type = 2;
  }

  void JC(double A, double B, double a, double c, double epdot0, double T0, double Tmelt, double M) { // JOHNSON_COOK
    C[0] = A;
    C[1] = B;
    C[2] = a;
    C[3] = c;
    C[4] = epdot0;
    C[5] = T0;
    C[6] = Tmelt;
    C[7] = M;
    type = 3;
  }

  void linear_plastic(double sigma0, double H) {
    C[0] = sigma0;
    C[1] = H;
    type = 4;
  }

  double evaluate(double ep) {
    switch (type) {
    case 0: // LH
      return C[0] + C[1] * pow(ep, C[2]);
    case 1: // VOCE
      return C[0] - C[1] * exp(-C[2] * ep) - C[3] * exp(-C[4] * ep);
    case 2: // SWIFT
      return C[0] + C[1] * pow(ep - C[3], C[2]);
    case 3: // JC
      return C[0] + C[1] * pow(ep, C[2]);
    case 4: // linear plastic
      return C[0] + C[1] * ep;
    }
  }

  double evaluate(double ep, double epdot) {
    double sigmay = evaluate(ep);
    switch (type) {
    case 1: // VOCE
      if (C[6] > 0.0)
	return sigmay * (1.0 + C[5] * log(MAX(epdot / C[6], 1.0)));
      else return sigmay;
    case 3: // JC
      if (C[4] > 0.0)
	return sigmay * pow(1.0 + MAX(epdot / C[4], 1.0), C[3]);
      else return sigmay;
    default:
      return sigmay;
    }
  }
  double evaluate_derivative(double ep) {
    switch (type) {
    case 0: // LH
      return C[1] * C[2] * pow(ep, C[2] - 1.0);
    case 1: // VOCE
      return -C[1] * C[2] * exp(-C[2] * ep) - C[3] * C[4] * exp(-C[4] * ep);
    case 2: // SWIFT
      return C[1] * C[2] * pow(ep - C[3], C[2] - 1.0);
    case 3: // JC
      return C[1] * C[2] * pow(ep, C[2] - 1.0);
    case 4: // linear plastic
      return C[1];
    }
  }

private:
  double C[8]; // array of constants.
  int type;
};

/*
 * EOS models
 */
void LinearEOS(const double lambda, const double pInitial, const double d, const double dt, double &pFinal, double &p_rate);
void LinearEOSwithDamage(const double rho, const double rho0, const double K, const double pInitial, const double dt, double &pFinal, double &p_rate, const double damage);
void ShockEOS(const double rho, const double rho0, const double e, const double e0, const double c0, const double S, const double Gamma, const double pInitial, const double dt,
	      double &pFinal, double &p_rate, const double damage);
void polynomialEOS(const double rho, const double rho0, const double e, const double C0, const double C1, const double C2, const double C3, const double C4, const double C5, const double C6,
		   const double pInitial, const double dt, double &pFinal, double &p_rate, const double damage);
void TaitEOS_density(const double exponent, const double c0_reference, const double rho_reference, const double rho_current,
		double &pressure, double &sound_speed);
void PerfectGasEOS(const double gamma, const double vol, const double mass, const double energy, double &pFinal__, double &c0);

/*
 * Material strength models
 */
void LinearStrength(const double mu, const Matrix3d sigmaInitial_dev, const Matrix3d d_dev, const double dt,
		Matrix3d &sigmaFinal_dev__, Matrix3d &sigma_dev_rate__);
void LinearPlasticStrength(const double G, double yieldStress, const Matrix3d sigmaInitial_dev, const Matrix3d d_dev,
			   const double dt, Matrix3d &sigmaFinal_dev__, Matrix3d &sigma_dev_rate__, double &plastic_strain_increment, const double damage);
void JohnsonCookStrength(const double G, const double cp, const double espec, const double A, const double B, const double a,
		const double C, const double epdot0, const double T0, const double Tmelt, const double M, const double dt, const double ep,
		const double epdot, const Matrix3d sigmaInitial_dev, const Matrix3d d_dev, Matrix3d &sigmaFinal_dev__,
		Matrix3d &sigma_dev_rate__, double &plastic_strain_increment, const double damage);
double GTNStrength(const double G, FlowStress flowstress, const double Q1, const double Q2,
		   const double fcr, const double fF, const double FN, const double inverse_sN, const double epsN, const double Komega,
		   const double dt, const double damage, const double ep, const Matrix3d sigmaInitial_dev, const Matrix3d d_dev,
		   const double pInitial, double &pFinal, Matrix3d &sigmaFinal_dev__, Matrix3d &sigma_dev_rate__,
		   double &plastic_strain_increment, const bool coupling, const int tag);

/*
 * Damage models
 */

bool IsotropicMaxStrainDamage(const Matrix3d E, const double maxStrain);
bool IsotropicMaxStressDamage(const Matrix3d E, const double maxStrain);
double JohnsonCookDamageIncrement(const double p, const Matrix3d Sdev, const double d1, const double d2, const double d3,
				  const double d4, const double epdot0, const double epdot, const double plastic_strain_increment);
double GTNDamageIncrement(const double Q1, const double Q2, const double An, const double Komega, const double pressure, const Matrix3d Sdev, const Matrix3d stress,
			  const double plastic_strain_increment, const double damage, const double fcr, const double yieldstress);
double CockcroftLathamDamageIncrement(const Matrix3d S, const double W, const double plastic_strain_increment);

#endif /* SMD_MATERIAL_MODELS_H_ */
