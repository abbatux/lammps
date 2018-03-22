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
void LinearPlasticStrength(const double G, const double yieldStress, const Matrix3d sigmaInitial_dev, const Matrix3d d_dev,
			   const double dt, Matrix3d &sigmaFinal_dev__, Matrix3d &sigma_dev_rate__, double &plastic_strain_increment, const double damage);
void JohnsonCookStrength(const double G, const double cp, const double espec, const double A, const double B, const double a,
		const double C, const double epdot0, const double T0, const double Tmelt, const double M, const double dt, const double ep,
		const double epdot, const Matrix3d sigmaInitial_dev, const Matrix3d d_dev, Matrix3d &sigmaFinal_dev__,
		Matrix3d &sigma_dev_rate__, double &plastic_strain_increment, const double damage);
double GTNStrength(const double G, const double Q1, const double Q2, const double dt, const double damage,
		   const Matrix3d sigmaInitial_dev, const Matrix3d d_dev, const double pFinal, const double yieldStress_undamaged,
		   Matrix3d &sigmaFinal_dev__, Matrix3d &sigma_dev_rate__, double &plastic_strain_increment);

/*
 * Damage models
 */

bool IsotropicMaxStrainDamage(const Matrix3d E, const double maxStrain);
bool IsotropicMaxStressDamage(const Matrix3d E, const double maxStrain);
double JohnsonCookDamageIncrement(const double p, const Matrix3d Sdev, const double d1, const double d2, const double d3,
				  const double d4, const double epdot0, const double epdot, const double plastic_strain_increment);
double GTNDamageIncrement(const double Q1, const double Q2, const double An, const double Komega, const double pressure, const Matrix3d Sdev, const Matrix3d stress,
			  const double eff_plastic_strain, const double plastic_strain_increment, const double damage, const Matrix3d Fdot, const double yieldstress, const double hM);


#endif /* SMD_MATERIAL_MODELS_H_ */
