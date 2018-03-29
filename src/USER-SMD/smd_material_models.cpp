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
#include <iostream>
#include "math_special.h"
#include "smd_math.h"
#include <stdio.h>

#include <Eigen/Eigen>

using namespace LAMMPS_NS::MathSpecial;
using namespace SMD_Math;
using namespace std;
using namespace Eigen;

#define MIN(A,B) ((A) < (B) ? (A) : (B))
#define MAX(A,B) ((A) > (B) ? (A) : (B))

/* ----------------------------------------------------------------------
 linear EOS for use with linear elasticity
 input: initial pressure pInitial, isotropic part of the strain rate d, time-step dt
 output: final pressure pFinal, pressure rate p_rate
 ------------------------------------------------------------------------- */
void LinearEOS(const double lambda, const double pInitial, const double d, const double dt, double &pFinal, double &p_rate) {

	/*
	 * pressure rate
	 */
	p_rate = lambda * d;

	pFinal = pInitial + dt * p_rate; // increment pressure using pressure rate
	//cout << "hurz" << endl;

}

/* ----------------------------------------------------------------------
 Linear EOS when there is damage integration point wise
 input:
 current density rho
 reference density rho0
 reference bulk modulus K
 initial pressure pInitial
 time step dt
 current damage

 output:
 pressure rate p_rate
 final pressure pFinal

 ------------------------------------------------------------------------- */
void LinearEOSwithDamage(const double rho, const double rho0, const double K, const double pInitial, const double dt, double &pFinal, double &p_rate, const double damage) {

  double mu = rho / rho0 - 1.0;

  if ((damage > 0.0) && (mu < 0.0)) {
    if (damage >= 1.0) {
      pFinal = 0.0;
    } else {
      mu = (1 - damage) * mu;
      pFinal = -(1 - damage) * K * mu;
    }
  } else {
    pFinal = -K * mu;
  }

  p_rate = (pFinal - pInitial) / dt;
}

/* ----------------------------------------------------------------------
 shock EOS
 input:
 current density rho
 reference density rho0
 current energy density e
 reference energy density e0
 reference speed of sound c0
 shock Hugoniot parameter S
 Grueneisen parameter Gamma
 initial pressure pInitial
 time step dt

 output:
 pressure rate p_rate
 final pressure pFinal

 ------------------------------------------------------------------------- */
void ShockEOS(const double rho, const double rho0, const double e, const double e0, const double c0, const double S, const double Gamma, const double pInitial, const double dt,
	      double &pFinal, double &p_rate, const double damage) {

	double mu = rho / rho0 - 1.0;
	double pH = rho0 * square(c0) * mu * (1.0 + mu) / square(1.0 - (S - 1.0) * mu);

	pFinal = -(pH + rho * Gamma * (e - e0));

	if ( damage > 0.0 ) {
	  if ( pFinal > 0.0 ) {
	    if ( damage >= 1.0) {
	      pFinal = -rho0 * Gamma * (e - e0);
	    } else {
	      double mu_damaged = (1.0 - damage) * mu;
	      double pH_damaged = rho0 * (1.0 - damage) * square(c0) * mu_damaged * (1.0 + mu_damaged) / square(1.0 - (S - 1.0) * mu_damaged);
	      pFinal = (-pH_damaged + rho0 * (1 + mu_damaged) * Gamma * (e - e0));;
	    }
	  }
	}

	//printf("shock EOS: rho = %g, rho0 = %g, Gamma=%f, c0=%f, S=%f, e=%f, e0=%f\n", rho, rho0, Gamma, c0, S, e, e0);
	//printf("pFinal = %f\n", pFinal);
	p_rate = (pFinal - pInitial) / dt;

}

/* ----------------------------------------------------------------------
 polynomial EOS
 input:
 current density rho
 reference density rho0
 coefficients 0 .. 6
 initial pressure pInitial
 time step dt

 output:
 pressure rate p_rate
 final pressure pFinal

 ------------------------------------------------------------------------- */
void polynomialEOS(const double rho, const double rho0, const double e, const double C0, const double C1, const double C2, const double C3, const double C4, const double C5, const double C6,
		   const double pInitial, const double dt, double &pFinal, double &p_rate, const double damage) {

	double mu = rho / rho0 - 1.0;

	if (mu > 0.0) {
		pFinal = C0 + C1 * mu + C2 * mu * mu + C3 * mu * mu * mu; // + (C4 + C5 * mu + C6 * mu * mu) * e;
	} else {
		pFinal = C0 + C1 * mu + C3 * mu * mu * mu; //  + (C4 + C5 * mu) * e;
	}
	pFinal = -pFinal; // we want the mean stress, not the pressure.

	if ( damage > 0.0 ) {
	  double mu_damaged = (1.0 - damage) * mu;
	  double pFinal_damaged;
	  if (mu_damaged > 0.0) {
	    pFinal_damaged = C0 + C1 * mu_damaged + C2 * mu_damaged * mu_damaged + C3 * mu_damaged * mu_damaged * mu_damaged; // + (C4 + C5 * mu_damaged + C6 * mu_damaged * mu_damaged) * e;
	  } else {
	    pFinal_damaged = C0 + C1 * mu_damaged + C3 * mu_damaged * mu_damaged * mu_damaged; //  + (C4 + C5 * mu_damaged) * e;
	  }
	  pFinal_damaged = -pFinal_damaged;
	  pFinal = MIN(pFinal, pFinal_damaged);
	}

	pFinal = -pFinal; // we want the mean stress, not the pressure.

	//printf("pFinal = %f\n", pFinal);
	p_rate = (pFinal - pInitial) / dt;

}

/* ----------------------------------------------------------------------
 Tait EOS based on current density vs. reference density.

 input: (1) reference sound speed
 (2) equilibrium mass density
 (3) current mass density

 output:(1) pressure
 (2) current speed of sound
 ------------------------------------------------------------------------- */
void TaitEOS_density(const double exponent, const double c0_reference, const double rho_reference, const double rho_current,
		double &pressure, double &sound_speed) {

	double B = rho_reference * c0_reference * c0_reference / exponent;
	double tmp = pow(rho_current / rho_reference, exponent);
	pressure = B * (tmp - 1.0);
	double bulk_modulus = B * tmp * exponent; // computed as rho * d(pressure)/d(rho)
	sound_speed = sqrt(bulk_modulus / rho_current);

//	if (fabs(pressure) > 0.01) {
//		printf("tmp = %f, press=%f, K=%f\n", tmp, pressure, bulk_modulus);
//	}

}

/* ----------------------------------------------------------------------
 perfect gas EOS
 input: gamma -- adiabatic index (ratio of specific heats)
 J -- determinant of deformation gradient
 volume0 -- reference configuration volume of particle
 energy -- energy of particle
 pInitial -- initial pressure of the particle
 d -- isotropic part of the strain rate tensor,
 dt -- time-step size

 output: final pressure pFinal, pressure rate p_rate
 ------------------------------------------------------------------------- */
void PerfectGasEOS(const double gamma, const double vol, const double mass, const double energy, double &pFinal, double &c0) {

	/*
	 * perfect gas EOS is p = (gamma - 1) rho e
	 */

	if (energy > 0.0) {

		pFinal = (1.0 - gamma) * energy / vol;
//printf("gamma = %f, vol%f, e=%g ==> p=%g\n", gamma, vol, energy, *pFinal__/1.0e-9);

		c0 = sqrt((gamma - 1.0) * energy / mass);

	} else {
		pFinal = c0 = 0.0;
	}

}

/* ----------------------------------------------------------------------
 linear strength model for use with linear elasticity
 input: lambda, mu : Lame parameters
 input: sigmaInitial_dev, d_dev: initial stress deviator, deviatoric part of the strain rate tensor
 input: dt: time-step
 output:  sigmaFinal_dev, sigmaFinal_dev_rate__: final stress deviator and its rate.
 ------------------------------------------------------------------------- */
void LinearStrength(const double mu, const Matrix3d sigmaInitial_dev, const Matrix3d d_dev, const double dt,
		Matrix3d &sigmaFinal_dev__, Matrix3d &sigma_dev_rate__) {

	/*
	 * deviatoric rate of unrotated stress
	 */
	sigma_dev_rate__ = 2.0 * mu * d_dev;

	/*
	 * elastic update to the deviatoric stress
	 */
	sigmaFinal_dev__ = sigmaInitial_dev + dt * sigma_dev_rate__;
}

/* ----------------------------------------------------------------------
 linear strength model for use with linear elasticity
 input: lambda, mu : Lame parameters
 input: F: deformation gradient
 output:  total stress tensor, deviator + pressure
 ------------------------------------------------------------------------- */
//void PairTlsph::LinearStrengthDefgrad(double lambda, double mu, Matrix3d F, Matrix3d *T) {
//	Matrix3d E, PK2, eye, sigma, S, tau;
//
//	eye.setIdentity();
//
//	E = 0.5 * (F * F.transpose() - eye); // strain measure E = 0.5 * (B - I) = 0.5 * (F * F^T - I)
//	tau = lambda * E.trace() * eye + 2.0 * mu * E; // Kirchhoff stress, work conjugate to above strain
//	sigma = tau / F.determinant(); // convert Kirchhoff stress to Cauchy stress
//
////printf("l=%f, mu=%f, sigma xy = %f\n", lambda, mu, sigma(0,1));
//
////    E = 0.5 * (F.transpose() * F - eye); // Green-Lagrange Strain E = 0.5 * (C - I)
////    S = lambda * E.trace() * eye + 2.0 * mu * Deviator(E); // PK2 stress
////    tau = F * S * F.transpose(); // convert PK2 to Kirchhoff stress
////    sigma = tau / F.determinant();
//
//	//*T = sigma;
//
//	/*
//	 * neo-hookean model due to Bonet
//	 */
////    lambda = mu = 100.0;
////    // left Cauchy-Green Tensor, b = F.F^T
//	double J = F.determinant();
//	double logJ = log(J);
//	Matrix3d b;
//	b = F * F.transpose();
//
//	sigma = (mu / J) * (b - eye) + (lambda / J) * logJ * eye;
//	*T = sigma;
//}
/* ----------------------------------------------------------------------
 linear strength model for use with linear elasticity
 input: lambda, mu : Lame parameters
 input: sigmaInitial_dev, d_dev: initial stress deviator, deviatoric part of the strain rate tensor
 input: dt: time-step
 output:  sigmaFinal_dev, sigmaFinal_dev_rate__: final stress deviator and its rate.
 ------------------------------------------------------------------------- */
void LinearPlasticStrength(const double G, const double yieldStress, const Matrix3d sigmaInitial_dev, const Matrix3d d_dev,
			   const double dt, Matrix3d &sigmaFinal_dev__, Matrix3d &sigma_dev_rate__, double &plastic_strain_increment, const double damage) {

	Matrix3d sigmaTrial_dev, dev_rate;
	double J2;
	double Gd = (1 - damage) * G;

	/*
	 * deviatoric rate of unrotated stress
	 */
	dev_rate = 2.0 * Gd * d_dev;

	/*
	 * perform a trial elastic update to the deviatoric stress
	 */
	sigmaTrial_dev = sigmaInitial_dev + dt * dev_rate; // increment stress deviator using deviatoric rate

	/*
	 * check yield condition
	 */
	J2 = sqrt(3. / 2.) * sigmaTrial_dev.norm();

	if (J2 < yieldStress) {
		/*
		 * no yielding has occured.
		 * final deviatoric stress is trial deviatoric stress
		 */
		sigma_dev_rate__ = dev_rate;
		sigmaFinal_dev__ = sigmaTrial_dev;
		plastic_strain_increment = 0.0;
		//printf("no yield\n");

	} else {
		//printf("yiedl\n");
		/*
		 * yielding has occured
		 */

		plastic_strain_increment = (J2 - yieldStress) / (3.0 * Gd);
		/*
		 * new deviatoric stress:
		 * obtain by scaling the trial stress deviator
		 */
		sigmaFinal_dev__ = (yieldStress / J2) * sigmaTrial_dev;

		/*
		 * new deviatoric stress rate
		 */
		sigma_dev_rate__ = sigmaFinal_dev__ - sigmaInitial_dev;
		//printf("yielding has occured.\n");
	}
}

/* ----------------------------------------------------------------------
 Johnson Cook Material Strength model
 input:
 G : shear modulus
 cp : heat capacity
 espec : energy / mass
 A : initial yield stress under quasi-static / room temperature conditions
 B : proportionality factor for plastic strain dependency
 a : exponent for plastic strain dpendency
 C : proportionality factor for logarithmic plastic strain rate dependency
 epdot0 : dimensionality factor for plastic strain rate dependency
 T : current temperature
 T0 : reference (room) temperature
 Tmelt : melting temperature
 input: sigmaInitial_dev, d_dev: initial stress deviator, deviatoric part of the strain rate tensor
 input: dt: time-step
 output:  sigmaFinal_dev, sigmaFinal_dev_rate__: final stress deviator and its rate.
 ------------------------------------------------------------------------- */
void JohnsonCookStrength(const double G, const double cp, const double espec, const double A, const double B, const double a,
		const double C, const double epdot0, const double T0, const double Tmelt, const double M, const double dt, const double ep,
		const double epdot, const Matrix3d sigmaInitial_dev, const Matrix3d d_dev, Matrix3d &sigmaFinal_dev__,
			 Matrix3d &sigma_dev_rate__, double &plastic_strain_increment, const double damage) {

	double yieldStress;

	double TH = espec / (cp * (Tmelt - T0));
	TH = MAX(TH, 0.0);
	double epdot_ratio = epdot / epdot0;
	epdot_ratio = MAX(epdot_ratio, 1.0);
	//printf("current temperature delta is %f, TH=%f\n", deltaT, TH);
	
	yieldStress = (A + B * pow(ep, a)) * (1.0 + C * log(epdot_ratio)); // * (1.0 - pow(TH, M));
	
	LinearPlasticStrength(G, yieldStress, sigmaInitial_dev, d_dev, dt, sigmaFinal_dev__, sigma_dev_rate__, plastic_strain_increment, damage);
}

/* ----------------------------------------------------------------------
 Gurson - Tvergaard - Needleman (GTN) Material Strength model
 input:
 G : shear modulus
 Q1, Q2: two model parameters
 damage
 
 input: sigmaInitial_dev, d_dev: initial stress deviator, deviatoric part of the strain rate tensor
 input: dt: time-step
 output:  sigmaFinal_dev, sigmaFinal_dev_rate__: final stress deviator and its rate.
 ------------------------------------------------------------------------- */
double GTNStrength(const double G, const double Q1, const double Q2, const double dt, const double damage, const double fcr,
		 const Matrix3d sigmaInitial_dev, const Matrix3d d_dev, const double p, const double yieldStress_undamaged,
		   Matrix3d &sigmaFinal_dev__, Matrix3d &sigma_dev_rate__, double &plastic_strain_increment, const bool coupling) {
  
  if ((damage == 0.0 ) || (Q1 == 0.0)) {
    LinearPlasticStrength(G, yieldStress_undamaged, sigmaInitial_dev, d_dev, dt, sigmaFinal_dev__, sigma_dev_rate__, plastic_strain_increment, damage);
    return yieldStress_undamaged;
  } else {

    Matrix3d sigmaTrial_dev, dev_rate;
    double yieldStress;
    double J2, Phi;
    double Gd = G;
    if (coupling == true) Gd *= (1-damage); 
    double f = damage * fcr;
    double Q1f = Q1 * f;
    double Q1fSq = Q1f * Q1f;
    /*
     * deviatoric rate of unrotated stress
     */
    dev_rate = 2.0 * Gd * d_dev;
    
    /*
     * perform a trial elastic update to the deviatoric stress
     */
    sigmaTrial_dev = sigmaInitial_dev + dt * dev_rate; // increment stress deviator using deviatoric rate

    J2 = sqrt(3. / 2.) * sigmaTrial_dev.norm();

    /*
     * NEWTON - RAPHSON METHOD TO DETERMINE THE YIELD STRESS:
     */

    // determine stress triaxiality
    double triax = 0.0;
    if (p != 0.0 && J2 != 0.0) {
      triax = -p / (J2 + 0.01 * fabs(p)); // have softening in denominator to avoid divison by zero
    }
    double Q2triax = 1.5 * Q2 * triax;
    double Q1fQ2triax = Q1f * Q2triax;

    double x = 1.0; // x = yieldStress / yieldStress_undamaged
    double dx = 1.0; // dx = x_{n+1} - x_{n} initiated at a value higher than the accepted error margin.
    double error = 0.001;

    double F, Fprime, Q2triaxx;

    while ((dx > error) || (dx < -error)) {
      Q2triaxx = Q2triax * x;
      F = x*x + 2 * Q1f * cosh(Q2triaxx) - (1 + Q1fSq);
      Fprime = 2 * (x + Q1fQ2triax * sinh(Q2triaxx));

      dx = -F/Fprime;
      x += dx;
    }

    yieldStress = x * yieldStress_undamaged;

    if (isnan(yieldStress) || yieldStress < 0.0) {
      cout << "yieldStress = " << yieldStress << "\tF = " << F << "\tFprime = " << Fprime << "\tdx = " << dx << endl;
      cout << "G=" << G << "\tQ1=" <<  Q1 << "\tQ2=" << Q2 << "\tdt=" <<  dt << "\tdamage=" <<  damage << "\tf=" <<  f << "\tJ2=" << J2 << "\tp=" << p << "\ttriax=" << triax << "yieldStress_undamaged = " << yieldStress_undamaged << endl << "\tsigmaInitial_dev=" << endl << sigmaInitial_dev << "d_dev = " << endl << d_dev << endl;
    }
    
    LinearPlasticStrength(G, yieldStress, sigmaInitial_dev, d_dev, dt, sigmaFinal_dev__, sigma_dev_rate__, plastic_strain_increment, damage);

    plastic_strain_increment *= x/(1 - f);

    return yieldStress;
  }
}
/* ----------------------------------------------------------------------
 isotropic maximum strain damage model
 input:
 current strain
 maximum value of allowed principal strain

 output:
 return value is true if any eigenvalue of the current strain exceeds the allowed principal strain

 ------------------------------------------------------------------------- */

bool IsotropicMaxStrainDamage(const Matrix3d E, const double maxStrain) {

	/*
	 * compute Eigenvalues of strain matrix
	 */
	SelfAdjointEigenSolver < Matrix3d > es;
	es.compute(E); // compute eigenvalue and eigenvectors of strain

	double max_eigenvalue = es.eigenvalues().maxCoeff();

	if (max_eigenvalue > maxStrain) {
		return true;
	} else {
		return false;
	}
}

/* ----------------------------------------------------------------------
 isotropic maximum stress damage model
 input:
 current stress
 maximum value of allowed principal stress

 output:
 return value is true if any eigenvalue of the current stress exceeds the allowed principal stress

 ------------------------------------------------------------------------- */

bool IsotropicMaxStressDamage(const Matrix3d S, const double maxStress) {

	/*
	 * compute Eigenvalues of strain matrix
	 */
	SelfAdjointEigenSolver < Matrix3d > es;
	es.compute(S); // compute eigenvalue and eigenvectors of strain

	double max_eigenvalue = es.eigenvalues().maxCoeff();

	if (max_eigenvalue > maxStress) {
		return true;
	} else {
		return false;
	}
}

/* ----------------------------------------------------------------------
 Johnson-Cook failure model
 input:


 output:


 ------------------------------------------------------------------------- */

double JohnsonCookDamageIncrement(const double p, const Matrix3d Sdev, const double d1, const double d2, const double d3,
				  const double d4, const double epdot0, const double epdot, const double plastic_strain_increment) {



	double vm = sqrt(3. / 2.) * Sdev.norm(); // von-Mises equivalent stress
	if (vm < 0.0) {
		cout << "this is sdev " << endl << Sdev << endl;
		printf("vm=%f < 0.0, surely must be an error\n", vm);
		exit(1);
	}

	// determine stress triaxiality
	double triax = 0.0;
	if (p != 0.0 && vm != 0.0) {
	  triax = -p / (vm + 0.01 * fabs(p)); // have softening in denominator to avoid divison by zero
	}
	if (triax > 3.0) {                                                                                                                                                                                
	  triax = 3.0;
	}
	// Johnson-Cook failure strain, dependence on stress triaxiality
	if (triax >= -1.0/3.0) {
	  double jc_failure_strain = d1 + d2 * exp(d3 * triax);
	  //printf("d1=%f, d2=%f, d3 = %f, triax = %f, jc_failure_strain = %f\n", d1, d2, d3, triax, jc_failure_strain);
	  // include strain rate dependency if parameter d4 is defined and current plastic strain rate exceeds reference strain rate
	  if (d4 > 0.0) { //
	    if (epdot > epdot0) {
	      double epdot_ratio = epdot / epdot0;
	      jc_failure_strain *= (1.0 + d4 * log(epdot_ratio));
	      //printf("epsdot=%f, epsdot0=%f, factor = %f\n", epdot, epdot0, (1.0 + d4 * log(epdot_ratio)));
	      //exit(1);
	      
	    }
	  }
	  return plastic_strain_increment/jc_failure_strain;
	} else {
	  return 0;
	}

}

/* ----------------------------------------------------------------------
 Gurson-Tvergaard-Needleman damage evolution model
 input:
 An and Komega: parameters
 equivalent plastic strain increment
 Finc
 dt
 damage

 output:
 damage increment

 ------------------------------------------------------------------------- */

double GTNDamageIncrement(const double Q1, const double Q2, const double An, const double Komega, const double pressure, const Matrix3d Sdev, const Matrix3d stress, const double plastic_strain_increment, const double damage, const double fcr, const double yieldstress_undamaged) { // Following K. Nahshon, J.W. Hutchinson / European Journal of Mechanics A/Solids 27 (2008) 1–17
  if (damage >= 1.0) return 0.0;
  if (plastic_strain_increment == 0.0) return 0.0;

  double fn_increment = 0;

  if (An != 0) fn_increment = An * plastic_strain_increment; // rate of void nucleation

  if (damage == 0.0) return fn_increment;

  double fs_increment = 0;
  double f = damage * fcr;
  
  if (Komega != 0) {
    double vm, inverse_sM, J3, omega;
    double lambda_increment, tmp1, sinh_tmp1;

    vm = sqrt(3. / 2.) * Sdev.norm(); // von-Mises equivalent stress
    if (vm < 0.0) {
      cout << "this is sdev " << endl << Sdev << endl;
      printf("vm=%f < 0.0, surely must be an error\n", vm);
      exit(1);
    }

    if ( vm == 0.0 ) return 0.0;

    inverse_sM = yieldstress_undamaged;
    J3 = Sdev.determinant();
    //printf("vm = %f, yieldstress_undamaged = %f, J3 = %f\n", vm, yieldstress_undamaged, J3);

    omega = 1 - 182.25 * J3 * J3/(vm * vm * vm * vm * vm * vm);

    if (omega < 0.0) {
      // printf("omega=%10.e < 0.0, surely must be an error\n", omega);
      // cout << "vm = " << vm << "\t";
      // cout << "J3 = " << J3 << "\t";
      // cout << "J3 * J3/(vm * vm * vm * vm * vm * vm) = " << J3 * J3/(vm * vm * vm * vm * vm * vm) << endl;
      // cout << "Here is S:" << endl << Sdev << endl;
      omega = 0;
    }
    else if (omega > 1.0) {
      // printf("omega=%10.e > 1.0, surely must be an error\n", omega);
      // cout << "vm = " << vm << "\t";
      // cout << "J3 = " << J3 << "\t";
      // cout << "J3 * J3/(vm * vm * vm * vm * vm * vm) = " << J3 * J3/(vm * vm * vm * vm * vm * vm) << endl;
      // cout << "Here is S:" << endl << Sdev << endl;
      omega = 1.0;
    }

    tmp1 = -1.5 * Q2 * pressure * inverse_sM;
    sinh_tmp1 = sinh(tmp1);
    lambda_increment = 2 * yieldstress_undamaged * plastic_strain_increment * (1 - f) / (vm * vm * inverse_sM * inverse_sM + Q1 * f * tmp1 * sinh_tmp1);

    fs_increment = lambda_increment * f * inverse_sM * ((1 - f) * 3 * Q1 * Q2 * sinh_tmp1 + Komega * omega * 2 * vm * inverse_sM);

    if (isnan(fs_increment) || isnan(-fs_increment)) {
      printf("GTN f increment: %10.e\n", fs_increment);
      cout << "vm = " << vm << "\t";
      cout << "yieldstress_undamaged = " << yieldstress_undamaged << "\t";
      cout << "tmp1 = " << tmp1 << endl;
      cout << "f = " << f << endl;
      cout << "omega = " << omega << endl;
      cout << "F = " << vm * vm * inverse_sM * inverse_sM + 2 * Q1 * f * cosh(tmp1) - (1 + Q1 * Q1 * f * f) << endl;
      cout << "plastic_strain_increment = " << plastic_strain_increment << endl;
    }

    if (fs_increment < 0.0) fs_increment = 0.0;
  }
  double f_increment = fn_increment + fs_increment;
  if (isnan(f_increment) || isnan(-f_increment)){
    cout << "fs_increment = " << fs_increment << "\t" << "fn_increment = " << fn_increment << endl;
  }
  return f_increment / fcr;
}
