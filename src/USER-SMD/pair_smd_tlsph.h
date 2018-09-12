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

#ifdef PAIR_CLASS

PairStyle(smd/tlsph,PairTlsph)

#else

#ifndef LMP_TLSPH_NEW_H
#define LMP_TLSPH_NEW_H

#include "pair.h"
#include <Eigen/Eigen>
#include "smd_material_models.h"

namespace LAMMPS_NS {

class PairTlsph: public Pair {
public:

	PairTlsph(class LAMMPS *);
	virtual ~PairTlsph();
	virtual void compute(int, int);
	void settings(int, char **);
	void coeff(int, char **);
	double init_one(int, int);
	void init_style();
	void init_list(int, class NeighList *);
	void write_restart_settings(FILE *) {
	}
	void read_restart_settings(FILE *) {
	}
	virtual double memory_usage();
	void compute_shape_matrix(void);
	void material_model(void);
	void *extract(const char *, int &);
	int pack_forward_comm(int, int *, double *, int, int *);
	void unpack_forward_comm(int, int, double *);
	void AssembleStress();

	void PreCompute();
	void ComputeForces(int eflag, int vflag);
	void effective_longitudinal_modulus(const int itype, const double dt, const double d_iso, const double p_rate,
			const Eigen::Matrix3d d_dev, const Eigen::Matrix3d sigma_dev_rate, const double damage, double &K_eff, double &mu_eff, double &M_eff);

	void ComputePressure(const int i, const double rho, const double mass_specific_energy, const double vol_specific_energy,
			const double pInitial, const double d_iso, double &pFinal, double &p_rate);
	void ComputeStressDeviator(const int i, const double mass_specific_energy, const Eigen::Matrix3d sigmaInitial_dev, const Eigen::Matrix3d d_dev, Eigen::Matrix3d &sigmaFinal_dev,
				   Eigen::Matrix3d &sigma_dev_rate, double &plastic_strain_increment, const double pInitial, double &pFinal);
	void ComputeDamage(const int i, const Eigen::Matrix3d strain, const Eigen::Matrix3d sigmaFinal, Eigen::Matrix3d &sigma_damaged, double plastic_strain_increment);
	void UpdateDegradation();
	double CalculateScale(const float degradation, const int itype);
	Matrix3d CreateOrthonormalBasisFromOneVector(Vector3d sU);
	void AdjustStressForZeroForceBC(const Matrix3d sigma, const Vector3d sU, Matrix3d &sigmaBC);
protected:
	void allocate();
	char *suffix;

	/*
	 * per-type arrays
	 */
	int *strengthModel, *eos;
	double *onerad_dynamic, *onerad_frozen, *maxrad_dynamic, *maxrad_frozen;

	/*
	 * per atom arrays
	 */
	Eigen::Matrix3d *K,  *PK1, *Fdot, *Fincr; 
	Eigen::Matrix3d *R; // rotation matrix
	Eigen::Matrix3d *FincrInv;
	Eigen::Matrix3d *D, *W; // strain rate and spin tensor
	Eigen::Vector3d *smoothVelDifference;
	Eigen::Matrix3d *CauchyStress;
	double *detF, *particle_dt;
	double *vijSq_max;
	double *hourglass_error;
	int *numNeighsRefConfig;
	double *damage_increment;

	int nmax; // max number of atoms on this proc
	double hMin; // minimum kernel radius for two particles
        double* rSqMin;
	double dtCFL;
	double dtRelative; // relative velocity of two particles, divided by sound speed
	int updateFlag;
	double update_threshold; // updateFlage is set to one if the relative displacement of a pair exceeds update_threshold
	double cut_comm;

	enum {
		UPDATE_NONE = 5000, UPDATE_CONSTANT_THRESHOLD = 5001, UPDATE_PAIRWISE_RATIO = 5002,
	};

	enum {
		LINEAR_DEFGRAD = 0,
		STRENGTH_LINEAR = 1,
		STRENGTH_LINEAR_PLASTIC = 2,
		STRENGTH_JOHNSON_COOK = 3,
		STRENGTH_NONE = 4,
		EOS_LINEAR = 5,
		EOS_SHOCK = 6,
		EOS_POLYNOMIAL = 7,
		EOS_NONE = 8,
		REFERENCE_DENSITY = 9,
		YOUNGS_MODULUS = 10,
		POISSON_RATIO = 11,
		HOURGLASS_CONTROL_AMPLITUDE = 12,
		HEAT_CAPACITY = 13,
		LAME_LAMBDA = 14,
		SHEAR_MODULUS = 15,
		M_MODULUS = 16,
		SIGNAL_VELOCITY = 17,
		BULK_MODULUS = 18,
		VISCOSITY_Q1 = 19,
		VISCOSITY_Q2 = 20,
		YIELD_STRESS = 21,
		FAILURE_MAX_PLASTIC_STRAIN_THRESHOLD = 22,
		JC_A = 23,
		JC_B = 24,
		JC_a = 25,
		JC_C = 26,
		JC_epdot0 = 27,
		JC_T0 = 28,
		JC_Tmelt = 29,
		JC_M = 30,
		EOS_SHOCK_C0 = 31,
		EOS_SHOCK_S = 32,
		EOS_SHOCK_GAMMA = 33,
		HARDENING_PARAMETER = 34,
		FAILURE_MAX_PRINCIPAL_STRAIN_THRESHOLD = 35,
		FAILURE_MAX_PRINCIPAL_STRESS_THRESHOLD = 36,
		FAILURE_MAX_PAIRWISE_STRAIN_THRESHOLD = 37,
		EOS_POLYNOMIAL_C0 = 38,
		EOS_POLYNOMIAL_C1 = 39,
		EOS_POLYNOMIAL_C2 = 40,
		EOS_POLYNOMIAL_C3 = 41,
		EOS_POLYNOMIAL_C4 = 42,
		EOS_POLYNOMIAL_C5 = 43,
		EOS_POLYNOMIAL_C6 = 44,

		FAILURE_JC_D1 = 45,
		FAILURE_JC_D2 = 46,
		FAILURE_JC_D3 = 47,
		FAILURE_JC_D4 = 48,
		FAILURE_JC_EPDOT0 = 49,

		CRITICAL_ENERGY_RELEASE_RATE = 50,

		STRENGTH_LUDWICK_HOLLOMON = 51,
		STRENGTH_SWIFT = 52,
		STRENGTH_VOCE = 53,
		
		LH_A = 54,
		LH_B = 55,
		LH_n = 56,
		SWIFT_A = 57,
		SWIFT_B = 58,
		SWIFT_n = 59,
		SWIFT_eps0 = 60,
		VOCE_A = 61,
		VOCE_Q1 = 62,
		VOCE_n1 = 63,
		VOCE_Q2 = 64,
		VOCE_n2 = 65,
		VOCE_C = 66,
		VOCE_epsdot0 = 67,

		GTN_Q1 = 68,
		GTN_Q2 = 69,
		GTN_FN = 70,
		GTN_inverse_sN = 71,
		GTN_epsN = 72,
		GTN_f0 = 73,
		GTN_fcr = 74,
		GTN_fF = 75,
		GTN_Komega = 76,
		CL_W = 77,

		VISCOSITY_Q1_times_SIGNAL_VELOCITY = 78,
		LONGITUDINAL_MODULUS = 79,
		HOURGLASS_CONTROL_AMPLITUDE_times_YOUNGS_MODULUS = 80,

		MAX_KEY_VALUE = 81
	};

	struct failure_types { // this is defined per type and determines which failure/damage model is active
		bool failure_none;
		bool failure_max_principal_strain;
		bool failure_max_principal_stress;
		bool failure_max_plastic_strain;
		bool failure_johnson_cook;
		bool failure_gtn;
		bool failure_cockcroft_latham;
		bool failure_max_pairwise_strain;
		bool integration_point_wise; // true if failure model applies to stress/strain state of integration point
		bool failure_energy_release_rate;

		failure_types() {
			failure_none = true;
			failure_max_principal_strain = false;
			failure_max_principal_stress = false;
			failure_max_plastic_strain = false;
			failure_johnson_cook = false;
			failure_gtn = false;
			failure_max_pairwise_strain = false;
			integration_point_wise = false;
			failure_energy_release_rate = false;
			//printf("constructed failure type\n");
		}
	};
	failure_types *failureModel;

	int ifix_tlsph;
	int update_method;

	class FixSMD_TLSPH_ReferenceConfiguration *fix_tlsph_reference_configuration;

private:
	double **Lookup; // holds per-type material parameters for the quantities defined in enum statement above.
	bool first; // if first is true, do not perform any computations, beacuse reference configuration is not ready yet.
	FlowStress flowstress; // holds the flow stress equations
};

}

#endif
#endif

/*
 * materialCoeffs array for EOS parameters:
 * 1: rho0
 *
 *
 * materialCoeffs array for strength parameters:
 *
 * Common
 * 10: maximum strain threshold for damage model
 * 11: maximum stress threshold for damage model
 *
 * Linear Plasticity model:
 * 12: plastic yield stress
 *
 *
 * Blei: rho = 11.34e-6, c0=2000, s=1.46, Gamma=2.77
 * Stahl 1403: rho = 7.86e-3, c=4569, s=1.49, Gamma=2.17
 */

