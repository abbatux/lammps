#ifndef SMD_FLOW_STRESS_H_
#define SMD_FLOW_STRESS_H_

#include <iostream>
#include <stdio.h>

using namespace std;

#define MAX(A,B) ((A) > (B) ? (A) : (B))

class FlowStress {
 public:
  void LH(double a, double b, double n) { // LUDWICK_HOLLOMON
    C[0] = a;
    C[1] = b;
    C[2] = n;
    type = 0;
  }

  void VOCE(double sigma0, double Q1, double n1, double Q2, double n2, double c, double epsdot0) {
    C[0] = sigma0;
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
      return C[0] + C[1] * (1.0 - exp(-C[2] * ep)) + C[3] * (1.0 - exp(-C[4] * ep));
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

#endif
