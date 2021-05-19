#pragma once

#include "hoomd/VectorMath.h"

//! Declaration of evaluator function
__device__ float eval(const vec3<float>& r_ij,
                      unsigned int type_i,
                      const quat<float>& q_i,
                      float d_i,
                      float charge_i,
                      unsigned int type_j,
                      const quat<float>& q_j,
                      float d_j,
                      float charge_j);

//! Function pointer type
typedef float (*eval_func)(const vec3<float>& r_ij,
                           const unsigned int typ_i,
                           const quat<float>& orientation_i,
                           const float diameter_i,
                           const float charge_i,
                           const unsigned int typ_j,
                           const quat<float>& orientation_j,
                           const float diameter_j,
                           const float charge_j);
