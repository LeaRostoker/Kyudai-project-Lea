#ifndef __CUDA_UTILITIES_H
#define __CUDA_UTILITIES_H

#include <stdio.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "device_launch_parameters.h"
//#include "device_functions.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/remove.h>
#include <thrust/generate.h>
#include <thrust/detail/type_traits.h>

#include <math.h>
#include <algorithm>
#include <time.h>
#include <limits.h>
#include <vector>

//#include <opencv2/core/core.hpp>
//#include <opencv2/core/cuda.hpp>
//#include "opencv2/core/devmem2d.hpp"

// Utilities and timing functions
//#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include "include_gpu/helper_cuda.h"         // helper functions for CUDA error check
//#include "helper_cusolver.h"
//#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop
#include "include_gpu/cutils_math.h"
#include "include_gpu/cutils_matrix.h"


// Preprocessor definitions for width and height of color and depth streams
#define THREAD_SIZE_X 8
#define THREAD_SIZE_Y 8
#define THREAD_SIZE_Z 8
#define THREAD_SIZE_L_X 8
#define THREAD_SIZE_L_Y 8
#define THREAD_SIZE THREAD_SIZE_L_X*THREAD_SIZE_L_Y
#define STRIDE 512
#define MAXTOLERANCE 0.2
// #define EPSILON 0.1
#define ALPHA 0.8 

//#define PI 3.1415926535897932384626433832795

#define MAX_DEPTH 100.0

#define divUp(x,y) (x%y) ? ((x+y-1)/y) : (x/y)

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

inline float vec_norminf(int n, const float* x)
{
    float norminf = 0;
    for (int j = 0; j < n; j++) {
        float x_abs = fabs(x[j]);
        norminf = (norminf > x_abs) ? norminf : x_abs;
    }
    return norminf;
}

//using namespace std;

struct Quaternion_dev {
    float4 value;
    
    __device__  __host__ Quaternion_dev(float X, float Y, float Z, float W) {
        value.x = X;
        value.y = Y;
        value.z = Z;
        value.w = W;
    }
    
    __device__ __host__ Quaternion_dev (float3 t, float W) {
        value.x = t.x;
        value.y = t.y;
        value.z = t.z;
        value.w = W;
    }

    __device__ __host__ Quaternion_dev(float3 r) {
        float norm_r = sqrt(r.x * r.x + r.y * r.y + r.z * r.z);
        if (norm_r == 0.0f) {
            value.x = 0.0f;
            value.y = 0.0f;
            value.z = 0.0f;
            value.w = 1.0f;
            return;
        }
        float f = sin(norm_r / 2.0f) / norm_r;
        value.x = f * r.x;
        value.y = f * r.y;
        value.z = f * r.z;
        value.w = cos(norm_r / 2.0f); 
    }

    __device__ __host__ Quaternion_dev(float3x3 rot) {
        float trace = rot(0, 0) + rot(1, 1) + rot(2, 2);
        if (trace > 0) {
            float s = 0.5f / sqrtf(trace + 1.0f);
            value.w = 0.25f / s;
            value.x = (rot(2, 1) - rot(1, 2)) * s;
            value.y = (rot(0, 2) - rot(2, 0)) * s;
            value.z = (rot(1, 0) - rot(0, 1)) * s;
        }
        else {
            if (rot(0, 0) > rot(1, 1) && rot(0, 0) > rot(2, 2)) {
                float s = 2.0f * sqrtf(1.0f + rot(0, 0) - rot(1, 1) - rot(2, 2));
                value.w = (rot(2, 1) - rot(1, 2)) / s;
                value.x = 0.25f * s;
                value.y = (rot(0, 1) + rot(1, 0)) / s;
                value.z = (rot(0, 2) + rot(2, 0)) / s;
            }
            else if (rot(1, 1) > rot(2, 2)) {
                float s = 2.0f * sqrtf(1.0f + rot(1, 1) - rot(0, 0) - rot(2, 2));
                value.w = (rot(0, 2) - rot(2, 0)) / s;
                value.x = (rot(0, 1) + rot(1, 0)) / s;
                value.y = 0.25f * s;
                value.z = (rot(1, 2) + rot(2, 1)) / s;
            }
            else {
                float s = 2.0f * sqrtf(1.0f + rot(2, 2) - rot(0, 0) - rot(1, 1));
                value.w = (rot(1, 0) - rot(0, 1)) / s;
                value.x = (rot(0, 2) + rot(2, 0)) / s;
                value.y = (rot(1, 2) + rot(2, 1)) / s;
                value.z = 0.25f * s;
            }
        }
    }

    // Inverse
    __device__ __host__ Quaternion_dev Inv (Quaternion_dev a) {
        float norm = a.value.x*a.value.x + a.value.y*a.value.y + a.value.z*a.value.z + a.value.w*a.value.w;
        if (norm == 0.0f)
            return Quaternion_dev(0.0,0.0,0.0,1.0);
        return Quaternion_dev(-a.value.x/norm, -a.value.y/norm, -a.value.z/norm, a.value.w/norm);
    }
        
    __device__ __host__ float3 Vector () {
        float3 r;
        r.x = value.x;
        r.y = value.y;
        r.z = value.z;
        return r;
    }
    
    __device__ __host__ float Scalar () {
        return value.w;
    }
    
    // Scalar Multiplication
    __device__ __host__ Quaternion_dev operator* (float s) {
        float3 v1;
        v1.x = s*value.x;
        v1.y = s*value.y;
        v1.z = s*value.z;
        return Quaternion_dev (v1, s*value.w);
    }
    
    // Multiplication
    __device__ __host__ Quaternion_dev operator* (Quaternion_dev b) {
        float3 v1;
        v1.x = value.x;
        v1.y = value.y;
        v1.z = value.z;
        float w1 = value.w;
        float3 v2 = b.Vector();
        float w2 = b.Scalar();
        
        return Quaternion_dev (w1*v2 + w2*v1 + cross(v1,v2), w1*w2 - dot(v1,v2));
    }
    
    // Addition
    __device__ __host__ Quaternion_dev operator+ (Quaternion_dev b) {
        float3 v1;
        v1.x = value.x;
        v1.y = value.y;
        v1.z = value.z;
        float w1 = value.w;
        float3 v2 = b.Vector();
        float w2 = b.Scalar();
        
        return Quaternion_dev (v1 + v2, w1 + w2);
    }
    
    // Conjugate
    __device__ __host__ Quaternion_dev Conjugate () {
        return Quaternion_dev(-value.x, -value.y, -value.z, value.w);
    }
    
    // Magnitude
    __device__ __host__ float Magnitude () {
        return sqrt(value.x*value.x + value.y*value.y + value.z*value.z + value.w*value.w);
    }
    
    __device__ __host__ float Dot(Quaternion_dev b) {
        float3 v1;
        v1.x = value.x;
        v1.y = value.y;
        v1.z = value.z;
        float w1 = value.w;
        float3 v2 = b.Vector();
        float w2 = b.Scalar();
        
        return w1*w2 + dot(v1,v2);
    }
    
    //Normalize
    __device__ __host__ Quaternion_dev Normalize ( ) {
        float norm = sqrt(value.x*value.x + value.y*value.y + value.z*value.z + value.w*value.w);
        return Quaternion_dev(value.x/norm, value.y/norm, value.z/norm, value.w/norm);
    }

    //norm
    __device__ __host__ float Norm() {
        float norm = sqrt(value.x * value.x + value.y * value.y + value.z * value.z + value.w * value.w);
        return norm;
    }

    __device__ __host__ void Print()
    {
        printf("%f, %f, %f, %f\n", value.x, value.y, value.z, value.w);
    }
};


struct DualQuaternion_dev {
    Quaternion_dev m_real = Quaternion_dev(0.0,0.0,0.0,1.0);
    Quaternion_dev m_dual = Quaternion_dev(0.0,0.0,0.0,0.0);

    __device__ __host__ DualQuaternion_dev() {
        m_real = Quaternion_dev(0.0,0.0,0.0,1.0);
        m_dual = Quaternion_dev(0.0,0.0,0.0,0.0);
    }
        
    __device__ __host__ DualQuaternion_dev(Quaternion_dev r, Quaternion_dev d) {
        //if (r.Magnitude() > 0.0f)
        //    m_real = r.Normalize();
        m_real = r;
        m_dual = d;
    }
    
    __device__ __host__ DualQuaternion_dev(Quaternion_dev r, float3 t) {
        if (r.Magnitude() > 0.0f)
            m_real = r.Normalize();
        m_dual = (Quaternion_dev(t,0.0) * m_real) * 0.5;
    }

    __device__ __host__ DualQuaternion_dev(float3 r, float3 t) {
        m_real = Quaternion_dev(r);
        m_dual = (Quaternion_dev(t, 0.0f) * m_real) * 0.5f;
    }

    __device__ __host__ DualQuaternion_dev(float4x4 transfo) {
        float3x3 rotation;
        rotation(0, 0) = transfo(0, 0); rotation(0, 1) = transfo(0, 1); rotation(0, 2) = transfo(0, 2);
        rotation(1, 0) = transfo(1, 0); rotation(1, 1) = transfo(1, 1); rotation(1, 2) = transfo(1, 2);
        rotation(2, 0) = transfo(2, 0); rotation(2, 1) = transfo(2, 1); rotation(2, 2) = transfo(2, 2);
        m_real = Quaternion_dev(rotation);

        float3 t = make_float3(transfo(0, 3), transfo(1, 3), transfo(2, 3));

        m_dual = (Quaternion_dev(t, 0.0) * m_real) * 0.5;
    }
    
    __device__ __host__ Quaternion_dev Real() {
        return m_real;
    }
    
    __device__ __host__ Quaternion_dev Dual () {
        return m_dual;
    }
    
    __device__ __host__ DualQuaternion_dev Identity() {
        return DualQuaternion_dev(Quaternion_dev(make_float3(0.0f), 1.0f), Quaternion_dev(make_float3(0.0f), 0.0f));
    }

    // Inverse
    __device__ __host__ DualQuaternion_dev Inv (DualQuaternion_dev a) {
        if(a.m_real.Magnitude() == 0)
            return DualQuaternion_dev();

        Quaternion_dev p_1 = a.m_real.Inv(a.m_real);
        Quaternion_dev p_2 = a.m_dual * p_1;
        DualQuaternion_dev q_1 = DualQuaternion_dev(p_1, Quaternion_dev(0.0,0.0,0.0,0.0));
        DualQuaternion_dev q_2 = DualQuaternion_dev(Quaternion_dev(0.0,0.0,0.0,1.0), p_2 * (-1.0f));

        return q_1 * q_2;
    }
    
    //Addition
    __device__ __host__ DualQuaternion_dev operator+ (DualQuaternion_dev b) {
        return DualQuaternion_dev(m_real + b.Real(), m_dual + b.Dual());
    }
    
    // Scalar multiplication
    __device__ __host__ DualQuaternion_dev operator* (float s) {
        return DualQuaternion_dev(m_real*s, m_dual*s);
    }
    
    // Multiplication
    __device__ __host__ DualQuaternion_dev operator* (DualQuaternion_dev b) {
        return DualQuaternion_dev(m_real*b.Real(), m_real*b.Dual() + m_dual*b.Real());
    }
    
    // Division
    __device__ __host__ DualQuaternion_dev operator/ (DualQuaternion_dev b) {
        if(m_real.Magnitude() == 0.0f)
            return DualQuaternion_dev();
        DualQuaternion_dev c = Inv(b);
        return DualQuaternion_dev(m_real*c.Real(), m_real*c.Dual() + m_dual*c.Real());
    }
    
    // Conjugate
    __device__ __host__ DualQuaternion_dev Conjugate () {
        return DualQuaternion_dev (m_real.Conjugate(), m_dual.Conjugate());
    }
    
    // Conjugate
    __device__ DualQuaternion_dev DualConjugate1 () {
        return DualQuaternion_dev (m_real, m_dual * (-1.0f));
    }
    
    // Conjugate
    __device__ __host__ DualQuaternion_dev DualConjugate2 () {
        return DualQuaternion_dev (m_real.Conjugate(), m_dual.Conjugate() * (-1.0f));
    }
    
    __device__ __host__ float Dot (DualQuaternion_dev b) {
        return m_real.Dot(b.Real());
    }
    
    // Magnitude
    __device__ __host__ float Magnitude () {
        return m_real.Dot(m_real);
    }

    __device__ __host__ float Norm () {
        return m_real.Magnitude();
        /*float norm_a = m_real.Magnitude();
        if (norm_a == 0.0f)
            return DualQuaternion_dev();
        return DualQuaternion_dev();
        Quaternion_dev b = (m_real.Conjugate() * m_dual + m_dual.Conjugate() * m_real) * (1.0f/(2.0f*norm_a));
        return DualQuaternion_dev(Quaternion_dev(0.0,0.0,0.0,norm_a), Quaternion_dev(0.0,0.0,0.0,b.Scalar()));*/
    }

    __device__ __host__ DualQuaternion_dev Normalize () {
        float norm_a = m_real.Norm();
        if (norm_a == 0.0f)
            return DualQuaternion_dev();
        Quaternion_dev real_part = m_real * (1.0f / norm_a);
        Quaternion_dev dual_factor = (m_dual * m_real.Conjugate() + Quaternion_dev(0.0, 0.0, 0.0, -1.0f * m_real.Dot(m_dual))) * (1.0f / (norm_a * norm_a));
        return DualQuaternion_dev(real_part, dual_factor * real_part);
    }
    __device__ __host__ Quaternion_dev GetRotation () {
        return m_real;
    }
    
    __device__ __host__ float3 GetTranslation () {
        Quaternion_dev t = (m_dual * 2.0f) * m_real.Conjugate();
        return t.Vector();
    }
    
    __device__ __host__ float4x4 DualQuaternionToMatrix () {
        float4x4 M;
        
        float mag = m_real.Dot(m_real);
        if (mag < 0.000001f)
            return M;
        DualQuaternion_dev q = DualQuaternion_dev(m_real*(1.0f/mag), m_dual*(1.0f/mag));
        
        float w = q.m_real.Scalar();
        float3 v = q.m_real.Vector();
        
        M(0,0) = w*w + v.x*v.x - v.y*v.y - v.z*v.z;
        M(1,0) = 2.0f*v.x*v.y + 2.0f*w*v.z;
        M(2,0) = 2.0f*v.x*v.z - 2.0f*w*v.y;
        M(3,0) = 0.0f;
        
        M(0,1) = 2.0f*v.x*v.y - 2.0f*w*v.z;
        M(1,1) = w*w + v.y*v.y - v.x*v.x - v.z*v.z;
        M(2,1) = 2.0f*v.y*v.z + 2.0f*w*v.x;
        M(3,1) = 0.0f;
        
        M(0,2) = 2.0f*v.x*v.z + 2.0f*w*v.y;
        M(1,2) = 2.0f*v.y*v.z - 2.0f*w*v.x;
        M(2,2) = w*w + v.z*v.z - v.x*v.x - v.y*v.y;
        M(3,2) = 0.0f;
        
        Quaternion_dev t = (m_dual * 2.0f) * m_real.Conjugate();
        float3 t_v = t.Vector();
        M(0,3) = t_v.x;
        M(1,3) = t_v.y;
        M(2,3) = t_v.z;
        M(3,3) = 1.0f;
        
        return M;
    }

    __device__ __host__ void Print() {
        m_real.Print();
        m_dual.Print();
    }
};

inline __device__ __host__ Quaternion_dev InvQ(Quaternion_dev a) {
    float norm = a.value.x * a.value.x + a.value.y * a.value.y + a.value.z * a.value.z + a.value.w * a.value.w;
    if (norm == 0.0f)
        return Quaternion_dev(0.0, 0.0, 0.0, 0.0);
    return Quaternion_dev(-a.value.x / norm, -a.value.y / norm, -a.value.z / norm, a.value.w / norm);
}

inline __device__ __host__ float3 logOfQuaternion(Quaternion_dev qi) {
    float3 thetai;
    float3 v = qi.Vector();
    float cosineAbs = qi.Scalar();

    // float sin_squared  = v.x * v.x + v.y * v.y + v.z * v.z ;

    float sin_theta = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);// theta

    if (sin_theta == 0.0f) {
        thetai.x = 0.0f;
        thetai.y = 0.0f;
        thetai.z = 0.0f;
        return thetai;
    }

    float k = 2.0f * acos(cosineAbs) / sin_theta; //std::atan2(sin_theta, cosineAbs)/sin_theta;//acos(cosineAbs) / sin_theta;//std::atan2(sin_theta, cosineAbs)/sin_theta; //;

    thetai.x = k * v.x;
    thetai.y = k * v.y;
    thetai.z = k * v.z;

    return thetai;
}

// outer product
inline __device__ __host__ void outer_prodB(float3 a, float3 b, float3 res[3]) {
    res[0].x = a.x * b.x; res[1].x = a.x * b.y; res[2].x = a.x * b.z;
    res[0].y = a.y * b.x; res[1].y = a.y * b.y; res[2].y = a.y * b.z;
    res[0].z = a.z * b.x; res[1].z = a.z * b.y; res[2].z = a.z * b.z;
}

inline __device__ __host__ void outer_prod_quat_orderB(float4 a, float4 b, float4 res[4]) {
    res[0].x = a.w * b.w; res[1].x = a.w * b.x; res[2].x = a.w * b.y; res[3].x = a.w * b.z;
    res[0].y = a.x * b.w; res[1].y = a.x * b.x; res[2].y = a.x * b.y; res[3].y = a.x * b.z;
    res[0].z = a.y * b.w; res[1].z = a.y * b.x; res[2].z = a.y * b.y; res[3].z = a.y * b.z;
    res[0].w = a.z * b.w; res[1].w = a.z * b.x; res[2].w = a.z * b.y; res[3].w = a.z * b.z;
}

inline __device__ __host__ float3x3 outer_prod(float3 a, float3 b) {
    float3x3 res;
    res(0, 0) = a.x * b.x; res(0, 1) = a.x * b.y; res(0, 2) = a.x * b.z;
    res(1, 0) = a.y * b.x; res(1, 1) = a.y * b.y; res(1, 2) = a.y * b.z;
    res(2, 0) = a.z * b.x; res(2, 1) = a.z * b.y; res(2, 2) = a.z * b.z;
    return res;
}

inline __device__ __host__ float4x4 outer_prod_quat_order(float4 a, float4 b) {
    float4x4 res;
    res(0, 0) = a.w * b.w; res(0, 1) = a.w * b.x; res(0, 2) = a.w * b.y; res(0, 3) = a.w * b.z;
    res(1, 0) = a.x * b.w; res(1, 1) = a.x * b.x; res(1, 2) = a.x * b.y; res(1, 3) = a.x * b.z;
    res(2, 0) = a.y * b.w; res(2, 1) = a.y * b.x; res(2, 2) = a.y * b.y; res(2, 3) = a.y * b.z;
    res(3, 0) = a.z * b.w; res(3, 1) = a.z * b.x; res(3, 2) = a.z * b.y; res(3, 3) = a.z * b.z;
    return res;
}

//Matrix multiplications
inline __device__ void mult(float3 a[3], float b, float3 res[3]) {
    res[0].x = a[0].x * b;
    res[0].y = a[0].y * b;
    res[0].z = a[0].z * b;

    res[1].x = a[1].x * b;
    res[1].y = a[1].y * b;
    res[1].z = a[1].z * b;

    res[2].x = a[2].x * b;
    res[2].y = a[2].y * b;
    res[2].z = a[2].z * b;
}

inline __device__ void mult(float4 a[4], float b, float4 res[4]) {
    res[0].x = a[0].x * b;
    res[0].y = a[0].y * b;
    res[0].z = a[0].z * b;
    res[0].w = a[0].w * b;

    res[1].x = a[1].x * b;
    res[1].y = a[1].y * b;
    res[1].z = a[1].z * b;
    res[1].w = a[1].w * b;

    res[2].x = a[2].x * b;
    res[2].y = a[2].y * b;
    res[2].z = a[2].z * b;
    res[2].w = a[2].w * b;

    res[3].x = a[3].x * b;
    res[3].y = a[3].y * b;
    res[3].z = a[3].z * b;
    res[3].w = a[3].w * b;
}

inline __device__ void mult(float3 a[3], float3 b[3], float3 res[3]) {
    res[0].x = a[0].x * b[0].x + a[1].x * b[0].y + a[2].x * b[0].z; 
    res[0].y = a[0].y * b[0].x + a[1].y * b[0].y + a[2].y * b[0].z;
    res[0].z = a[0].z * b[0].x + a[1].z * b[0].y + a[2].z * b[0].z;

    res[1].x = a[0].x * b[1].x + a[1].x * b[1].y + a[2].x * b[1].z;
    res[1].y = a[0].y * b[1].x + a[1].y * b[1].y + a[2].y * b[1].z;
    res[1].z = a[0].z * b[1].x + a[1].z * b[1].y + a[2].z * b[1].z;

    res[2].x = a[0].x * b[2].x + a[1].x * b[2].y + a[2].x * b[2].z;
    res[2].y = a[0].y * b[2].x + a[1].y * b[2].y + a[2].y * b[2].z;
    res[2].z = a[0].z * b[2].x + a[1].z * b[2].y + a[2].z * b[2].z;
}

inline __device__ void mult(float4 a[4], float4 b[4], float4 res[4]) {
    res[0].x = a[0].x * b[0].x + a[1].x * b[0].y + a[2].x * b[0].z + a[3].x * b[0].w;
    res[0].y = a[0].y * b[0].x + a[1].y * b[0].y + a[2].y * b[0].z + a[3].y * b[0].w;
    res[0].z = a[0].z * b[0].x + a[1].z * b[0].y + a[2].z * b[0].z + a[3].z * b[0].w;
    res[0].w = a[0].w * b[0].x + a[1].w * b[0].y + a[2].w * b[0].z + a[3].w * b[0].w;

    res[1].x = a[0].x * b[1].x + a[1].x * b[1].y + a[2].x * b[1].z + a[3].x * b[1].w;
    res[1].y = a[0].y * b[1].x + a[1].y * b[1].y + a[2].y * b[1].z + a[3].y * b[1].w;
    res[1].z = a[0].z * b[1].x + a[1].z * b[1].y + a[2].z * b[1].z + a[3].z * b[1].w;
    res[1].w = a[0].w * b[0].x + a[1].w * b[0].y + a[2].w * b[0].z + a[3].w * b[1].w;

    res[2].x = a[0].x * b[2].x + a[1].x * b[2].y + a[2].x * b[2].z + a[3].x * b[2].w;
    res[2].y = a[0].y * b[2].x + a[1].y * b[2].y + a[2].y * b[2].z + a[3].y * b[2].w;
    res[2].z = a[0].z * b[2].x + a[1].z * b[2].y + a[2].z * b[2].z + a[3].z * b[2].w;
    res[2].w = a[0].w * b[0].x + a[1].w * b[0].y + a[2].w * b[0].z + a[3].w * b[2].w;

    res[3].x = a[0].x * b[2].x + a[1].x * b[2].y + a[2].x * b[2].z + a[3].x * b[3].w;
    res[3].y = a[0].y * b[2].x + a[1].y * b[2].y + a[2].y * b[2].z + a[3].y * b[3].w;
    res[3].z = a[0].z * b[2].x + a[1].z * b[2].y + a[2].z * b[2].z + a[3].z * b[3].w;
    res[3].w = a[0].w * b[0].x + a[1].w * b[0].y + a[2].w * b[0].z + a[3].w * b[3].w;
}

//matrix additions
inline __device__ void add(float3 a[3], float3 b[3], float3 res[3]) {
    res[0].x = a[0].x + b[0].x;
    res[0].y = a[0].y + b[0].y;
    res[0].z = a[0].z + b[0].z;

    res[1].x = a[1].x + b[1].x;
    res[1].y = a[1].y + b[1].y;
    res[1].z = a[1].z + b[1].z;

    res[2].x = a[2].x + b[2].x;
    res[2].y = a[2].y + b[2].y;
    res[2].z = a[2].z + b[2].z;
}

inline __device__ void add(float4 a[4], float4 b[4], float4 res[4]) {
    res[0].x = a[0].x + b[0].x;
    res[0].y = a[0].y + b[0].y;
    res[0].z = a[0].z + b[0].z;
    res[0].w = a[0].w + b[0].w;

    res[1].x = a[1].x + b[1].x;
    res[1].y = a[1].y + b[1].y;
    res[1].z = a[1].z + b[1].z;
    res[1].w = a[1].w + b[1].w;

    res[2].x = a[2].x + b[2].x;
    res[2].y = a[2].y + b[2].y;
    res[2].z = a[2].z + b[2].z;
    res[2].w = a[2].w + b[2].w;

    res[3].x = a[3].x + b[3].x;
    res[3].y = a[3].y + b[3].y;
    res[3].z = a[3].z + b[3].z;
    res[3].w = a[3].w + b[3].w;
}

//matrix substraction
inline __device__ void sub(float3 a[3], float3 b[3], float3 res[3]) {
    res[0].x = a[0].x - b[0].x;
    res[0].y = a[0].y - b[0].y;
    res[0].z = a[0].z - b[0].z;

    res[1].x = a[1].x - b[1].x;
    res[1].y = a[1].y - b[1].y;
    res[1].z = a[1].z - b[1].z;

    res[2].x = a[2].x - b[2].x;
    res[2].y = a[2].y - b[2].y;
    res[2].z = a[2].z - b[2].z;
}

inline __device__ void sub(float4 a[3], float4 b[3], float4 res[4]) {
    res[0].x = a[0].x - b[0].x;
    res[0].y = a[0].y - b[0].y;
    res[0].z = a[0].z - b[0].z;
    res[0].w = a[0].w - b[0].w;

    res[1].x = a[1].x - b[1].x;
    res[1].y = a[1].y - b[1].y;
    res[1].z = a[1].z - b[1].z;
    res[1].w = a[1].w - b[1].w;

    res[2].x = a[2].x - b[2].x;
    res[2].y = a[2].y - b[2].y;
    res[2].z = a[2].z - b[2].z;
    res[2].w = a[2].w - b[2].w;

    res[3].x = a[3].x - b[3].x;
    res[3].y = a[3].y - b[3].y;
    res[3].z = a[3].z - b[3].z;
    res[3].w = a[3].w - b[3].w;
}
 
#endif