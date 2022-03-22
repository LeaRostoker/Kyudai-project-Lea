#ifndef __UTILITIES_H
#define __UTILITIES_H

#pragma once

#ifdef _CUDA_
    #define NOMINMAX
#endif

/*** Include files for OpenGL to work ***/
/*#ifdef _APPLE_
    #define GLFW_INCLUDE_GLCOREARB
    #include <OpenGL/gl3.h>
    //#include <OpenGL/glu.h>
    #include <OpenGL/glext.h>
    //#include <GLFW/glfw3.h>
    #include <GLUT/glut.h>
#else
    #include <GL/glew.h>
    #include <GL/glut.h>
    #include <GLFW/glfw3.h>
#endif*/

/*** Standard include files for manipulating vectors, files etc... ***/
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <chrono>
#include <random>
#include <functional>
#include <future>
#include <thread>

#include <time.h>
#include <Windows.h>

/*#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc.hpp"*/
//#include <opencv/cv.hpp>

/*** Include files to manipulate matrices ***/
#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Cholesky>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/Sparse>


//#include <glm/glm.hpp>

/*** Include files for gpu operations on vectors ***/
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/remove.h>
#include <thrust/generate.h>
#include <thrust/detail/type_traits.h>

/*** Include files for CUDA to work ***/
//#define __CUDACC__
#include <cuda_runtime.h>
#include <vector_types.h>
#include <cuda_gl_interop.h>
#include <cusparse.h>
#include "cusolverSp.h"
#include <cusolverDn.h>
#include <cublas_v2.h> 

// Utilities and timing functions
//#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
// CUDA helper functions
#include <include_gpu/helper_cuda.h>         // helper functions for CUDA error check
//#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop

/*** Include files for cuda kernels ***/
#include "include_gpu/LevelSet_gpu.cuh"


/*** Global constant variables ***/
//#define VERBOSE        // Display or not intermediate messages
//#define BASIC_TETRA
#define INC_SIZE 25
#define INC_STEP 64
#define GAP  25             /* gap between subwindows */
#define lighting true
#define TIMING false
//#define PI 3.1415926535897932384626433832795
#define DISPLAY_FRAME_IN true
#define MIN_SIZE_PLAN 5000
#define THRESH 0.1f

#define PATH_DATA string("F:/Data/TetraModel/")

using namespace std;

Eigen::Matrix4f Interpolate(Eigen::Matrix4f A, Eigen::Matrix4f B, float lambda);

Eigen::Matrix4f InverseTransfo(Eigen::Matrix4f A);

Eigen::Matrix4f euler2matrix(float Rx, float Ry, float Rz);

Eigen::Matrix3f rodrigues2matrix(float *Rodrigues);

Eigen::Matrix4f quaternion2matrix(float *q);

// Function to get cofactor of A[p][q] in temp[][]. n is current
// dimension of A[][]
void getCofactor(float * A, float * temp, int p, int q, int n);
  
/* Recursive function for finding determinant of matrix.
   n is current dimension of A[][].
 The method is the Cofactor expansion*/
float determinant(float * A, int n);
  
// Function to get adjoint of A[N][N] in adj[N][N].
void adjoint(float * A, float *adj, int N);
  
// Function to calculate and store inverse, returns false if
// matrix is singular
bool inverse(float * A, float *inverse, int N);
  
// Generic function to display the matrix.  We use it to display
// both adjoin and inverse. adjoin is integer matrix and inverse
// is a float.
void display(float * A, int N);

struct my_int2 : int2 {
    my_int2() {
        x = 0;
        y = 0;
    }

    my_int2(int a, int b) {
        x = a;
        y = b;
    }
};
inline static my_int2 make_my_int2(int X, int Y) {
    return my_int2(X, Y);
}

struct my_int3 : int3 {
    my_int3() {
        x = 0;
        y = 0;
        z = 0;
    }
    my_int3(int a, int b, int c) {
        x = a;
        y = b;
        z = c;
    }

    //print
    void print() {
        cout << x << ", " << y << ", " << z << endl;
    }
};

inline static my_int3 make_my_int3(int X, int Y, int Z) {
    return my_int3(X, Y, Z);
}

struct my_int4 : int4 {
    my_int4() {
        x = 0;
        y = 0;
        z = 0;
        w = 0;
    }
    my_int4(int a, int b, int c, int d) {
        x = a;
        y = b;
        z = c;
        w = d;
    }

    //print
    void print() {
        cout << x << ", " << y << ", " << z << ", " << w << endl;
    }
};

inline static my_int4 make_my_int4(int X, int Y, int Z, int W) {
    return my_int4(X, Y, Z, W);
}

struct my_float3 : float3 {
    // Initializer
    my_float3() {
        x = 0.0f;
        y = 0.0f;
        z = 0.0f;
    }

    my_float3(float a) {
        x = a;
        y = a;
        z = a;
    }

    my_float3(float X, float Y, float Z) {
        x = X;
        y = Y;
        z = Z;
    }

    // Scalar Multiplication
    my_float3 operator* (float s) {
        return my_float3(s * x, s * y, s * z);
    }

    // outer product
    Eigen::Matrix3f operator* (my_float3 v) {
        Eigen::Matrix3f res;
        res << x * v.x, x* v.y, x* v.z,
            y* v.x, y* v.y, y* v.z,
            z* v.x, z* v.y, z* v.z;
        return res;
    }

    //matrix product
    my_float3 operator* (Eigen::Matrix3f M) {
        return my_float3(x * M(0, 0) + y * M(0, 1) + z * M(0, 2),
            x * M(1, 0) + y * M(1, 1) + z * M(1, 2),
            x * M(2, 0) + y * M(2, 1) + z * M(2, 2));
    }

    // Addition
    my_float3 operator+ (my_float3 a) {
        return my_float3(a.x + x, a.y + y, a.z + z);
    }

    // Subtraction
    my_float3 operator- (my_float3 a) {
        return my_float3(x - a.x, y - a.y, z - a.z);
    }

};

inline static my_float3 make_my_float3(float X, float Y, float Z) {
    return my_float3(X, Y, Z);
}

inline static my_float3 make_my_float3(float X) {
    return my_float3(X);
}

static my_float3 cross(my_float3 a, my_float3 b) {
    return my_float3(a.y * b.z - a.z * b.y,
        -a.x * b.z + a.z * b.x,
        a.x * b.y - a.y * b.x);
}

static float dot(my_float3 a, my_float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static float norm(my_float3 a) {
    return sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}

static Eigen::Matrix3f outer_prod_tri_order(my_float3 a, my_float3 b) {
    Eigen::Matrix3f res;
    res << a.x * b.x, a.x* b.y, a.x* b.z,
        a.y* b.x, a.y* b.y, a.y* b.z,
        a.z* b.x, a.z* b.y, a.z* b.z;

    return res;
}

struct my_float4 : float4 {
    my_float4() {
        x = 0.0f;
        y = 0.0f;
        z = 0.0f;
        w = 0.0f;
    }
    my_float4(float X, float Y, float Z, float W)  {
        x = X;
        y = Y;
        z = Z;
        w = W;
    }
};

inline static my_float4 make_my_float4(float X, float Y, float Z, float W) {
    return my_float4(X, Y, Z, W);
}

static Eigen::Matrix4f outer_prod(my_float4 a, my_float4 b) {
    Eigen::Matrix4f res;
    res << a.x * b.x, a.y* b.x, a.z* b.x, a.w* b.x,
        a.x* b.y, a.y* b.y, a.z* b.y, a.w* b.y,
        a.x* b.z, a.y* b.z, a.z* b.z, a.w* b.z,
        a.x* b.w, a.y* b.w, a.z* b.w, a.w* b.w;

    return res;
}

static Eigen::Matrix4f outer_prod_quat_order(my_float4 a, my_float4 b) {
    Eigen::Matrix4f res;
    res << a.w * b.w, a.w* b.x, a.w* b.y, a.w* b.z,
        a.x* b.w, a.x* b.x, a.x* b.y, a.x* b.z,
        a.y* b.w, a.y* b.x, a.y* b.y, a.y* b.z,
        a.z* b.w, a.z* b.x, a.z* b.y, a.z* b.z;

    return res;
}

float IsInterectingRayTriangle3D(my_float3 ray, my_float3 p0, my_float3 p1, my_float3 p2, my_float3 p3, my_float3 n);

float DistancePointFace3D(my_float3 p0, my_float3 p1, my_float3 p2, my_float3 p3, my_float3 n, bool approx = false);

void SkinWeightsFromFace3D(float *s_w, float *w1, float *w2, float *w3, my_float3 p0, my_float3 p1, my_float3 p2, my_float3 p3, my_float3 n);

struct Quaternion {
    my_float4 value;
    
    Quaternion(float X, float Y, float Z, float W) {
        value.x = X;
        value.y = Y;
        value.z = Z;
        value.w = W;
    }
    
    Quaternion (my_float3 t, float W) {
        value.x = t.x;
        value.y = t.y;
        value.z = t.z;
        value.w = W;
    }
    
    Quaternion (my_float3 r) {
        float norm_r = sqrt(r.x*r.x + r.y*r.y + r.z*r.z);
        if (norm_r == 0.0f) {
            value.x = 0.0f;
            value.y = 0.0f;
            value.z = 0.0f;
            value.w = 1.0f;
            return;
        }
        float f = sin(norm_r/2.0f)/norm_r;
        value.x = f*r.x;
        value.y = f*r.y;
        value.z = f*r.z;
        value.w = cos(norm_r/2.0f);
    }
    
    Quaternion(Eigen::Matrix3f rot) {
        float trace = rot(0,0) + rot(1,1) + rot(2,2);
        if( trace > 0 ) {
            float s = 0.5f / sqrtf(trace + 1.0f);
            value.w = 0.25f / s;
            value.x = ( rot(2,1) - rot(1,2) ) * s;
            value.y = ( rot(0,2) - rot(2,0) ) * s;
            value.z = ( rot(1,0) - rot(0,1) ) * s;
        } else {
            if ( rot(0,0) > rot(1,1) && rot(0,0) > rot(2,2) ) {
                float s = 2.0f * sqrtf( 1.0f + rot(0,0) - rot(1,1) - rot(2,2));
                value.w = (rot(2,1) - rot(1,2) ) / s;
                value.x = 0.25f * s;
                value.y = (rot(0,1) + rot(1,0) ) / s;
                value.z = (rot(0,2) + rot(2,0) ) / s;
            } else if (rot(1,1) > rot(2,2)) {
                float s = 2.0f * sqrtf( 1.0f + rot(1,1) - rot(0,0) - rot(2,2));
                value.w = (rot(0,2)- rot(2,0) ) / s;
                value.x = (rot(0,1) + rot(1,0)) / s;
                value.y = 0.25f * s;
                value.z = (rot(1,2) + rot(2,1) ) / s;
            } else {
                float s = 2.0f * sqrtf( 1.0f + rot(2,2) - rot(0,0) - rot(1,1) );
                value.w = (rot(1,0) - rot(0,1) ) / s;
                value.x = (rot(0,2) + rot(2,0) ) / s;
                value.y = (rot(1,2) + rot(2,1) ) / s;
                value.z = 0.25f * s;
            }
        }
    }
    
    // Inverse
    Quaternion Inv (Quaternion a) {
        float norm = a.value.x*a.value.x + a.value.y*a.value.y + a.value.z*a.value.z + a.value.w*a.value.w;
        if (norm == 0.0f)
            return Quaternion(0.0,0.0,0.0,1.0);
        return Quaternion(-a.value.x/norm, -a.value.y/norm, -a.value.z/norm, a.value.w/norm);
    }
    
    my_float3 Vector () {
        my_float3 r;
        r.x = value.x;
        r.y = value.y;
        r.z = value.z;
        return r;
    }
    
    float Scalar () {
        return value.w;
    }
    
    // Scalar Multiplication
    Quaternion operator* (float s) {
        my_float3 v1;
        v1.x = s*value.x;
        v1.y = s*value.y;
        v1.z = s*value.z;
        return Quaternion (v1, s*value.w);
    }
    
    // Multiplication
    Quaternion operator* (Quaternion b) {
        my_float3 v1;
        v1.x = value.x;
        v1.y = value.y;
        v1.z = value.z;
        float w1 = value.w;
        my_float3 v2 = b.Vector();
        float w2 = b.Scalar();
        
        return Quaternion (v2*w1 + v1*w2 + cross(v1,v2), w1*w2 - dot(v1,v2));
    }
    
    // Addition
    Quaternion operator+ (Quaternion b) {
        my_float3 v1;
        v1.x = value.x;
        v1.y = value.y;
        v1.z = value.z;
        float w1 = value.w;
        my_float3 v2 = b.Vector();
        float w2 = b.Scalar();
        
        return Quaternion (v1 + v2, w1 + w2);
    }
    
    // Conjugate
    Quaternion Conjugate () {
        return Quaternion(-value.x, -value.y, -value.z, value.w);
    }
    
    // Magnitude
    float Magnitude () {
        return sqrt(value.x*value.x + value.y*value.y + value.z*value.z + value.w*value.w);
    }
    
    float Dot(Quaternion b) {
        my_float3 v1;
        v1.x = value.x;
        v1.y = value.y;
        v1.z = value.z;
        float w1 = value.w;
        my_float3 v2 = b.Vector();
        float w2 = b.Scalar();
        
        return w1*w2 + dot(v1,v2);
    }
    
    //Normalize
    Quaternion Normalize ( ) {
        float norm = sqrt(value.x*value.x + value.y*value.y + value.z*value.z + value.w*value.w);
        return Quaternion(value.x/norm, value.y/norm, value.z/norm, value.w/norm);
    }

    void Print()
    {
        cout << value.x << ", " << value.y << ", " << value.z << ", " << value.w << endl;
    }
    //norm
    float Norm(){
        float norm = sqrt(value.x*value.x + value.y*value.y + value.z*value.z + value.w*value.w);
        return norm;
    }
};

// Inverse
Quaternion InvQ (Quaternion a);

//
struct DualQuaternion {
    Quaternion m_real = Quaternion(0.0,0.0,0.0,1.0);
    Quaternion m_dual = Quaternion(0.0,0.0,0.0,0.0);

    DualQuaternion() {
        m_real = Quaternion(0.0,0.0,0.0,1.0);
        m_dual = Quaternion(0.0,0.0,0.0,0.0);
    }
        
    DualQuaternion(Quaternion r, Quaternion d) {
        //m_real = r.Normalize();
        m_real = r;
        m_dual = d;
    }
    
    DualQuaternion(Quaternion r, my_float3 t) {
        if (r.Magnitude() > 0.0f)
            m_real = r.Normalize();
        m_dual = (Quaternion(t,0.0) * m_real) * 0.5;
    }
    
    DualQuaternion(my_float3 r, my_float3 t) {
        m_real = Quaternion(r);
        m_dual = (Quaternion(t,0.0) * m_real) * 0.5;
    }
    
    DualQuaternion(Eigen::Matrix4f transfo) {
        Eigen::Matrix3f rotation;
        rotation << transfo(0,0), transfo(0,1), transfo(0,2),
                    transfo(1,0), transfo(1,1), transfo(1,2),
                    transfo(2,0), transfo(2,1), transfo(2,2);
        m_real = Quaternion(rotation);

        my_float3 t = make_my_float3(transfo(0,3), transfo(1,3), transfo(2,3));

        m_dual = (Quaternion(t,0.0) * m_real) * 0.5;
    }
    
    Quaternion Real() {
        return m_real;
    }
    
    Quaternion Dual () {
        return m_dual;
    }
    
    DualQuaternion Identity() {
        return DualQuaternion(Quaternion(make_my_float3(0.0f), 1.0f), Quaternion(make_my_float3(0.0f), 0.0f));
    }
    
    // Inverse
    DualQuaternion Inv (DualQuaternion a) {
        if(a.m_real.Magnitude() == 0.0f)
            return DualQuaternion();

        Quaternion p_1 = InvQ(a.m_real);
        Quaternion p_2 = a.m_dual * p_1;
        DualQuaternion q_1 = DualQuaternion(p_1, Quaternion(0.0,0.0,0.0,0.0));
        DualQuaternion q_2 = DualQuaternion(Quaternion(0.0,0.0,0.0,1.0), p_2 * (-1.0f));

        return q_1 * q_2;
    }
    
    //Addition
    DualQuaternion operator+ (DualQuaternion b) {
        return DualQuaternion(m_real + b.Real(), m_dual + b.Dual());
    }
    
    // Scalar multiplication
    DualQuaternion operator* (float s) {
        return DualQuaternion(m_real*s, m_dual*s);
    }
    
    // Multiplication
    DualQuaternion operator* (DualQuaternion b) {
        return DualQuaternion(m_real*b.Real(), m_real*b.Dual() + m_dual*b.Real());
    }
    
    // Division
    DualQuaternion operator/ (DualQuaternion b) {
        if(m_real.Magnitude() == 0.0f)
            return DualQuaternion();
        DualQuaternion c = Inv(b);
        return DualQuaternion(m_real*c.Real(), m_real*c.Dual() + m_dual*c.Real());
    }
    
    // Conjugate
    DualQuaternion Conjugate () {
        return DualQuaternion (m_real.Conjugate(), m_dual.Conjugate());
    }
    
    // Conjugate
    DualQuaternion DualConjugate1 () {
        return DualQuaternion (m_real, m_dual * (-1.0f));
    }
    
    // Conjugate
    DualQuaternion DualConjugate2 () {
        return DualQuaternion (m_real.Conjugate(), m_dual.Conjugate() * (-1.0f));
    }
    
    float Dot (DualQuaternion b) {
        return m_real.Dot(b.Real());
    }
    
    // Magnitude
    float Magnitude () {
        return m_real.Dot(m_real);
    }
    
    float Norm () {
        return m_real.Magnitude();
    }
    
    DualQuaternion Normalize () {
        float norm_a = m_real.Norm();//if one uses m_real.Magnitude()) it is the squared norm
        if (norm_a == 0.0f)
            return DualQuaternion();
        Quaternion real_part = m_real * (1.0f/norm_a);
        Quaternion dual_factor = (m_dual * m_real.Conjugate()+ Quaternion(0.0,0.0,0.0,-1.0f*m_real.Dot(m_dual)) ) * (1.0f/(norm_a*norm_a));
        return DualQuaternion (real_part, dual_factor*real_part);
    }
    
    Quaternion GetRotation () {
        return m_real;
    }
    
    my_float3 GetTranslation () {
        Quaternion t = (m_dual * 2.0f) * m_real.Conjugate();
        return t.Vector();
    }
    
    /*my_float4x4 DualQuaternionToMatrix () {
        my_float4x4 M;*/
    Eigen::Matrix4f DualQuaternionToMatrix () {
        Eigen::Matrix4f M;
        
        float mag = m_real.Dot(m_real);
        if (mag < 0.000001f)
            return M;
        DualQuaternion q = DualQuaternion(m_real*(1.0f/mag), m_dual*(1.0f/mag));
        
        float w = q.m_real.Scalar();
        my_float3 v = q.m_real.Vector();
        
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
        
        Quaternion t = (m_dual * 2.0f) * m_real.Conjugate();
        my_float3 t_v = t.Vector();
        M(0,3) = t_v.x;
        M(1,3) = t_v.y;
        M(2,3) = t_v.z;
        M(3,3) = 1.0f;
        
        return M;
    }
    void Print()
    {
        cout << "Real: "; m_real.Print();
        cout << "Dual: "; m_dual.Print();
    }
};


my_float3 BicubiInterpolation(my_float3 vertex, float ***tsdf_grid, my_float3 ***grad_grid, my_int3 dim_grid, my_float3 center, float res, float iso = 0.0f);

my_float3 ***VolumetricGrad(float ***tsdf_grid, my_int3 dim_grid);

bool isOnSurface(float ***sdf, float iso, int i, int j, int k, float shift_x = 0.0f, float shift_y = 0.0f,  float shift_z = 0.0f);

float VolumeTetra(float *Nodes, int *Tetra, int tet);

float AreaFace(float *Nodes, int *Faces, int face);
float AreaFace(float *Nodes, int i1, int i2, int i3);

void reduce_points(float *points_out, float *points, int nb_points, int rate);

vector<int> closest_nodes(my_float3 v, float *nodes, int nb_nodes, int num_nearest_nodes=9);

vector<vector<int>> make_adjlist(float *nodes_s, int nb_nodes_s, float *nodes_t, int nb_nodes_t, int num_nearest_nodes = 9, bool allow_self_connection=false);

void list2csv(string save_dir, vector<vector<int>>  list);

float ScTP(my_float3 a, my_float3 b, my_float3 c);
#endif
