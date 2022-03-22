#ifndef __LEVELSET_GPU_H
#define __LEVELSET_GPU_H

#include "include_gpu/Utilities.h"
#include "include_gpu/cudaTypes.cuh"

/**** Function definitions ****/
std::pair<float***, int***> LevelSet_gpu(float* vertices, int* labels, int* faces, float* normals, int nb_vertices, int nb_faces, int3 size_grid, float3 center_grid, float res_x, float res_y, float res_z, float disp);

#endif