//
//  CubicMesh.h
//  DEEPANIM
//
//  Created by Diego Thomas on 2021/02/12.
//

#ifndef CubicMesh_h
#define CubicMesh_h

#include "include_gpu/Utilities.h"

class CubicMesh {
private:
    int _Nb_voxels = 0;
    my_int3 _Dim;
    
    float ***_tsdf;
    my_float3 ***_voxels;
    
public:
    CubicMesh() {
        _Nb_voxels = 0;
        _Dim = make_my_int3(0,0,0);
        _tsdf = NULL;
        _voxels = NULL;
    }
    
    CubicMesh(my_int3 dim, my_float3 center, float res) {
        _Nb_voxels = dim.x*dim.y*dim.z;
        _Dim = make_my_int3(dim.x,dim.y,dim.z);
        
        _voxels = new my_float3 **[_Dim.x];
        _tsdf = new float **[_Dim.x];
        for (int i = 0; i < _Dim.x; i++) {
            _voxels[i] = new my_float3 *[_Dim.y];
            _tsdf[i] = new float *[_Dim.y];
            for (int j = 0; j < _Dim.y; j++) {
                _voxels[i][j] = new my_float3[_Dim.z];
                _tsdf[i][j] = new float[_Dim.z];
                for (int k = 0; k < _Dim.z; k++) {
                    _voxels[i][j][k] = make_my_float3(center.x + (float(i) - float(dim.x)/2.0f)*res,
                                                      center.y + (float(j) - float(dim.y)/2.0f)*res,
                                                      center.z + (float(k) - float(dim.z)/2.0f)*res);
                }
            }
        }
    }
       
    ~CubicMesh() {
        for (int i = 0; i < _Dim.x; i++) {
            for (int j = 0; j < _Dim.y; j++) {
                delete []_voxels[i][j];
                delete []_tsdf[i][j];
            }
            delete []_voxels[i];
            delete []_tsdf[i];
        }
        delete []_voxels;
        delete []_tsdf;
    }
    
    inline my_int3 Dim() {return _Dim;}
    
    inline my_float3 ***Voxels() {return _voxels;}
    
    inline float *LevelSet(float *vertices, int *faces, float *normals_face, int nb_faces) {
    }
    
    inline int *MC(float *TSDF, float *Vertices, int *Faces, float m_iso = 0.0) {
    }
    
    void FitToTSDF(float ***tsdf_grid, my_int3 dim_grid, my_float3 center, float res, int outerloop_maxiter = 50, int innerloop_maxiter = 100, float delta = 0.1f, float delta_elastic = 0.1f);

};

#endif /* CubicMesh_h */