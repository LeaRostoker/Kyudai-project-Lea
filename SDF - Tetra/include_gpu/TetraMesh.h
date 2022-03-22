//
//  TetraMesh.h
//  DEEPANIM
//
//  Created by Diego Thomas on 2021/01/19.
//

#ifndef TetraMesh_h
#define TetraMesh_h

#include "MarchingTets.h"
#include "include_gpu/Utilities.h"

#include <list>

bool my_compare (const pair<float, int>& first, const pair<float, int>& second)
{
  return ( first.first < second.first );
}

//void ComputeRBFInnerLoop(float *Nodes, float *Vertices, int *Tetra, float *RBF, int *RBF_id, int i, int nb_nodes, int nb_vertices) {
void ComputeRBFInnerLoop(float *Nodes, float *Vertices, float *RBF, int *RBF_id, int i, int nb_nodes, int nb_vertices) {
    for (int sum1 = i; sum1 < i + nb_nodes/100; sum1++) {
        if (i == 0)
            cout << 100.0f*float(sum1-i)/float(nb_nodes/100) << "%\r";
        
        /*if (isSurface(sum1))
            continue;
        
        // Go through all tetrahedras to find the list of tets that has the current node as summit
        vector<int> TetraList;
        for (int t = 0; t < nb_tetra; t++) {
            int s0 = Tetra[4*t];
            int s1 = Tetra[4*t+1];
            int s2 = Tetra[4*t+2];
            int s3 = Tetra[4*t+3];
            if (s0 == sum1 || s1 == sum1 || s2 == sum1 || s3 == sum1)
                TetraList.push_back(t);
        }
        
        //Compute the full area made by the syrrounding tetraedra
        float vol_tot = 0.0f;
        for (vector<int>::iterator it = TetraList.begin(); it != TetraList.end(); it++) {
            int t = (*it);
            vol_tot += VolumeTetra(Nodes, Tetra[4*t], Tetra[4*t+1], Tetra[4*t+2], Tetra[4*t+3]);
        }
        
        //For each summit of each tetra compute the weight
        int count = 0;
        for (vector<int>::iterator it = TetraList.begin(); it != TetraList.end(); it++) {
            int t = (*it);
            for (int j = 0; j < 4; j++) {
                if (Tetra[4*t+j] == sum1)
                    continue;
                
                //remove the volume of tets that contain j
                float vol_curr = vol_tot;
                for (vector<int>::iterator it2 = TetraList.begin(); it2 != TetraList.end(); it2++) {
                    int t2 = (*it2);
                    if (Tetra[4*t2] == j || Tetra[4*t2+1] == j || Tetra[4*t2+2] == j || Tetra[4*t2+3] == j) {
                        vol_curr -= VolumeTetra(Nodes, Tetra[4*t2], Tetra[4*t2+1], Tetra[4*t2+2], Tetra[4*t2+3]);
                    }
                }
                
                RBF[20*sum1+count] = vol_curr/vol_tot;
                RBF_id [20*sum1+count] = Tetra[4*t+j];
                count++;
            }
        }*/
        
        
        
        my_float3 prev_sum1 = make_my_float3(Nodes[3*sum1], Nodes[3*sum1+1], Nodes[3*sum1+2]);
        list<pair<float, int>> closest;
        
        for (int s = 0; s < nb_vertices; s++) {
            my_float3 prev_vertex = make_my_float3(Vertices[3*s], Vertices[3*s+1], Vertices[3*s+2]);
            float dist = norm(prev_vertex - prev_sum1);
            closest.push_back(make_pair(dist, s+nb_nodes));
        }
                    
        for (int sum2 = 0; sum2 < nb_nodes; sum2++) {
            my_float3 prev_sum2 = make_my_float3(Nodes[3*sum2], Nodes[3*sum2+1], Nodes[3*sum2+2]);
            float dist = norm(prev_sum1 - prev_sum2);
            closest.push_back(make_pair(dist, sum2));
        }
        
        closest.sort(my_compare);
        
        int count = 0;
        list<pair<float, int>>::iterator it;
        for (it=closest.begin(); it!=closest.end(); ++it){
            pair<float, int> elem = (*it);
            RBF[20*sum1+count] = elem.first;
            RBF_id[20*sum1+count] = elem.second;
            count++;
            if (count > 19)
                break;
        }
        
        closest.clear();
        
        //for (int j = 0; j < 20; j++)
        //    cout << "RBF: " << RBF[20*sum1+j] << ", RBF id: " << RBF_id [20*sum1+j] << endl;
    }
}

void SkinWeightsInnerLoop(float *volume_weights, int i, float *vertices, float *skin_weights, int *faces, float *normals, int nb_nodes, int nb_faces, float *nodes) {

    my_float3 ray = make_my_float3(0.0f, 0.0f, 1.0f);
    
    for (int idx = i; idx < i + nb_nodes/100; idx++) {
        if (idx >= nb_nodes)
            break;
        //if (i == 0)
        //    cout << 100.0f*float(idx-i)/float(nb_nodes/100) << "%\r";
        
        my_float3 p0 = make_my_float3(nodes[3*idx], nodes[3*idx+1], nodes[3*idx+2]);
        
        // Compute the smallest distance to the faces
        float min_dist = 1.0e32f;
        float *s_w = new float[24];
        for (int f = 0; f < nb_faces; f++) {
            // Compute distance point to face
            my_float3 n = make_my_float3(normals[3*f], normals[3*f+1], normals[3*f+2]);
            my_float3 p1 = make_my_float3(vertices[3*faces[3*f]], vertices[3*faces[3*f]+1], vertices[3*faces[3*f]+2]);
            my_float3 p2 = make_my_float3(vertices[3*faces[3*f+1]], vertices[3*faces[3*f+1]+1], vertices[3*faces[3*f+1]+2]);
            my_float3 p3 = make_my_float3(vertices[3*faces[3*f+2]], vertices[3*faces[3*f+2]+1], vertices[3*faces[3*f+2]+2]);
                        
            my_float3 center = (p1 + p2 + p3) * (1.0f/3.0f);
            
            // Compute point to face distance
            float curr_dist = DistancePointFace3D(p0, p1, p2, p3, n, false);
            if (curr_dist < min_dist) {
                min_dist = curr_dist;
                // Compute skin weights
                SkinWeightsFromFace3D(s_w, &skin_weights[24*faces[3*f]], &skin_weights[24*faces[3*f+1]], &skin_weights[24*faces[3*f+2]], p0, p1, p2, p3, n);
            }
        }
    
        for (int k = 0; k < 24; k++)
            volume_weights[24*idx+k] = s_w[k];
        
        delete[] s_w;
        
    }
}

class TetraMesh {
private:
    float *_Normals = NULL;
    float *_Nodes = NULL;
    int *_Faces = NULL;
    int *_Tetra = NULL;
    int *_Surface = NULL;
    int *_Inside = NULL;
    int *_Edges_row_ptr = NULL;
    int *_Edges_columns = NULL;
    float *_RBF = NULL;
    int *_RBF_id = NULL;
    float *_SkinWeights = NULL;
    
    int _Nb_tetra;
    int _Nb_Nodes;
    int _Nb_Surface;
    int _Nb_Inside;
    int _Nb_Faces;
    int _Nb_Edges;
    
    float _res = 1.0f;
    
public:
    
    TetraMesh() {
        _Nb_tetra = 0;
        _Nb_Surface = 0;
        _Nb_Faces = 0;
        _Nb_Nodes = 0;
    }
    
    TetraMesh(string path, float res = 1.0f): _res(res) {
        int *res_tetra = LoadPLY_Tet(path+string("TetraShell-Star.ply"), &_Nodes, &_Faces, &_Tetra);
        _Nb_Nodes = res_tetra[0];
        _Nb_Faces = res_tetra[1];
        _Nb_tetra = res_tetra[2];
        
        GetSurfaceAndNormals();
        
        //int *res_tetra = LoadPLY_Tet(filename, &_Vertices, &_Normals, &_Faces, &_Nodes, &_Surface_edges, &_Tetra);
        //int *Faces = NULL;
        //int *res_tetra = LoadPLY_PyMesh(filename, &_Nodes, &Faces, &_Tetra);
        delete []res_tetra;
        cout << "The OuterShell file has " << _Nb_Surface << " vertices and " << _Nb_Nodes << " nodes and " << _Nb_tetra << " tetrahedra" << endl;
        //SaveTetraMeshToPLY(string("Test.ply"), _Nodes, _Tetra, res_tetra[0], res_tetra[2]);
        
        ComputeWeights(path);
        
        _RBF = new float[16*_Nb_Nodes];
        _RBF_id = new int[16*_Nb_Nodes];
        for (int n = 0; n < _Nb_Nodes; n++) {
            for (int i = 0; i < 16; i++)
            _RBF_id[16*n + i] = -1;
        }
        
        cout << _Nb_Surface << ", " << _Nb_Inside << ", " << _Nb_Nodes << endl;
        
        ComputeRBFSurface();
        ComputeRBFInside();
        
        // Create List of edges
        Eigen::SparseMatrix<int, Eigen::RowMajor> Edges(_Nb_Nodes, _Nb_Nodes);
        std::vector<Eigen::Triplet<int>> coefficientsE;
        // go through all voxels
        int vtx_id = 0;
        for (int vox_id = 0; vox_id < _Nb_tetra; vox_id++) {
            // get the edges
            for (int sum_1 = 0; sum_1 < 4; sum_1++) {
                for (int sum_2 = sum_1 + 1; sum_2 < 4; sum_2++) {
                    if (_Tetra[4 * vox_id + sum_1] < _Tetra[4 * vox_id + sum_2])
                        coefficientsE.push_back(Eigen::Triplet<int>(_Tetra[4*vox_id + sum_1], _Tetra[4 * vox_id + sum_2], _Tetra[4 * vox_id + sum_1]));
                    else
                        coefficientsE.push_back(Eigen::Triplet<int>(_Tetra[4 * vox_id + sum_2], _Tetra[4 * vox_id + sum_1], _Tetra[4 * vox_id + sum_2]));
                    vtx_id++;
                }
            }
        }
        Edges.setFromTriplets(coefficientsE.begin(), coefficientsE.end(), [](const int&, const int& b) { return b; });

        _Nb_Edges = Edges.nonZeros();
        int num_rows = Edges.outerSize();
        int num_cols = Edges.innerSize();
        cout << num_rows << endl;
        cout << num_cols << endl;
        cout << _Nb_Edges << ", " << vtx_id << endl;
        
        _Edges_row_ptr = new int[num_rows+1];
        memcpy((void*)_Edges_row_ptr, Edges.outerIndexPtr(), ((long long)num_rows + 1) * sizeof(int));
        //_Edges_values = new int[_nb_edges];
        //memcpy((void*)_Edges_values, Edges.valuePtr(), _nb_edges * sizeof(int));
        _Edges_columns = new int[_Nb_Edges];
        memcpy((void*)_Edges_columns, Edges.innerIndexPtr(), _Nb_Edges * sizeof(int));
        
    }
    
    ~TetraMesh() {
        if (_Nodes != NULL) {
            delete []_Surface;
            delete []_Inside;
            delete []_Normals;
            delete []_Faces;
            delete []_Nodes;
            delete []_Tetra;
            delete []_SkinWeights;
        }
        
        if (_Edges_row_ptr != NULL) {
            delete []_Edges_row_ptr;
            delete []_Edges_columns;
        }
        
        if (_RBF != NULL) {
            delete []_RBF;
            delete []_RBF_id;
        }
    }
    
    inline void SetRes(float res) {
        _res = res;
    }
    
    inline int NBFaces() {
        return _Nb_Faces;
    }
    inline int NBEdges() {
        return _Nb_Edges;
    }
    
    inline int NBTets() {
        return _Nb_tetra;
    }
    
    inline int NBSurface() {
        return _Nb_Surface;
    }
    
    inline int NBNodes() {
        return _Nb_Nodes;
    }
    
    inline int *Surface() {
        return _Surface;
    }
    
    inline float *Normals() {
        return _Normals;
    }
    
    inline int *Faces() {
        return _Faces;
    }
    
    inline int *Tetras() {
        return _Tetra;
    }
    
    inline float *Nodes() {
        return _Nodes;
    }
    
    inline void GetSurfaceAndNormals() {
        
        bool *surface_label = new bool[_Nb_Nodes];
        _Normals = new float[3*_Nb_Nodes];
        for (int n = 0; n < _Nb_Nodes; n++) {
            surface_label[n] = false;
            _Normals[3*n] = 0.0f;
            _Normals[3*n+1] = 0.0f;
            _Normals[3*n+2] = 0.0f;
        }
        
        for (int f = 0; f < _Nb_Faces; f++) {
            surface_label[_Faces[3*f]] = true;
            surface_label[_Faces[3*f+1]] = true;
            surface_label[_Faces[3*f+2]] = true;
            
            my_float3 s1 = make_my_float3(_Nodes[3*_Faces[3*f]], _Nodes[3*_Faces[3*f]+1], _Nodes[3*_Faces[3*f]+2]);
            my_float3 s2 = make_my_float3(_Nodes[3*_Faces[3*f+1]], _Nodes[3*_Faces[3*f+1]+1], _Nodes[3*_Faces[3*f+1]+2]);
            my_float3 s3 = make_my_float3(_Nodes[3*_Faces[3*f+2]], _Nodes[3*_Faces[3*f+2]+1], _Nodes[3*_Faces[3*f+2]+2]);
            
            my_float3 nrmle = cross(s2-s1, s3-s1);
            
            _Normals[3*_Faces[3*f]] = _Normals[3*_Faces[3*f]] + nrmle.x;
            _Normals[3*_Faces[3*f]+1] = _Normals[3*_Faces[3*f]+1] + nrmle.y;
            _Normals[3*_Faces[3*f]+2] = _Normals[3*_Faces[3*f]+2] + nrmle.z;
            
            _Normals[3*_Faces[3*f+1]] = _Normals[3*_Faces[3*f+1]] + nrmle.x;
            _Normals[3*_Faces[3*f+1]+1] = _Normals[3*_Faces[3*f+1]+1] + nrmle.y;
            _Normals[3*_Faces[3*f+1]+2] = _Normals[3*_Faces[3*f+1]+2] + nrmle.z;
            
            _Normals[3*_Faces[3*f+2]] = _Normals[3*_Faces[3*f+2]] + nrmle.x;
            _Normals[3*_Faces[3*f+2]+1] = _Normals[3*_Faces[3*f+2]+1] + nrmle.y;
            _Normals[3*_Faces[3*f+2]+2] = _Normals[3*_Faces[3*f+2]+2] + nrmle.z;
        }
        
        vector<int> surface_list;
        vector<int> inside_list;
        for (int n = 0; n < _Nb_Nodes; n++) {
            if (surface_label[n])
                surface_list.push_back(n);
            else
                inside_list.push_back(n);
            
            my_float3 nrmle = make_my_float3(_Normals[3*n], _Normals[3*n+1], _Normals[3*n+2]);
            float mag = norm(nrmle);
            _Normals[3*n] = _Normals[3*n]/mag;
            _Normals[3*n+1] = _Normals[3*n+1]/mag;
            _Normals[3*n+2] = _Normals[3*n+2]/mag;
            
        }
        
        _Nb_Surface = surface_list.size();
        _Surface = new int[_Nb_Surface];
        memcpy(_Surface, surface_list.data(), _Nb_Surface*sizeof(int));
        surface_list.clear();
        
        _Nb_Inside = inside_list.size();
        _Inside = new int[_Nb_Inside];
        memcpy(_Inside, inside_list.data(), _Nb_Inside*sizeof(int));
        inside_list.clear();
        
        delete[] surface_label;
    }
    
    int *GetSurface(float **vertices, float **normals, int **faces) {
        *vertices = new float[3*_Nb_Surface];
        *normals = new float[3*_Nb_Surface];
        int *index = new int[_Nb_Nodes];
        
        float *Vbuff = *vertices;
        float *Nbuff = *normals;
        
        for (int s = 0; s < _Nb_Nodes; s++) {
            index[s] = -1;
        }
        
        for (int s = 0; s < _Nb_Surface; s++) {
            Vbuff[3*s] = _Nodes[3*_Surface[s]];
            Vbuff[3*s+1] = _Nodes[3*_Surface[s]+1];
            Vbuff[3*s+2] = _Nodes[3*_Surface[s]+2];
            
            Nbuff[3*s] = _Normals[3*_Surface[s]];
            Nbuff[3*s+1] = _Normals[3*_Surface[s]+1];
            Nbuff[3*s+2] = _Normals[3*_Surface[s]+2];
            
            index[_Surface[s]] = s;
        }
        
        *faces = new int[3*_Nb_Faces];
        int *Fbuff = *faces;
        
        for (int f = 0; f < _Nb_Faces; f++) {
            Fbuff[3*f] = index[_Faces[3*f]];
            Fbuff[3*f+1] = index[_Faces[3*f+1]];
            Fbuff[3*f+2] = index[_Faces[3*f+2]];
        }
        
        delete[] index;
        
        int *res = new int[2];
        res[0] = _Nb_Surface;
        res[1] = _Nb_Faces;
        return res;
    };
    
    inline float *LevelSet(float *vertices, int *faces, float *normals_face, int nb_vertices, int nb_faces) {
        /**volume_weights = new float[24*_Nb_Nodes];
        return TetLevelSet(*volume_weights, vertices, skin_weights, faces, normals_face, nb_faces, _Nodes, _Nb_Nodes);*/
        return TetLevelSet_gpu(vertices, faces, normals_face, _Nodes, nb_vertices, nb_faces, _Nb_Nodes);
    }
    
    inline int *MT(float *TSDF, float *volume_weights, float *Vertices, float *skin_weights, int *Faces, float m_iso = 0.0) {
        return MarchingTetrahedra(TSDF, volume_weights, Vertices, skin_weights, Faces, _Nodes, _Tetra, _Edges_row_ptr, _Edges_columns, _Nb_Nodes, _Nb_tetra, _Nb_Edges, m_iso);
        //return MarchingTetrahedraNaive(TSDF, Vertices, Faces, _Nodes, _Tetra, _Nb_Nodes, _Nb_tetra, m_iso);
    }
    
    void FitToTSDF(float ***tsdf_grid, my_int3 dim_grid, my_float3 center, float res, float iso = 0.0f, int outerloop_maxiter = 20, int innerloop_surf_maxiter = 1, int innerloop_inner_maxiter = 10, float alpha = 0.01f, float delta = 0.1f, float delta_elastic = 0.01f) {
        // Compute gradient of tsdf
        my_float3 ***grad_grid = VolumetricGrad(tsdf_grid, dim_grid);
        
        my_float3 *force = new my_float3 [_Nb_Nodes];
        for (int n = 0; n < _Nb_Nodes; n++) {
            force[n] = make_my_float3(0.0f);
        }
        
        // Copy of vertex position
        float *nodes_copy;
        nodes_copy = new float [3*_Nb_Nodes];
        memcpy(nodes_copy, _Nodes, 3*_Nb_Nodes*sizeof(float));
        
        // Prev of vertex position
        float *nodes_prev;
        nodes_prev = new float [3*_Nb_Nodes];
        memcpy(nodes_prev, _Nodes, 3*_Nb_Nodes*sizeof(float));
                
        //############## Outer loop ##################
        float alpha_curr = alpha;
        for (int OuterIter = 0; OuterIter < outerloop_maxiter; OuterIter++) {
            // Copy current state of voxels
            memcpy(nodes_prev, _Nodes, 3*_Nb_Nodes*sizeof(float));
            
            // Increment deplacement of surface vertices
            for (int s = 0; s < _Nb_Surface; s++) {
                my_float3 vertex = make_my_float3(_Nodes[3*_Surface[s]], _Nodes[3*_Surface[s]+1], _Nodes[3*_Surface[s]+2]);
                my_float3 normal = make_my_float3(_Normals[3*_Surface[s]], _Normals[3*_Surface[s]+1], _Normals[3*_Surface[s]+2]);
                my_float3 grad = BicubiInterpolation(vertex, tsdf_grid, grad_grid, dim_grid, center, res, iso);
                
                if (dot(grad, normal) < 0.0f)
                    vertex = vertex + grad * delta - normal * norm(grad) * alpha_curr;
                else
                    vertex = vertex + grad * delta + normal * norm(grad) * alpha_curr;
                _Nodes[3*_Surface[s]] = vertex.x;
                _Nodes[3*_Surface[s]+1] = vertex.y;
                _Nodes[3*_Surface[s]+2] = vertex.z;
            }
            
            for (int inner_iter = 0; inner_iter < innerloop_surf_maxiter; inner_iter++) {
                // Copy current state of voxels
                memcpy(nodes_copy, _Nodes, 3*_Nb_Nodes*sizeof(float));
                for (int s = 0; s < _Nb_Surface; s++) {
                    float x = 0.0f;
                    float y = 0.0f;
                    float z = 0.0f;
                    for (int k = 0; k < 16; k++) {
                        if (_RBF_id[16*_Surface[s] + k] == -1)
                            break;
                        
                        //cout << _RBF_id[16*_Surface[s] + k] << endl;
                        x += nodes_copy[3*_RBF_id[16*_Surface[s] + k]]*_RBF[16*_Surface[s] + k];
                        y += nodes_copy[3*_RBF_id[16*_Surface[s] + k]+1]*_RBF[16*_Surface[s] + k];
                        z += nodes_copy[3*_RBF_id[16*_Surface[s] + k]+2]*_RBF[16*_Surface[s] + k];
                    }
                    
                    if (x != 0.0f && y != 0.0f && z != 0.0f) {
                        _Nodes[3*_Surface[s]] = x;
                        _Nodes[3*_Surface[s]+1] = y;
                        _Nodes[3*_Surface[s]+2] = z;
                    }
                }
            }
                
            for (int inner_iter = 0; inner_iter < innerloop_inner_maxiter; inner_iter++) {
                // Copy current state of voxels
                memcpy(nodes_copy, _Nodes, 3*_Nb_Nodes*sizeof(float));
                
                for (int s = 0; s < _Nb_Inside; s++) {
                    float x = 0.0f;
                    float y = 0.0f;
                    float z = 0.0f;
                    for (int k = 0; k < 16; k++) {
                        if (_RBF_id[16*_Inside[s] + k] == -1)
                            break;
                        
                        //cout << _RBF_id[16*_Surface[s] + k] << endl;
                        x += nodes_copy[3*_RBF_id[16*_Inside[s] + k]]*_RBF[16*_Inside[s] + k];
                        y += nodes_copy[3*_RBF_id[16*_Inside[s] + k]+1]*_RBF[16*_Inside[s] + k];
                        z += nodes_copy[3*_RBF_id[16*_Inside[s] + k]+2]*_RBF[16*_Inside[s] + k];
                    }
                    
                    if (x != 0.0f && y != 0.0f && z != 0.0f) {
                        _Nodes[3*_Inside[s]] = x;
                        _Nodes[3*_Inside[s]+1] = y;
                        _Nodes[3*_Inside[s]+2] = z;
                    }
                }
            }
        }
            
        UpdateNormals(_Nodes, _Normals, _Faces, _Nb_Nodes, _Nb_Faces);
        
        for (int i = 1; i < dim_grid.x-1; i++) {
            for (int j = 1; j < dim_grid.y-1; j++) {
                delete []grad_grid[i][j];
            }
            delete []grad_grid[i];
        }
        delete []grad_grid;
        
        delete []nodes_copy;
        delete []nodes_prev;
        delete []force;
    }
            
    void SaveRBF(string path) {        
        auto file = std::fstream(path, std::ios::out | std::ios::binary);
        
        file.write((char*)_RBF, 20*_Nb_Nodes*sizeof(float));
        file.write((char*)_RBF_id, 20*_Nb_Nodes*sizeof(int));
        
        file.close();
    }
    
    void LoadRBF(string path) {
        _RBF = new float[20*_Nb_Nodes];
        _RBF_id = new int[20*_Nb_Nodes];
        
        auto file = std::fstream(path, std::ios::in | std::ios::binary);
        if (!file.is_open()) {
            cout << "Could not load RBF" << endl;
            file.close();
            return;
        }
        
        file.read((char*)_RBF, 20*_Nb_Nodes*sizeof(float));
        file.read((char*)_RBF_id, 20*_Nb_Nodes*sizeof(int));
        
        for (int i = 0; i < 100; i++) {
            for (int j = 0; j < 20; j++)
                cout << "RBF: " << _RBF[20*i+j] << ", RBF id: " << _RBF_id [20*i+j] << endl;
        }
        
        file.close();
    }
    
    void ComputeRBFSurface() {
        // 1. for each surface node get list of triangles that is attached to the node
        vector<vector<int>> neigh_face;
        vector<vector<int>> neigh_n;
        for (int n = 0; n < _Nb_Nodes; n++) {
            vector<int> tmp;
            tmp.clear();
            neigh_face.push_back(tmp);
            vector<int> tmp2;
            tmp2.clear();
            neigh_n.push_back(tmp2);
        }
        
        for (int f = 0; f < _Nb_Faces; f++) {
            neigh_face[_Faces[3*f]].push_back(f);
            neigh_face[_Faces[3*f+1]].push_back(f);
            neigh_face[_Faces[3*f+2]].push_back(f);
            
            bool valid1 = true;
            bool valid2 = true;
            for (vector<int>::iterator it_n = neigh_n[_Faces[3*f]].begin(); it_n != neigh_n[_Faces[3*f]].end(); it_n++) {
                if (_Faces[3*f+1] == (*it_n))
                    valid1 = false;
                
                if (_Faces[3*f+2] == (*it_n))
                    valid2 = false;
            }
            if (valid1)
                neigh_n[_Faces[3*f]].push_back(_Faces[3*f+1]);
            
            if (valid2)
                neigh_n[_Faces[3*f]].push_back(_Faces[3*f+2]);
            
            valid1 = true;
            valid2 = true;
            for (vector<int>::iterator it_n = neigh_n[_Faces[3*f+1]].begin(); it_n != neigh_n[_Faces[3*f+1]].end(); it_n++) {
                if (_Tetra[3*f] == (*it_n))
                    valid1 = false;
                
                if (_Tetra[3*f+2] == (*it_n))
                    valid2 = false;
            }
            if (valid1)
                neigh_n[_Faces[3*f+1]].push_back(_Faces[3*f]);
            
            if (valid2)
                neigh_n[_Faces[3*f+1]].push_back(_Faces[3*f+2]);
            
            valid1 = true;
            valid2 = true;
            for (vector<int>::iterator it_n = neigh_n[_Faces[3*f+2]].begin(); it_n != neigh_n[_Faces[3*f+2]].end(); it_n++) {
                if (_Faces[3*f+1] == (*it_n))
                    valid1 = false;
                
                if (_Faces[3*f] == (*it_n))
                    valid2 = false;
            }
            if (valid1)
                neigh_n[_Faces[3*f+2]].push_back(_Faces[3*f+1]);
            
            if (valid2)
                neigh_n[_Faces[3*f+2]].push_back(_Faces[3*f]);
        }
        
        // 2. Go through each node
        for (int n = 0; n < _Nb_Surface; n++) {
            if (neigh_n[_Surface[n]].size() > 16)
                cout << "nb tet in vicinity S: " << n << ", " << neigh_n[_Surface[n]].size() << endl;
            
            // compute the total volume of the surrounding tetras.
            float area_tot = 1.0f;
            for (vector<int>::iterator it = neigh_face[_Surface[n]].begin(); it != neigh_face[_Surface[n]].end(); it++) {
                int face = (*it);
                area_tot *= 100.0f*AreaFace(_Nodes, _Faces[3*face], _Faces[3*face+1], _Faces[3*face+2]);
            }
            //cout << "area_tot = " << area_tot << endl;
            
            // for each neighborhing summit
            float tot_weight = 0.0f;
            int count = 0;
            for (vector<int>::iterator it_n = neigh_n[_Surface[n]].begin(); it_n != neigh_n[_Surface[n]].end(); it_n++) {
                float area_cur = area_tot;
                int curr_n = (*it_n);
                int s1 = -1;
                int s2 = -1;
        
                for (vector<int>::iterator it = neigh_face[_Surface[n]].begin(); it != neigh_face[_Surface[n]].end(); it++) {
                    int face = (*it);
                    assert(_Faces[3*face] == _Surface[n] || _Faces[3*face+1] == _Surface[n] || _Faces[3*face+2] == _Surface[n]);
                    if (_Faces[3*face] == curr_n || _Faces[3*face+1] == curr_n || _Faces[3*face+2] == curr_n) {
                        area_cur = area_cur/(100.0f*AreaFace(_Nodes, _Faces[3*face], _Faces[3*face+1], _Faces[3*face+2]));
                        if (s1 == -1) {
                            if (_Faces[3*face] != _Surface[n] && _Faces[3*face] != curr_n) {
                                s1 = _Faces[3*face];
                            } else if (_Faces[3*face+1] != _Surface[n] && _Faces[3*face+1] != curr_n) {
                                s1 = _Faces[3*face+1];
                            } else if (_Faces[3*face+2] != _Surface[n] && _Faces[3*face+2] != curr_n) {
                                s1 = _Faces[3*face+2];
                            }
                        } else {
                            if (_Faces[3*face] != _Surface[n] && _Faces[3*face] != curr_n) {
                                s2 = _Faces[3*face];
                            } else if (_Faces[3*face+1] != _Surface[n] && _Faces[3*face+1] != curr_n) {
                                s2 = _Faces[3*face+1];
                            } else if (_Faces[3*face+2] != _Surface[n] && _Faces[3*face+2] != curr_n) {
                                s2 = _Faces[3*face+2];
                            }
                        }
                    }
                }
                
                if (s1 == -1) {
                    //cout << "s1 == -1" << endl;
                    continue;
                }
                if (s2 == -1) {
                    //cout << "s2 == -1" << endl;
                    continue;
                    cout << _Surface[n] << endl;
                    cout << curr_n << endl;
                    cout << neigh_face[_Surface[n]].size() << endl;
                    for (vector<int>::iterator it = neigh_face[_Surface[n]].begin(); it != neigh_face[_Surface[n]].end(); it++) {
                        int face = (*it);
                        cout << _Faces[3*face] << ", " << _Faces[3*face+1] << ", " << _Faces[3*face+2] << endl;
                    }
                }
                assert(s1 != -1);
                assert(s2 != -1);
                
                _RBF[16*_Surface[n] + count] = 100.0f*AreaFace(_Nodes, s1, curr_n, s2) * area_cur;
                _RBF_id[16*_Surface[n] + count] = curr_n;
                //cout << n << ", " << curr_n << endl;
                tot_weight += _RBF[16*_Surface[n] + count];
                count++;
            }
            
            for (int k = 0; k < 16; k++) {
                if (_RBF_id[16*_Surface[n] + k] == -1)
                    break;
                _RBF[16*_Surface[n] + k] = _RBF[16*_Surface[n] + k]/tot_weight;
                //cout << n << ", " << _RBF[16*_Surface[n] + k] << endl;
            }
        }
        
    }
    
    void ComputeRBFInside() {
        // 1. for each inside node get list of tetrahedra that is attached to the node
        vector<vector<int>> neigh_tet;
        vector<vector<int>> neigh_n;
        for (int n = 0; n < _Nb_Nodes; n++) {
            vector<int> tmp;
            tmp.clear();
            neigh_tet.push_back(tmp);
            vector<int> tmp2;
            tmp2.clear();
            neigh_n.push_back(tmp2);
        }
        
        for (int t = 0; t < _Nb_tetra; t++) {
            neigh_tet[_Tetra[4*t]].push_back(t);
            neigh_tet[_Tetra[4*t+1]].push_back(t);
            neigh_tet[_Tetra[4*t+2]].push_back(t);
            neigh_tet[_Tetra[4*t+3]].push_back(t);
            
            bool valid1 = true;
            bool valid2 = true;
            bool valid3 = true;
            for (vector<int>::iterator it_n = neigh_n[_Tetra[4*t]].begin(); it_n != neigh_n[_Tetra[4*t]].end(); it_n++) {
                if (_Tetra[4*t+1] == (*it_n))
                    valid1 = false;
                
                if (_Tetra[4*t+2] == (*it_n))
                    valid2 = false;
            
                if (_Tetra[4*t+3] == (*it_n))
                    valid3 = false;
            }
            if (valid1)
                neigh_n[_Tetra[4*t]].push_back(_Tetra[4*t+1]);
            
            if (valid2)
                neigh_n[_Tetra[4*t]].push_back(_Tetra[4*t+2]);
            
            if (valid3)
                neigh_n[_Tetra[4*t]].push_back(_Tetra[4*t+3]);
            
            valid1 = true;
            valid2 = true;
            valid3 = true;
            for (vector<int>::iterator it_n = neigh_n[_Tetra[4*t+1]].begin(); it_n != neigh_n[_Tetra[4*t+1]].end(); it_n++) {
                if (_Tetra[4*t] == (*it_n))
                    valid1 = false;
                
                if (_Tetra[4*t+2] == (*it_n))
                    valid2 = false;
            
                if (_Tetra[4*t+3] == (*it_n))
                    valid3 = false;
            }
            if (valid1)
                neigh_n[_Tetra[4*t+1]].push_back(_Tetra[4*t]);
            
            if (valid2)
                neigh_n[_Tetra[4*t+1]].push_back(_Tetra[4*t+2]);
            
            if (valid3)
                neigh_n[_Tetra[4*t+1]].push_back(_Tetra[4*t+3]);
            
            valid1 = true;
            valid2 = true;
            valid3 = true;
            for (vector<int>::iterator it_n = neigh_n[_Tetra[4*t+2]].begin(); it_n != neigh_n[_Tetra[4*t+2]].end(); it_n++) {
                if (_Tetra[4*t+1] == (*it_n))
                    valid1 = false;
                
                if (_Tetra[4*t] == (*it_n))
                    valid2 = false;
            
                if (_Tetra[4*t+3] == (*it_n))
                    valid3 = false;
            }
            if (valid1)
                neigh_n[_Tetra[4*t+2]].push_back(_Tetra[4*t+1]);
            
            if (valid2)
                neigh_n[_Tetra[4*t+2]].push_back(_Tetra[4*t]);
            
            if (valid3)
                neigh_n[_Tetra[4*t+2]].push_back(_Tetra[4*t+3]);
            
            valid1 = true;
            valid2 = true;
            valid3 = true;
            for (vector<int>::iterator it_n = neigh_n[_Tetra[4*t+3]].begin(); it_n != neigh_n[_Tetra[4*t+3]].end(); it_n++) {
                if (_Tetra[4*t+1] == (*it_n))
                    valid1 = false;
                
                if (_Tetra[4*t+2] == (*it_n))
                    valid2 = false;
            
                if (_Tetra[4*t] == (*it_n))
                    valid3 = false;
            }
            if (valid1)
                neigh_n[_Tetra[4*t+3]].push_back(_Tetra[4*t+1]);
            
            if (valid2)
                neigh_n[_Tetra[4*t+3]].push_back(_Tetra[4*t+2]);
            
            if (valid3)
                neigh_n[_Tetra[4*t+3]].push_back(_Tetra[4*t]);
            
        }
        
        // 2. Go through each node
        for (int n = 0; n < _Nb_Inside; n++) {
            if (neigh_n[_Inside[n]].size() > 16)
                cout << "nb tet in vicinity: " << n << ", " << neigh_n[_Inside[n]].size() << endl;
            
            // Identify 8 neighbors that are at sqrt(res) distance
            
            my_float3 pt = make_my_float3(_Nodes[3*_Inside[n]], _Nodes[3*_Inside[n]+1], _Nodes[3*_Inside[n]+2]);
            int count = 0;
            for (vector<int>::iterator it_n = neigh_n[_Inside[n]].begin(); it_n != neigh_n[_Inside[n]].end(); it_n++) {
                my_float3 pt_n = make_my_float3(_Nodes[3*(*it_n)], _Nodes[3*(*it_n)+1], _Nodes[3*(*it_n)+2]);
                
                float dist = norm(pt - pt_n);
                if (fabs(dist - sqrt(3.0f*_res*_res)/2.0f) < 1.0e-5) {
                    _RBF[16*_Inside[n] + count] = 1.0f;
                    _RBF_id[16*_Inside[n] + count] = (*it_n);
                    count++;
                }
            }
            //cout << "nb nei: " << count << endl;
            
            for (int i = 0; i < 16; i++) {
                if (_RBF_id[16*_Inside[n] + i] != -1) {
                    _RBF[16*_Inside[n] + i] = 1.0f/float(count);
                }
            }
            
            /*
            
            // compute the total volume of the surrounding tetras.
            float vol_tot = 0.0f;
            for (vector<int>::iterator it = neigh_tet[_Inside[n]].begin(); it != neigh_tet[_Inside[n]].end(); it++) {
                int tet = (*it);
                vol_tot += VolumeTetra(_Nodes, _Tetra, tet);
            }
            
            // for each neighborhing summit
            // remove the volume of each tetra that contains the summit
            int count = 0;
            for (vector<int>::iterator it_n = neigh_n[_Inside[n]].begin(); it_n != neigh_n[_Inside[n]].end(); it_n++) {
                float vol_cur = vol_tot;
                int curr_n = (*it_n);
        
                for (vector<int>::iterator it = neigh_tet[_Inside[n]].begin(); it != neigh_tet[_Inside[n]].end(); it++) {
                    int tet = (*it);
                    if (_Tetra[4*tet] == curr_n || _Tetra[4*tet+1] == curr_n || _Tetra[4*tet+2] == curr_n || _Tetra[4*tet+3] == curr_n)
                        vol_cur -= VolumeTetra(_Nodes, _Tetra, tet);
                }
                _RBF[16*_Nb_Surface + 16*_Inside[n] + count] = vol_cur/vol_tot;
                _RBF_id[16*_Nb_Surface + 16*_Inside[n] + count] = curr_n;
                count++;
            }*/
        }
    }
    
    void ComputeAdjacencies(string savedir, int *rates, int adjnodes) {
        cout << "#########Create adjlists#########" << endl;
                
        // Reduce vertices for full connection

        float *coarse1 = new float[3*_Nb_Nodes/rates[0]];
        reduce_points(coarse1, _Nodes, _Nb_Nodes, rates[0]);
        int nb_coarse1 = _Nb_Nodes/rates[0];
        SavePCasPLY(savedir + "/coarse1.ply", coarse1, nb_coarse1);
        float *coarse2 = new float[3*nb_coarse1/rates[1]];
        reduce_points(coarse2, coarse1, nb_coarse1, rates[1]);
        int nb_coarse2 = nb_coarse1/rates[1];
        SavePCasPLY(savedir + "/coarse2.ply", coarse2, nb_coarse2);
        float *coarse3 = new float[3*nb_coarse2/rates[2]];
        reduce_points(coarse3, coarse2, nb_coarse2, rates[2]);
        int nb_coarse3 = nb_coarse2/rates[2];
        SavePCasPLY(savedir + "/coarse3.ply", coarse3, nb_coarse3);
        float *coarse4 = new float[3*nb_coarse3/rates[3]];
        reduce_points(coarse4, coarse3, nb_coarse3, rates[3]);
        int nb_coarse4 = nb_coarse3/rates[3];
        SavePCasPLY(savedir + "/coarse4.ply", coarse4, nb_coarse4);
        
        cout << "Generate node adj list" << endl;

        // Note: utils.make_adjlist returns a list of n nearest nodes (L2 norm)
        cout << "Node reduction rates: " << rates[0] << ", " << rates[1] << ", " << rates[2] << ", " << rates[3] << endl;
        cout << "Adjacent Nodes: " << adjnodes << endl;
        
        // For partial connection
        cout << "coarse 4 to 3: " << nb_coarse4 << " -> " << nb_coarse3 << endl;
        vector<vector<int>> adjlist4 = make_adjlist(coarse4, nb_coarse4, coarse3, nb_coarse3, adjnodes, true);
        list2csv(savedir + "/adjlist_4to3.csv", adjlist4);
        
        cout << "coarse 3 to 2: " << nb_coarse3 << " -> " << nb_coarse2 << endl;
        vector<vector<int>> adjlist3 = make_adjlist(coarse3, nb_coarse3, coarse2, nb_coarse2, adjnodes, true);
        list2csv(savedir + "/adjlist_3to2.csv", adjlist3);
        
        
        cout << "coarse 2 to 1: " << nb_coarse2 << " -> " << nb_coarse1 << endl;
        vector<vector<int>> adjlist2 = make_adjlist(coarse2, nb_coarse2, coarse1, nb_coarse1, adjnodes, true);
        list2csv(savedir + "/adjlist_2to1.csv", adjlist2);
        
        
        cout << "coarse 1 to original: " << nb_coarse1 << " -> " << _Nb_Nodes << endl;
        vector<vector<int>> adjlist1 = make_adjlist(coarse1, nb_coarse1, _Nodes, _Nb_Nodes, adjnodes, true);
        list2csv(savedir + "/adjlist_1to0.csv", adjlist1);
        
        delete[] coarse1;
        delete[] coarse2;
        delete[] coarse3;
        delete[] coarse4;
    }
    
    void ComputeWeights(string path) {
        // Load SMPL model
        float *vertices;
        float *normals_face;
        int *faces;
        int *res = LoadPLY_Mesh(path+string("Template.ply"), &vertices, &normals_face, &faces);
        cout << "The PLY file has " << res[0] << " vertices and " << res[1] << " faces" << endl;
        
        // Load skinning weights
        float *skin_weights = new float[24*res[0]];
        
        ifstream SWfile (path+"weights_template.bin", ios::binary);
        if (!SWfile.is_open()) {
            cout << "Could not load skin weights" << endl;
            SWfile.close();
            return;
        }

        SWfile.read((char *) skin_weights, 24*res[0]*sizeof(float));
        SWfile.close();
        
        _SkinWeights = new float[24*_Nb_Nodes];
        // Compute the signed distance to the mesh for each voxel
        std::vector< std::thread > my_threads;
        for (int i = 0; i < 101; i++) {
            my_threads.push_back( std::thread(SkinWeightsInnerLoop, _SkinWeights, i*(_Nb_Nodes/100), vertices, skin_weights, faces, normals_face, _Nb_Nodes, res[1], _Nodes) );
        }
        std::for_each(my_threads.begin(), my_threads.end(), std::mem_fn(&std::thread::join));
        
        delete[] vertices;
        delete[] normals_face;
        delete[] faces;
        delete[] skin_weights;
    }
    
    void Skin(DualQuaternion *Skeleton) {
        for (int n = 0; n < _Nb_Nodes; n++) {
            // Blend the transformations with iterative algorithm
            DualQuaternion Transfo = DualQuaternion(Quaternion(0.0,0.0,0.0,0.0), Quaternion(0.0,0.0,0.0,0.0));
            for (int j = 0; j < 24; j++) {
                if (_SkinWeights[24*n+j] > 0.0) {
                    Transfo = Transfo + (Skeleton[j] * _SkinWeights[24*n+j]);
                }
            }
                    
            Transfo = Transfo.Normalize();
            DualQuaternion point = DualQuaternion(Quaternion(0.0,0.0,0.0,1.0), Quaternion(_Nodes[3*n], _Nodes[3*n+1], _Nodes[3*n+2], 0.0f));
            point  = Transfo * point * Transfo.DualConjugate2();

            my_float3 vtx = point.Dual().Vector();
            _Nodes[3*n] = vtx.x;
            _Nodes[3*n+1] = vtx.y;
            _Nodes[3*n+2] = vtx.z;
        }
        
    }
        
};

int *TetraFromGrid(float ***sdf, float **nodes_out, int **tetra_out, float iso, my_int3 size_grid, my_float3 center_grid, float res) {
        
    // Create the list of regular tetrahedral grid with negative levelset
    vector<my_int4> Tetra;
    vector<my_float3> Nodes;
    vector<int> Index;
    
    for (int i = 0; i < size_grid.x-2; i++) {
        cout << 100.0f*float(i)/float(size_grid.x) << "%\r";
        for (int j = 0; j < size_grid.y-2; j++) {
            for (int k = 0; k < size_grid.z-2; k++) {
                // create the six tetrahedra inside the cube {left, right, front, back, up, down}
                float sdf_center = (sdf[i][j][k] + sdf[i+1][j][k] + sdf[i+1][j+1][k] + sdf[i+1][j+1][k+1] +
                                    sdf[i][j+1][k] + sdf[i][j+1][k+1] + sdf[i][j][k+1] + sdf[i+1][j][k+1])/8.0f;
                
                if (sdf_center > iso)
                    continue;
                
                float sdf_center_1 = (sdf[i][j+1][k] + sdf[i+1][j+1][k] + sdf[i+1][j+2][k] + sdf[i+1][j+2][k+1] +
                                      sdf[i][j+2][k] + sdf[i][j+2][k+1] + sdf[i][j+1][k+1] + sdf[i+1][j+1][k+1])/8.0f;
                
                float sdf_center_2 = (sdf[i+1][j][k] + sdf[i+2][j][k] + sdf[i+2][j+1][k] + sdf[i+2][j+1][k+1] +
                                      sdf[i+1][j+1][k] + sdf[i+1][j+1][k+1] + sdf[i+1][j][k+1] + sdf[i+2][j][k+1])/8.0f;
                
                float sdf_center_3 = (sdf[i][j][k+1] + sdf[i+1][j][k+1] + sdf[i+1][j+1][k+1] + sdf[i+1][j+1][k+2] +
                                      sdf[i][j+1][k+1] + sdf[i][j+1][k+2] + sdf[i][j][k+2] + sdf[i+1][j][k+2])/8.0f;
                                
                int idx_c0 = Nodes.size();
                Nodes.push_back(make_my_float3(center_grid.x + (float(i) + 0.5f - float(size_grid.x)/2.0f)*res,
                                               center_grid.y + (float(j) + 0.5f - float(size_grid.y)/2.0f)*res,
                                               center_grid.z + (float(k) + 0.5f - float(size_grid.z)/2.0f)*res));
                Index.push_back(-1);
                                
                int idx_c1, idx_c2, idx_c3;
                int idx_s0, idx_s1, idx_s2, idx_s3, idx_s4, idx_s5, idx_s6, idx_s7;
                
                
                if (sdf_center_1 <= iso) {
                    idx_c1 = Nodes.size();
                    Nodes.push_back(make_my_float3(center_grid.x+ (float(i) + 0.5f - float(size_grid.x)/2.0f)*res,
                                                   center_grid.y+ (float(j) + 1.5f - float(size_grid.y)/2.0f)*res,
                                                   center_grid.z+ (float(k) + 0.5f - float(size_grid.z)/2.0f)*res));
                    Index.push_back(-1);
                }
                if (sdf_center_2 <= iso) {
                    idx_c2 = Nodes.size();
                    Nodes.push_back(make_my_float3(center_grid.x+ (float(i) + 1.5f - float(size_grid.x)/2.0f)*res,
                                                   center_grid.y+ (float(j) + 0.5f - float(size_grid.y)/2.0f)*res,
                                                   center_grid.z+ (float(k) + 0.5f - float(size_grid.z)/2.0f)*res));
                    Index.push_back(-1);
                }
                if (sdf_center_3 <= iso) {
                    idx_c3 = Nodes.size();
                    Nodes.push_back(make_my_float3(center_grid.x+ (float(i) + 0.5f - float(size_grid.x)/2.0f)*res,
                                                   center_grid.y+ (float(j) + 0.5f - float(size_grid.y)/2.0f)*res,
                                                   center_grid.z+ (float(k) + 1.5f - float(size_grid.z)/2.0f)*res));
                    Index.push_back(-1);
                }
                
                if (sdf[i+1][j][k] <= iso) {
                    idx_s1 = Nodes.size();
                    Nodes.push_back(make_my_float3(center_grid.x+ (float(i+1) - float(size_grid.x)/2.0f)*res,
                                                   center_grid.y+ (float(j) - float(size_grid.y)/2.0f)*res,
                                                   center_grid.z+ (float(k) - float(size_grid.z)/2.0f)*res));
                    Index.push_back(-1);
                }
                
                if (sdf[i+1][j+1][k] <= iso) {
                    idx_s2 = Nodes.size();
                    Nodes.push_back(make_my_float3(center_grid.x+ (float(i+1) - float(size_grid.x)/2.0f)*res,
                                                   center_grid.y+ (float(j+1) - float(size_grid.y)/2.0f)*res,
                                                   center_grid.z+ (float(k) - float(size_grid.z)/2.0f)*res));
                    Index.push_back(-1);
                }
                
                if (sdf[i][j+1][k] <= iso) {
                    idx_s3 = Nodes.size();
                    Nodes.push_back(make_my_float3(center_grid.x+ (float(i) - float(size_grid.x)/2.0f)*res,
                                                   center_grid.y+ (float(j+1) - float(size_grid.y)/2.0f)*res,
                                                   center_grid.z+ (float(k) - float(size_grid.z)/2.0f)*res));
                    Index.push_back(-1);
                }
                
                if (sdf[i][j][k+1] <= iso) {
                    idx_s4 = Nodes.size();
                    Nodes.push_back(make_my_float3(center_grid.x+ (float(i) - float(size_grid.x)/2.0f)*res,
                                                   center_grid.y+ (float(j) - float(size_grid.y)/2.0f)*res,
                                                   center_grid.z+ (float(k+1) - float(size_grid.z)/2.0f)*res));
                    Index.push_back(-1);
                }
                
                if (sdf[i+1][j][k+1] <= iso) {
                    idx_s5 = Nodes.size();
                    Nodes.push_back(make_my_float3(center_grid.x+ (float(i+1) - float(size_grid.x)/2.0f)*res,
                                                   center_grid.y+ (float(j) - float(size_grid.y)/2.0f)*res,
                                                   center_grid.z+ (float(k+1) - float(size_grid.z)/2.0f)*res));
                    Index.push_back(-1);
                }
                
                if (sdf[i+1][j+1][k+1] <= iso) {
                    idx_s6 = Nodes.size();
                    Nodes.push_back(make_my_float3(center_grid.x+ (float(i+1) - float(size_grid.x)/2.0f)*res,
                                                   center_grid.y+ (float(j+1) - float(size_grid.y)/2.0f)*res,
                                                   center_grid.z+ (float(k+1) - float(size_grid.z)/2.0f)*res));
                    Index.push_back(-1);
                }
                
                if (sdf[i][j+1][k+1] <= iso) {
                    idx_s7 = Nodes.size();
                    Nodes.push_back(make_my_float3(center_grid.x+ (float(i) - float(size_grid.x)/2.0f)*res,
                                                   center_grid.y+ (float(j+1) - float(size_grid.y)/2.0f)*res,
                                                   center_grid.z+ (float(k+1) - float(size_grid.z)/2.0f)*res));
                    Index.push_back(-1);
                }
                
                
                /// Turn around first edge (= 4 tetrahedra with same s3, s4)
                // first tetrahedron
                if (sdf[i][j+1][k+1] <= iso && sdf[i+1][j+1][k+1] <= iso &&
                    sdf_center_1 <= iso) {
                    Tetra.push_back(make_my_int4(idx_s7, idx_s6, idx_c0, idx_c1));
                }
            
                // second tetrahedron                
                if (sdf[i+1][j+1][k] <= iso && sdf[i+1][j+1][k+1] <= iso &&
                    sdf_center_1 <= iso) {
                    Tetra.push_back(make_my_int4(idx_s6, idx_s2, idx_c0, idx_c1));
                }

                // third tetrahedron
                if (sdf[i][j+1][k] <= iso && sdf[i+1][j+1][k] <= iso &&
                    sdf_center_1 <= iso) {
                    Tetra.push_back(make_my_int4(idx_s2, idx_s3, idx_c0, idx_c1));
                }

                // fourth tetrahedron
                if (sdf[i][j+1][k] <= iso && sdf[i][j+1][k+1] <= iso &&
                    sdf_center_1 <= iso) {
                    
                    Tetra.push_back(make_my_int4(idx_s3, idx_s7, idx_c0, idx_c1));
                }
                            
                ///#################Turn around second edge (= 4 tetrahedra with same s3, s4)
                // fifth tetrahedron
                if (sdf[i+1][j][k+1] <= iso && sdf[i+1][j+1][k+1] <= iso &&
                    sdf_center_2 <= iso) {
                    Tetra.push_back(make_my_int4(idx_s6, idx_s5, idx_c0, idx_c2));
                }
                
                // sixth tetrahedron
                if (sdf[i+1][j][k] <= iso && sdf[i+1][j][k+1] <= iso &&
                    sdf_center_2 <= iso) {
                    
                    Tetra.push_back(make_my_int4(idx_s5, idx_s1, idx_c0, idx_c2));
                }
                
                // seventh tetrahedron
                if (sdf[i+1][j][k] <= iso && sdf[i+1][j+1][k] <= iso &&
                    sdf_center_2 <= iso) {
                    
                    Tetra.push_back(make_my_int4(idx_s1, idx_s2, idx_c0, idx_c2));
                }
                    
                // eighth tetrahedron
                if (sdf[i+1][j+1][k] <= iso && sdf[i+1][j+1][k+1] <= iso &&
                    sdf_center_2 <= iso) {
                    Tetra.push_back(make_my_int4(idx_s2, idx_s6, idx_c0, idx_c2));
                }
                
                //###############// Turn around third edge (= 4 tetrahedra with same s3, s4)
                // nineth tetrahedron
                if (sdf[i+1][j][k+1] <= iso && sdf[i+1][j+1][k+1] <= iso &&
                    sdf_center_3 <= iso) {
                    Tetra.push_back(make_my_int4(idx_s5, idx_s6, idx_c0, idx_c3));
                }                
                
                // tenth tetrahedron
                if (sdf[i][j+1][k+1] <= iso && sdf[i+1][j+1][k+1] <= iso &&
                    sdf_center_3 <= iso) {
                    int curr_count = Nodes.size();
                    
                    Tetra.push_back(make_my_int4(idx_s6, idx_s7, idx_c0, idx_c3));
                }
                
                
                // eleventh tetrahedron
                if (sdf[i][j][k+1] <= iso && sdf[i][j+1][k+1] <= iso &&
                    sdf_center_3 <= iso) {
                    
                    Tetra.push_back(make_my_int4(idx_s7, idx_s4, idx_c0, idx_c3));
                }
                    
                
                // twelvth tetrahedron
                if (sdf[i][j][k+1] <= iso && sdf[i+1][j][k+1] <= iso &&
                    sdf_center_3 <= iso) {
                    Tetra.push_back(make_my_int4(idx_s4, idx_s5, idx_c0, idx_c3));
                }
            }
        }
    }
    
    cout << "TetraHedra created" << endl;
       
    // Merge nodes
    vector<float> merged_nodes;
    int count_curr = 0;
    int count_merge = 0;
    for (vector<my_float3>::iterator it = Nodes.begin(); it != Nodes.end(); it++) {
        my_float3 curr_pt = (*it);
        if (Index[count_curr] != -1) {
            count_curr++;
            continue;
        }
        
        merged_nodes.push_back(curr_pt.x);
        merged_nodes.push_back(curr_pt.y);
        merged_nodes.push_back(curr_pt.z);
        Index[count_curr] = count_merge;
        
        int count = 0;
        for (vector<my_float3>::iterator it2 = Nodes.begin(); it2 != Nodes.end(); it2++) {
            my_float3 next_pt = (*it2);
            if (Index[count] == -1) {
                if (curr_pt.x == next_pt.x && curr_pt.y == next_pt.y && curr_pt.z == next_pt.z) {
                    Index[count] = count_merge;
                }
            }
            count++;
        }
        count_merge++;
        count_curr++;
    }
    cout << "nodes merged" << endl;
        
    *nodes_out = new float[merged_nodes.size()];
    memcpy(*nodes_out, merged_nodes.data(), merged_nodes.size()*sizeof(float));
    
    // Reorganize tetrahedra
    *tetra_out = new int[4*Tetra.size()];
    int count = 0;
    for (vector<my_int4>::iterator it = Tetra.begin(); it != Tetra.end(); it++) {
        my_int4 curr_tetra = (*it);
        
        (*tetra_out)[4*count] = Index[curr_tetra.x];
        (*tetra_out)[4*count+1] = Index[curr_tetra.y];
        (*tetra_out)[4*count+2] = Index[curr_tetra.z];
        (*tetra_out)[4*count+3] = Index[curr_tetra.w];
        
        count++;
    }
        
    int *dim = new int[3];
    dim[0] = merged_nodes.size()/3;
    dim[1] = Tetra.size();
    return dim;
}

int *SurfaceFromTetraGrid(float ***sdf, float **surface_out, int **faces_out, float iso, my_int3 size_grid, my_float3 center_grid, float res) {
        
    // Create the list of regular tetrahedral grid
    vector<my_int4> Tetra;
    vector<my_float3> Nodes;
    vector<int> Index;
    vector<float> SDFList;
    
    for (int i = 0; i < size_grid.x-2; i++) {
        cout << 100.0f*float(i)/float(size_grid.x) << "%\r";
        for (int j = 0; j < size_grid.y-2; j++) {
            for (int k = 0; k < size_grid.z-2; k++) {
                // create the six tetrahedra inside the cube {left, right, front, back, up, down}
                float sdf_center = (sdf[i][j][k] + sdf[i+1][j][k] + sdf[i+1][j+1][k] + sdf[i+1][j+1][k+1] +
                                    sdf[i][j+1][k] + sdf[i][j+1][k+1] + sdf[i][j][k+1] + sdf[i+1][j][k+1])/8.0f;
                    
                if (sdf_center > 2.0f*res)
                    continue;
                
                float sdf_center_1 = (sdf[i][j+1][k] + sdf[i+1][j+1][k] + sdf[i+1][j+2][k] + sdf[i+1][j+2][k+1] +
                                      sdf[i][j+2][k] + sdf[i][j+2][k+1] + sdf[i][j+1][k+1] + sdf[i+1][j+1][k+1])/8.0f;
                
                float sdf_center_2 = (sdf[i+1][j][k] + sdf[i+2][j][k] + sdf[i+2][j+1][k] + sdf[i+2][j+1][k+1] +
                                      sdf[i+1][j+1][k] + sdf[i+1][j+1][k+1] + sdf[i+1][j][k+1] + sdf[i+2][j][k+1])/8.0f;
                
                float sdf_center_3 = (sdf[i][j][k+1] + sdf[i+1][j][k+1] + sdf[i+1][j+1][k+1] + sdf[i+1][j+1][k+2] +
                                      sdf[i][j+1][k+1] + sdf[i][j+1][k+2] + sdf[i][j][k+2] + sdf[i+1][j][k+2])/8.0f;
                                
                int idx_c0 = Nodes.size();
                Nodes.push_back(make_my_float3(center_grid.x + (float(i) + 0.5f - float(size_grid.x)/2.0f)*res,
                                               center_grid.y + (float(j) + 0.5f - float(size_grid.y)/2.0f)*res,
                                               center_grid.z + (float(k) + 0.5f - float(size_grid.z)/2.0f)*res));
                Index.push_back(-1);
                if (sdf_center > 0.0f)
                    SDFList.push_back(1.0e32);
                else
                    SDFList.push_back(-1.0e-6);
                
                int idx_c1, idx_c2, idx_c3;
                int idx_s0, idx_s1, idx_s2, idx_s3, idx_s4, idx_s5, idx_s6, idx_s7;
                
                idx_c1 = Nodes.size();
                Nodes.push_back(make_my_float3(center_grid.x+ (float(i) + 0.5f - float(size_grid.x)/2.0f)*res,
                                               center_grid.y+ (float(j) + 1.5f - float(size_grid.y)/2.0f)*res,
                                               center_grid.z+ (float(k) + 0.5f - float(size_grid.z)/2.0f)*res));
                Index.push_back(-1);
                if (sdf_center_1 > 0.0f)
                    SDFList.push_back(1.0e32);
                else
                    SDFList.push_back(-1.0e-6);
                                
                idx_c2 = Nodes.size();
                Nodes.push_back(make_my_float3(center_grid.x+ (float(i) + 1.5f - float(size_grid.x)/2.0f)*res,
                                               center_grid.y+ (float(j) + 0.5f - float(size_grid.y)/2.0f)*res,
                                               center_grid.z+ (float(k) + 0.5f - float(size_grid.z)/2.0f)*res));
                Index.push_back(-1);
                
                if (sdf_center_2 > 0.0f)
                    SDFList.push_back(1.0e32);
                else
                    SDFList.push_back(-1.0e-6);
                
                idx_c3 = Nodes.size();
                Nodes.push_back(make_my_float3(center_grid.x+ (float(i) + 0.5f - float(size_grid.x)/2.0f)*res,
                                               center_grid.y+ (float(j) + 0.5f - float(size_grid.y)/2.0f)*res,
                                               center_grid.z+ (float(k) + 1.5f - float(size_grid.z)/2.0f)*res));
                Index.push_back(-1);
                if (sdf_center_3 > 0.0f)
                    SDFList.push_back(1.0e32);
                else
                    SDFList.push_back(-1.0e-6);
                
                idx_s1 = Nodes.size();
                Nodes.push_back(make_my_float3(center_grid.x+ (float(i+1) - float(size_grid.x)/2.0f)*res,
                                               center_grid.y+ (float(j) - float(size_grid.y)/2.0f)*res,
                                               center_grid.z+ (float(k) - float(size_grid.z)/2.0f)*res));
                Index.push_back(-1);
                if (sdf[i+1][j][k] > 0.0f)
                    SDFList.push_back(1.0e32);
                else
                    SDFList.push_back(-1.0e-6);
                
                idx_s2 = Nodes.size();
                Nodes.push_back(make_my_float3(center_grid.x+ (float(i+1) - float(size_grid.x)/2.0f)*res,
                                               center_grid.y+ (float(j+1) - float(size_grid.y)/2.0f)*res,
                                               center_grid.z+ (float(k) - float(size_grid.z)/2.0f)*res));
                Index.push_back(-1);
                if (sdf[i+1][j+1][k] > 0.0f)
                    SDFList.push_back(1.0e32);
                else
                    SDFList.push_back(-1.0e-6);
                
                idx_s3 = Nodes.size();
                Nodes.push_back(make_my_float3(center_grid.x+ (float(i) - float(size_grid.x)/2.0f)*res,
                                               center_grid.y+ (float(j+1) - float(size_grid.y)/2.0f)*res,
                                               center_grid.z+ (float(k) - float(size_grid.z)/2.0f)*res));
                Index.push_back(-1);
                if (sdf[i][j+1][k] > 0.0f)
                    SDFList.push_back(1.0e32);
                else
                    SDFList.push_back(-1.0e-6);
                
                idx_s4 = Nodes.size();
                Nodes.push_back(make_my_float3(center_grid.x+ (float(i) - float(size_grid.x)/2.0f)*res,
                                               center_grid.y+ (float(j) - float(size_grid.y)/2.0f)*res,
                                               center_grid.z+ (float(k+1) - float(size_grid.z)/2.0f)*res));
                Index.push_back(-1);
                if (sdf[i][j][k+1] > 0.0f)
                    SDFList.push_back(1.0e32);
                else
                    SDFList.push_back(-1.0e-6);
                
                idx_s5 = Nodes.size();
                Nodes.push_back(make_my_float3(center_grid.x+ (float(i+1) - float(size_grid.x)/2.0f)*res,
                                               center_grid.y+ (float(j) - float(size_grid.y)/2.0f)*res,
                                               center_grid.z+ (float(k+1) - float(size_grid.z)/2.0f)*res));
                Index.push_back(-1);
                if (sdf[i+1][j][k+1] > 0.0f)
                    SDFList.push_back(1.0e32);
                else
                    SDFList.push_back(-1.0e-6);
                
                idx_s6 = Nodes.size();
                Nodes.push_back(make_my_float3(center_grid.x+ (float(i+1) - float(size_grid.x)/2.0f)*res,
                                               center_grid.y+ (float(j+1) - float(size_grid.y)/2.0f)*res,
                                               center_grid.z+ (float(k+1) - float(size_grid.z)/2.0f)*res));
                Index.push_back(-1);
                if (sdf[i+1][j+1][k+1] > 0.0f)
                    SDFList.push_back(float(1.0e32));
                else
                    SDFList.push_back(float(-1.0e-6));
                
                idx_s7 = Nodes.size();
                Nodes.push_back(make_my_float3(center_grid.x+ (float(i) - float(size_grid.x)/2.0f)*res,
                                               center_grid.y+ (float(j+1) - float(size_grid.y)/2.0f)*res,
                                               center_grid.z+ (float(k+1) - float(size_grid.z)/2.0f)*res));
                Index.push_back(-1);
                if (sdf[i][j+1][k+1] > 0.0f)
                    SDFList.push_back(1.0e32);
                else
                    SDFList.push_back(-1.0e-6);
                
                
                /// Turn around first edge (= 4 tetrahedra with same s3, s4)
                // first tetrahedron
                Tetra.push_back(make_my_int4(idx_s7, idx_s6, idx_c0, idx_c1));
            
                // second tetrahedron
                Tetra.push_back(make_my_int4(idx_s6, idx_s2, idx_c0, idx_c1));
                
                // third tetrahedron
                Tetra.push_back(make_my_int4(idx_s2, idx_s3, idx_c0, idx_c1));

                // fourth tetrahedron
                Tetra.push_back(make_my_int4(idx_s3, idx_s7, idx_c0, idx_c1));
                            
                ///#################Turn around second edge (= 4 tetrahedra with same s3, s4)
                // fifth tetrahedron
                Tetra.push_back(make_my_int4(idx_s6, idx_s5, idx_c0, idx_c2));
                
                // sixth tetrahedron
                Tetra.push_back(make_my_int4(idx_s5, idx_s1, idx_c0, idx_c2));
                
                // seventh tetrahedron
                Tetra.push_back(make_my_int4(idx_s1, idx_s2, idx_c0, idx_c2));
                    
                // eighth tetrahedron
                Tetra.push_back(make_my_int4(idx_s2, idx_s6, idx_c0, idx_c2));
                
                //###############// Turn around third edge (= 4 tetrahedra with same s3, s4)
                // nineth tetrahedron
                Tetra.push_back(make_my_int4(idx_s5, idx_s6, idx_c0, idx_c3));
                
                // tenth tetrahedron
                Tetra.push_back(make_my_int4(idx_s6, idx_s7, idx_c0, idx_c3));
                
                // eleventh tetrahedron
                Tetra.push_back(make_my_int4(idx_s7, idx_s4, idx_c0, idx_c3));
                
                // twelvth tetrahedron
                Tetra.push_back(make_my_int4(idx_s4, idx_s5, idx_c0, idx_c3));
            }
        }
    }
    
    cout << "TetraHedra created: " << Nodes.size() << " nodes and " << Tetra.size() << " Tetras" << endl;
            
    // Merge nodes
    vector<float> merged_nodes;
    vector<float> merged_sdf;
    int count_curr = 0;
    int count_merge = 0;
    for (vector<my_float3>::iterator it = Nodes.begin(); it != Nodes.end(); it++) {
        if (count_curr % 1000 == 0 )
            cout << 100.0f*float(count_curr)/float(Nodes.size()) << endl;
        
        my_float3 curr_pt = (*it);
        if (Index[count_curr] != -1) {
            count_curr++;
            continue;
        }
        
        merged_nodes.push_back(curr_pt.x);
        merged_nodes.push_back(curr_pt.y);
        merged_nodes.push_back(curr_pt.z);
        merged_sdf.push_back(SDFList[count_curr]);
        Index[count_curr] = count_merge;
                
        int count = count_curr;
        for (vector<my_float3>::iterator it2 = it/*Nodes.begin()*/; it2 != Nodes.end(); it2++) {
            my_float3 next_pt = (*it2);
            if (Index[count] == -1) {
                if (curr_pt.x == next_pt.x && curr_pt.y == next_pt.y && curr_pt.z == next_pt.z) {
                    Index[count] = count_merge;
                }
            }
            count++;
        }
        count_merge++;
        count_curr++;
    }
    cout << "nodes merged: " << merged_nodes.size() << endl;
        
    float *nodes = new float[merged_nodes.size()];
    memcpy(nodes, merged_nodes.data(), merged_nodes.size()*sizeof(float));
    
    // Reorganize tetrahedra
    int *tetra = new int[4*Tetra.size()];
    int count = 0;
    for (vector<my_int4>::iterator it = Tetra.begin(); it != Tetra.end(); it++) {
        my_int4 curr_tetra = (*it);
        
        tetra[4*count] = Index[curr_tetra.x];
        tetra[4*count+1] = Index[curr_tetra.y];
        tetra[4*count+2] = Index[curr_tetra.z];
        tetra[4*count+3] = Index[curr_tetra.w];
        
        count++;
    }
        
    //############################## Compute surface indexes and faces ######################################
    
    // Create List of edges
    int Nb_Nodes = merged_nodes.size()/3;
    Eigen::SparseMatrix<int, Eigen::RowMajor> Edges(Nb_Nodes, Nb_Nodes);
    std::vector<Eigen::Triplet<int>> coefficientsE;
    // go through all voxels
    int vtx_id = 0;
    for (int vox_id = 0; vox_id < Tetra.size(); vox_id++) {
        // get the edges
        for (int sum_1 = 0; sum_1 < 4; sum_1++) {
            for (int sum_2 = sum_1 + 1; sum_2 < 4; sum_2++) {
                if (tetra[4 * vox_id + sum_1] < tetra[4 * vox_id + sum_2])
                    coefficientsE.push_back(Eigen::Triplet<int>(tetra[4*vox_id + sum_1], tetra[4 * vox_id + sum_2], tetra[4 * vox_id + sum_1]));
                else
                    coefficientsE.push_back(Eigen::Triplet<int>(tetra[4 * vox_id + sum_2], tetra[4 * vox_id + sum_1], tetra[4 * vox_id + sum_2]));
                vtx_id++;
            }
        }
    }
    Edges.setFromTriplets(coefficientsE.begin(), coefficientsE.end(), [](const int&, const int& b) { return b; });

    int Nb_Edges = Edges.nonZeros();
    int num_rows = Edges.outerSize();
    int num_cols = Edges.innerSize();
    cout << num_rows << endl;
    cout << num_cols << endl;
    cout << Nb_Edges << ", " << vtx_id << endl;
    
    int *Edges_row_ptr = new int[num_rows+1];
    memcpy((void*)Edges_row_ptr, Edges.outerIndexPtr(), ((long long)num_rows + 1) * sizeof(int));
    int *Edges_columns = new int[Nb_Edges];
    memcpy((void*)Edges_columns, Edges.innerIndexPtr(), Nb_Edges * sizeof(int));
    
    // Run tetraMesh on the merged sdf to get surface vertices and faces.
    float *sdf_data = new float[merged_sdf.size()];
    memcpy(sdf_data, merged_sdf.data(), merged_sdf.size()*sizeof(float));
    float *buff_skinweights = new float[24*merged_sdf.size()];
    
    *surface_out = new float[3*Nb_Edges];
    float *SkinWeightsTet = new float[24*Nb_Edges];
    *faces_out = new int[3*3*Tetra.size()];
    
    int *res_surface = MarchingTetrahedra(sdf_data, buff_skinweights, (*surface_out), SkinWeightsTet, (*faces_out), nodes, tetra, Edges_row_ptr, Edges_columns, Nb_Nodes, Tetra.size(), Nb_Edges);
    
    delete[] Edges_row_ptr;
    delete[] Edges_columns;
    delete[] sdf_data;
    delete[] buff_skinweights;
    delete[] SkinWeightsTet;
    
    int *dim = new int[3];
    dim[0] = res_surface[0];
    dim[1] = res_surface[1];
    return dim;
}

int ReorganizeSurface(float *vertices_surf, int *faces, float *nodes, int nb_surface, int nb_faces, int Nb_Nodes) {
    //############################## Compute surface indexes and faces ######################################
    int *SurfaceIdx = new int[nb_surface];
    for (int v = 0; v < nb_surface; v++) {
        my_float3 pt = make_my_float3(vertices_surf[3*v], vertices_surf[3*v+1], vertices_surf[3*v+2]);
        
        SurfaceIdx[v] = -1;
        // find corresponding index in the list of nodes
        for (int curr_v = 0; curr_v < Nb_Nodes; curr_v++) {
            my_float3 curr_pt = make_my_float3(nodes[3*curr_v], nodes[3*curr_v+1], nodes[3*curr_v+2]);
            if (norm(pt - curr_pt) < 1.0e-5) {
                SurfaceIdx[v] = curr_v;
                break;
            }
        }
    }
    
    // Reassign faces
    vector<int> merged_faces;
    for (int f = 0; f < nb_faces; f++) {
        if ((SurfaceIdx[faces[3*f]] != SurfaceIdx[faces[3*f+1]] && SurfaceIdx[faces[3*f]] != SurfaceIdx[faces[3*f+2]] && SurfaceIdx[faces[3*f+1]] != SurfaceIdx[faces[3*f+2]])) {
            merged_faces.push_back(SurfaceIdx[faces[3*f]]);
            merged_faces.push_back(SurfaceIdx[faces[3*f+1]]);
            merged_faces.push_back(SurfaceIdx[faces[3*f+2]]);
        }
    }
    memcpy(faces, merged_faces.data(), merged_faces.size()*sizeof(int));
    int res = merged_faces.size()/3;
    merged_faces.clear();
    
    delete[] SurfaceIdx;
    return res;
}

void LinkSurfaceToTetra(int **surface_edges, float *vertices, float *nodes, int nb_vertices, int nb_nodes) {
    *surface_edges = new int[2*nb_vertices];
    
    for (int v = 0; v < nb_vertices; v++) {
        my_float3 curr_v = make_my_float3(vertices[3*v], vertices[3*v+1], vertices[3*v+2]);
        
        float min_dist = float(1.0e32);
        int min_n = 0;
        for (int n = 0; n < nb_nodes; n++) {
            my_float3 curr_n = make_my_float3(nodes[3*n], nodes[3*n+1], nodes[3*n+2]);
            
            float dist = norm(curr_v - curr_n);
            if (dist < min_dist) {
                min_dist = dist;
                min_n = n;
            }
        }
        
        (*surface_edges)[2*v] = v;
        (*surface_edges)[2*v+1] = min_n;
    }
}


#endif /* TetraMesh_h */
