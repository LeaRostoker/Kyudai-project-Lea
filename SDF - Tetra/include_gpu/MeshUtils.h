//
//  MeshUtils.h
//  DEEPANIM
//
//  Created by Diego Thomas on 2021/01/18.
//  Updated by Lea Rostoker on 2022/02/17
//

#ifndef MeshUtils_h
#define MeshUtils_h

#include "include_gpu/Utilities.h"

void SaveMeshToPLY(string path, float* Vertices, float* Colors, float* Normals, int* Faces, int nbPoints, int nbFaces) {
    ofstream plyfile(path);
    cout << "Saving the mesh to ply" << endl;
    if (plyfile.is_open()) {
        plyfile << "ply" << endl;
        plyfile << "format ascii 1.0" << endl;
        plyfile << "comment ply file created by Diego Thomas" << endl;
        plyfile << "element vertex " << nbPoints << endl;
        plyfile << "property float x" << endl;
        plyfile << "property float y" << endl;
        plyfile << "property float z" << endl;
        plyfile << "property float nx" << endl;
        plyfile << "property float ny" << endl;
        plyfile << "property float nz" << endl;
        plyfile << "property uchar red" << endl;
        plyfile << "property uchar green" << endl;
        plyfile << "property uchar blue" << endl;
        plyfile << "element face " << nbFaces << endl;
        plyfile << "property list uchar int vertex_index" << endl;
        plyfile << "end_header" << endl;

        for (int i = 0; i < nbPoints; i++) {
            
            plyfile << Vertices[3 * i] << " " << Vertices[3 * i + 1] << " " << Vertices[3 * i + 2] << " " <<
                Normals[3 * i] << " " << Normals[3 * i + 1] << " " << Normals[3 * i + 2] << " " <<
                100 << " " << 100 << " " << 100 << endl;
        }
        for (int f = 0; f < nbFaces; f++) {
            plyfile << 3 << " " << Faces[3 * f] << " " << Faces[3 * f + 1] << " " << Faces[3 * f + 2] << endl;
        }
    }

    plyfile.close();
}

void SaveLevelSet(string path, float ***tsdf, my_int3 dim) {     
    auto file = std::fstream(path, std::ios::out | std::ios::binary);
    int size = dim.x*dim.y*dim.z;
    
    for (int i = 0; i < dim.x; i++) {
        for (int j = 0; j < dim.y; j++) {
            file.write((char*)tsdf[i][j], dim.z*sizeof(float));
        }
    }
    
    file.close();
}

void SaveLevelSet(string path, int*** tsdf, my_int3 dim) {
    auto file = std::fstream(path, std::ios::out | std::ios::binary);
    int size = dim.x * dim.y * dim.z;

    for (int i = 0; i < dim.x; i++) {
        for (int j = 0; j < dim.y; j++) {            
            /*for (int k = 0; k < dim.y; k++)
            {
                std::cout <<tsdf[i][j][k] << endl;
            }*/
            file.write((char*)tsdf[i][j], dim.z * sizeof(int));
        }
    }

    file.close();
}

int* LoadPLY_TetraMesh(string path, float** Vertices, float** NormalsF, float** NormalsT, int** Faces, int** Tetras) { //KEEP
    int* result = new int[3];
    result[0] = -1; result[1] = -1; result[2] = -1;
    cout << "hello load tetmesh " << endl;
    string line;
    ifstream objfile(path, ios::binary);

    vector<float> vertices;
    vector<int> faces;
    vector<int> tetras; 

    if (objfile.is_open()) {

        getline(objfile, line);
        getline(objfile, line);
        getline(objfile, line);
        getline(objfile, line);

        string word;
        string* words = new string[3];
        std::istringstream iss(line);
        int i = 0;
        while (iss >> word) {
            words[i] = word;
            cout << words[i] << endl;
            i++;
        }

        result[0] = std::stoi(words[2]);

        getline(objfile, line);
        getline(objfile, line);
        getline(objfile, line);
        getline(objfile, line);

        string word2;
        string* words2 = new string[3];
        std::istringstream iss2(line);
        int j = 0;
        while (iss2 >> word2) {
            words2[j] = word2;
            cout << words2[j] << endl;
            j++;
        }

        result[1] = std::stoi(words2[2]);

        getline(objfile, line);
        getline(objfile, line);

        string word3;
        string* words3 = new string[3];
        std::istringstream iss3(line);
        int k = 0;
        while (iss3 >> word3) {
            words3[k] = word3;
            cout << words3[k] << endl;
            k++;
        }

        result[2] = std::stoi(words3[2]);

        getline(objfile, line); 
        getline(objfile, line); 
        //end of header


        for (int i = 0; i < result[0]; i++) {
            getline(objfile, line);
            string word;
            string* words = new string[3];
            std::istringstream iss(line);
            int k = 0;
            while (iss >> word) {
                words[k] = word;
                k++;

            }
            
            /*vertices.push_back(std::stof(words[0]));
            vertices.push_back(std::stof(words[2]));
            float z = std::stof(words[1]);
            vertices.push_back(z);*/
            vertices.push_back(std::stof(words[0]));
            vertices.push_back(std::stof(words[1]));
            float z = std::stof(words[2]);
            vertices.push_back(z);
        }

        for (int i = 0; i < result[1]; i++) {
            getline(objfile, line);
            string word;
            string* words = new string[4];
            std::istringstream iss(line);
            int k = 0;
            while (iss >> word) {
                words[k] = word;
                k++;
            }
            faces.push_back(std::stoi(words[1]));
            faces.push_back(std::stoi(words[2]));
            faces.push_back(std::stoi(words[3]));
        }

        for (int i = 0; i < result[2]; i++) {
            getline(objfile, line);
            string word;
            string* words = new string[5];
            std::istringstream iss(line);
            int k = 0;
            while (iss >> word) {
                words[k] = word;
                k++;
            }
            tetras.push_back(std::stoi(words[1]));
            tetras.push_back(std::stoi(words[2]));
            tetras.push_back(std::stoi(words[3]));
            tetras.push_back(std::stoi(words[4]));
        }


        cout << line << endl;

    }

    objfile.close();

    //Allocate data

    *Vertices = new float[3*result[0]]();
    float* Buff_vertices = *Vertices;
    memcpy(Buff_vertices, vertices.data(), 3 * result[0] * sizeof(float));

    *NormalsF = new float[3 * result[1]]();
    float* Buff_normalsF = *NormalsF;

    *NormalsT = new float[3*4*result[2]]();
    float* Buff_normalsT = *NormalsT;

    *Faces = new int[3*result[1]]();
    int* Buff_faces = *Faces;
    memcpy(Buff_faces, faces.data(), 3 * result[1] * sizeof(int));
   
    *Tetras = new int[4*result[2]]();
    int* Buff_tetras = *Tetras;
    memcpy(Buff_tetras, tetras.data(), 4 * result[2] * sizeof(int));


    // Compute normals
       
    for (int f = 0; f < result[1]; f++) {
        my_float3 p1f = make_my_float3(Buff_vertices[3 * Buff_faces[3 * f + 0]], Buff_vertices[3 * Buff_faces[3 * f + 0] + 1], Buff_vertices[3 * Buff_faces[3 * f + 0] + 2]);
        my_float3 p2f = make_my_float3(Buff_vertices[3 * Buff_faces[3 * f + 1]], Buff_vertices[3 * Buff_faces[3 * f + 1] + 1], Buff_vertices[3 * Buff_faces[3 * f + 1] + 2]);
        my_float3 p3f = make_my_float3(Buff_vertices[3 * Buff_faces[3 * f + 2]], Buff_vertices[3 * Buff_faces[3 * f + 2] + 1], Buff_vertices[3 * Buff_faces[3 * f + 2] + 2]);
       
        my_float3 normal = cross(p3f - p1f, p2f - p1f);
        float norm = sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
        
        Buff_normalsF[3 * f] = normal.x / norm;
        Buff_normalsF[3 * f + 1] = normal.y / norm;
        Buff_normalsF[3 * f + 2] = normal.z / norm;
    }
    

    // Compute normal and check they are pointing outside of the tetrahedras
   
    for (int t = 0; t < result[2]; t++) {
        my_float3 p0 = make_my_float3(Buff_vertices[3 * Buff_tetras[4 * t + 0] + 0], Buff_vertices[3 * Buff_tetras[4 * t + 0] + 1], Buff_vertices[3 * Buff_tetras[4 * t + 0] + 2]);
        my_float3 p1 = make_my_float3(Buff_vertices[3 * Buff_tetras[4 * t + 1] + 0], Buff_vertices[3 * Buff_tetras[4 * t + 1] + 1], Buff_vertices[3 * Buff_tetras[4 * t + 1] + 2]);
        my_float3 p2 = make_my_float3(Buff_vertices[3 * Buff_tetras[4 * t + 2] + 0], Buff_vertices[3 * Buff_tetras[4 * t + 2] + 1], Buff_vertices[3 * Buff_tetras[4 * t + 2] + 2]);
        my_float3 p3 = make_my_float3(Buff_vertices[3 * Buff_tetras[4 * t + 3] + 0], Buff_vertices[3 * Buff_tetras[4 * t + 3] + 1], Buff_vertices[3 * Buff_tetras[4 * t + 3] + 2]);


        int dir = 1;

        //FACE 012
        my_float3 normal_012 = cross(p1 - p0, p2 - p0);
        float norm_012 = sqrt(normal_012.x * normal_012.x + normal_012.y * normal_012.y + normal_012.z * normal_012.z);

            //dot with vector of remaining tetra point
        my_float3 p3p0 = p0 - p3;
        float dot_012 = dot(p3p0, normal_012);

        if (dot_012 <= 1e-6) {
            dir = -1;
        }

        Buff_normalsT[3 * 4 * t + 0] = dir * normal_012.x / norm_012;
        Buff_normalsT[3 * 4 * t + 1] = dir * normal_012.y / norm_012;
        Buff_normalsT[3 * 4 * t + 2] = dir * normal_012.z / norm_012;

        //FACE 013
        dir = 1;
        my_float3 normal_013 = cross(p1 - p0, p3 - p0);
        float norm_013 = sqrt(normal_013.x * normal_013.x + normal_013.y * normal_013.y + normal_013.z * normal_013.z);

            //dot with vector of remaining tetra point
        my_float3 p2p0 = p0 - p2;
        float dot_013 = dot(p2p0, normal_013);
        
        if (dot_013 <= 1e-6) {
            dir = -1;
        }

        Buff_normalsT[3 * 4 * t + 3] = dir * normal_013.x / norm_013;
        Buff_normalsT[3 * 4 * t + 4] = dir * normal_013.y / norm_013;
        Buff_normalsT[3 * 4 * t + 5] = dir * normal_013.z / norm_013;

        //FACE 023
        dir = 1;
        my_float3 normal_023 = cross(p2 - p0, p3 - p0);
        float norm_023 = sqrt(normal_023.x * normal_023.x + normal_023.y * normal_023.y + normal_023.z * normal_023.z);

            //dot with vector of remaining tetra point
        my_float3 p1p0 = p0 - p1;
        float dot_023 = dot(p1p0, normal_023);
        
        if (dot_023 <= 1e-6) {
            dir = -1;
        }

        Buff_normalsT[3 * 4 * t + 6] = dir * normal_023.x / norm_023;
        Buff_normalsT[3 * 4 * t + 7] = dir * normal_023.y / norm_023;
        Buff_normalsT[3 * 4 * t + 8] = dir * normal_023.z / norm_023;

        //FACE 123
        dir = 1;
        my_float3 normal_123 = cross(p2 - p1, p3 - p1);
        float norm_123 = sqrt(normal_123.x * normal_123.x + normal_123.y * normal_123.y + normal_123.z * normal_123.z);

            //dot with vector of remaining tetra point
        my_float3 p0p1 = p1 - p0;
        float dot_10 = dot(p0p1, normal_123);
        if (dot_10 <= 1e-6) {
            dir = -1;
        }

        Buff_normalsT[3 * 4 * t +  9] = dir * normal_123.x / norm_123;
        Buff_normalsT[3 * 4 * t + 10] = dir * normal_123.y / norm_123;
        Buff_normalsT[3 * 4 * t + 11] = dir * normal_123.z / norm_123;

    }
    return result;
}

#endif /* MeshUtils_h */
