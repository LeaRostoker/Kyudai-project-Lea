//
//  MeshUtils.h
//  DEEPANIM
//
//  Created by Diego Thomas on 2021/01/18.
//

#ifndef MeshUtils_h
#define MeshUtils_h

#include "include_gpu/Utilities.h"

void SaveColoredPC(string path, float *PC, float* Colors, int nbPoints){
    ofstream pcfile (path);
    if (pcfile.is_open()){
        for(int i=0; i< nbPoints; i++){
            pcfile << PC[3*i] << " " << PC[3*i+1] << " " << PC[3*i+2] <<" " << Colors[3*i] << " " << Colors[3*i+1] << " " << Colors[3*i+2]  << endl;
        }
        pcfile.close();
    }
}

void SavePCasPLY(string path, float *PC, int nbPoints) {
    ofstream plyfile (path);
    if (plyfile.is_open()) {
        plyfile << "ply" << endl;
        plyfile << "format ascii 1.0" << endl;
        plyfile << "comment ply file created by Diego Thomas" << endl;
        plyfile << "element vertex " << nbPoints << endl;
        plyfile << "property float x" << endl;
        plyfile << "property float y" << endl;
        plyfile << "property float z" << endl;
        plyfile << "end_header" << endl;
        //property float nx\nproperty float ny\nproperty float nz\nelement face \(nbFaces) \nproperty list uchar int vertex_index \nend_header\n" << endl;

        for (int i = 0; i < nbPoints; i++) {
            plyfile << PC[3*i] << " " << PC[3*i+1] << " " << PC[3*i+2] << endl;
        }
    }

      plyfile.close();
}

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
        //property float nx\nproperty float ny\nproperty float nz\nelement face \(nbFaces) \nproperty list uchar int vertex_index \nend_header\n" << endl;

        for (int i = 0; i < nbPoints; i++) {
            /*plyfile << Vertices[3 * i] << " " << Vertices[3 * i + 1] << " " << Vertices[3 * i + 2] << " " <<
                Normals[3 * i] << " " << Normals[3 * i + 1] << " " << Normals[3 * i + 2] <<  " " <<
                unsigned char(Colors[3 * i] * 255.0f) << " " << unsigned char(Colors[3 * i + 1] * 255.0f) << " " << unsigned char(Colors[3 * i + 2] * 255.0f) << endl;*/

            plyfile << Vertices[3 * i] << " " << Vertices[3 * i + 1] << " " << Vertices[3 * i + 2] << " " <<
                Normals[3 * i] << " " << Normals[3 * i + 1] << " " << Normals[3 * i + 2] << " " <<
                100 << " " << 100 << " " << 100 << endl;

           /* std::cout << Vertices[3 * i] << " " << Vertices[3 * i + 1] << " " << Vertices[3 * i + 2] << " " <<
                Normals[3 * i] << " " << Normals[3 * i + 1] << " " << Normals[3 * i + 2] << " " <<
                unsigned char(Colors[3 * i] * 255.0f) << " " << unsigned char(Colors[3 * i + 1] * 255.0f) << " " << unsigned char(Colors[3 * i + 2] * 255.0f) << endl;
            int tmp;
            std::cin >> tmp;*/
        }
        //cout << "points written" << endl;
        for (int f = 0; f < nbFaces; f++) {
            plyfile << 3 << " " << Faces[3 * f] << " " << Faces[3 * f + 1] << " " << Faces[3 * f + 2] << endl;
        }
        //cout << "faces written" << endl;
    }

    plyfile.close();
}

void SaveMeshToPLY(string path, float *Vertices, float *Normals, int *Faces, int nbPoints, int nbFaces) {
    ofstream plyfile (path);
    cout<<"Saving the mesh to ply"<<endl;
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
        plyfile << "element face " << nbFaces << endl;
        plyfile << "property list uchar int vertex_index" << endl;
        plyfile << "end_header" << endl;
        //property float nx\nproperty float ny\nproperty float nz\nelement face \(nbFaces) \nproperty list uchar int vertex_index \nend_header\n" << endl;

        for (int i = 0; i < nbPoints; i++) {
            plyfile << Vertices[3*i] << " " << Vertices[3*i+1] << " " << Vertices[3*i+2] << " " << Normals[3*i] << " " << Normals[3*i+1] << " " << Normals[3*i+2] << endl;
        }
        //cout << "points written" << endl;
        for (int f = 0; f < nbFaces; f++) {
            plyfile << 3 << " " << Faces[3*f] << " " << Faces[3*f+1] << " " << Faces[3*f+2] << endl;
        }
        //cout << "faces written" << endl;
    }

      plyfile.close();
}

void SaveMeshToPLY(string path, float *Vertices, int *Faces, int nbPoints, int nbFaces) {
    ofstream plyfile (path);
    cout<<"Saving the mesh to ply"<<endl;
    if (plyfile.is_open()) {
        plyfile << "ply" << endl;
        plyfile << "format ascii 1.0" << endl;
        plyfile << "comment ply file created by Diego Thomas" << endl;
        plyfile << "element vertex " << nbPoints << endl;
        plyfile << "property float x" << endl;
        plyfile << "property float y" << endl;
        plyfile << "property float z" << endl;
        plyfile << "element face " << nbFaces << endl;
        plyfile << "property list uchar int vertex_index" << endl;
        plyfile << "end_header" << endl;
        //property float nx\nproperty float ny\nproperty float nz\nelement face \(nbFaces) \nproperty list uchar int vertex_index \nend_header\n" << endl;

        for (int i = 0; i < nbPoints; i++) {
            plyfile << Vertices[3*i] << " " << Vertices[3*i+1] << " " << Vertices[3*i+2] << endl;
        }
        //cout << "points written" << endl;
        for (int f = 0; f < nbFaces; f++) {
            plyfile << 3 << " " << Faces[3*f] << " " << Faces[3*f+1] << " " << Faces[3*f+2] << endl;
        }
        //cout << "faces written" << endl;
    }

      plyfile.close();
}

void SaveTetraMeshToPLY(string path, float *Nodes, float *Normals, int *Faces, int *Tetra, int nbNodes, int nbTetra, int nbFaces) {
    ofstream plyfile (path);
    cout<<"Saving the mesh to ply"<<endl;
    if (plyfile.is_open()) {
        plyfile << "ply" << endl;
        plyfile << "format ascii 1.0" << endl;
        plyfile << "comment ply file created by Diego Thomas" << endl;
        plyfile << "element vertex " << nbNodes << endl;
        plyfile << "property float x" << endl;
        plyfile << "property float y" << endl;
        plyfile << "property float z" << endl;
        plyfile << "property float nx" << endl;
        plyfile << "property float ny" << endl;
        plyfile << "property float nz" << endl;
        plyfile << "element face " << nbFaces << endl;
        plyfile << "property list uchar int vertex_index" << endl;
        /*plyfile << "element face " << 4*nbTetra << endl;
        plyfile << "property list uchar int vertex_index" << endl;*/
        plyfile << "element voxel " << nbTetra << endl;
        plyfile << "property list uchar int vertex_index" << endl;
        plyfile << "end_header" << endl;
        //property float nx\nproperty float ny\nproperty float nz\nelement face \(nbFaces) \nproperty list uchar int vertex_index \nend_header\n" << endl;

        for (int i = 0; i < nbNodes; i++) {
            plyfile << Nodes[3*i] << " " << Nodes[3*i+1] << " " << Nodes[3*i+2] << " " << Normals[3*i] << " " << Normals[3*i+1] << " " << Normals[3*i+2] << endl;
        }
        //cout << "points written" << endl;
        
        for (int f = 0; f < nbFaces; f++) {
            plyfile << 3 << " " << Faces[3*f] << " " << Faces[3*f+1] << " " << Faces[3*f+2] << endl;
        }
        //cout << "faces written" << endl;
        
        for (int f = 0; f < nbTetra; f++) {
            plyfile << 4 << " " << Tetra[4*f] << " " << Tetra[4*f+1] << " " << Tetra[4*f+2] << " " << Tetra[4*f+3] << endl;
        }
        //cout << "voxels written" << endl;
    }

      plyfile.close();
}

void SaveTetraAndMeshToPLY(string path, float *Vertices, float *Normals, int *Faces, float *Nodes, int *Surface, int *Tetra, int nbNodes, int nbSurface, int nbTetra, int nbPoints, int nbFaces) {
    ofstream plyfile (path);
    cout<<"Saving the mesh to ply"<<endl;
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
        plyfile << "element face " << nbFaces << endl;
        plyfile << "property list uchar int vertex_index" << endl;
        /*plyfile << "element face " << 4*nbTetra << endl;
        plyfile << "property list uchar int vertex_index" << endl;*/
        plyfile << "element nodes " << nbNodes << endl;
        plyfile << "property float x" << endl;
        plyfile << "property float y" << endl;
        plyfile << "property float z" << endl;
        plyfile << "element voxel " << nbTetra << endl;
        plyfile << "property list uchar int vertex_index" << endl;
        plyfile << "element surface " << nbSurface << endl;
        plyfile << "property list uchar int vertex_index" << endl;
        plyfile << "end_header" << endl;
        //property float nx\nproperty float ny\nproperty float nz\nelement face \(nbFaces) \nproperty list uchar int vertex_index \nend_header\n" << endl;

        for (int i = 0; i < nbPoints; i++) {
            plyfile << Vertices[3*i] << " " << Vertices[3*i+1] << " " << Vertices[3*i+2] << " " << Normals[3*i] << " " << Normals[3*i+1] << " " << Normals[3*i+2] << endl;
        }
        //cout << "points written" << endl;
        
        for (int f = 0; f < nbFaces; f++) {
            plyfile << 3 << " " << Faces[3*f] << " " << Faces[3*f+1] << " " << Faces[3*f+2] << endl;
        }
        //cout << "faces written" << endl;
                
        for (int i = 0; i < nbNodes; i++) {
            plyfile << Nodes[3*i] << " " << Nodes[3*i+1] << " " << Nodes[3*i+2] << endl;
        }
        //cout << "nodes written" << endl;
        
        /*for (int v = 0; v < nbTetra; v++) {
            plyfile << 3 << " " << Tetra[4*v] << " " << Tetra[4*v+1] << " " << Tetra[4*v+2] << endl;
            plyfile << 3 << " " << Tetra[4*v] << " " << Tetra[4*v+1] << " " << Tetra[4*v+3] << endl;
            plyfile << 3 << " " << Tetra[4*v] << " " << Tetra[4*v+2] << " " << Tetra[4*v+3] << endl;
            plyfile << 3 << " " << Tetra[4*v+1] << " " << Tetra[4*v+2] << " " << Tetra[4*v+3] << endl;
        }
        cout << "faces written" << endl;*/
        
        for (int f = 0; f < nbTetra; f++) {
            plyfile << 4 << " " << Tetra[4*f] << " " << Tetra[4*f+1] << " " << Tetra[4*f+2] << " " << Tetra[4*f+3] << endl;
        }
        //cout << "voxels written" << endl;
        
        for (int s = 0; s < nbSurface; s++) {
            plyfile << 2 << " " << Surface[2*s] << " " << Surface[2*s+1] << endl;
        }
        //cout << "surface written" << endl;
    }

      plyfile.close();
}


void SaveTetraSurfaceToPLY(string path, float *Vertices, int *Surface, int nbSurface) {
    ofstream plyfile (path);
    cout<<"Saving the mesh to ply"<<endl;
    if (plyfile.is_open()) {
        plyfile << "ply" << endl;
        plyfile << "format ascii 1.0" << endl;
        plyfile << "comment ply file created by Diego Thomas" << endl;
        plyfile << "element vertex " << nbSurface << endl;
        plyfile << "property float x" << endl;
        plyfile << "property float y" << endl;
        plyfile << "property float z" << endl;
        plyfile << "end_header" << endl;

        for (int i = 0; i < nbSurface; i++) {
            plyfile << Vertices[3*Surface[i]] << " " << Vertices[3*Surface[i]+1] << " " << Vertices[3*Surface[i]+2] << endl;
        }
        //cout << "points written" << endl;
    }

    plyfile.close();
}

void SaveCubicMeshToPLY(string path, my_float3 ***Voxels, my_int3 dim) {
    ofstream plyfile (path);
    cout<<"Saving the mesh to ply"<<endl;
    if (plyfile.is_open()) {
        plyfile << "ply" << endl;
        plyfile << "format ascii 1.0" << endl;
        plyfile << "comment ply file created by Diego Thomas" << endl;
        plyfile << "element vertex " << dim.x*dim.y*6 << endl;
        plyfile << "property float x" << endl;
        plyfile << "property float y" << endl;
        plyfile << "property float z" << endl;
        plyfile << "end_header" << endl;
        
        //face 1
        for (int i = 0; i < dim.x; i++) {
            for (int j = 0; j < dim.y; j++) {
                plyfile << Voxels[i][j][0].x << " " << Voxels[i][j][0].y << " " << Voxels[i][j][0].z << endl;
            }
        }
        
        //face 2
        for (int i = 0; i < dim.x; i++) {
            for (int j = 0; j < dim.y; j++) {
                plyfile << Voxels[i][j][dim.z-1].x << " " << Voxels[i][j][dim.z-1].y << " " << Voxels[i][j][dim.z-1].z << endl;
            }
        }
        
        //face 3
        for (int i = 0; i < dim.x; i++) {
            for (int k = 0; k < dim.z; k++) {
                plyfile << Voxels[i][0][k].x << " " << Voxels[i][0][k].y << " " << Voxels[i][0][k].z << endl;
            }
        }
        
        //face 4
        for (int i = 0; i < dim.x; i++) {
            for (int k = 0; k < dim.z; k++) {
                plyfile << Voxels[i][dim.y-1][k].x << " " << Voxels[i][dim.y-1][k].y << " " << Voxels[i][dim.y-1][k].z << endl;
            }
        }
        
        //face 5
        for (int j = 0; j < dim.y; j++) {
            for (int k = 0; k < dim.z; k++) {
                plyfile << Voxels[0][j][k].x << " " << Voxels[0][j][k].y << " " << Voxels[0][j][k].z << endl;
            }
        }
        
        //face 6
        for (int j = 0; j < dim.y; j++) {
            for (int k = 0; k < dim.z; k++) {
                plyfile << Voxels[dim.x-1][j][k].x << " " << Voxels[dim.x-1][j][k].y << " " << Voxels[dim.x-1][j][k].z << endl;
            }
        }
        
        
        /*for (int i = 1; i < dim.x-1; i++) {
            for (int j = 1; j < dim.y-1; j++) {
                for (int k = 1; k < dim.z-1; k++) {
                    plyfile << Voxels[i][j][k].x << " " << Voxels[i][j][k].y << " " << Voxels[i][j][k].z << endl;
                }
            }
        }*/

       //cout << "points written" << endl; 
    }

    plyfile.close();
}

void SaveLevelSet(string path, float ***tsdf, my_int3 dim) {
    
    auto file = std::fstream(path, std::ios::out | std::ios::binary);
    int size = dim.x*dim.y*dim.z;
    //file.write((char*)&size, sizeof(int));
    
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
    //file.write((char*)&size, sizeof(int));

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

void SaveLevelSet(string path, float *tsdf, int nb_nodes) {
    std::cout << "nb nodes: " << nb_nodes << endl;
    auto file = std::fstream(path, std::ios::out | std::ios::binary);
    file.write((char*)&nb_nodes, sizeof(int));
    file.write((char*)tsdf, nb_nodes*sizeof(float));
    file.close();
    
}

float *LoadTetLevelSet(string path, int nb_nodes) {
    
    auto file = std::fstream(path, std::ios::in | std::ios::binary);
    
    int size_sdf;
    file.read((char*)&size_sdf, sizeof(int));
    cout << "tsdf sie: " << size_sdf << endl;
    
    float *tsdf = new float[nb_nodes];
    file.read((char*)tsdf, nb_nodes*sizeof(float));
    file.close();
        
    return tsdf;
}

float ***LoadLevelSet(string path, my_int3 dim) {
    
    auto file = std::fstream(path, std::ios::in | std::ios::binary);
    
    int size_sdf;
    file.read((char*)&size_sdf, sizeof(int));
    
    float ***tsdf = new float **[dim.x];
    for (int i = 0; i < dim.x; i++) {
        tsdf[i] = new float *[dim.y];
        for (int j = 0; j < dim.y; j++) {
            tsdf[i][j] = new float[dim.z];
            //file.seekg((i*dim.y + j)*dim.z*sizeof(float));
            file.read((char*)tsdf[i][j], dim.z*sizeof(float));
        }
    }
    
    file.close();
    
    
    return tsdf;
}


int LoadPLY(string path, float **Vertices) {
      string line;
    ifstream plyfile (path);
    if (plyfile.is_open()) {

        // read Header
        getline (plyfile,line); // PLY
        getline (plyfile,line); // format ascii 1.0
        getline (plyfile,line); // comment ply file
        getline (plyfile,line); // element vertex ??

        // get number of vertices
        std::istringstream iss(line);
        std::vector<std::string> words((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
        int nb_vertices = std::stoi(words[2]);
        cout << "number of vertices: " << nb_vertices << endl;

        while (line != string("end_header")) {
            getline (plyfile,line);
        }

        /*getline (plyfile,line); // property float x
        getline (plyfile,line); // property float y
        getline (plyfile,line); // property float z
        getline (plyfile,line); // property float nx
        getline (plyfile,line); // property float ny
        getline (plyfile,line); // property float nz
        getline (plyfile,line); // element face 0
        getline (plyfile,line); // property list uchar int vertex_indices
        getline (plyfile,line); // end_header*/

        // Allocate data to store vertices
        *Vertices = new float[3*nb_vertices]();
        float *Buff = *Vertices;
        //int i = 0;

        for (int i = 0; i < nb_vertices; i++)
          //while ( getline (plyfile,line) )
        {
            getline (plyfile,line);
            std::istringstream iss(line);
            std::vector<std::string> words((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
            Buff[3*i] = std::stof(words[0]);
            Buff[3*i+1] = std::stof(words[1]);
            Buff[3*i+2] = std::stof(words[2]);
            //i += 3;
        }

          plyfile.close();
        return nb_vertices;
    }

      plyfile.close();
    cout << "could not load file: " << path << endl;
    return -1;
}

// The normals correspond to the normals of the faces
int *LoadPLY_Mesh(string path, float **Vertices, float **Normals, int **Faces){
    int *result = new int[2];
    result[0] = -1; result[1] = -1;

    string line;
    ifstream plyfile (path, ios::binary);
    if (plyfile.is_open()) {
        //Read header
        getline (plyfile,line); // PLY
        //cout << line << endl;
        std::istringstream iss(line);
        std::vector<std::string> words((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
        
        while (words[0].compare(string("end_header")) != 0) {
            if (words[0].compare(string("element")) == 0) {
                if (words[1].compare(string("vertex")) == 0) {
                    result[0] = std::stoi(words[2]);
                } else if (words[1].compare(string("face")) == 0) {
                    result[1] = std::stoi(words[2]);
                }
            }
            
            getline (plyfile,line); // PLY
            //cout << line << endl;
            iss = std::istringstream(line);
            words.clear();
            words = std::vector<std::string>((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
        }
        
        // Allocate data
        *Vertices = new float[3*result[0]]();
        float *Buff_vtx = *Vertices;
        
        *Normals = new float[3*result[1]]();
        float *Buff_nmls = *Normals;
        
        *Faces = new int[3*result[1]]();
        int *Buff_faces = *Faces;

        float min_x = 1.0e32;
        float min_y = 1.0e32;
        float min_z = 1.0e32;
        float max_x = 0.0f;
        float max_y = 0.0f;
        float max_z = 0.0f;
        for (int i = 0; i < result[0]; i++) {
            getline (plyfile,line); // PLY
            std::istringstream iss(line);
            std::vector<std::string> words((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
            Buff_vtx[3*i] = std::stof(words[0]);
            Buff_vtx[3*i+1] = std::stof(words[1]);
            Buff_vtx[3*i+2] = std::stof(words[2]);
            //cout << Buff_vtx[3*i] << ", " << Buff_vtx[3*i+1] << ", " << Buff_vtx[3*i+2] << endl;
            if (Buff_vtx[3 * i] < min_x)
                min_x = Buff_vtx[3 * i];
            if (Buff_vtx[3 * i + 1] < min_y)
                min_y = Buff_vtx[3 * i + 1];
            if (Buff_vtx[3 * i + 2] < min_z)
                min_z = Buff_vtx[3 * i + 2];
            if (Buff_vtx[3 * i] > max_x)
                max_x = Buff_vtx[3 * i];
            if (Buff_vtx[3 * i + 1] > max_y)
                max_y = Buff_vtx[3 * i + 1];
            if (Buff_vtx[3 * i + 2] > max_z)
                max_z = Buff_vtx[3 * i + 2];
        }
        cout << "min: " << min_x << ", " << min_y << ", " << min_z << endl;
        cout << "max: " << max_x << ", " << max_y << ", " << max_z << endl;

        for (int i = 0; i < result[1]; i++) {
            getline (plyfile,line); // PLY
            std::istringstream iss(line);
            std::vector<std::string> words((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
            
            if (std::stoi(words[0]) != 3) {
                cout << "Error non triangle faces" << endl;
            }
            
            Buff_faces[3*i] = std::stoi(words[1]);
            Buff_faces[3*i+1] = std::stoi(words[2]);
            Buff_faces[3*i+2] = std::stoi(words[3]);
        }
        
        plyfile.close();
        
        cout << "Computing normals" << endl;
        
        for (int f = 0; f < result[1]; f++) {
            // compute face normal
            my_float3 p1 = make_my_float3(Buff_vtx[3*Buff_faces[3*f]], Buff_vtx[3*Buff_faces[3*f]+1], Buff_vtx[3*Buff_faces[3*f]+2]);
            my_float3 p2 = make_my_float3(Buff_vtx[3*Buff_faces[3*f+1]], Buff_vtx[3*Buff_faces[3*f+1]+1], Buff_vtx[3*Buff_faces[3*f+1]+2]);
            my_float3 p3 = make_my_float3(Buff_vtx[3*Buff_faces[3*f+2]], Buff_vtx[3*Buff_faces[3*f+2]+1], Buff_vtx[3*Buff_faces[3*f+2]+2]);
            
            my_float3 nml = cross(p2-p1, p3-p1);
            float mag = sqrt(nml.x*nml.x + nml.y*nml.y + nml.z*nml.z);
            
            Buff_nmls[3*f] = nml.x/mag;
            Buff_nmls[3*f+1] = nml.y/mag;
            Buff_nmls[3*f+2] = nml.z/mag;
        }
        
        return result;
    }

    plyfile.close();
    cout << "could not load file: " << path << endl;
    return result;
}

int *LoadPLY_PyMesh(string path, float **Vertices, int **Faces, int **Voxels){
    int *result = new int[3];
    result[0] = -1; result[1] = -1; result[2] = -1;

      string line;
    ifstream plyfile (path, ios::binary);
    if (plyfile.is_open()) {
        
        getline (plyfile,line); // PLY
        getline (plyfile,line); // PLY
        getline (plyfile,line); // PLY
        getline (plyfile,line); // PLY
        // get number of vertices
        std::istringstream iss(line);
        std::vector<std::string> words((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
        result[0] = std::stoi(words[2]);
        cout << "number of vertices: " << result[0] << endl;

        getline (plyfile,line); // PLY
        getline (plyfile,line); // PLY
        getline (plyfile,line); // PLY
        getline (plyfile,line); // PLY
        // get number of faces
        std::istringstream iss2(line);
        std::vector<std::string> words2((std::istream_iterator<std::string>(iss2)), std::istream_iterator<std::string>());
        result[1]  = std::stoi(words2[2]);
        cout << "number of faces: " << result[1]  << endl;

        getline (plyfile,line); // PLY
        getline (plyfile,line); // PLY
        std::istringstream iss3(line);
        std::vector<std::string> words3((std::istream_iterator<std::string>(iss3)), std::istream_iterator<std::string>());
        result[2] = std::stoi(words3[2]);
        cout << "number of voxels: " << result[2] << endl;

        getline (plyfile,line); // PLY
        getline (plyfile,line); // PLY
        cout << line << endl;

        // Allocate data
        *Vertices = new float[3*result[0]]();
        float *Buff_vtx = *Vertices;
        
        *Faces = new int[3*result[1]]();
        int *Buff_faces = *Faces;
        
        *Voxels = new int[4*result[2]]();
        int *Buff_vox = *Voxels;

        for (int i = 0; i < result[0]; i++) {
            double x,y,z;
            plyfile.read((char *) &x, 8);
            plyfile.read((char *) &y, 8);
            plyfile.read((char *) &z, 8);
            Buff_vtx[3*i] = float(x);
            Buff_vtx[3*i+1] = float(y);
            Buff_vtx[3*i+2] = float(z);
        }

        for (int i = 0; i < result[1]; i++) {
            unsigned char a;
            int b;
            plyfile.read((char *) &a, 1);

            if (int(a) != 3) {
                cout << "Error non triangle faces" << endl;
            }

            plyfile.read((char *) &b, 4);
            Buff_faces[3*i] = b;
            plyfile.read((char *) &b, 4);
            Buff_faces[3*i+1] = b;
            plyfile.read((char *) &b, 4);
            Buff_faces[3*i+2] = b;
        }

        for (int i = 0; i < result[2]; i++) {
            unsigned char a;
            int b;
            plyfile.read((char *) &a, 1);

            if (int(a) != 4) {
                cout << "Error non tetrahedral voxel" << endl;
            }

            plyfile.read((char *) &b, 4);
            Buff_vox[4*i] = b;
            plyfile.read((char *) &b, 4);
            Buff_vox[4*i+1] = b;
            plyfile.read((char *) &b, 4);
            Buff_vox[4*i+2] = b;
            plyfile.read((char *) &b, 4);
            Buff_vox[4*i+3] = b;
        }

        plyfile.close();
        return result;
    }

      plyfile.close();
    cout << "could not load file: " << path << endl;
    return result;
}

int *LoadOBJ_Mesh(string path, float **Vertices, float **Normals, int **Faces){
    int *result = new int[2];
    result[0] = -1; result[1] = -1; 
    cout << "hello load mesh" << endl;
    string line;
    ifstream objfile (path, ios::binary);
    if (objfile.is_open()) {
        vector<float> vertices;
        vector<int> faces;
        
        while(getline (objfile,line)) {
            //cout << line << endl;
            std::istringstream iss(line);
            std::vector<std::string> words((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
            
            if (words.size() == 0)
                continue;

            if (words[0].compare(string("v")) == 0) {
                vertices.push_back(std::stof(words[3]));
                vertices.push_back(std::stof(words[1]));
                vertices.push_back(std::stof(words[2]));
            }
            
            if (words[0].compare(string("f")) == 0) {
                //cout << words.size() << endl;
                //cout << words[1] << endl;
                if (words.size() == 4) {
                    faces.push_back(std::stoi(words[1]) - 1);
                    faces.push_back(std::stoi(words[2]) - 1);
                    faces.push_back(std::stoi(words[3]) - 1);
                }
                else if (words.size() == 5) {
                    faces.push_back(std::stoi(words[1]) - 1);
                    faces.push_back(std::stoi(words[2]) - 1);
                    faces.push_back(std::stoi(words[3]) - 1);

                    faces.push_back(std::stoi(words[1]) - 1);
                    faces.push_back(std::stoi(words[3]) - 1);
                    faces.push_back(std::stoi(words[4]) - 1);
                }                 
            }
        }
        
        objfile.close();
        
        result[0] = vertices.size()/3;
        result[1] = faces.size()/3;
        
        // Allocate data
        *Vertices = new float[3*result[0]]();
        float *Buff_vtx = *Vertices;
        memcpy(Buff_vtx, vertices.data(), 3*result[0]*sizeof(float));
        
        *Normals = new float[3*result[1]]();
        float *Buff_nmls = *Normals;
        
        *Faces = new int[3*result[1]]();
        int *Buff_faces = *Faces;
        memcpy(Buff_faces, faces.data(), 3*result[1]*sizeof(int));
     
        std::cout << "Computing normals" << endl;
        
        for (int f = 0; f < result[1]; f++) {
            // compute face normal
            //cout << Buff_faces[3*f] << ", " << Buff_faces[3*f+1] << ", " << Buff_faces[3*f+2] << endl;
            my_float3 p1 = make_my_float3(Buff_vtx[3*Buff_faces[3*f]], Buff_vtx[3*Buff_faces[3*f]+1], Buff_vtx[3*Buff_faces[3*f]+2]);
            my_float3 p2 = make_my_float3(Buff_vtx[3*Buff_faces[3*f+1]], Buff_vtx[3*Buff_faces[3*f+1]+1], Buff_vtx[3*Buff_faces[3*f+1]+2]);
            my_float3 p3 = make_my_float3(Buff_vtx[3*Buff_faces[3*f+2]], Buff_vtx[3*Buff_faces[3*f+2]+1], Buff_vtx[3*Buff_faces[3*f+2]+2]);
            
            my_float3 nml = cross(p2-p1, p3-p1);
            float mag = sqrt(nml.x*nml.x + nml.y*nml.y + nml.z*nml.z);
            
            Buff_nmls[3*f] = nml.x/mag;
            Buff_nmls[3*f+1] = nml.y/mag;
            Buff_nmls[3*f+2] = nml.z/mag;
        }
        
        return result;
    }

    objfile.close();
    cout << "could not load file: " << path << endl;
    return result;
}

int* LoadPLY_TetraMesh(string path, float** Vertices, float** NormalsF, float** NormalsT, int** Faces, int** Tetras) {
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
        //cout << "result[0] : " << result[0] << endl;

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
        //cout << "result[1] : " << result[1] << endl;

        getline(objfile, line);
        getline(objfile, line);
        //cout << "line : " << line << endl;

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
        //cout << "result[2] : " << result[2];

        getline(objfile, line); 
        getline(objfile, line); 
        cout << line << endl; //end of header


        for (int i = 0; i < result[0]; i++) {
            getline(objfile, line);
            string word;
            string* words = new string[3];
            std::istringstream iss(line);
            int k = 0;
            while (iss >> word) {
                words[k] = word;
                //cout << words[k] << endl;
                k++;

            }
            vertices.push_back(std::stof(words[0]));
            vertices.push_back(std::stof(words[1]));
            vertices.push_back(std::stof(words[2]));
        }

        for (int i = 0; i < result[1]; i++) {
            getline(objfile, line);
            string word;
            string* words = new string[4];
            std::istringstream iss(line);
            int k = 0;
            while (iss >> word) {
                words[k] = word;
                /*cout << words[k] << endl;*/
                k++;
            }
            faces.push_back(std::stoi(words[1]));
            faces.push_back(std::stoi(words[2]));
            faces.push_back(std::stoi(words[3]));
            /*cout << faces[0] << endl;
            cout << faces[1] << endl;
            cout << faces[2] << endl << endl;*/
        }

        for (int i = 0; i < result[2]; i++) {
            getline(objfile, line);
            string word;
            string* words = new string[5];
            std::istringstream iss(line);
            int k = 0;
            while (iss >> word) {
                words[k] = word;
                /*cout << words[k] << endl;*/
                k++;
            }
            tetras.push_back(std::stoi(words[1]));
            tetras.push_back(std::stoi(words[2]));
            tetras.push_back(std::stoi(words[3]));
            tetras.push_back(std::stoi(words[4]));
            /*cout << "t from tetras = " << i << " : " << tetras[4 * i + 0] << ", ";
            cout << tetras[4 * i + 1] << ", ";
            cout << tetras[4 * i + 2] << ", ";
            cout << tetras[4 * i + 3] << endl << endl;*/
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
        //cout << "FACE " << f << " : " << Buff_faces[3 * f + 0] << ", " << Buff_faces[3 * f + 1] << ", " << Buff_faces[3 * f + 2] << endl;
        my_float3 p1f = make_my_float3(Buff_vertices[3 * Buff_faces[3 * f + 0]], Buff_vertices[3 * Buff_faces[3 * f + 0] + 1], Buff_vertices[3 * Buff_faces[3 * f + 0] + 2]);
        my_float3 p2f = make_my_float3(Buff_vertices[3 * Buff_faces[3 * f + 1]], Buff_vertices[3 * Buff_faces[3 * f + 1] + 1], Buff_vertices[3 * Buff_faces[3 * f + 1] + 2]);
        my_float3 p3f = make_my_float3(Buff_vertices[3 * Buff_faces[3 * f + 2]], Buff_vertices[3 * Buff_faces[3 * f + 2] + 1], Buff_vertices[3 * Buff_faces[3 * f + 2] + 2]);
        /*cout << "p1 : " << p1f.x << ", " << p1f.y << ", " << p1f.z << endl;
        cout << "p2 : " << p2f.x << ", " << p2f.y << ", " << p2f.z << endl;
        cout << "p3 : " << p3f.x << ", " << p3f.y << ", " << p3f.z << endl;*/


        my_float3 normal = cross(p3f - p1f, p2f - p1f);

        //cout << "n : " << normal.x << ", " << normal.y << ", " << normal.z << endl;
        float norm = sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
        
        Buff_normalsF[3 * f] = normal.x / norm;
        Buff_normalsF[3 * f + 1] = normal.y / norm;
        Buff_normalsF[3 * f + 2] = normal.z / norm;


        //cout << "n : " << normal.x / norm << ", " << normal.y / norm << ", " << normal.z / norm << endl << endl;
    }
    

    // Compute normal and check they are pointing outside of the tetrahedras
   
    for (int t = 0; t < result[2]; t++) {
        //cout << "TETRA = " << t << "-->" << Buff_tetras[4 * t] << ", " << Buff_tetras[4 * t + 1] << ", " << Buff_tetras[4 * t + 2] << ", " << Buff_tetras[4 * t + 3] << endl;
        my_float3 p0 = make_my_float3(Buff_vertices[3 * Buff_tetras[4 * t + 0] + 0], Buff_vertices[3 * Buff_tetras[4 * t + 0] + 1], Buff_vertices[3 * Buff_tetras[4 * t + 0] + 2]);
        my_float3 p1 = make_my_float3(Buff_vertices[3 * Buff_tetras[4 * t + 1] + 0], Buff_vertices[3 * Buff_tetras[4 * t + 1] + 1], Buff_vertices[3 * Buff_tetras[4 * t + 1] + 2]);
        my_float3 p2 = make_my_float3(Buff_vertices[3 * Buff_tetras[4 * t + 2] + 0], Buff_vertices[3 * Buff_tetras[4 * t + 2] + 1], Buff_vertices[3 * Buff_tetras[4 * t + 2] + 2]);
        my_float3 p3 = make_my_float3(Buff_vertices[3 * Buff_tetras[4 * t + 3] + 0], Buff_vertices[3 * Buff_tetras[4 * t + 3] + 1], Buff_vertices[3 * Buff_tetras[4 * t + 3] + 2]);

        /*cout << "p0 : " << p0.x << ", " << p0.y << ", " << p0.z << endl;
        cout << "p1 : " << p1.x << ", " << p1.y << ", " << p1.z << endl;
        cout << "p2 : " << p2.x << ", " << p2.y << ", " << p2.z << endl;
        cout << "p3 : " << p3.x << ", " << p3.y << ", " << p3.z << endl;*/

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


        //cout << "normal_012 : " << normal_012.x << ", " << normal_012.y << ", " << normal_012.z << endl;

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


        //cout << "normal_013 : " << normal_013.x << ", " << normal_013.y << ", " << normal_013.z << endl;

        //FACE 023
        dir = 1;
        my_float3 normal_023 = cross(p2 - p0, p3 - p0);
        float norm_023 = sqrt(normal_023.x * normal_023.x + normal_023.y * normal_023.y + normal_023.z * normal_023.z);

            //dot with vector of remaining tetra point
        my_float3 p1p0 = p0 - p1;
        float dot_023 = dot(p1p0, normal_023);
        //cout << "dot_023 : " << dot_023 << endl;
        
        if (dot_023 <= 1e-6) {
            dir = -1;
        }

        Buff_normalsT[3 * 4 * t + 6] = dir * normal_023.x / norm_023;
        Buff_normalsT[3 * 4 * t + 7] = dir * normal_023.y / norm_023;
        Buff_normalsT[3 * 4 * t + 8] = dir * normal_023.z / norm_023;


        //cout << "normal_023 : " << Buff_normalsT[3 * 4 * t + 6] << ", " << Buff_normalsT[3 * 4 * t + 7] << ", " << Buff_normalsT[3 * 4 * t + 8] << endl;

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


        //cout << "normal_123 : " << normal_123.x << ", " << normal_123.y << ", " << normal_123.z << endl << endl;
    }
    //cout << "could not load file: " << path << endl;
    return result;
}

float** GetAvatarBbox(string path) {

    cout << endl << "---------- Get the avatar's bounding box ----------" << endl;
    string line;
    ifstream objfile(path, ios::binary);

    float maxX = -1000.0f;
    float minX = 1000.0f;
    float maxY = -1000.0f;
    float minY = 1000.0f;
    float maxZ = -1000.0f;
    float minZ = 1000.0f;

    float** bbox = new float* [3]; 
    for (int i = 0; i < 3; i++)
    {
        bbox[i] = new float[2];
        for (int j = 0; j < 2; j++)
        {
            bbox[i][j] = 0.0f;
        }
    }

    if (objfile.is_open()) {

        getline(objfile, line);
        //cout << "line1 : " << line << endl << endl;

        while (getline(objfile, line))
        {
            string word;
            std::istringstream iss(line);
            std::vector<std::string> words((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());

            if (words.size() == 0)
                continue;

            float x = 0;
            float y = 0;
            float z = 0;

            if (words[0].compare(string("v")) == 0) {
                x = std::stof(words[1]);
                y = std::stof(words[2]);
                z = std::stof(words[3]);
                //cout << x << ", " << y << ", " << z << endl;

                if (x >= maxX)
                {
                    maxX = x;
                    //cout << " new maxX : " << maxX << endl;
                }
                if (x <= minX)
                {
                    minX = x;
                    //cout << " new minX : " << minX << endl;
                }

                if (y >= maxY)
                {
                    maxY = y;
                    //cout << " new maxY : " << maxY << endl;
                }
                if (y <= minY)
                {
                    minY = y;
                    //cout << " new minY : " << minY << endl;
                }

                if (z >= maxZ)
                {
                    maxZ = z;
                    //cout << " new maxZ : " << maxZ << endl;
                }
                if (z <= minZ)
                {
                    minZ = z;
                    //cout << " new minZ : " << minZ << endl;
                }
            }

            if (words[0].compare(string("f")) == 0) {
                continue;
            }

        }

        //cout << "maxX = " << maxX << ", " << "minX = " << minX << ", " << "maxY = " << maxY << ", " << "minY = " << minY << ", " << "maxZ = " << maxZ << ", " << "minZ = " << minZ << endl;
    }

    objfile.close();
    bbox[0][0] = minX;
    bbox[0][1] = maxX;
    bbox[1][0] = minY;
    bbox[1][1] = maxY;
    bbox[2][0] = minZ;
    bbox[2][1] = maxZ;

    //cout << bbox[0][0] << endl;

    return bbox;
}

void LoadSkeletonVIBE(string template_skeleton, string filename_pose, Eigen::Matrix4f *Joints, int *KinTree_Table, int index) {
    
    float *Joints_T = new float[24*3]();
    ifstream Tfile (template_skeleton, ios::binary);
    if (!Tfile.is_open()) {
        cout << "Could not load skeleton of the model" << endl;
        Tfile.close();
        return;
    }
    Tfile.read((char *) Joints_T, 24*3*sizeof(float));
    Tfile.close();
    
    /*
     ########### Search for the pose that face the camera the most
     */
    float *Angles = new float[24*3]();
    for (int j = 0; j < 3*24; j++)
        Angles[j] = 0.0f;
    float *Angles_tmp = new float[24*3]();
    float min_angle = 1.0e32;
    int best_cam = 0;
    for (int c = 1; c < 9; c++) {
        ifstream sklfile (filename_pose + "skeleton_cam_" + to_string(c) + "_" + to_string(index) + ".bin");
        if (!sklfile.is_open()) {
            cout << "Could not load skeleton of the model" << endl;
            sklfile.close();
            return;
        }
        sklfile.read((char *) Angles_tmp, 24*3*sizeof(float));
        sklfile.close();
        
        for (int j = 0; j < 3*24; j++)
            Angles[j] = Angles[j] + Angles_tmp[j];
        
        if (Angles[0] < min_angle) {
            min_angle = Angles[0];
            best_cam = c;
        }
    }
    
    delete[] Angles_tmp;
    
    for (int j = 0; j < 3*24; j++)
        Angles[j] = Angles[j]/8.0f;
    
    /*cout << "best cam: " << best_cam << endl;
    
    ifstream sklfile (filename_pose + "skeleton_cam_" + to_string(best_cam) + "_" + to_string(index) + ".bin");
    if (!sklfile.is_open()) {
        cout << "Could not load skeleton of the model" << endl;
        sklfile.close();
        return;
    }
    sklfile.read((char *) Angles, 24*3*sizeof(float));
    sklfile.close();*/
    
    /*for (int j = 0; j < 24; j++) {
        cout << "angle " << j << " : " << Angles[3*j] << ", " << Angles[3*j+1] << ", " << Angles[3*j+2] << endl;
    }*/
    
    //Eigen::Matrix3f rotation = rodrigues2matrix(&Template_rotations[0]);
    //Eigen::Matrix4f rotation = euler2matrix(PI-Angles[0], Angles[1], Angles[2]);
    Eigen::Matrix4f rotation = euler2matrix(0.0f, 0.0f, 0.0f);
                
    Eigen::Matrix4f *Kinematics = new Eigen::Matrix4f[24]();
    Kinematics[0] << rotation(0,0), rotation(0,1), rotation(0,2), Joints_T[0],
                    rotation(1,0), rotation(1,1), rotation(1,2), Joints_T[1],
                    rotation(2,0), rotation(2,1), rotation(2,2), Joints_T[2],
                    0.0f, 0.0f, 0.0f, 1.0f;

    //Eigen::Matrix4f jnt0_loc = _Pose.inverse() * Shift * Kinematics[0];
    Eigen::Matrix4f jnt0_loc = Kinematics[0];
    Joints[0] = jnt0_loc;

    for (int j = 1; j < 24; j++) {
        // transform current bone with angle following the kinematic table
        // Get rotation matrix

        //Eigen::Matrix3f rotation = rodrigues2matrix(&Angles[3*j]);
        Eigen::Matrix4f rotation = euler2matrix(Angles[3*j], Angles[3*j+1], Angles[3*j+2]);

        Eigen::Matrix4f Transfo;
        Transfo << rotation(0,0), rotation(0,1), rotation(0,2), Joints_T[3*KinTree_Table[2*j+1]] - Joints_T[3*KinTree_Table[2*j]],
                    rotation(1,0), rotation(1,1), rotation(1,2), Joints_T[3*KinTree_Table[2*j+1]+1] - Joints_T[3*KinTree_Table[2*j]+1],
                    rotation(2,0), rotation(2,1), rotation(2,2), Joints_T[3*KinTree_Table[2*j+1]+2] - Joints_T[3*KinTree_Table[2*j]+2],
                    0.0f, 0.0f, 0.0f, 1.0f;

        Kinematics[KinTree_Table[2*j+1]]= Kinematics[KinTree_Table[2*j]] * Transfo;

        //Eigen::Matrix4f jnt_loc = _Pose.inverse() * Shift * Kinematics[KinTree_Table[2*j+1]];
        Eigen::Matrix4f jnt_loc = Kinematics[KinTree_Table[2*j+1]];
        Joints[KinTree_Table[2*j+1]] = jnt_loc;
    }

    delete []KinTree_Table;
    delete []Kinematics;
    delete []Angles;
    delete []Joints_T;
}

void LoadSkeletonTEMPLATE(string template_skeleton, string filename_pose, Eigen::Matrix4f *Joints) {
    
    float *Template_rotations = new float[27*3]();
    ifstream sklfile (template_skeleton + "Rotations.bin", ios::binary);
    if (!sklfile.is_open()) {
        cout << "Could not load skeleton of the model" << endl;
        sklfile.close();
        return;
    }
    sklfile.read((char *) Template_rotations, 27*3*sizeof(float));
    sklfile.close();
    
    float *Template_translations = new float[27*3]();
    ifstream Tfile (template_skeleton + "Translations.bin", ios::binary);
    if (!Tfile.is_open()) {
        cout << "Could not load skeleton of the model" << endl;
        Tfile.close();
        return;
    }
    Tfile.read((char *) Template_translations, 27*3*sizeof(float));
    Tfile.close();
        
    int *KinTree_Table = new int[27*2]();
    ifstream Kfile (template_skeleton + "KinematicTable.bin", ios::binary);
    if (!Kfile.is_open()) {
        cout << "Could not load skeleton of the model" << endl;
        Kfile.close();
        return;
    }
    Kfile.read((char *) KinTree_Table, 27*2*sizeof(int));
    Kfile.close();
        
    //Eigen::Matrix3f rotation = rodrigues2matrix(&Template_rotations[0]);
    Eigen::Matrix4f rotation = euler2matrix(PI*Template_rotations[0]/180.0f, PI*Template_rotations[1]/180.0f, PI*Template_rotations[2]/180.0f);

    Joints[0] << rotation(0,0), rotation(0,1), rotation(0,2), Template_translations[0],
                    rotation(1,0), rotation(1,1), rotation(1,2), Template_translations[1],
                    rotation(2,0), rotation(2,1), rotation(2,2), Template_translations[2],
                    0.0f, 0.0f, 0.0f, 1.0f;
    cout << Joints[0] << endl;
    
    // build the template skeleton
    for (int j = 1; j < 27; j++) {
        // transform current bone with angle following the kinematic table
        // Get rotation matrix
        //Eigen::Matrix3f rotation = rodrigues2matrix(&Template_rotations[3*j]);
        Eigen::Matrix4f rotation = euler2matrix(PI*Template_rotations[3*j]/180.0f, PI*Template_rotations[3*j+1]/180.0f, PI*Template_rotations[3*j+2]/180.0f);
        //cout << Template_rotations[3*j] << ", " << Template_rotations[3*j+1] << ", " << Template_rotations[3*j+2] << endl;
        //cout << rotation << endl;

        Eigen::Matrix4f Transfo;
        Transfo << rotation(0,0), rotation(0,1), rotation(0,2), Template_translations[3*j],
                    rotation(1,0), rotation(1,1), rotation(1,2), Template_translations[3*j+1],
                    rotation(2,0), rotation(2,1), rotation(2,2), Template_translations[3*j+2],
                    0.0f, 0.0f, 0.0f, 1.0f;
        
        //cout << KinTree_Table[2*j] << ", " << KinTree_Table[2*j+1] << endl;
        
        Joints[KinTree_Table[2*j+1]] = Joints[KinTree_Table[2*j]] * Transfo;
        //cout << Joints[KinTree_Table[2*j+1]]  << endl;
    }

    delete []Template_rotations;
    delete []Template_translations;
    delete []KinTree_Table;
}

void LoadSkeletonARTICULATED(string template_skeleton, string filename_pose, Eigen::Matrix4f *Joints, int index) {
    
    float *Template_rotations = new float[27*3]();
    ifstream sklfileT (template_skeleton + "Rotations.bin", ios::binary);
    if (!sklfileT.is_open()) {
        cout << "Could not load skeleton of the model" << endl;
        sklfileT.close();
        return;
    }
    sklfileT.read((char *) Template_rotations, 27*3*sizeof(float));
    sklfileT.close();
    
    float *Template_translations = new float[27*3]();
    ifstream Tfile (template_skeleton + "Translations.bin", ios::binary);
    if (!Tfile.is_open()) {
        cout << "Could not load skeleton of the model" << endl;
        Tfile.close();
        return;
    }
    Tfile.read((char *) Template_translations, 27*3*sizeof(float));
    Tfile.close();
    
    ifstream sklfile (filename_pose);
    if (!sklfile.is_open()) {
        cout << "Could not load skeleton of the model" << endl;
        sklfile.close();
        return;
    }
    
    string line;
    getline (sklfile,line);
    getline (sklfile,line);
    getline (sklfile,line);
    for (int i = 0; i < index; i++)
        getline (sklfile,line);
    
    //cout << line << endl;
    std::istringstream iss(line);
    std::vector<std::string> words((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
    
    float *Rotations = new float[27*3]();
    Rotations[0] = stof(words[0]); Rotations[1] = stof(words[1]); Rotations[2] = stof(words[2]); // root
    Rotations[3] = stof(words[3]); Rotations[4] = stof(words[4]); Rotations[5] = stof(words[5]); // waist
    Rotations[6] = stof(words[6]); Rotations[7] = -stof(words[8]); Rotations[8] = stof(words[7]); // neck
    Rotations[9] = 0.0f; Rotations[10] = 0.0f; Rotations[11] = 0.0f; // head
    Rotations[12] = 0.0f; Rotations[13] = 0.0f; Rotations[14] = 0.0f; // head right
    Rotations[15] = 0.0f; Rotations[16] = 0.0f; Rotations[17] = 0.0f; // head left
    Rotations[18] = 0.0f; Rotations[19] = 0.0f; Rotations[20] = 0.0f; // head front
    Rotations[21] = 0.0f; Rotations[22] = 0.0f; Rotations[23] = 0.0f; // head back
    Rotations[24] = 0.0f; Rotations[25] = 0.0f; Rotations[26] = 0.0f; // head up
    Rotations[27] = stof(words[9]); Rotations[28] = stof(words[10]); Rotations[29] = 0.0f; // left sternum
    Rotations[30] = stof(words[11]); Rotations[31] = stof(words[12]); Rotations[32] = stof(words[13]); // left shoulder
    Rotations[33] = 0.0f; Rotations[34] = stof(words[14]); Rotations[35] = stof(words[15]); // left elbow
    Rotations[36] = -stof(words[16]); Rotations[37] = stof(words[17]); Rotations[38] = 0.0f; // left wrist
    Rotations[39] = 0.0f; Rotations[40] = 0.0f; Rotations[41] = 0.0f; // left hand
    Rotations[42] = stof(words[18]); Rotations[43] = stof(words[19]); Rotations[44] = 0.0f; // right sternum
    Rotations[45] = stof(words[20]); Rotations[46] = -stof(words[21]); Rotations[47] = -stof(words[22]); // right shoulder
    Rotations[48] = 0.0f; Rotations[49] = stof(words[23]); Rotations[50] = -stof(words[24]); // right elbow
    Rotations[51] = stof(words[25]); Rotations[52] = 0.0f; Rotations[53] = stof(words[26]); // right wrist
    Rotations[54] = 0.0f; Rotations[55] = 0.0f; Rotations[56] = 0.0f; // right hand
    Rotations[57] = stof(words[27]); Rotations[58] = stof(words[28]); Rotations[59] = stof(words[29]); // left hip
    Rotations[60] = stof(words[30]); Rotations[61] = 0.0f; Rotations[62] = stof(words[31]); // left knee
    Rotations[63] = stof(words[32]); Rotations[64] = 0.0f; Rotations[65] = 0.0f; // left ankle
    Rotations[66] = 0.0f; Rotations[67] = 0.0f; Rotations[68] = 0.0f; // left toe
    Rotations[69] = stof(words[33]); Rotations[70] = stof(words[34]); Rotations[71] = stof(words[35]); // right hip
    Rotations[72] = stof(words[36]); Rotations[73] = 0.0f; Rotations[74] = stof(words[37]); // right knee
    Rotations[75] = stof(words[38]); Rotations[76] = 0.0f; Rotations[77] = 0.0f; // right ankle
    Rotations[78] = 0.0f; Rotations[79] = 0.0f; Rotations[80] = 0.0f; // right toe
        
    Template_translations[0] = stof(words[39]); Template_translations[1] = stof(words[40]); Template_translations[2] = stof(words[41]);
    //Template_translations[0] = 0.0f; Template_translations[1] = 0.0f; Template_translations[2] = 0.0f;
        
    sklfile.close();
        
    int *KinTree_Table = new int[27*2]();
    ifstream Kfile (template_skeleton + "KinematicTable.bin", ios::binary);
    if (!Kfile.is_open()) {
        cout << "Could not load skeleton of the model" << endl;
        Kfile.close();
        return;
    }
    Kfile.read((char *) KinTree_Table, 27*2*sizeof(int));
    Kfile.close();
        
    //Eigen::Matrix3f rotation = rodrigues2matrix(&Template_rotations[0]);
    Eigen::Matrix4f rotationP = euler2matrix(Rotations[0], Rotations[1], Rotations[2]);
    cout << Rotations[0] << ", " << Rotations[1] << ", " << Rotations[2] << endl;
    
    Eigen::Matrix4f rotation = euler2matrix(PI*Template_rotations[0]/180.0f, PI*Template_rotations[1]/180.0f, PI*Template_rotations[2]/180.0f);
    cout << Template_rotations[0] << ", " << Template_rotations[1] << ", " << Template_rotations[2] << endl;

    Joints[0] << rotation(0,0), rotation(0,1), rotation(0,2), Template_translations[0],
                    rotation(1,0), rotation(1,1), rotation(1,2), Template_translations[1],
                    rotation(2,0), rotation(2,1), rotation(2,2), Template_translations[2],
                    0.0f, 0.0f, 0.0f, 1.0f;
    Joints[0] = Joints[0]*rotationP;
    /*Joints[0](0,3) = stof(words[39]);
    Joints[0](1,3) = stof(words[40]);
    Joints[0](2,3) = stof(words[41]);*/
    cout << Joints[0] << endl;
    
    // build the template skeleton
    for (int j = 1; j < 27; j++) {
        // transform current bone with angle following the kinematic table
        // Get rotation matrix
        //Eigen::Matrix3f rotation = rodrigues2matrix(&Template_rotations[3*j]);
        Eigen::Matrix4f rotationP = euler2matrix(Rotations[3*j], Rotations[3*j+1], Rotations[3*j+2]);
        Eigen::Matrix4f rotation = euler2matrix(PI*Template_rotations[3*j]/180.0f, PI*Template_rotations[3*j+1]/180.0f, PI*Template_rotations[3*j+2]/180.0f);
        cout << Rotations[3*j] << ", " << Rotations[3*j+1] << ", " << Rotations[3*j+2] << endl;
        //cout << rotation << endl;

        Eigen::Matrix4f Transfo;
        Transfo << rotation(0,0), rotation(0,1), rotation(0,2), Template_translations[3*j],
                    rotation(1,0), rotation(1,1), rotation(1,2), Template_translations[3*j+1],
                    rotation(2,0), rotation(2,1), rotation(2,2), Template_translations[3*j+2],
                    0.0f, 0.0f, 0.0f, 1.0f;
        
        //cout << KinTree_Table[2*j] << ", " << KinTree_Table[2*j+1] << endl;
        
        Joints[KinTree_Table[2*j+1]] = Joints[KinTree_Table[2*j]] * Transfo * rotationP;
        //cout << Joints[KinTree_Table[2*j+1]]  << endl;
    }

    //delete []Template_rotations;
    //delete []Template_translations;
    delete []KinTree_Table;
}

void LoadSkeleton(string filename_joints, string filename_pose, Eigen::Matrix4f *Joints, int *KinTree_Table) {
    float *Joints_T = new float[24*3]();
    ifstream sklfile (filename_joints, ios::binary);
    if (!sklfile.is_open()) {
        cout << "Could not load skeleton of the model" << endl;
        sklfile.close();
        return;
    }
    sklfile.read((char *) Joints_T, 24*3*sizeof(float));
    sklfile.close();
    
    ifstream psefile (filename_pose, ios::binary);
    if (!psefile.is_open()) {
        cout << "Could not load pose: " << filename_pose  << endl;
        psefile.close();
        return;
    }
    float *Angles = new float[3*24]();
    psefile.read((char *) Angles, 24*3*sizeof(float));
    psefile.close();
    
    /*for (int i = 0; i < 24; i++) {
        cout << "i: " << Angles[3*i] << ", " << Angles[3*i+1] << ", " << Angles[3*i+2] << endl;
    }*/

    //Eigen::Matrix3f rotation = _Rectify * rodrigues2matrix(&Angles[0]);
    Eigen::Matrix3f rotation = rodrigues2matrix(&Angles[0]);

    Eigen::Matrix4f *Kinematics = new Eigen::Matrix4f[24]();
    Kinematics[0] << rotation(0,0), rotation(0,1), rotation(0,2), Joints_T[0],
                    rotation(1,0), rotation(1,1), rotation(1,2), Joints_T[1],
                    rotation(2,0), rotation(2,1), rotation(2,2), Joints_T[2],
                    0.0f, 0.0f, 0.0f, 1.0f;

    //Eigen::Matrix4f jnt0_loc = _Pose.inverse() * Shift * Kinematics[0];
    Eigen::Matrix4f jnt0_loc = Kinematics[0];
    Joints[0] = jnt0_loc;

    for (int j = 1; j < 24; j++) {
        // transform current bone with angle following the kinematic table
        // Get rotation matrix

        Eigen::Matrix3f rotation = rodrigues2matrix(&Angles[3*j]);

        Eigen::Matrix4f Transfo;
        Transfo << rotation(0,0), rotation(0,1), rotation(0,2), Joints_T[3*KinTree_Table[2*j+1]] - Joints_T[3*KinTree_Table[2*j]],
                    rotation(1,0), rotation(1,1), rotation(1,2), Joints_T[3*KinTree_Table[2*j+1]+1] - Joints_T[3*KinTree_Table[2*j]+1],
                    rotation(2,0), rotation(2,1), rotation(2,2), Joints_T[3*KinTree_Table[2*j+1]+2] - Joints_T[3*KinTree_Table[2*j]+2],
                    0.0f, 0.0f, 0.0f, 1.0f;

        Kinematics[KinTree_Table[2*j+1]]= Kinematics[KinTree_Table[2*j]] * Transfo;

        //Eigen::Matrix4f jnt_loc = _Pose.inverse() * Shift * Kinematics[KinTree_Table[2*j+1]];
        Eigen::Matrix4f jnt_loc = Kinematics[KinTree_Table[2*j+1]];
        Joints[KinTree_Table[2*j+1]] = jnt_loc;
    }

    delete []Kinematics;
    delete []Angles;
    delete []Joints_T;
}

void LoadStarSkeleton(string filename, Eigen::Matrix4f *Joints_T, int *KinTree_Table, float angle_star = 0.5f) {
    float *Joints = new float[24*3]();
    ifstream sklfile (filename, ios::binary);
    if (!sklfile.is_open()) {
        cout << "Could not load skeleton of the model" << endl;
        sklfile.close();
        return;
    }
    sklfile.read((char *) Joints, 24*3*sizeof(float));
    sklfile.close();

    for (int i = 0; i < 24; i++) {
        Joints_T[i] << 1.0, 0.0, 0.0, Joints[3*i],
                         0.0, 1.0, 0.0, Joints[3*i+1],
                         0.0, 0.0, 1.0, Joints[3*i+2],
                          0.0, 0.0, 0.0, 1.0;
    }
    

    // Take the A pose
    float *Angles = new float[3*24]();

    for (int i = 0; i < 3*24; i++) {
        Angles[i] = 0.0f;
    }

    Angles[5] = angle_star;
    Angles[8] = -angle_star;
    
    /*for (int i = 0; i < 24; i++) {
        cout << "i: " << Angles[3*i] << ", " << Angles[3*i+1] << ", " << Angles[3*i+2] << endl;
    }*/

    Eigen::Matrix3f rotation = rodrigues2matrix(&Angles[0]);

    Eigen::Matrix4f *Kinematics = new Eigen::Matrix4f[24]();
    Kinematics[0] << rotation(0,0), rotation(0,1), rotation(0,2), Joints[0],
                    rotation(1,0), rotation(1,1), rotation(1,2), Joints[1],
                    rotation(2,0), rotation(2,1), rotation(2,2), Joints[2],
                    0.0f, 0.0f, 0.0f, 1.0f;

    Eigen::Matrix4f jnt0_loc = Kinematics[0];
    Joints_T[0] = jnt0_loc;


    for (int j = 1; j < 24; j++) {
        // transform current bone with angle following the kinematic table
        // Get rotation matrix
        Eigen::Matrix3f rotation = rodrigues2matrix(&Angles[3*j]);

        Eigen::Matrix4f Transfo;
        Transfo << rotation(0,0), rotation(0,1), rotation(0,2), Joints[3*KinTree_Table[2*j+1]] - Joints[3*KinTree_Table[2*j]],
                    rotation(1,0), rotation(1,1), rotation(1,2), Joints[3*KinTree_Table[2*j+1]+1] - Joints[3*KinTree_Table[2*j]+1],
                    rotation(2,0), rotation(2,1), rotation(2,2), Joints[3*KinTree_Table[2*j+1]+2] - Joints[3*KinTree_Table[2*j]+2],
                    0.0f, 0.0f, 0.0f, 1.0f;

        Kinematics[KinTree_Table[2*j+1]]= Kinematics[KinTree_Table[2*j]] * Transfo;

        Eigen::Matrix4f jnt_loc = Kinematics[KinTree_Table[2*j+1]];
        Joints_T[KinTree_Table[2*j+1]] = jnt_loc;
    }
    return;
    
    // Compute joints orientations
    for (int i = 1; i < 7; i++) {
        Joints_T[i](0,0) = -Joints_T[i](0,0);
        Joints_T[i](1,0) = -Joints_T[i](1,0);
        Joints_T[i](2,0) = -Joints_T[i](2,0);
        
        Joints_T[i](0,1) = -Joints_T[i](0,1);
        Joints_T[i](1,1) = -Joints_T[i](1,1);
        Joints_T[i](2,1) = -Joints_T[i](2,1);
    }
    
    for (int i = 7; i < 12; i++) {
        if (i == 9)
            continue;
        float a = Joints_T[i](0,2);
        float b = Joints_T[i](1,2);
        float c = Joints_T[i](2,2);
        Joints_T[i](0,0) = -Joints_T[i](0,0);
        Joints_T[i](1,0) = -Joints_T[i](1,0);
        Joints_T[i](2,0) = -Joints_T[i](2,0);
        
        Joints_T[i](0,2) = Joints_T[i](0,1);
        Joints_T[i](1,2) = Joints_T[i](1,1);
        Joints_T[i](2,2) = Joints_T[i](2,1);
        
        Joints_T[i](0,1) = a;
        Joints_T[i](1,1) = b;
        Joints_T[i](2,1) = c;
    }
    
    for (int i = 3; i < 9; i++) {
        if (i == 1 || i == 2 || i == 4 || i == 5 || i == 7 || i == 8 || i == 10 || i == 11)
            continue;
        Joints_T[i](0,0) = -Joints_T[i](0,0);
        Joints_T[i](1,0) = -Joints_T[i](1,0);
        Joints_T[i](2,0) = -Joints_T[i](2,0);
        
        Joints_T[i](0,1) = -Joints_T[i](0,1);
        Joints_T[i](1,1) = -Joints_T[i](1,1);
        Joints_T[i](2,1) = -Joints_T[i](2,1);
    }
    
    for (int i = 13; i < 24; i++) {
        if (i == 15)
            continue;
        if (i == 13 || i == 16 || i == 18 || i == 20 || i == 22) {
            float a = Joints_T[i](0,0);
            float b = Joints_T[i](1,0);
            float c = Joints_T[i](2,0);
            Joints_T[i](0,0) = -Joints_T[i](0,1);
            Joints_T[i](1,0) = -Joints_T[i](1,1);
            Joints_T[i](2,0) = -Joints_T[i](2,1);
            
            Joints_T[i](0,1) = a;
            Joints_T[i](1,1) = b;
            Joints_T[i](2,1) = c;
        } else {
            float a = Joints_T[i](0,0);
            float b = Joints_T[i](1,0);
            float c = Joints_T[i](2,0);
            Joints_T[i](0,0) = Joints_T[i](0,1);
            Joints_T[i](1,0) = Joints_T[i](1,1);
            Joints_T[i](2,0) = Joints_T[i](2,1);
            
            Joints_T[i](0,1) = -a;
            Joints_T[i](1,1) = -b;
            Joints_T[i](2,1) = -c;
        }
    }
    
    delete []Angles;
    delete []Kinematics;
    delete []Joints;
}

void LoadTSkeleton(string filename, Eigen::Matrix4f *Joints_T, int *KinTree_Table) {
    float *Joints = new float[24*3]();
    ifstream sklfile (filename, ios::binary);
    if (!sklfile.is_open()) {
        cout << "Could not load skeleton of the model" << endl;
        sklfile.close();
        return;
    }
    sklfile.read((char *) Joints, 24*3*sizeof(float));
    sklfile.close();

    for (int i = 0; i < 24; i++) {
        Joints_T[i] << 1.0, 0.0, 0.0, Joints[3*i],
                         0.0, 1.0, 0.0, Joints[3*i+1],
                         0.0, 0.0, 1.0, Joints[3*i+2],
                          0.0, 0.0, 0.0, 1.0;
    }
    

    // Take the A pose
    float *Angles = new float[3*24]();

    for (int i = 0; i < 3*24; i++) {
        Angles[i] = 0.0f;
    }

    Eigen::Matrix3f rotation = rodrigues2matrix(&Angles[0]);

    Eigen::Matrix4f *Kinematics = new Eigen::Matrix4f[24]();
    Kinematics[0] << rotation(0,0), rotation(0,1), rotation(0,2), Joints[0],
                    rotation(1,0), rotation(1,1), rotation(1,2), Joints[1],
                    rotation(2,0), rotation(2,1), rotation(2,2), Joints[2],
                    0.0f, 0.0f, 0.0f, 1.0f;

    Eigen::Matrix4f jnt0_loc = Kinematics[0];
    Joints_T[0] = jnt0_loc;


    for (int j = 1; j < 24; j++) {
        // transform current bone with angle following the kinematic table
        // Get rotation matrix
        Eigen::Matrix3f rotation = rodrigues2matrix(&Angles[3*j]);

        Eigen::Matrix4f Transfo;
        Transfo << rotation(0,0), rotation(0,1), rotation(0,2), Joints[3*KinTree_Table[2*j+1]] - Joints[3*KinTree_Table[2*j]],
                    rotation(1,0), rotation(1,1), rotation(1,2), Joints[3*KinTree_Table[2*j+1]+1] - Joints[3*KinTree_Table[2*j]+1],
                    rotation(2,0), rotation(2,1), rotation(2,2), Joints[3*KinTree_Table[2*j+1]+2] - Joints[3*KinTree_Table[2*j]+2],
                    0.0f, 0.0f, 0.0f, 1.0f;

        Kinematics[KinTree_Table[2*j+1]]= Kinematics[KinTree_Table[2*j]] * Transfo;

        Eigen::Matrix4f jnt_loc = Kinematics[KinTree_Table[2*j+1]];
        Joints_T[KinTree_Table[2*j+1]] = jnt_loc;
    }
    return;
    
    // Compute joints orientations
    for (int i = 1; i < 7; i++) {
        Joints_T[i](0,0) = -Joints_T[i](0,0);
        Joints_T[i](1,0) = -Joints_T[i](1,0);
        Joints_T[i](2,0) = -Joints_T[i](2,0);
        
        Joints_T[i](0,1) = -Joints_T[i](0,1);
        Joints_T[i](1,1) = -Joints_T[i](1,1);
        Joints_T[i](2,1) = -Joints_T[i](2,1);
    }
    
    for (int i = 7; i < 12; i++) {
        if (i == 9)
            continue;
        float a = Joints_T[i](0,2);
        float b = Joints_T[i](1,2);
        float c = Joints_T[i](2,2);
        Joints_T[i](0,0) = -Joints_T[i](0,0);
        Joints_T[i](1,0) = -Joints_T[i](1,0);
        Joints_T[i](2,0) = -Joints_T[i](2,0);
        
        Joints_T[i](0,2) = Joints_T[i](0,1);
        Joints_T[i](1,2) = Joints_T[i](1,1);
        Joints_T[i](2,2) = Joints_T[i](2,1);
        
        Joints_T[i](0,1) = a;
        Joints_T[i](1,1) = b;
        Joints_T[i](2,1) = c;
    }
    
    for (int i = 3; i < 9; i++) {
        if (i == 1 || i == 2 || i == 4 || i == 5 || i == 7 || i == 8 || i == 10 || i == 11)
            continue;
        Joints_T[i](0,0) = -Joints_T[i](0,0);
        Joints_T[i](1,0) = -Joints_T[i](1,0);
        Joints_T[i](2,0) = -Joints_T[i](2,0);
        
        Joints_T[i](0,1) = -Joints_T[i](0,1);
        Joints_T[i](1,1) = -Joints_T[i](1,1);
        Joints_T[i](2,1) = -Joints_T[i](2,1);
    }
    
    for (int i = 13; i < 24; i++) {
        if (i == 15)
            continue;
        if (i == 13 || i == 16 || i == 18 || i == 20 || i == 22) {
            float a = Joints_T[i](0,0);
            float b = Joints_T[i](1,0);
            float c = Joints_T[i](2,0);
            Joints_T[i](0,0) = -Joints_T[i](0,1);
            Joints_T[i](1,0) = -Joints_T[i](1,1);
            Joints_T[i](2,0) = -Joints_T[i](2,1);
            
            Joints_T[i](0,1) = a;
            Joints_T[i](1,1) = b;
            Joints_T[i](2,1) = c;
        } else {
            float a = Joints_T[i](0,0);
            float b = Joints_T[i](1,0);
            float c = Joints_T[i](2,0);
            Joints_T[i](0,0) = Joints_T[i](0,1);
            Joints_T[i](1,0) = Joints_T[i](1,1);
            Joints_T[i](2,0) = Joints_T[i](2,1);
            
            Joints_T[i](0,1) = -a;
            Joints_T[i](1,1) = -b;
            Joints_T[i](2,1) = -c;
        }
    }
    
    delete []Angles;
    delete []Kinematics;
    delete []Joints;
}

int CleanFaces(float *Vertices, int *Faces, int nbFaces) {
    vector<int> merged_faces;
    for (int f = 0; f < nbFaces; f++) {
        if (Faces[3*f] == -1)
            continue;
        
        float dist1 = (Vertices[3*Faces[3*f]]-Vertices[3*Faces[3*f+1]])*(Vertices[3*Faces[3*f]]-Vertices[3*Faces[3*f+1]]) +
                        (Vertices[3*Faces[3*f]+1]-Vertices[3*Faces[3*f+1]+1])*(Vertices[3*Faces[3*f]+1]-Vertices[3*Faces[3*f+1]+1]) +
                        (Vertices[3*Faces[3*f]+2]-Vertices[3*Faces[3*f+1]+2])*(Vertices[3*Faces[3*f]+2]-Vertices[3*Faces[3*f+1]+2]);
        
        float dist2 = (Vertices[3*Faces[3*f]]-Vertices[3*Faces[3*f+2]])*(Vertices[3*Faces[3*f]]-Vertices[3*Faces[3*f+2]]) +
                        (Vertices[3*Faces[3*f]+1]-Vertices[3*Faces[3*f+2]+1])*(Vertices[3*Faces[3*f]+1]-Vertices[3*Faces[3*f+2]+1]) +
                        (Vertices[3*Faces[3*f]+2]-Vertices[3*Faces[3*f+2]+2])*(Vertices[3*Faces[3*f]+2]-Vertices[3*Faces[3*f+2]+2]);
        
        float dist3 = (Vertices[3*Faces[3*f+1]]-Vertices[3*Faces[3*f+2]])*(Vertices[3*Faces[3*f+1]]-Vertices[3*Faces[3*f+2]]) +
                        (Vertices[3*Faces[3*f+1]+1]-Vertices[3*Faces[3*f+2]+1])*(Vertices[3*Faces[3*f+1]+1]-Vertices[3*Faces[3*f+2]+1]) +
                        (Vertices[3*Faces[3*f+1]+2]-Vertices[3*Faces[3*f+2]+2])*(Vertices[3*Faces[3*f+1]+2]-Vertices[3*Faces[3*f+2]+2]);
        
        if (dist1 < 1.0e-8 || dist2 < 1.0e-8 || dist3 < 1.0e-8) {
            Faces[3*f] = -1;
            continue;
        }
        
        cout << f << " / " << nbFaces << endl;
        merged_faces.push_back(Faces[3*f]);
        merged_faces.push_back(Faces[3*f+1]);
        merged_faces.push_back(Faces[3*f+2]);
        
        for (int f2 = f+1; f2 < nbFaces; f2++) {
            if (Faces[3*f2] == -1)
                continue;
            
            float dist11 = (Vertices[3*Faces[3*f]]-Vertices[3*Faces[3*f2]])*(Vertices[3*Faces[3*f]]-Vertices[3*Faces[3*f2]]) +
                            (Vertices[3*Faces[3*f]+1]-Vertices[3*Faces[3*f2]+1])*(Vertices[3*Faces[3*f]+1]-Vertices[3*Faces[3*f2]+1]) +
                            (Vertices[3*Faces[3*f]+2]-Vertices[3*Faces[3*f2]+2])*(Vertices[3*Faces[3*f]+2]-Vertices[3*Faces[3*f2]+2]);
            
            float dist12 = (Vertices[3*Faces[3*f]]-Vertices[3*Faces[3*f2+1]])*(Vertices[3*Faces[3*f]]-Vertices[3*Faces[3*f2+1]]) +
                            (Vertices[3*Faces[3*f]+1]-Vertices[3*Faces[3*f2+1]+1])*(Vertices[3*Faces[3*f]+1]-Vertices[3*Faces[3*f2+1]+1]) +
                            (Vertices[3*Faces[3*f]+2]-Vertices[3*Faces[3*f2+1]+2])*(Vertices[3*Faces[3*f]+2]-Vertices[3*Faces[3*f2+1]+2]);
            
            float dist13 = (Vertices[3*Faces[3*f]]-Vertices[3*Faces[3*f2+2]])*(Vertices[3*Faces[3*f]]-Vertices[3*Faces[3*f2+2]]) +
                            (Vertices[3*Faces[3*f]+1]-Vertices[3*Faces[3*f2+2]+1])*(Vertices[3*Faces[3*f]+1]-Vertices[3*Faces[3*f2+2]+1]) +
                            (Vertices[3*Faces[3*f]+2]-Vertices[3*Faces[3*f2+2]+2])*(Vertices[3*Faces[3*f]+2]-Vertices[3*Faces[3*f2+2]+2]);
            
            float dist21 = (Vertices[3*Faces[3*f+1]]-Vertices[3*Faces[3*f2]])*(Vertices[3*Faces[3*f+1]]-Vertices[3*Faces[3*f2]]) +
                            (Vertices[3*Faces[3*f+1]+1]-Vertices[3*Faces[3*f2]+1])*(Vertices[3*Faces[3*f+1]+1]-Vertices[3*Faces[3*f2]+1]) +
                            (Vertices[3*Faces[3*f+1]+2]-Vertices[3*Faces[3*f2]+2])*(Vertices[3*Faces[3*f+1]+2]-Vertices[3*Faces[3*f2]+2]);
            
            float dist22 = (Vertices[3*Faces[3*f+1]]-Vertices[3*Faces[3*f2+1]])*(Vertices[3*Faces[3*f+1]]-Vertices[3*Faces[3*f2+1]]) +
                            (Vertices[3*Faces[3*f+1]+1]-Vertices[3*Faces[3*f2+1]+1])*(Vertices[3*Faces[3*f+1]+1]-Vertices[3*Faces[3*f2+1]+1]) +
                            (Vertices[3*Faces[3*f+1]+2]-Vertices[3*Faces[3*f2+1]+2])*(Vertices[3*Faces[3*f+1]+2]-Vertices[3*Faces[3*f2+1]+2]);
            
            float dist23 = (Vertices[3*Faces[3*f+1]]-Vertices[3*Faces[3*f2+2]])*(Vertices[3*Faces[3*f+1]]-Vertices[3*Faces[3*f2+2]]) +
                            (Vertices[3*Faces[3*f+1]+1]-Vertices[3*Faces[3*f2+2]+1])*(Vertices[3*Faces[3*f+1]+1]-Vertices[3*Faces[3*f2+2]+1]) +
                            (Vertices[3*Faces[3*f+1]+2]-Vertices[3*Faces[3*f2+2]+2])*(Vertices[3*Faces[3*f+1]+2]-Vertices[3*Faces[3*f2+2]+2]);
            
            float dist31 = (Vertices[3*Faces[3*f+2]]-Vertices[3*Faces[3*f2]])*(Vertices[3*Faces[3*f+2]]-Vertices[3*Faces[3*f2]]) +
                            (Vertices[3*Faces[3*f+2]+1]-Vertices[3*Faces[3*f2]+1])*(Vertices[3*Faces[3*f+2]+1]-Vertices[3*Faces[3*f2]+1]) +
                            (Vertices[3*Faces[3*f+2]+2]-Vertices[3*Faces[3*f2]+2])*(Vertices[3*Faces[3*f+2]+2]-Vertices[3*Faces[3*f2]+2]);
            
            float dist32 = (Vertices[3*Faces[3*f+2]]-Vertices[3*Faces[3*f2+1]])*(Vertices[3*Faces[3*f+2]]-Vertices[3*Faces[3*f2+1]]) +
                            (Vertices[3*Faces[3*f+2]+1]-Vertices[3*Faces[3*f2+1]+1])*(Vertices[3*Faces[3*f+2]+1]-Vertices[3*Faces[3*f2+1]+1]) +
                            (Vertices[3*Faces[3*f+2]+2]-Vertices[3*Faces[3*f2+1]+2])*(Vertices[3*Faces[3*f+2]+2]-Vertices[3*Faces[3*f2+1]+2]);
            
            float dist33 = (Vertices[3*Faces[3*f+2]]-Vertices[3*Faces[3*f2+2]])*(Vertices[3*Faces[3*f+2]]-Vertices[3*Faces[3*f2+2]]) +
                            (Vertices[3*Faces[3*f+2]+1]-Vertices[3*Faces[3*f2+2]+1])*(Vertices[3*Faces[3*f+2]+1]-Vertices[3*Faces[3*f2+2]+1]) +
                            (Vertices[3*Faces[3*f+2]+2]-Vertices[3*Faces[3*f2+2]+2])*(Vertices[3*Faces[3*f+2]+2]-Vertices[3*Faces[3*f2+2]+2]);
            
            
            if ((dist11 < 1.0e-8 || dist12 < 1.0e-8 || dist13 < 1.0e-8) &&
                (dist21 < 1.0e-8 || dist22 < 1.0e-8 || dist23 < 1.0e-8) &&
                (dist31 < 1.0e-8 || dist32 < 1.0e-8 || dist33 < 1.0e-8)) {
                cout << "F" << endl;
                cout << Vertices[3*Faces[3*f]] << ", " << Vertices[3*Faces[3*f]+1] << ", " << Vertices[3*Faces[3*f]+2] << endl;
                cout << Vertices[3*Faces[3*f+1]] << ", " << Vertices[3*Faces[3*f+1]+1] << ", " << Vertices[3*Faces[3*f+1]+2] << endl;
                cout << Vertices[3*Faces[3*f+2]] << ", " << Vertices[3*Faces[3*f+2]+1] << ", " << Vertices[3*Faces[3*f+2]+2] << endl;
                cout << "F2" << endl;
                cout << Vertices[3*Faces[3*f2]] << ", " << Vertices[3*Faces[3*f2]+1] << ", " << Vertices[3*Faces[3*f2]+2] << endl;
                cout << Vertices[3*Faces[3*f2+1]] << ", " << Vertices[3*Faces[3*f2+1]+1] << ", " << Vertices[3*Faces[3*f2+1]+2] << endl;
                cout << Vertices[3*Faces[3*f2+2]] << ", " << Vertices[3*Faces[3*f2+2]+1] << ", " << Vertices[3*Faces[3*f2+2]+2] << endl;
                Faces[3*f2] = -1;
            }
        }
    }
    memcpy(Faces, merged_faces.data(), merged_faces.size()*sizeof(int));
    int res = merged_faces.size()/3;
    merged_faces.clear();
    cout << nbFaces << " -> " << res << endl;
    return res;
}

float *ComputeNormalsFaces(float *Vertices, int *Faces, int nbFaces) {
    float *Normals = new float[3*nbFaces];
    
    for (int f = 0; f < nbFaces; f++) {
        // compute face normal
        my_float3 p1 = make_my_float3(Vertices[3*Faces[3*f]], Vertices[3*Faces[3*f]+1], Vertices[3*Faces[3*f]+2]);
        my_float3 p2 = make_my_float3(Vertices[3*Faces[3*f+1]], Vertices[3*Faces[3*f+1]+1], Vertices[3*Faces[3*f+1]+2]);
        my_float3 p3 = make_my_float3(Vertices[3*Faces[3*f+2]], Vertices[3*Faces[3*f+2]+1], Vertices[3*Faces[3*f+2]+2]);
        
        my_float3 nml = cross(p2-p1, p3-p1);
        float mag = sqrt(nml.x*nml.x + nml.y*nml.y + nml.z*nml.z);
        
        if (mag > 0.0f) {
            Normals[3*f] = nml.x/mag;
            Normals[3*f+1] = nml.y/mag;
            Normals[3*f+2] = nml.z/mag;
        } else {
            Normals[3*f] = 0.0f;
            Normals[3*f+1] = 0.0f;
            Normals[3*f+2] = 0.0f;
        }
    }
    return Normals;
}

void UpdateNormalsFaces(float *Vertices, float *Normals, int *Faces, int nbFaces) {
    for (int f = 0; f < nbFaces; f++) {
        // compute face normal
        my_float3 p1 = make_my_float3(Vertices[3*Faces[3*f]], Vertices[3*Faces[3*f]+1], Vertices[3*Faces[3*f]+2]);
        my_float3 p2 = make_my_float3(Vertices[3*Faces[3*f+1]], Vertices[3*Faces[3*f+1]+1], Vertices[3*Faces[3*f+1]+2]);
        my_float3 p3 = make_my_float3(Vertices[3*Faces[3*f+2]], Vertices[3*Faces[3*f+2]+1], Vertices[3*Faces[3*f+2]+2]);
        
        my_float3 nml = cross(p2-p1, p3-p1);
        float mag = sqrt(nml.x*nml.x + nml.y*nml.y + nml.z*nml.z);
        
        Normals[3*f] = nml.x/mag;
        Normals[3*f+1] = nml.y/mag;
        Normals[3*f+2] = nml.z/mag;
    }
}

void UpdateNormals(float *Vertices, float *Normals, int *Faces, int nbVertices, int nbFaces) {
    
    for (int f = 0; f < nbVertices; f++) {
        Normals[3*f] = 0.0f;
        Normals[3*f+1] = 0.0f;
        Normals[3*f+2] = 0.0f;
    }
    
    for (int f = 0; f < nbFaces; f++) {
        // compute face normal
        my_float3 p1 = make_my_float3(Vertices[3*Faces[3*f]], Vertices[3*Faces[3*f]+1], Vertices[3*Faces[3*f]+2]);
        my_float3 p2 = make_my_float3(Vertices[3*Faces[3*f+1]], Vertices[3*Faces[3*f+1]+1], Vertices[3*Faces[3*f+1]+2]);
        my_float3 p3 = make_my_float3(Vertices[3*Faces[3*f+2]], Vertices[3*Faces[3*f+2]+1], Vertices[3*Faces[3*f+2]+2]);
        
        my_float3 nml = cross(p2-p1, p3-p1);
        Normals[3*Faces[3*f]] = Normals[3*Faces[3*f]] + nml.x;
        Normals[3*Faces[3*f]+1] = Normals[3*Faces[3*f]+1] + nml.y;
        Normals[3*Faces[3*f]+2] = Normals[3*Faces[3*f]+2] + nml.z;
        
        Normals[3*Faces[3*f+1]] = Normals[3*Faces[3*f+1]] + nml.x;
        Normals[3*Faces[3*f+1]+1] = Normals[3*Faces[3*f+1]+1] + nml.y;
        Normals[3*Faces[3*f+1]+2] = Normals[3*Faces[3*f+1]+2] + nml.z;
        
        Normals[3*Faces[3*f+2]] = Normals[3*Faces[3*f+2]] + nml.x;
        Normals[3*Faces[3*f+2]+1] = Normals[3*Faces[3*f+2]+1] + nml.y;
        Normals[3*Faces[3*f+2]+2] = Normals[3*Faces[3*f+2]+2] + nml.z;
    }
    
    for (int f = 0; f < nbVertices; f++) {
        my_float3 nml = make_my_float3(Normals[3*f], Normals[3*f+1], Normals[3*f+2]);
        float mag = sqrt(nml.x*nml.x + nml.y*nml.y + nml.z*nml.z);
        
        if (mag > 0.0f) {
            Normals[3*f] = Normals[3*f]/mag;
            Normals[3*f+1] = Normals[3*f+1]/mag;
            Normals[3*f+2] = Normals[3*f+2]/mag;
        }
    }
}

void SkinArticulatedMesh(float *Vertices, float *SkinWeights, DualQuaternion *Skeleton, int nbVertices) {
    for (int v = 0; v < nbVertices; v++) {
        if (Vertices[3*v] == 0.0f && Vertices[3*v+1] == 0.0f && Vertices[3*v+2] == 0.0f)
            continue;
            
        // Blend the transformations with iterative algorithm
        DualQuaternion Transfo = DualQuaternion(Quaternion(0.0,0.0,0.0,0.0), Quaternion(0.0,0.0,0.0,0.0));
        for (int j = 0; j < 24; j++) {
            if (SkinWeights[24*v+j] > 0.0)
                Transfo = Transfo + (Skeleton[j] * SkinWeights[24*v+j]);
        }
                
        Transfo = Transfo.Normalize();
        DualQuaternion point = DualQuaternion(Quaternion(0.0,0.0,0.0,1.0), Quaternion(Vertices[3*v], Vertices[3*v+1], Vertices[3*v+2], 0.0f));
        point  = Transfo * point * Transfo.DualConjugate2();

        my_float3 vtx = point.Dual().Vector();
        Vertices[3*v] = vtx.x;
        Vertices[3*v+1] = vtx.y;
        Vertices[3*v+2] = vtx.z;
    }
}

void SkinMesh(float *Vertices, float *SkinWeights, DualQuaternion *Skeleton, int nbVertices) {
    for (int v = 0; v < nbVertices; v++) {
        if (Vertices[3*v] == 0.0f && Vertices[3*v+1] == 0.0f && Vertices[3*v+2] == 0.0f)
            continue;
            
        // Blend the transformations with iterative algorithm
        DualQuaternion Transfo = DualQuaternion(Quaternion(0.0,0.0,0.0,0.0), Quaternion(0.0,0.0,0.0,0.0));
        for (int j = 0; j < 24; j++) {
            if (SkinWeights[24*v+j] > 0.0)
                Transfo = Transfo + (Skeleton[j] * SkinWeights[24*v+j]);
        }
                
        Transfo = Transfo.Normalize();
        DualQuaternion point = DualQuaternion(Quaternion(0.0,0.0,0.0,1.0), Quaternion(Vertices[3*v], Vertices[3*v+1], Vertices[3*v+2], 0.0f));
        point  = Transfo * point * Transfo.DualConjugate2();

        my_float3 vtx = point.Dual().Vector();
        Vertices[3*v] = vtx.x;
        Vertices[3*v+1] = vtx.y;
        Vertices[3*v+2] = vtx.z;
    }
}

void SkinMeshARTICULATED(float *Vertices, float *SkinWeights, DualQuaternion *Skeleton, int nbVertices) {
    for (int v = 0; v < nbVertices; v++) {
        if (Vertices[3*v] == 0.0f && Vertices[3*v+1] == 0.0f && Vertices[3*v+2] == 0.0f)
            continue;
            
        // Blend the transformations with iterative algorithm
        DualQuaternion Transfo = DualQuaternion(Quaternion(0.0,0.0,0.0,0.0), Quaternion(0.0,0.0,0.0,0.0));
        for (int j = 0; j < 26; j++) {
            if (SkinWeights[26*v+j] > 0.0)
                Transfo = Transfo + (Skeleton[j] * SkinWeights[26*v+j]);
        }
                
        Transfo = Transfo.Normalize();
        DualQuaternion point = DualQuaternion(Quaternion(0.0,0.0,0.0,1.0), Quaternion(Vertices[3*v], Vertices[3*v+1], Vertices[3*v+2], 0.0f));
        point  = Transfo * point * Transfo.DualConjugate2();

        my_float3 vtx = point.Dual().Vector();
        Vertices[3*v] = vtx.x;
        Vertices[3*v+1] = vtx.y;
        Vertices[3*v+2] = vtx.z;
    }
}

void transferSkinWeights(float *VerticesA, float *SkinWeightsA, float *VerticesB, float *SkinWeightsB, int nbVerticesA, int nbVerticesB) {
    for (int v1 = 0; v1 < nbVerticesA; v1++) {
        my_float3 p1 = make_my_float3(VerticesA[3*v1], VerticesA[3*v1+1], VerticesA[3*v1+2]);
        float min_dist = 1.0e32;
        for (int v2 = 0; v2 < nbVerticesB; v2++) {
            my_float3 p2 = make_my_float3(VerticesB[3*v2], VerticesB[3*v2+1], VerticesB[3*v2+2]);
            float dist = norm(p1-p2);
            if (dist < min_dist) {
                min_dist = dist;
                for (int s = 0; s < 24; s++) {
                    SkinWeightsA[24*v1+s] = SkinWeightsB[24*v2+s];
                }
            }
        }
    }
}

// The normals correspond to the normals of the faces
int *LoadPLY_TetAndSurface(string path, float **Vertices, float **Normals, int **Faces, float **Nodes, int **Surface_edges, int **Tetra){
    int *result = new int[5];
    result[0] = -1; result[1] = -1; result[2] = -1; result[3] = -1; result[4] = -1;
    
    string line;
    ifstream plyfile (path, ios::binary);
    if (plyfile.is_open()) {
        //Read header
        getline (plyfile,line); // PLY
        //cout << line << endl;
        std::istringstream iss(line);
        std::vector<std::string> words((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
        
        while (words[0].compare(string("end_header")) != 0) {
            if (words[0].compare(string("element")) == 0) {
                if (words[1].compare(string("vertex")) == 0) {
                    result[0] = std::stoi(words[2]);
                    cout << "nb vertices: " << result[0] << endl;
                } else if (words[1].compare(string("face")) == 0) {
                    result[1] = std::stoi(words[2]);
                    cout << "nb face: " << result[1] << endl;
                } else if (words[1].compare(string("nodes")) == 0) {
                    result[2] = std::stoi(words[2]);
                    cout << "nb nodes: " << result[2] << endl;
                } else if (words[1].compare(string("surface")) == 0) {
                    result[3] = std::stoi(words[2]);
                    cout << "nb surface: " << result[3] << endl;
                } else if (words[1].compare(string("voxel")) == 0) {
                    result[4] = std::stoi(words[2]);
                    cout << "nb voxel: " << result[4] << endl;
                }
            }
            
            getline (plyfile,line); // PLY
            //cout << line << endl;
            iss = std::istringstream(line);
            words.clear();
            words = std::vector<std::string>((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
        }
        
        // Allocate data
        *Vertices = new float[3*result[0]]();
        float *Buff_vtx = *Vertices;
        
        *Normals = new float[3*result[0]]();
        float *Buff_nml = *Normals;
        
        *Faces = new int[3*result[1]]();
        int *Buff_faces = *Faces;
        
        *Nodes = new float[3*result[2]]();
        float *Buff_nodes = *Nodes;
        
        *Surface_edges = new int[2*result[3]]();
        int *Buff_surf = *Surface_edges;
                
        *Tetra = new int[4*result[4]]();
        int *Buff_tetra = *Tetra;

        for (int i = 0; i < result[0]; i++) {
            getline (plyfile,line); // PLY
            std::istringstream iss(line);
            std::vector<std::string> words((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
            Buff_vtx[3*i] = std::stof(words[0]);
            Buff_vtx[3*i+1] = std::stof(words[1]);
            Buff_vtx[3*i+2] = std::stof(words[2]);
            Buff_nml[3*i] = std::stof(words[3]);
            Buff_nml[3*i+1] = std::stof(words[4]);
            Buff_nml[3*i+2] = std::stof(words[5]);
        }        
        
        for (int i = 0; i < result[1]; i++) {
            getline (plyfile,line); // PLY
            std::istringstream iss(line);
            std::vector<std::string> words((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
            
            if (std::stoi(words[0]) != 3) {
                cout << "Error non triangular face" << endl;
            }
            
            Buff_faces[3*i] = std::stoi(words[1]);
            Buff_faces[3*i+1] = std::stoi(words[2]);
            Buff_faces[3*i+2] = std::stoi(words[3]);
        }
        
        for (int i = 0; i < result[2]; i++) {
            getline (plyfile,line); // PLY
            std::istringstream iss(line);
            std::vector<std::string> words((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
            Buff_nodes[3*i] = std::stof(words[0]);
            Buff_nodes[3*i+1] = std::stof(words[1]);
            Buff_nodes[3*i+2] = std::stof(words[2]);
        }
        

        for (int i = 0; i < result[4]; i++) {
            getline (plyfile,line); // PLY
            std::istringstream iss(line);
            std::vector<std::string> words((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
            
            if (std::stoi(words[0]) != 4) {
                cout << "Error non tetrahedron" << endl;
            }
            
            Buff_tetra[4*i] = std::stoi(words[1]);
            Buff_tetra[4*i+1] = std::stoi(words[2]);
            Buff_tetra[4*i+2] = std::stoi(words[3]);
            Buff_tetra[4*i+3] = std::stoi(words[4]);
        }
        
        for (int i = 0; i < result[3]; i++) {
            getline (plyfile,line); // PLY
            //cout << line << endl;
            std::istringstream iss(line);
            std::vector<std::string> words((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
            
            if (std::stoi(words[0]) != 2) {
                cout << "Error non edge" << endl;
            }
            
            //cout << i << ": " << words[1] << ", " << words[2] << endl;
            
            Buff_surf[2*i] = std::stoi(words[1]);
            Buff_surf[2*i+1] = std::stoi(words[2]);
        }
        
        plyfile.close();
        return result;
    }

    plyfile.close();
    cout << "could not load file: " << path << endl;
    return result;
}

int *LoadPLY_Tet(string path, float **Nodes, int **Faces, int **Tetra){
    int *result = new int[3];
    result[0] = -1; result[1] = -1; result[2] = -1;
    
    string line;
    ifstream plyfile (path, ios::binary);
    if (plyfile.is_open()) {
        //Read header
        getline (plyfile,line); // PLY
        //cout << line << endl;
        std::istringstream iss(line);
        std::vector<std::string> words((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
        
        while (words[0].compare(string("end_header")) != 0) {
            if (words[0].compare(string("element")) == 0) {
                if (words[1].compare(string("vertex")) == 0) {
                    result[0] = std::stoi(words[2]);
                    cout << "nb vertices: " << result[0] << endl;
                } else if (words[1].compare(string("face")) == 0) {
                    result[1] = std::stoi(words[2]);
                    cout << "nb face: " << result[1] << endl;
                } else if (words[1].compare(string("voxel")) == 0) {
                    result[2] = std::stoi(words[2]);
                    cout << "nb voxel: " << result[2] << endl;
                }
            }
            
            getline (plyfile,line); // PLY
            //cout << line << endl;
            iss = std::istringstream(line);
            words.clear();
            words = std::vector<std::string>((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
        }
        
        // Allocate data
        *Nodes = new float[3*result[0]]();
        float *Buff_nodes = *Nodes;
        
        *Faces = new int[3*result[1]]();
        int *Buff_faces = *Faces;
                
        *Tetra = new int[4*result[2]]();
        int *Buff_tetra = *Tetra;

        for (int i = 0; i < result[0]; i++) {
            getline (plyfile,line); // PLY
            std::istringstream iss(line);
            std::vector<std::string> words((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
            Buff_nodes[3*i] = std::stof(words[0]);
            Buff_nodes[3*i+1] = std::stof(words[1]);
            Buff_nodes[3*i+2] = std::stof(words[2]);
        }
        
        for (int i = 0; i < result[1]; i++) {
            getline (plyfile,line); // PLY
            std::istringstream iss(line);
            std::vector<std::string> words((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
            
            if (std::stoi(words[0]) != 3) {
                cout << "Error non triangular face" << endl;
            }
            
            Buff_faces[3*i] = std::stoi(words[1]);
            Buff_faces[3*i+1] = std::stoi(words[2]);
            Buff_faces[3*i+2] = std::stoi(words[3]);
        }

        for (int i = 0; i < result[2]; i++) {
            getline (plyfile,line); // PLY
            std::istringstream iss(line);
            std::vector<std::string> words((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
            
            if (std::stoi(words[0]) != 4) {
                cout << "Error non tetrahedron" << endl;
            }
            
            Buff_tetra[4*i] = std::stoi(words[1]);
            Buff_tetra[4*i+1] = std::stoi(words[2]);
            Buff_tetra[4*i+2] = std::stoi(words[3]);
            Buff_tetra[4*i+3] = std::stoi(words[4]);
        }
        
        plyfile.close();
        return result;
    }

    plyfile.close();
    cout << "could not load file: " << path << endl;
    return result;
}

void LoadLevelSet(string path, float *tsdf, float **weights, int nb_nodes) {
    
    ifstream file (path, ios::binary);
    file.read((char*) tsdf, nb_nodes*sizeof(float));
    //file.write((char*)weights, 24*nb_nodes*sizeof(float));
    file.close();
    
}

void CenterMesh(float *Vertices, float *Root, int nbVertices) {
    for (int v = 0; v < nbVertices; v++) {
        if (Vertices[3*v] == 0.0f && Vertices[3*v+1] == 0.0f && Vertices[3*v+2] == 0.0f)
            continue;
        
        Vertices[3*v] = Vertices[3*v] - Root[0];
        Vertices[3*v+1] = Vertices[3*v+1] - Root[1];
        Vertices[3*v+2] = Vertices[3*v+2] - Root[2];
    }
}

#endif /* MeshUtils_h */
