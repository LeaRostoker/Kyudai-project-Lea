#include "include_gpu/Utilities.h"
#include "include_gpu/MarchingCubes.h" 
#include "include_gpu/LevelSet.h"
#include "include_gpu/MeshUtils.h"
#include "include_gpu/CubicMesh.h"

using namespace std;
namespace fs = std::filesystem;

SYSTEMTIME current_time, last_time;

string root_path = "C:/Users/learo/Desktop/SDF/";

//int GRID_SIZE = 256;
const int GRID_SIZE = 256; 
float GRID_RES_X = 0.05f;
float GRID_RES_Y = 0.05f;
float GRID_RES_Z = 0.05f;

constexpr unsigned int str2int(const char* str, int h = 0)
{
    return !str[h] ? 5381 : (str2int(str, h + 1) * 33) ^ str[h];
}

int LabelParser(string name) {
    switch (str2int(name.c_str())) {
    case str2int("Wall"):
        return 1;
    case str2int("wall"):
        return 1;
    case str2int("Floor"):
        return 2;
    case str2int("floor"):
        return 2;
    case str2int("Chair"):
        return 3;
    case str2int("chair"):
        return 3;
    case str2int("Seat"):
        return 34;
    case str2int("sofa"):
        return 10; 
    case str2int("Door"):
        return 4;
    case str2int("Table"):
        return 5;
    case str2int("Desk"):
        return 5;
    case str2int("Glass"):
        return 6;
    case str2int("Ceiling"):
        return 17;
    case str2int("ceiling"):
        return 17;
    case str2int("Shelf"):
        return 31;
    case str2int("shelf"):
        return 31;
    case str2int("wagon"):
        return 31;
    case str2int("misc"):
        return 40;
    default:
        return 41; //unlabelled
    }
    return 0;
}

void CreateLevelSet(string input_path, string output_path, string output_path_tsdf) {

    // CREATE LEVEL SET FROM TETRA MESH
    /**
            0. Search all mesh files in the directory
     */

    vector<float*> vertices_list;
    vector<float*> normalsF_list;
    vector<float*> normalsT_list;
    vector<int*> faces_list;
    vector<int*> tetras_list;
    vector<int*> res_list;
    vector<int*> labels_list;
    int tot_v = 0;
    int tot_f = 0;
    int tot_t = 0;

    std::string delimiter = ".";
    std::string delimiter2 = "/";
    std::string delimiter3 = "_";
    std::string delimiter4 = " ";


    for (const auto& entry : fs::directory_iterator(input_path + "today/tetrahedralized/")) {
        std::string file_ext = entry.path().string();
        auto start = 0U;
        auto end = file_ext.find(delimiter);
        cout << "file_ext : " << file_ext << endl;
        cout << "end : " << end << endl;

        while (end != std::string::npos)
        {
            start = end + delimiter2.length();
            end = file_ext.find(delimiter, start);
        }
        file_ext = file_ext.substr(start, end - start);

        std::string file_name = entry.path().string();
        size_t pos = file_name.find(delimiter);
        file_name = file_name.substr(0, pos);
        if (str2int(file_ext.c_str()) == str2int("ply")) {

            auto start2 = 0U;
            auto end2 = file_name.find(delimiter2);
            while (end2 != std::string::npos)
            {
                start2 = end2 + delimiter2.length();
                end2 = file_name.find(delimiter2, start2);
            }
            file_name = file_name.substr(start2, end2 - start2);

            auto start3 = 0U;
            auto end3 = file_name.find(delimiter3);
            file_name = file_name.substr(start3, end3 - start3);

            auto start4 = 0U;
            auto end4 = file_name.find(delimiter4);
            while (end4 != std::string::npos)
            {
                start4 = end4 + delimiter4.length();
                end4 = file_name.find(delimiter4, start4);
            }
            file_name = file_name.substr(start4, end4 - start4);

            int label = LabelParser(file_name);
            cout << file_name << " ==> " << label << endl;

            float* vertices;
            float* normals_face;
            float* normals_tet;
            int* faces;
            int* tetras;
            int* res = LoadPLY_TetraMesh(entry.path().string(), &vertices, &normals_face, &normals_tet, &faces, &tetras);
            cout << "The PLY file has " << res[0] << " vertices, " << res[1] << " faces and " << res[2] << " tetrahedras" << endl;
            int* labels_v = new int[res[0]];
            for (int i = 0; i < res[0]; i++) {
                labels_v[i] = label;
            }
            
            cout << "label : " << label << endl;
            vertices_list.push_back(vertices);
            normalsF_list.push_back(normals_face);
            normalsT_list.push_back(normals_tet);
            faces_list.push_back(faces);
            tetras_list.push_back(tetras);
            res_list.push_back(res);
            labels_list.push_back(labels_v);

            tot_v += res[0];
            tot_f += res[1];
            tot_t += res[2];

        }
    }

    cout << "total vertices: " << tot_v << ", total faces: " << tot_f << ", total tetras : " << tot_t << endl << endl;

    float* vertices = new float[3 * tot_v];
    int* labels = new int[tot_v];
    float* normals_face = new float[3 * tot_f];
    float* normals_tet = new float[4 * 3 * tot_t];
    int* faces = new int[3 * tot_f];
    int* tetras = new int[4 * tot_t];
    int* res = new int[3];

    int offset_v = 0;
    int offset_l = 0;
    int offset_f = 0;
    int offset_t = 0;

    for (int i = 0; i < vertices_list.size(); i++) {
        memcpy(&vertices[offset_v], vertices_list[i], 3 * res_list[i][0] * sizeof(float));
        memcpy(&labels[offset_l], labels_list[i], res_list[i][0] * sizeof(int));
        memcpy(&normals_face[offset_f], normalsF_list[i], 3 * res_list[i][1] * sizeof(float));        
        memcpy(&normals_tet[3 * offset_t], normalsT_list[i], 4 * res_list[i][2] * 3 * sizeof(float));

        for (int f = 0; f < res_list[i][1]; f++) {
            faces[offset_f + 3 * f + 0] = offset_v / 3 + faces_list[i][3 * f + 0];
            faces[offset_f + 3 * f + 1] = offset_v / 3 + faces_list[i][3 * f + 1];
            faces[offset_f + 3 * f + 2] = offset_v / 3 + faces_list[i][3 * f + 2];
        }

        for (int t = 0; t < res_list[i][2]; t++) {
            tetras[offset_t + 4 * t + 0] = offset_v / 3 + tetras_list[i][4 * t + 0];
            tetras[offset_t + 4 * t + 1] = offset_v / 3 + tetras_list[i][4 * t + 1];
            tetras[offset_t + 4 * t + 2] = offset_v / 3 + tetras_list[i][4 * t + 2];
            tetras[offset_t + 4 * t + 3] = offset_v / 3 + tetras_list[i][4 * t + 3];
        }
        
        offset_t += 4 * res_list[i][2];
        offset_f += 3 * res_list[i][1];
        offset_v += 3 * res_list[i][0];
        offset_l += res_list[i][0];
        delete[]vertices_list[i];
        delete[]labels_list[i];
        delete[]tetras_list[i];
        delete[]normalsF_list[i];
        delete[]normalsT_list[i];
        delete[]faces_list[i];
        delete[]res_list[i];
    }

    res[0] = tot_v;
    res[1] = tot_f;
    res[2] = tot_t;

    /*
            1. Load the 3D scan (it should be a 2D manifold)
     */

    my_int3 size_grid = make_my_int3(GRID_SIZE, GRID_SIZE, GRID_SIZE);
    my_float3 center_grid = make_my_float3(0.0f, 0.0f, 0.0f);
    pair<float***, int***> lvl_set_labels = LevelSet_gpu(vertices, labels, faces, normals_face, tetras, normals_tet, res[0], res[1], res[2], size_grid, center_grid, GRID_RES_X, GRID_RES_Y, GRID_RES_Z, 0.0f);
    float*** sdf = lvl_set_labels.first;
    int*** sdf_l = lvl_set_labels.second;

    float* vertices_tet_out;
    float* colors_tet_out;
    float* normals_tet_out;
    int* faces_tet_out;
    int* res_tet_out = MarchingCubes(sdf, sdf_l, &vertices_tet_out, &colors_tet_out, &normals_tet_out, &faces_tet_out, size_grid, center_grid, GRID_RES_X, 0.0f);
    SaveMeshToPLY(output_path, vertices_tet_out, colors_tet_out, normals_tet_out, faces_tet_out, res_tet_out[0], res_tet_out[1]);

    // save the level set to file
    SaveLevelSet(output_path_tsdf + "sdf.bin", sdf, size_grid);
    SaveLevelSet(output_path_tsdf + "sdf_sem.bin", sdf_l, size_grid);



    for (int i = 0; i < size_grid.x; i++) {
        for (int j = 0; j < size_grid.y; j++) {
            delete[]sdf[i][j];
            delete[]sdf_l[i][j];
        }
        delete[]sdf[i];
        delete[]sdf_l[i];
    }
    delete[]sdf;
    delete[]sdf_l;

    delete[]vertices_tet_out;
    delete[]colors_tet_out;
    delete[]normals_tet_out;
    delete[]faces_tet_out;

    delete[]vertices;
    delete[]normals_face;
    delete[]faces;
    return;
}

int main(int argc, char** argv)
{
    cout << "=======================Program by Diego Thomas, adapted by Lea Rostoker for this project==========================" << endl;

    // This will pick the best possible CUDA capable device
    int devID = findCudaDevice();
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    size_t free, total, initMem;
    cudaMemGetInfo(&free, &total);
    cout << "GPU " << devID << " memory: free=" << free << ", total=" << total << endl;

    CreateLevelSet(root_path + "public-room/", root_path + "public-room/output/office-model.ply", root_path + "public-room/output/");
   
    return 0;
}
