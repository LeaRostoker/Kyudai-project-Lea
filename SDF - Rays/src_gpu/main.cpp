#include "include_gpu/Utilities.h"
#include "include_gpu/MarchingCubes.h" 
#include "include_gpu/LevelSet.h"
#include "include_gpu/MeshUtils.h"
#include "include_gpu/CubicMesh.h"

using namespace std;
namespace fs = std::filesystem;

SYSTEMTIME current_time, last_time;

string root_path = "C:/Users/learo/Desktop/archi-main/";

const int GRID_SIZE = 256; 
float GRID_RES_X = 0.04f;
float GRID_RES_Y = 0.04f;
float GRID_RES_Z = 0.04f;


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
        return 3;
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
    case str2int("rp"):
        return 100;
    default:
        return 0;
    }
    return 0;
}


void UpdateSDF(string input_path, string output_path, string output_path_tsdf) {

    cout << endl << "----------- Adding character's SDF to scene's SDF ----------" << endl << endl;

    std::string delimiter = ".";
    std::string delimiter2 = "/";
    std::string delimiter3 = "_";
    std::string delimiter4 = " ";

    std::string scene_sdf_path = "";
    std::string scene_labels_path = "";
    std::string avatar_path = "";

    for (const auto& entry : fs::directory_iterator(input_path + "seiko/")) {
        std::string file_ext = entry.path().string();
        std::cout << "file_ext : " << file_ext << std::endl;
        auto start = 0U;
        auto end = file_ext.find(delimiter);

        while (end != std::string::npos)
        {
            start = end + delimiter2.length();
            end = file_ext.find(delimiter, start);
        }

        file_ext = file_ext.substr(start, end - start);
                
        std::string file_name = entry.path().string();

        if (str2int(file_ext.c_str()) == str2int("bin")) {
            cout << "bin file '" << file_name << endl;
            size_t pos = file_name.find(delimiter);
            string file_name_noext = file_name.substr(0, pos);
            char last_letter = file_name_noext.back();
            if (last_letter == 'f')
            {
                scene_sdf_path = file_name;                               
            }
            if (last_letter == 'l')
            {
                scene_labels_path = file_name;                                
            }          
        }

        size_t pos = file_name.find(delimiter);
        file_name = file_name.substr(0, pos);

        if (str2int(file_ext.c_str()) == str2int("obj")) {

            auto start2 = 0U;
            auto end2 = file_name.find(delimiter2);
            while (end2 != std::string::npos)
            {
                start2 = end2 + delimiter2.length();
                end2 = file_name.find(delimiter2, start2);
            }

            file_name = file_name.substr(start2, end2 - start2);
            cout << file_name << endl;

            auto start3 = 0U;
            auto end3 = file_name.find(delimiter3);
            file_name = file_name.substr(start3, end3 - start3);
            cout << file_name << endl;

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

            if (label =! 100)
                continue;

            avatar_path = entry.path().string();
        }
    }

    cout << "scene_sdf_path : " << scene_sdf_path << endl;
    cout << "scene_labels_path : " << scene_labels_path << endl;
    cout << "avatar_path : " << avatar_path << endl;

    float** bbox = GetAvatarBbox(avatar_path);
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            cout << bbox[i][j] << endl;
        }
    }

    float* sdf_list = new float[256*256*256];
    int * labels_list = new int[256*256*256];

    float sdf_c;
    std::ifstream sdf(scene_sdf_path, std::ios::binary);
    int count_sdf = 0;
    while (sdf.read(reinterpret_cast<char*>(&sdf_c), sizeof(float))) {
        sdf_list[count_sdf] = sdf_c;
        count_sdf++;
    }

    int labels_c;
    std::ifstream labels(scene_labels_path, std::ios::binary);
    int count_labels = 0;
    while (labels.read(reinterpret_cast<char*>(&labels_c), sizeof(int))) {
        labels_list[count_labels] = labels_c;
        count_labels++;
    }
  
    my_int3 size_grid = make_my_int3(GRID_SIZE, GRID_SIZE, GRID_SIZE);
    my_float3 center_grid = make_my_float3(0.0f, 0.0f, 0.0f);

    AddAvatarSDF(bbox, sdf_list, labels_list, size_grid, center_grid, GRID_RES_X, GRID_RES_Y, GRID_RES_Z, 0.0f);
    
    float* vertices_tet_out;
    float* colors_tet_out;
    float* normals_tet_out;
    int* faces_tet_out;

    float*** sdfs = new float** [size_grid.x];
    for (int i = 0; i < size_grid.x; i++) {
        sdfs[i] = new float* [size_grid.y];
        for (int j = 0; j < size_grid.y; j++) {
            sdfs[i][j] = new float[size_grid.z];
            for (int k = 0; k < size_grid.z; k++) {
                sdfs[i][j][k] = 1.0f;
            }
        }
    }

    int*** lbls = new int** [size_grid.x];
    for (int i = 0; i < size_grid.x; i++) {
        lbls[i] = new int* [size_grid.y];
        for (int j = 0; j < size_grid.y; j++) {
            lbls[i][j] = new int[size_grid.z];
            for (int k = 0; k < size_grid.z; k++) {
                lbls[i][j][k] = 2;
            }
        }
    }

    for (int i = 0; i < size_grid.x; i++) {
        for (int j = 0; j < size_grid.y; j++) {
            for (int k = 0; k < size_grid.z; k++){
                sdfs[i][j][k] = sdf_list[i * size_grid.y * size_grid.z + j * size_grid.z + k];
                lbls[i][j][k] = labels_list[i * size_grid.y * size_grid.z + j * size_grid.z + k];
            }
        }
    }


    int* res_tet_out = MarchingCubes(sdfs, lbls, &vertices_tet_out, &colors_tet_out, &normals_tet_out, &faces_tet_out, size_grid, center_grid, GRID_RES_X, 0.0f);
    //Save 3D mesh to the disk for visualization
    SaveMeshToPLY(output_path, vertices_tet_out, colors_tet_out, normals_tet_out, faces_tet_out, res_tet_out[0], res_tet_out[1]);
    
    // save the level set to file
    SaveLevelSet(output_path_tsdf + "sdf_avatar.bin", sdfs, size_grid);
    SaveLevelSet(output_path_tsdf + "label_avatar.bin", lbls, size_grid);
    return;
     

}



void CreateLevelSet (string input_path, string output_path, string output_path_tsdf) {
    /**
            0. Search all mesh files in the directory
     */

    vector<float*> vertices_list;
    vector<float*> normals_list;
    vector<int*> faces_list;
    vector<int*> res_list;
    vector<int*> labels_list;
    int tot_v = 0;
    int tot_f = 0;

    std::string delimiter = ".";
    std::string delimiter2 = "/";
    std::string delimiter3 = "_";
    std::string delimiter4 = " ";


    for (const auto& entry : fs::directory_iterator(input_path + "shawn/")) {
        std::string file_ext = entry.path().string();
        std::cout << "file_ext : " << file_ext << std::endl;
        auto start = 0U;
        auto end = file_ext.find(delimiter);
        
        
        while (end != std::string::npos)
        {
            start = end + delimiter2.length();
            end = file_ext.find(delimiter, start);

        }
        //std::cout << file_ext.substr(start, end - start) << std::endl;
        file_ext = file_ext.substr(start, end - start);

        std::string file_name = entry.path().string();
        size_t pos = file_name.find(delimiter);
        file_name = file_name.substr(0, pos);
        if (str2int(file_ext.c_str()) == str2int("obj")) {
            
            auto start2 = 0U;
            auto end2 = file_name.find(delimiter2);
            while (end2 != std::string::npos)
            {
                //std::cout << file_name.substr(start, end - start) << std::endl;
                start2 = end2 + delimiter2.length();
                end2 = file_name.find(delimiter2, start2);
            }
            //std::cout << file_name.substr(start2, end2 - start2) << std::endl;
            file_name = file_name.substr(start2, end2 - start2);
            cout << file_name << endl;

            auto start3 = 0U;
            auto end3 = file_name.find(delimiter3);
            file_name = file_name.substr(start3, end3 - start3);
            cout << file_name << endl;

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

            if (label == 0 || label == 1 || label == 4 || label == 17 || label == 31)
                continue;

            float* vertices;
            float* normals_face;
            int* faces;
            int* voxels;
            int* res = LoadOBJ_Mesh(entry.path().string(), &vertices, &normals_face, &faces);
            cout << "The OBJ file has " << res[0] << " vertices and " << res[1] << " faces" << endl;
            //SaveMeshToPLY(output_path, vertices, res_tet_out[0], res_tet_out[1]);
            int* labels_v = new int[res[0]];
            /*memset(labels_v, label, res[0] * sizeof(int)); */
            for (int i = 0; i < res[0]; i++) {
                labels_v[i] = label; 
            }

            cout << "label" << label << endl;
            vertices_list.push_back(vertices);
            normals_list.push_back(normals_face);
            faces_list.push_back(faces);
            res_list.push_back(res);
            labels_list.push_back(labels_v);

            tot_v += res[0];
            tot_f += res[1];
            
            
        }
        
    }

    cout << "total vertices: " << tot_v << ", total faces: " << tot_f << endl;

    float* vertices = new float[3*tot_v];
    int* labels = new int[tot_v];
    float* normals_face = new float[3 * tot_f];
    int* faces = new int[3 * tot_f];
    int* res = new int[2];

    int offset_v = 0;
    int offset_l = 0;
    int offset_f = 0;
    for (int i = 0; i < vertices_list.size(); i++) {
        memcpy(&vertices[offset_v], vertices_list[i], 3*res_list[i][0]*sizeof(float));
        /*for (int v = 0; v < res_list[i][0]; v++) {

            cout << "vertices[v] : " << vertices[v]<< endl;
        }*/
        memcpy(&labels[offset_l], labels_list[i], res_list[i][0] * sizeof(int));
        //cout << "*labels_list[i]" << *labels_list[i] << endl;
        memcpy(&normals_face[offset_f], normals_list[i], 3 * res_list[i][1] * sizeof(float));
        for (int f = 0; f < res_list[i][1]; f++) {
            faces[offset_f + 3 * f] = offset_v / 3 + faces_list[i][3 * f];
            faces[offset_f + 3 * f + 1] = offset_v / 3 + faces_list[i][3 * f + 1];
            faces[offset_f + 3 * f + 2] = offset_v / 3 + faces_list[i][3 * f + 2];

          /* cout << "faces[offset_f + 3 * f] : " << faces[offset_f + 3 * f] << endl;
            cout << "faces[offset_f + 3 * f + 1] : " << faces[offset_f + 3 * f] << endl;
            cout << "faces[offset_f + 3 * f + 2] : " << faces[offset_f + 3 * f] << endl;
        */}


        offset_f += 3 * res_list[i][1];
        offset_v += 3 * res_list[i][0];
        offset_l += res_list[i][0];
        delete[]vertices_list[i];
        delete[]labels_list[i];
        delete[]normals_list[i];
        delete[]faces_list[i];
        delete[]res_list[i];
    }

    res[0] = tot_v;
    res[1] = tot_f;

    //SaveMeshToPLY(output_path, vertices, vertices, faces, tot_v, tot_f);

    /**
            1. Load the 3D scan (it should be a 2D manifold)
     */
    /*float* vertices;
    float* normals_face;
    int* faces;
    int* res = LoadOBJ_Mesh(input_path + "office model_0524.obj", &vertices, &normals_face, &faces);
    cout << "The OBJ file has " << res[0] << " vertices and " << res[1] << " faces" << endl;*/

    
    
    my_int3 size_grid = make_my_int3(GRID_SIZE, GRID_SIZE, GRID_SIZE);
    //my_float3 center_grid = make_my_float3(0.0f, 0.0f, 0.0f);
    my_float3 center_grid = make_my_float3(0.0f, 0.0f, 0.0f);

    pair<float***, int***> lvl_set_labels = LevelSet_gpu(vertices, labels, faces, normals_face, res[0], res[1], size_grid, center_grid, GRID_RES_X, GRID_RES_Y, GRID_RES_Z, 0.0f); 
    //pair<float***, int***> lvl_set_labels = LevelSet_test(vertices, labels, faces, normals_face, res[0], res[1], size_grid, center_grid, GRID_RES_X, GRID_RES_Y, GRID_RES_Z, 0.0f); 
    float*** sdf = lvl_set_labels.first;
    int*** sdf_l = lvl_set_labels.second;


    //float*** sdf = LevelSet_gpu(vertices, faces, normals_face, res[0], res[1], size_grid, center_grid, GRID_RES, 0.0f);


    float* vertices_tet_out;
    float* colors_tet_out;
    float* normals_tet_out;
    int* faces_tet_out;
    int* res_tet_out = MarchingCubes(sdf, sdf_l , &vertices_tet_out, &colors_tet_out, &normals_tet_out, &faces_tet_out, size_grid, center_grid, GRID_RES_X, 0.0f);
    // Save 3D mesh to the disk for visualization
    SaveMeshToPLY(output_path, vertices_tet_out, colors_tet_out, normals_tet_out, faces_tet_out, res_tet_out[0], res_tet_out[1]);
    //SaveMeshToPLY(output_path, vertices, vertices, faces, tot_v, tot_f);

    // save the level set to file
    SaveLevelSet(output_path_tsdf+"sdf.bin", sdf, size_grid);
    SaveLevelSet(output_path_tsdf+"label.bin", sdf_l, size_grid);
    


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

//void CreateLevelSet(string input_path, string output_path, string output_path_tsdf) {
//
//    // CREATE LEVEL SET FROM TETRA MESH
//    /**
//            0. Search all mesh files in the directory
//     */
//
//    vector<float*> vertices_list;
//    vector<float*> normalsF_list;
//    vector<float*> normalsT_list;
//    vector<int*> faces_list;
//    vector<int*> tetras_list;
//    vector<int*> res_list;
//    vector<int*> labels_list;
//    int tot_v = 0;
//    int tot_f = 0;
//    int tot_t = 0;
//
//    std::string delimiter = ".";
//    std::string delimiter2 = "/";
//    std::string delimiter3 = "_";
//    std::string delimiter4 = " ";
//
//
//    for (const auto& entry : fs::directory_iterator(input_path + "meshes2 - sceneTetra/")) {
//        std::string file_ext = entry.path().string();
//        auto start = 0U;
//        auto end = file_ext.find(delimiter);
//        cout << "file_ext : " << file_ext << endl;
//        cout << "end : " << end << endl;
//
//        while (end != std::string::npos)
//        {
//            start = end + delimiter2.length();
//            end = file_ext.find(delimiter, start);
//        }
//        file_ext = file_ext.substr(start, end - start);
//
//        std::string file_name = entry.path().string();
//        size_t pos = file_name.find(delimiter);
//        file_name = file_name.substr(0, pos);
//        if (str2int(file_ext.c_str()) == str2int("ply")) {
//
//            auto start2 = 0U;
//            auto end2 = file_name.find(delimiter2);
//            while (end2 != std::string::npos)
//            {
//                start2 = end2 + delimiter2.length();
//                end2 = file_name.find(delimiter2, start2);
//            }
//            file_name = file_name.substr(start2, end2 - start2);
//
//            auto start3 = 0U;
//            auto end3 = file_name.find(delimiter3);
//            file_name = file_name.substr(start3, end3 - start3);
//
//            auto start4 = 0U;
//            auto end4 = file_name.find(delimiter4);
//            while (end4 != std::string::npos)
//            {
//                start4 = end4 + delimiter4.length();
//                end4 = file_name.find(delimiter4, start4);
//            }
//            file_name = file_name.substr(start4, end4 - start4);
//
//            int label = LabelParser(file_name);
//            cout << file_name << " ==> " << label << endl;
//
//            float* vertices;
//            float* normals_face;
//            float* normals_tet;
//            int* faces;
//            int* tetras;
//            int* res = LoadPLY_TetraMesh(entry.path().string(), &vertices, &normals_face, &normals_tet, &faces, &tetras);
//            cout << "The PLY file has " << res[0] << " vertices, " << res[1] << " faces and " << res[2] << " tetrahedras" << endl;
//            int* labels_v = new int[res[0]];
//            for (int i = 0; i < res[0]; i++) {
//                labels_v[i] = label;
//            }
//            
//            cout << "label" << label << endl;
//            vertices_list.push_back(vertices);
//            normalsF_list.push_back(normals_face);
//            normalsT_list.push_back(normals_tet);
//            faces_list.push_back(faces);
//            tetras_list.push_back(tetras);
//            res_list.push_back(res);
//            labels_list.push_back(labels_v);
//
//            tot_v += res[0];
//            tot_f += res[1];
//            tot_t += res[2];
//
//        }
//    }
//
//    cout << "total vertices: " << tot_v << ", total faces: " << tot_f << ", total tetras : " << tot_t << endl << endl;
//
//    float* vertices = new float[3 * tot_v];
//    int* labels = new int[tot_v];
//    float* normals_face = new float[3 * tot_f];
//    float* normals_tet = new float[4 * 3 * tot_t];
//    int* faces = new int[3 * tot_f];
//    int* tetras = new int[4 * tot_t];
//    int* res = new int[3];
//
//    int offset_v = 0;
//    int offset_l = 0;
//    int offset_f = 0;
//    int offset_t = 0;
//
//    for (int i = 0; i < vertices_list.size(); i++) {
//        memcpy(&vertices[offset_v], vertices_list[i], 3 * res_list[i][0] * sizeof(float));
//        memcpy(&labels[offset_l], labels_list[i], res_list[i][0] * sizeof(int));
//        memcpy(&normals_face[offset_f], normalsF_list[i], 3 * res_list[i][1] * sizeof(float));        
//        memcpy(&normals_tet[3 * offset_t], normalsT_list[i], 4 * res_list[i][2] * 3 * sizeof(float));
//
//        for (int f = 0; f < res_list[i][1]; f++) {
//            faces[offset_f + 3 * f + 0] = offset_v / 3 + faces_list[i][3 * f + 0];
//            faces[offset_f + 3 * f + 1] = offset_v / 3 + faces_list[i][3 * f + 1];
//            faces[offset_f + 3 * f + 2] = offset_v / 3 + faces_list[i][3 * f + 2];
//        }
//
//        for (int t = 0; t < res_list[i][2]; t++) {
//            tetras[offset_t + 4 * t + 0] = offset_v / 3 + tetras_list[i][4 * t + 0];
//            tetras[offset_t + 4 * t + 1] = offset_v / 3 + tetras_list[i][4 * t + 1];
//            tetras[offset_t + 4 * t + 2] = offset_v / 3 + tetras_list[i][4 * t + 2];
//            tetras[offset_t + 4 * t + 3] = offset_v / 3 + tetras_list[i][4 * t + 3];
//        }
//
//        offset_t += 4 * res_list[i][2];
//        offset_f += 3 * res_list[i][1];
//        offset_v += 3 * res_list[i][0];
//        offset_l += res_list[i][0];
//        delete[]vertices_list[i];
//        delete[]labels_list[i];
//        delete[]tetras_list[i];
//        delete[]normalsF_list[i];
//        delete[]normalsT_list[i];
//        delete[]faces_list[i];
//        delete[]res_list[i];
//    }
//
//    res[0] = tot_v;
//    res[1] = tot_f;
//    res[2] = tot_t;
//
//
//    /*
//            1. Load the 3D scan (it should be a 2D manifold)
//     */
//
//    my_int3 size_grid = make_my_int3(GRID_SIZE, GRID_SIZE, GRID_SIZE);
//    //my_float3 center_grid = make_my_float3(0.0f, 0.0f, 0.0f);
//    my_float3 center_grid = make_my_float3(0.0f, 0.0f, 0.0f);
//    pair<float***, int***> lvl_set_labels = LevelSet_gpu(vertices, labels, faces, normals_face, res[0], res[1], size_grid, center_grid, GRID_RES_X, GRID_RES_Y, GRID_RES_Z, 0.0f);
//    float*** sdf = lvl_set_labels.first;
//    int*** sdf_l = lvl_set_labels.second;
//
//    float* vertices_tet_out;
//    float* colors_tet_out;
//    float* normals_tet_out;
//    int* faces_tet_out;
//    int* res_tet_out = MarchingCubes(sdf, sdf_l, &vertices_tet_out, &colors_tet_out, &normals_tet_out, &faces_tet_out, size_grid, center_grid, GRID_RES_X, 0.0f);
//    // Save 3D mesh to the disk for visualization
//    SaveMeshToPLY(output_path, vertices_tet_out, colors_tet_out, normals_tet_out, faces_tet_out, res_tet_out[0], res_tet_out[1]);
//
//    // save the level set to file
//    SaveLevelSet(output_path_tsdf + "sdf.bin", sdf, size_grid);
//    SaveLevelSet(output_path_tsdf + "sdf_sem.bin", sdf_l, size_grid);
//
//
//
//    for (int i = 0; i < size_grid.x; i++) {
//        for (int j = 0; j < size_grid.y; j++) {
//            delete[]sdf[i][j];
//            delete[]sdf_l[i][j];
//        }
//        delete[]sdf[i];
//        delete[]sdf_l[i];
//    }
//    delete[]sdf;
//    delete[]sdf_l;
//
//    delete[]vertices_tet_out;
//    delete[]colors_tet_out;
//    delete[]normals_tet_out;
//    delete[]faces_tet_out;
//
//    delete[]vertices;
//    delete[]normals_face;
//    delete[]faces;
//    return;
//}

//void CreateLevelSetFromPLYMesh(string input_path, string output_path, string output_path_tsdf) {
//
//    /**
//            1. Load the 3D scan (it should be a 2D manifold)
//     */
//    float* vertices;
//    float* normals_face;
//    int* faces;
//    int* res = LoadPLY_Mesh(input_path, &vertices, &normals_face, &faces);
//    cout << "The PLY file has " << res[0] << " vertices and " << res[1] << " faces" << endl;
//
//    int* labels = new int[res[0]];
//    memset(labels, 0, res[0] * sizeof(int));
//
//    SaveMeshToPLY(output_path, vertices, vertices, faces, res[0], res[1]);
//
//    my_int3 size_grid = make_my_int3(GRID_SIZE, GRID_SIZE, GRID_SIZE);
//    //my_float3 center_grid = make_my_float3(0.0f, 0.0f, 0.0f);
//    my_float3 center_grid = make_my_float3(1.57f, 2.36f, 0.71f);
//
//    pair<float***, int***> lvl_set_labels = LevelSet_gpu(vertices, labels, faces, normals_face, res[0], res[1], size_grid, center_grid, GRID_RES_X, GRID_RES_Y, GRID_RES_Z, 0.0f);
//    float*** sdf = lvl_set_labels.first;
//    int*** sdf_l = lvl_set_labels.second;
//
//    //float*** sdf = LevelSet_gpu(vertices, faces, normals_face, res[0], res[1], size_grid, center_grid, GRID_RES, 0.0f);
//
//    float* vertices_tet_out;
//    float* colors_tet_out;
//    float* normals_tet_out;
//    int* faces_tet_out;
//    int* res_tet_out = MarchingCubes(sdf, sdf_l, &vertices_tet_out, &colors_tet_out, &normals_tet_out, &faces_tet_out, size_grid, center_grid, GRID_RES_X, 0.0f);
//    // Save 3D mesh to the disk for visualization
//    SaveMeshToPLY(output_path, vertices_tet_out, colors_tet_out, normals_tet_out, faces_tet_out, res_tet_out[0], res_tet_out[1]);
//    //SaveMeshToPLY(output_path, vertices, vertices, faces, tot_v, tot_f);
//
//    // save the level set to file
//    SaveLevelSet(output_path_tsdf + "sdf.bin", sdf, size_grid);
//    SaveLevelSet(output_path_tsdf + "sdf_sem.bin", sdf_l, size_grid);
//    cout << "The coarse level set is generated" << endl;
//
//
//    for (int i = 0; i < size_grid.x; i++) {
//        for (int j = 0; j < size_grid.y; j++) {
//            delete[]sdf[i][j];
//            delete[]sdf_l[i][j];
//        }
//        delete[]sdf[i];
//        delete[]sdf_l[i];
//    }
//    delete[]sdf;
//    delete[]sdf_l;
//
//    delete[]vertices_tet_out;
//    delete[]colors_tet_out;
//    delete[]normals_tet_out;
//    delete[]faces_tet_out;
//
//    delete[]vertices;
//    delete[]normals_face;
//    delete[]faces;
//    return;
//}

//void Generate3DModel(string tsdf_path, string output_path) {
//    cout << "==============Generate the 3D mesh from the level sets==================" << endl;
//
//    /**
//            1. Load the coarse level set and fit the outer shell to it
//     */
//    my_int3 size_grid = make_my_int3(GRID_SIZE, GRID_SIZE, GRID_SIZE);
//    my_float3 center_grid = make_my_float3(0.0f, 0.0f, 0.0f);
//
//    float*** tsdf = LoadLevelSet(tsdf_path, size_grid);
//    cout << "The level set is generated" << endl;
//
//    float* vertices_tet_out;
//    float* normals_tet_out;
//    int* faces_tet_out;
//    int* res_tet_out;// = MarchingCubes(tsdf, &vertices_tet_out, &normals_tet_out, &faces_tet_out, size_grid, center_grid, GRID_RES, 0.0f);
//    // Save 3D mesh to the disk for visualization
//    SaveMeshToPLY(output_path, vertices_tet_out, normals_tet_out, faces_tet_out, res_tet_out[0], res_tet_out[1]);
//
//    for (int i = 0; i < size_grid.x; i++) {
//        for (int j = 0; j < size_grid.y; j++) {
//            delete[]tsdf[i][j];
//        }
//        delete[]tsdf[i];
//    }
//    delete[]tsdf;
//
//    delete[]vertices_tet_out;
//    delete[]normals_tet_out;
//    delete[]faces_tet_out;
//}

char* getCmdOption(char** begin, char** end, const std::string& option)
{
    char** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
    return std::find(begin, end, option) != end;
}

int main(int argc, char** argv)
{
    cout << "=======================Program by Diego Thomas, updated by Lea Rostoker ==========================" << endl;

    if (cmdOptionExists(argv, argv + argc, "-h"))
    {
        cout << "HELP" << endl;
        cout << "Use the different modes of teh program using command -mode [mode]" << endl;
        cout << "[mode] = CreateLevelSet -> Generate the coarse and fine level set from one or a sequence of centered .ply files with skeleton data" << endl;
        cout << "[mode] = Generate3DMesh -> Generate the 3D mesh for the given level set" << endl;
        return 1;
    }

    // This will pick the best possible CUDA capable device
    int devID = findCudaDevice();
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    size_t free, total, initMem;
    cudaMemGetInfo(&free, &total);
    cout << "GPU " << devID << " memory: free=" << free << ", total=" << total << endl;
       

    //CreateLevelSet(root_path + "public-room/", root_path + "public-room/output/office-model.ply", root_path + "public-room/output/");
    UpdateSDF(root_path + "public-room/", root_path + "public-room/output/office-model.ply", root_path + "public-room/output/");
    
    return 0;
}
