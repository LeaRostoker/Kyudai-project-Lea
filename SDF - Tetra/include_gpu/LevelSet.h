//
//  LevelSet.h
//  DEEPANIM
//
//  Created by Diego Thomas on 2021/01/14.
//

#ifndef LevelSet_h
#define LevelSet_h

#define PI 3.14159265f


float norm_test(float3 a) {
    return sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}

float DistancePointFace_test(float3 p0, float3 p1, float3 p2, float3 p3, float3 n, bool approx = false) {
    float3 center = (p1 + p2 + p3) * (1.0f / 3.0f);
    if (approx) {
        float d0 = sqrt(dot(p0 - center, p0 - center));
        float d1 = sqrt(dot(p0 - p1, p0 - p1));
        float d2 = sqrt(dot(p0 - p2, p0 - p2));
        float d3 = sqrt(dot(p0 - p3, p0 - p3));
        return min(d0, min(d1, min(d2, d3)));
    }

    // a. Project point onto the plane of the triangle
    float3 p1p0 = p0 - p1;
    float dot_prod = dot(p1p0, n);
    float3 proj = p0 - n * dot_prod;

    //p1p2p3
    float3 cross_p1p2p3 = cross(p2 - p1, p3 - p1);
    float area = norm_test(cross_p1p2p3) / 2.0f;
    if (area < 1.0e-12) {
        return 1.0e32;
    }

    // b. Test if projection is inside the triangle
    float3 C;

    // edge 0 = p1p2
    float3 edge0 = p2 - p1;
    float3 vp0 = proj - p1;
    C = cross(edge0, vp0);
    float w = (norm_test(C) / 2.0f) / area;
    if (dot(n, C) < 0.0f) {
        // P is on the right side of edge0
        // compute distance point to segment
        float curr_dist;
        float3 base = edge0 * (1.0f / norm_test(edge0));
        float Dt = dot(base, vp0);
        if (Dt < 0.0f) {
            curr_dist = norm_test(p0 - p1);
        }
        else if (Dt > norm_test(edge0)) {
            curr_dist = norm_test(p0 - p2);
        }
        else {
            curr_dist = norm_test(p0 - (p1 + base * Dt));
        }
        return curr_dist;
    }

    // edge 1 = p2p3
    float3 edge1 = p3 - p2;
    float3 vp1 = proj - p2;
    C = cross(edge1, vp1);
    float u = (norm_test(C) / 2.0f) / area;
    if (dot(n, C) < 0.0f) {
        // P is on the right side of edge1
        // compute distance point to segment
        // compute distance point to segment
        float curr_dist;
        float3 base = edge1 * (1.0f / norm_test(edge1));
        float Dt = dot(base, vp1);
        if (Dt < 0.0f) {
            curr_dist = norm_test(p0 - p2);
        }
        else if (Dt > norm_test(edge1)) {
            curr_dist = norm_test(p0 - p3);
        }
        else {
            curr_dist = norm_test(p0 - (p2 + base * Dt));
        }
        return curr_dist;
    }

    // edge 2 = p3p1
    float3 edge2 = p1 - p3;
    float3 vp2 = proj - p3;
    C = cross(edge2, vp2);
    float v = (norm_test(C) / 2.0f) / area;
    if (dot(n, C) < 0.0f) {
        // P is on the right side of edge 2;
        float curr_dist;
        float3 base = edge2 * (1.0f / norm_test(edge2));
        float Dt = dot(base, vp2);
        if (Dt < 0.0f) {
            curr_dist = norm_test(p0 - p3);
        }
        else if (Dt > norm_test(edge2)) {
            curr_dist = norm_test(p0 - p1);
        }
        else {
            curr_dist = norm_test(p0 - (p3 + base * Dt));
        }
        return curr_dist;
    }

    if (u <= 1.00001f && v <= 1.00001f && w <= 1.00001f) {
        return sqrt(dot(p0 - proj, p0 - proj));
    }
    else {
        return 1.0e32;
    }

    return 1.0e32;
}


void InnerLoopThin(float ***volume, int i, float *vertices, int *faces, float *normals, int nb_faces, my_int3 size_grid, my_float3 center_grid, float res, float disp) {
    my_float3 p0;
    p0.x = (float(i)-float(size_grid.x)/2.0f)*res + center_grid.x;
    
    my_float3 ray = make_my_float3(0.0f, 0.0f, 1.0f);
    //my_float3 ray = make_my_float3(1.0f, 1.0f, 1.0f)*(1.0f/sqrt(3.0f));
    
    for (int j = 0; j < size_grid.y; j++) {
        if (i == 0)
            cout << 100.0f*float(j)/float(size_grid.y) << "%\r";
        p0.y = (float(j)-float(size_grid.y)/2.0f)*res + center_grid.y;
        for (int k = 0; k < size_grid.z; k++) {
            //cout << i << ", " << j << ", " << k << endl;
            // Get the 3D coordinate
            p0.z = (float(k)-float(size_grid.z)/2.0f)*res + center_grid.z;
            
            // Compute the smallest distance to the faces
            float min_dist = 1.0e32f;
            float sdf = 1.0f;
            for (int f = 0; f < nb_faces; f++) {
                // Compute distance point to face
                my_float3 n = make_my_float3(normals[3*f], normals[3*f+1], normals[3*f+2]);
                my_float3 p1 = make_my_float3(vertices[3*faces[3*f]], vertices[3*faces[3*f]+1], vertices[3*faces[3*f]+2]);
                my_float3 p2 = make_my_float3(vertices[3*faces[3*f+1]], vertices[3*faces[3*f+1]+1], vertices[3*faces[3*f+1]+2]);
                my_float3 p3 = make_my_float3(vertices[3*faces[3*f+2]], vertices[3*faces[3*f+2]+1], vertices[3*faces[3*f+2]+2]);
                
                // Compute point to face distance
                float curr_dist = DistancePointFace3D(p0, p1, p2, p3, n);
                if (curr_dist < min_dist) {
                    min_dist = curr_dist;
                    float sign = dot(p1-p0, n);
                    sdf = sign > 0.0 ? -curr_dist : curr_dist;
                }
            }
            
            volume[i][j][k] = (sdf-disp)/0.05f;
        }
    }
}


void InnerLoop(float ***volume, int ***volume_l, int i, float *vertices, int* labels, int *faces, float *normals, int nb_faces, my_int3 size_grid, my_float3 center_grid, float res, float disp) {
    my_float3 p0;
    p0.x = (float(i)-float(size_grid.x)/2.0f)*res + center_grid.x;
    
    my_float3 ray = make_my_float3(0.0f, 0.0f, 1.0f);
    //my_float3 ray = make_my_float3(1.0f, 1.0f, 1.0f)*(1.0f/sqrt(3.0f));
    
    for (int j = 0; j < size_grid.y; j++) {
        if (i == 0) 
            cout << 100.0f*float(j)/float(size_grid.y) << "%\r";
        p0.y = (float(j)-float(size_grid.y)/2.0f)*res + center_grid.y;
        for (int k = 0; k < size_grid.z; k++) {
            //cout << i << ", " << j << ", " << k << endl;
            // Get the 3D coordinate
            p0.z = (float(k)-float(size_grid.z)/2.0f)*res + center_grid.z;
            
            // Compute the smallest distance to the faces
            float min_dist = 1.0e32f;
            float sdf = 1.0f;
            int lbl = 0;
            float intersections = 0.0f;
            for (int f = 0; f < nb_faces; f++) {
                // Compute distance point to face
                my_float3 n = make_my_float3(normals[3*f], normals[3*f+1], normals[3*f+2]);
                my_float3 p1 = make_my_float3(vertices[3*faces[3*f]], vertices[3*faces[3*f]+1], vertices[3*faces[3*f]+2]);
                my_float3 p2 = make_my_float3(vertices[3*faces[3*f+1]], vertices[3*faces[3*f+1]+1], vertices[3*faces[3*f+1]+2]);
                my_float3 p3 = make_my_float3(vertices[3*faces[3*f+2]], vertices[3*faces[3*f+2]+1], vertices[3*faces[3*f+2]+2]);
                            
            //    // Compute line plane intersection
            //    //if (IsInterectingRayTriangle3D(ray, p0, p1, p2, p3, n))
            //    //    intersections++;
            //    intersections += IsInterectingRayTriangle3D(ray, p0, p1, p2, p3, n);
            //    
            //    // Compute point to face distance
            //    float curr_dist = DistancePointFace3D(p0, p1, p2, p3, n);
            //    if (curr_dist < min_dist) {
            //        min_dist = curr_dist;
            //        sdf = curr_dist;///0.4f;
            //    }
            //}
            //
            ////cout << int(intersections) <<  endl;
            ////volume[i][j][k] = intersections % 2 == 0 ? sdf : -sdf;
            //volume[i][j][k] = int(intersections) % 2 == 0 ? sdf-disp : -sdf-disp;

            //TEST LEA 
                float curr_dist = DistancePointFace_test(p0, p1, p2, p3, n);

                if (curr_dist <= min_dist) {
                    if (dot(n, (p1 - p0)) <= 0.0f) //outside
                        sdf = curr_dist - disp;
                    else if (curr_dist < min_dist)
                        sdf = -curr_dist - disp;

                    min_dist = curr_dist - disp;
                    lbl = labels[faces[3 * f]];
                }
            }

            volume[i][j][k] = sdf;

        }
    }
}

void InnerLoopSDFFromLevelSet(float ***volume, int i, float ***sdf, my_int3 size_grid, my_float3 center_grid, float res) {
    my_float3 p0;
    p0.x = (float(i)-float(size_grid.x)/2.0f)*res + center_grid.x;
    
    my_float3 ray = make_my_float3(0.0f, 0.0f, 1.0f);
    //my_float3 ray = make_my_float3(1.0f, 1.0f, 1.0f)*(1.0f/sqrt(3.0f));
    
    for (int j = 0; j < size_grid.y; j++) {
        if (i == 0)
            cout << 100.0f*float(j)/float(size_grid.y) << "%\r";
        p0.y = (float(j)-float(size_grid.y)/2.0f)*res + center_grid.y;
        for (int k = 0; k < size_grid.z; k++) {
            // Get the 3D coordinate
            p0.z = (float(k)-float(size_grid.z)/2.0f)*res + center_grid.z;
            
            float sdf_val = sdf[i][j][k];
            float sdf_curr = sdf_val;
            
            int s = 1;
            while (sdf_val*sdf_curr > 0.0f && s < size_grid.x+1) {
                for (int a = max(0,i-s); a < min(size_grid.x,i+s+1); a++) {
                    for (int b = max(0,j-s); b < min(size_grid.y,j+s+1); b++) {
                        if (k-s >= 0) {
                            sdf_curr = sdf[a][b][k-s];
                            if (sdf_val*sdf_curr <= 0.0f)
                                break;
                        }
                        
                        if (k+s < size_grid.z) {
                            sdf_curr = sdf[a][b][k+s];
                            if (sdf_val*sdf_curr <= 0.0f)
                                break;
                        }
                    }
                }
                
                if (sdf_val*sdf_curr <= 0.0f)
                    break;
                
                
                for (int a = max(0,i-s); a < min(size_grid.x,i+s+1); a++) {
                    for (int c = max(0,k-s); c < min(size_grid.z,k+s+1); c++) {
                        if (j-s >= 0) {
                            sdf_curr = sdf[a][j-s][c];
                            if (sdf_val*sdf_curr <= 0.0f)
                                break;
                        }
                        
                        if (j+s < size_grid.y) {
                            sdf_curr = sdf[a][j+s][c];
                            if (sdf_val*sdf_curr <= 0.0f)
                                break;
                        }
                    }
                }
                
                if (sdf_val*sdf_curr <= 0.0f)
                    break;
                
                
                for (int c = max(0,k-s); c < min(size_grid.z,k+s+1); c++) {
                    for (int b = max(0,j-s); b < min(size_grid.y,j+s+1); b++) {
                        if (i-s >= 0) {
                            sdf_curr = sdf[i-s][b][c];
                            if (sdf_val*sdf_curr <= 0.0f)
                                break;
                        }
                        
                        if (i+s < size_grid.x) {
                            sdf_curr = sdf[i+s][b][c];
                            if (sdf_val*sdf_curr <= 0.0f)
                                break;
                        }
                    }
                }
                
                if (sdf_val*sdf_curr <= 0.0f)
                    break;
                
                s++;
            }
            
            volume[i][j][k] = sdf_val < 0.0f ? -float(s)*res : float(s)*res;
        }
    }
}

/**
 This function creates a volumetric level set from a mesh with define resolution (=size of 1 voxel)
 */
pair<float ***, int***> LevelSet(int* labels, float *vertices, int *faces, float *normals, int nb_faces, my_int3 size_grid, my_float3 center_grid, float res, float disp = 0.0f) {
    // Allocate data
    float ***volume = new float **[size_grid.x];
    for (int i = 0; i < size_grid.x; i++) {
        volume[i] = new float *[size_grid.y];
        for (int j = 0; j < size_grid.y; j++) {
            volume[i][j] = new float[size_grid.z];
            for (int k = 0; k < size_grid.z; k++) {
                volume[i][j][k] = 1.0f;
            }
        }
    }

    int*** volume_l = new int** [size_grid.x];
    for (int i = 0; i < size_grid.x; i++) {
        volume_l[i] = new int* [size_grid.y];
        for (int j = 0; j < size_grid.y; j++) {
            volume_l[i][j] = new int[size_grid.z];
            for (int k = 0; k < size_grid.z; k++) {
                volume_l[i][j][k] = 0;

            }
        }
    }
    
    // Compute the signed distance to the mesh for each voxel
    std::vector< std::thread > my_threads;
    for (int i = 0; i < size_grid.x; i++) {
        //InnerLoop(volume, i, vertices, faces, normals, nb_faces, size_grid, center_grid, res, disp);
        my_threads.push_back( std::thread(InnerLoop, volume, volume_l, i, vertices, labels, faces, normals, nb_faces, size_grid, center_grid, res, disp) );
    }
    std::for_each(my_threads.begin(), my_threads.end(), std::mem_fn(&std::thread::join));
    
    return pair<float***, int***>(volume, volume_l);
}

/**
 This function creates a volumetric level set of a sphere
 */
float ***LevelSet_sphere(my_float3 center, float radius, int *size_grid, float res) {
    // Allocate data
    float ***volume = new float **[size_grid[0]];
    for (int i = 0; i < size_grid[0]; i++) {
        volume[i] = new float *[size_grid[1]];
        for (int j = 0; j < size_grid[1]; j++) {
            volume[i][j] = new float[size_grid[2]];
        }
    }
    
    // Compute the signed distance to the mesh for each voxel
    my_float3 p0;
    for (int i = 0; i < size_grid[0]; i++) {
        p0.x = float(i)*res-float(size_grid[0])*res/2.0f;
        for (int j = 0; j < size_grid[1]; j++) {
            if (i == 0)
                cout << 100.0f*float(j)/float(size_grid[1]) << "%\r";
            p0.y = float(j)*res-float(size_grid[1])*res/2.0f;
            for (int k = 0; k < size_grid[2]; k++) {
                // Get the 3D coordinate
                p0.z = float(k)*res-float(size_grid[2])*res/2.0f;
                
                // Compute distance point to sphere
                volume[i][j][k] = (dot(p0-center, p0-center) - radius*radius);
            }
        }
    }
    
    return volume;
}

void TetInnerLoop(float *volume, float *volume_weights, int i, float *vertices, float *skin_weights, int *faces, float *normals, int nb_nodes, int nb_faces, float *nodes) {

    my_float3 ray = make_my_float3(0.0f, 0.0f, 1.0f);
    
    for (int idx = i; idx < i + nb_nodes/100; idx++) {
        if (idx >= nb_nodes)
            break;
        if (i == 0)
            cout << 100.0f*float(idx-i)/float(nb_nodes/100) << "%\r";
        
        my_float3 p0 = make_my_float3(nodes[3*idx], nodes[3*idx+1], nodes[3*idx+2]);
        
        // Compute the smallest distance to the faces
        float min_dist = 1.0e32f;
        float sdf = 1.0f;
        float *s_w = new float[24];
        int intersections = 0;
        for (int f = 0; f < nb_faces; f++) {
            // Compute distance point to face
            my_float3 n = make_my_float3(normals[3*f], normals[3*f+1], normals[3*f+2]);
            my_float3 p1 = make_my_float3(vertices[3*faces[3*f]], vertices[3*faces[3*f]+1], vertices[3*faces[3*f]+2]);
            my_float3 p2 = make_my_float3(vertices[3*faces[3*f+1]], vertices[3*faces[3*f+1]+1], vertices[3*faces[3*f+1]+2]);
            my_float3 p3 = make_my_float3(vertices[3*faces[3*f+2]], vertices[3*faces[3*f+2]+1], vertices[3*faces[3*f+2]+2]);
                        
            // Compute line plane intersection
            if (IsInterectingRayTriangle3D(ray, p0, p1, p2, p3, n))
                intersections++;
            
            my_float3 center = (p1 + p2 + p3) * (1.0f/3.0f);
            
            // Compute point to face distance
            float curr_dist = DistancePointFace3D(p0, p1, p2, p3, n, false);
            if (curr_dist < min_dist) {
                min_dist = curr_dist;
                //sdf = curr_dist/THRESH;
                sdf = min(1.0f, curr_dist/THRESH);
                // Compute skin weights
                //for (int k = 0; k < 24; k++)
                //    s_w[k] = skin_weights[24*faces[3*f] + k];
                // <===== For skin weights
                /*SkinWeightsFromFace3D(s_w, &skin_weights[24*faces[3*f]], &skin_weights[24*faces[3*f+1]], &skin_weights[24*faces[3*f+2]], p0, p1, p2, p3, n);*/
            }
        }
    
        volume[idx] = intersections % 2 == 0 ? sdf : -sdf;
        /*for (int k = 0; k < 24; k++)
            volume_weights[24*idx+k] = s_w[k];*/
        
        delete[] s_w;
        
    }
}
/**
 This function creates a tet volumetric level set from a mesh
 */
float *TetLevelSet(float *volume_weights, float *vertices, float *skin_weights, int *faces, float *normals, int nb_faces, float *nodes, int nb_nodes) {
    // Allocate data
    float *volume = new float [nb_nodes];
    for (int i = 0; i < nb_nodes; i++) {
            volume[i] = 1.0f;
    }
    
    // Compute the signed distance to the mesh for each voxel
    std::vector< std::thread > my_threads;
    for (int i = 0; i < 101; i++) {
        my_threads.push_back( std::thread(TetInnerLoop, volume, volume_weights, i*(nb_nodes/100), vertices, skin_weights, faces, normals, nb_nodes, nb_faces, nodes) );
    }
    std::for_each(my_threads.begin(), my_threads.end(), std::mem_fn(&std::thread::join));
    
    /*for (int i = 0; i < nb_nodes; i++) {
        if (i % 100 == 0)
            cout << 100.0f*float(i)/float(nb_nodes) << "%\r";
        TetInnerLoop(volume, volume_weights, i, vertices, skin_weights, faces, normals, nb_nodes, nb_faces, nodes);
    }*/
    
    return volume;
}

/**
 This function creates a volumetric level set of a sphere
 */
float *TetLevelSet_sphere(my_float3 center, float radius, float *nodes, int nb_nodes) {
    // Allocate data
    float *volume = new float [nb_nodes];
    for (int i = 0; i < nb_nodes; i++) {
            volume[i] = 1.0f;
    }
    
    // Compute the signed distance to the mesh for each voxel
    my_float3 p0;
    for (int i = 0; i < nb_nodes; i++) {
        // Get the 3D coordinate
        p0 = make_my_float3(nodes[3*i], nodes[3*i+1], nodes[3*i+2]);
        
        // Compute distance point to sphere
        volume[i] = (dot(p0-center, p0-center) - radius*radius);
    }
    
    return volume;
}


/**
 This function recompute the level set from the current 0 crossing
 */
float ***SDFFromLevelSet(float ***sdf, my_int3 size_grid, my_float3 center_grid, float res) {
    // Allocate data
    float ***volume = new float **[size_grid.x];
    for (int i = 0; i < size_grid.x; i++) {
        volume[i] = new float *[size_grid.y];
        for (int j = 0; j < size_grid.y; j++) {
            volume[i][j] = new float[size_grid.z];
            for (int k = 0; k < size_grid.z; k++) {
                volume[i][j][k] = 1.0f;
            }
        }
    }
    
    // Compute the signed distance to the mesh for each voxel
    std::vector< std::thread > my_threads;
    for (int i = 0; i < size_grid.x; i++) {
        //InnerLoop(volume, i, vertices, faces, normals, nb_faces, size_grid, center_grid, res, disp);
        my_threads.push_back( std::thread(InnerLoopSDFFromLevelSet, volume, i, sdf, size_grid, center_grid, res) );
    }
    std::for_each(my_threads.begin(), my_threads.end(), std::mem_fn(&std::thread::join));
    
    return volume;
}


//TESTS




pair<float***, int***> LevelSet_test(float* vertices, int* labels, int* faces, float* normals, int nb_vertices, int nb_faces, int3 size_grid, float3 center_grid, float res_x, float res_y, float res_z, float disp) {
    cout << "hello level set test" << endl;

    float*** volume = new float** [size_grid.x];
    int*** volume_l = new int** [size_grid.x];

    for (int i = 0; i < size_grid.x; i++) {
        volume[i] = new float* [size_grid.y];
        for (int j = 0; j < size_grid.y; j++) {
            volume[i][j] = new float[size_grid.z];
            for (int k = 0; k < size_grid.z; k++) {
                volume[i][j][k] = 1.0f;

            }
        }
    }

    cout << "hey" << endl;

    for (int i = 0; i < size_grid.x; i++) {
        volume_l[i] = new int* [size_grid.y];
        for (int j = 0; j < size_grid.y; j++) {
            volume_l[i][j] = new int[size_grid.z];
            for (int k = 0; k < size_grid.z; k++) {
                volume_l[i][j][k] = 0;

            }
        }
    }
    cout << "hey" << endl;

    // SDF Process
        
  /*  int a = 64;
    int b = 118;
    int c = 64;*/

    for (int i = 0; i < size_grid.x; i++) {
        for (int j = 0; j < size_grid.y; j++) {
            for (int k = 0; k < size_grid.z; k++) {

                if (i > size_grid.x - 1 || j > size_grid.y - 1 || k > size_grid.z - 1)
                    return pair<float***, int***>(volume, volume_l);

                // Get the 3D coordinate
                float3 p0;
                p0.x = (float(i) - float(size_grid.x) / 2.0f) * res_x + center_grid.x;
                p0.y = (float(j) - float(size_grid.y) / 2.0f) * res_y + center_grid.y;
                p0.z = (float(k) - float(size_grid.z) / 2.0f) * res_z + center_grid.z;

                float x = 0.0f;
                float y = 0.0f;
                float z = 1.0f; 
                float3 ray = make_float3(x, y, z);
                ray = ray / sqrt(x * x + y * y + z * z);

                // Compute the smallest distance to the faces
                float min_dist = 1.0e32f;
                float sdf = 1.0f;
                int lbl = 0;
                int intersections = 0;

                //bool show = false;
                //if (   (i == a && j == b && k == c) 
                //    || (i == a + 1 && j == b && k == c)
                //    || (i == a + 1 && j == b + 1 && k == c)
                //    || (i == a + 1 && j == b && k == c + 1) //
                //    || (i == a && j == b + 1 && k == c)
                //    || (i == a && j == b + 1 && k == c + 1)
                //    || (i == a - 1 && j == b && k == c)
                //    || (i == a - 1 && j == b - 1 && k == c)
                //    || (i == a - 1 && j == b && k == c - 1)
                //    || (i == a && j == b - 1 && k == c)
                //    || (i == a && j == b - 1 && k == c - 1)
                //    || (i == a && j == b - 1 && k == c + 1)
                //    || (i == a && j == b + 1 && k == c - 1)
                //    || (i == a - 1 && j == b && k == c - 1)
                //    || (i == a - 1 && j == b + 1 && k == c)
                //    || (i == a + 1 && j == b + 1 && k == c + 1)
                //    || (i == a - 1 && j == b - 1 && k == c - 1)
                //    || (i == a - 1 && j == b + 1 && k == c + 1)
                //    || (i == a - 1 && j == b - 1 && k == c + 1)
                //    || (i == a - 1 && j == b + 1 && k == c - 1)
                //    || (i == a + 1 && j == b - 1 && k == c - 1)
                //    || (i == a + 1 && j == b - 1 && k == c + 1)
                //    || (i == a + 1 && j == b + 1 && k == c - 1)
                //    || (i == a + 1 && j == b - 1 && k == c)
                //    || (i == a + 1 && j == b && k == c - 1)
                //    || (i == a - 1 && j == b && k == c + 1)
                //    || (i == a && j == b && k == c + 1)
                //    || (i == a && j == b && k == c - 1)
                //    ) {
                //    show = true;
                //    cout << " point (" << i << ", " << j << ", " << k << ")" << endl;
                //    cout << " p1 3D coord (" << p0.x << ", " << p0.y << ", " << p0.z << ")" << endl;
                //}
                

                for (int f = 0; f < nb_faces; f++) {
                    /*if (show==true)
                        cout << "FACE " << f << endl;*/
                    // Compute distance point to face
                    float3 n = make_float3(normals[3 * f], normals[3 * f + 1], normals[3 * f + 2]);
                    //if (fabs(n.z) < 1.0f)
                    //    continue;
                    float3 p1 = make_float3(vertices[3 * faces[3 * f + 0]], vertices[3 * faces[3 * f + 0] + 1], vertices[3 * faces[3 * f + 0] + 2]);
                    float3 p2 = make_float3(vertices[3 * faces[3 * f + 1]], vertices[3 * faces[3 * f + 1] + 1], vertices[3 * faces[3 * f + 1] + 2]);
                    float3 p3 = make_float3(vertices[3 * faces[3 * f + 2]], vertices[3 * faces[3 * f + 2] + 1], vertices[3 * faces[3 * f + 2] + 2]);
                    
                    //cout << " p1 3D coord (" << p1.x << ", " << p1.y << ", " << p1.z << ")" << endl;

                    // Compute line plane intersection
                    //intersections += IsInterectingRayTriangle3D_gpu(ray, p0, p1, p2, p3, n);

                    // Compute point to face distance
                    float curr_dist = DistancePointFace_test(p0, p1, p2, p3, n, false);
                    /*if (show == true)
                        cout << "curr_dist : " << curr_dist << endl;*/
                    //sdf = curr_dist;
                    float3 p1p0 = p0 - p1;
                    float dot_prod = dot(p1p0, n);
                    float3 proj = p0 - n * dot_prod;

                    /*if (show == true)
                        cout << "dot = " << dot(n, (proj - p0)) << endl;*/

                    /*if (show == true)
                        cout << "curr_dist : " << curr_dist << endl;*/

                    if ((curr_dist < min_dist) || fabs(curr_dist - min_dist) < 10e-6 ){
                        
                        if (dot(n, (proj - p0)) <= 10e-6) {
                            sdf = curr_dist;
                        } 
                        
                        else if (fabs(curr_dist - min_dist) > 10e-6) {
                             sdf = -curr_dist;
                        }
                            

                        min_dist = curr_dist;
                        lbl = labels[faces[3 * f]];
                    }

                
                }
                /*if (show == true)
                    cout << "   FINAL SDF = " << sdf << endl;*/
                volume[i][j][k] = sdf;
                volume_l[i][j][k] = lbl;
            }
        }
    }

    return pair<float***, int***>(volume, volume_l);
}

pair<float***, int***> LevelSet_testTetra(float* vertices, int* labels, int* faces, int* tetras, float* normalsF, float* normalsT, int nb_vertices, int nb_faces, int nb_tetras, int3 size_grid, float3 center_grid, float res_x, float res_y, float res_z, float disp) {
    cout << "hello tetra level set test" << endl;


    float*** volume = new float** [size_grid.x];
    int*** volume_l = new int** [size_grid.x];

    for (int i = 0; i < size_grid.x; i++) {
        volume[i] = new float* [size_grid.y];
        for (int j = 0; j < size_grid.y; j++) {
            volume[i][j] = new float[size_grid.z];
            for (int k = 0; k < size_grid.z; k++) {
                volume[i][j][k] = 1.0f;

            }
        }
    }

    for (int i = 0; i < size_grid.x; i++) {
        volume_l[i] = new int* [size_grid.y];
        for (int j = 0; j < size_grid.y; j++) {
            volume_l[i][j] = new int[size_grid.z];
            for (int k = 0; k < size_grid.z; k++) {
                volume_l[i][j][k] = 0;

            }
        }
    }

    for (int i = 0; i < size_grid.x; i++) {
        for (int j = 0; j < size_grid.y; j++) {
            for (int k = 0; k < size_grid.z; k++) {

                // Get the 3D coordinate
                float3 p_curr;
                p_curr.x = (float(i) - float(size_grid.x) / 2.0f) * res_x + center_grid.x;
                p_curr.y = (float(j) - float(size_grid.y) / 2.0f) * res_y + center_grid.y;
                p_curr.z = (float(k) - float(size_grid.z) / 2.0f) * res_z + center_grid.z;

                float x = 0.0f;
                float y = 0.0f;
                float z = 1.0f;

                // Compute the smallest distance to the faces
                float min_dist = 1.0e32f;
                float sdf = 1.0f;
                int lbl = 0;
                int intersections = 0;

                for (int f = 0; f < nb_faces; f++) {
                    float3 n  = make_float3(normalsF[3 * f], normalsF[3 * f + 1], normalsF[3 * f + 2]);

                    float3 p1 = make_float3(vertices[3 * faces[3 * f + 0]], vertices[3 * faces[3 * f + 0] + 1], vertices[3 * faces[3 * f + 0] + 2]);
                    float3 p2 = make_float3(vertices[3 * faces[3 * f + 1]], vertices[3 * faces[3 * f + 1] + 1], vertices[3 * faces[3 * f + 1] + 2]);
                    float3 p3 = make_float3(vertices[3 * faces[3 * f + 2]], vertices[3 * faces[3 * f + 2] + 1], vertices[3 * faces[3 * f + 2] + 2]);

                    float curr_dist = DistancePointFace_test(p_curr, p1, p2, p3, n, false);
                    
                    if ((curr_dist < min_dist) || fabs(curr_dist - min_dist) < 10e-6) {

                        sdf = fabs(curr_dist);

                        min_dist = curr_dist;
                        lbl = labels[faces[3 * f]];
                    }

                }
                //sdf = min(1.0f, sdf / 4.0f);

                int face_idx[12] = { 0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3 };

                for (int t = 0; t < nb_tetras; t++) {

                    // get normals of each faces and see on what side of the current point it is
                    int count = 0;
                    for (int f = 0; f < 4; f++) {
                        float3 nrml = make_float3(normalsT[12 * t + 3 * f + 0], normalsT[12 * t + 3 * f + 1], normalsT[12 * t + 3 * f + 2]);
                        //cout << "nrml : " << nrml.x << ", " << nrml.y << ", " << nrml.z << endl;

                        // get points of the face : 
                        float3 p1 = make_float3(vertices[3 * tetras[4 * t + face_idx[3 * f + 0]] + 0], vertices[3 * tetras[4 * t + face_idx[3 * f + 0]] + 1], vertices[3 * tetras[4 * t + face_idx[3 * f + 0]] + 2]);
                        float3 p2 = make_float3(vertices[3 * tetras[4 * t + face_idx[3 * f + 1]] + 0], vertices[3 * tetras[4 * t + face_idx[3 * f + 1]] + 1], vertices[3 * tetras[4 * t + face_idx[3 * f + 1]] + 2]);
                        float3 p3 = make_float3(vertices[3 * tetras[4 * t + face_idx[3 * f + 2]] + 0], vertices[3 * tetras[4 * t + face_idx[3 * f + 2]] + 1], vertices[3 * tetras[4 * t + face_idx[3 * f + 2]] + 2]);


                        // what side of the face the point is on ?
                        float3 p1pc = p_curr - p1;
                        float dot_dir = dot(p1pc, nrml);

                        if (dot_dir < 0.0f) {
                            count += 1;
                        }
                    }


                    if (count == 4) {
                        // point inside sdf should be < 0
                        sdf = -sdf;
                    }
                }

                volume[i][j][k] = sdf;
                volume_l[i][j][k] = lbl;
            }
        }
    }

    return pair<float***, int***>(volume, volume_l);
}

#endif /* LevelSet_h */
