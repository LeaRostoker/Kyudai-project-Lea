#include "include_gpu/Utilities.h"

using namespace std;

void list2csv(string path, vector<vector<int>> list) {
    ofstream ofs(path);
    for (int i = 0; i < list.size(); i++){
        ofs << list[i][0];
        for (int j = 1; j < list[i].size(); j++){
            ofs << "," << list[i][j];
        }
        ofs << "\n";
    }
    ofs.close();
    return;
}

void reduce_points(float *points_out, float *points, int nb_points, int rate){
    for (int i = 0; i < nb_points/rate; i++) {
        points_out[3*i] = points[3*(rate*i)];
        points_out[3*i+1] = points[3*(rate*i)+1];
        points_out[3*i+2] = points[3*(rate*i)+2];
    }
}

vector<int> closest_nodes(my_float3 v, float *nodes, int nb_nodes, int num_nearest_nodes) {
    vector<int> adj;

    vector<pair<float,int>> distlist;

    for (int i = 0; i < nb_nodes; i++) {
        my_float3 p  = make_my_float3(nodes[3*i], nodes[3*i+1], nodes[3*i+2]);
        float dist = norm(v - p);
        if (distlist.size() < num_nearest_nodes) {
            bool added = false;
            for (int k = 0; k < distlist.size(); k++) {
                if (dist < distlist[k].first) {
                    distlist.insert(distlist.begin()+k, pair<float,int>(norm(v - p),i));
                    added = true;
                    break;
                }
            }
            if (!added)
                distlist.push_back(pair<float,int>(dist, i));
        } else {
            for (int k = 0; k < num_nearest_nodes; k++) {
                if (dist < distlist[k].first) {
                    distlist.insert(distlist.begin()+k, pair<float,int>(norm(v - p),i));
                    break;
                }
            }
        }
    }
    
    for (int k = 0; k < num_nearest_nodes; k++)
        adj.push_back(distlist[k].second);
    
    return adj;
    
}

vector<vector<int>> make_adjlist(float *nodes_s, int nb_nodes_s, float *nodes_t, int nb_nodes_t, int num_nearest_nodes, bool allow_self_connection) {
    // find adjlist for each vertex
    vector<vector<int>> adjlist;

    for (int i = 0; i < nb_nodes_t; i++) {
        my_float3 v = make_my_float3(nodes_t[3*i], nodes_t[3*i+1], nodes_t[3*i+2]);
        vector<int> adj = closest_nodes(v, nodes_s, nb_nodes_s, num_nearest_nodes);
        if(!allow_self_connection) {
            adj.erase(std::remove(adj.begin(), adj.end(), i), adj.end());
        }
        //for (int k = 0; k < 9; k++)
         //   cout << adj[k] << endl;
        adjlist.push_back(adj);
    }
    return adjlist;
}
    

float IsInterectingRayTriangle3D(my_float3 ray, my_float3 p0, my_float3 p1, my_float3 p2, my_float3 p3, my_float3 n) {
    float den = dot(ray, n);
    if (fabs(den) == 1.0e-6f)
        return 0.0f;
    
    float fact = (dot(p1 - p0, n)/den);
    if (fact < 1.0e-6f)
        return 0.0f;
    
    my_float3 proj = p0 + ray * fact;
    // Compute if proj is inside the triangle
    // V = p1 + s(p2-p1) + t(p3-p1)
    // find s and t
    my_float3 u = p2 - p1;
    my_float3 v = p3 - p1;
    my_float3 w = proj - p1;
    
    float s = (dot(u,v)*dot(w,v) - dot(v,v)*dot(w,u)) / (dot(u,v)*dot(u,v) - dot(u,u)*dot(v,v));
    float t = (dot(u,v)*dot(w,u) - dot(u,u)*dot(w,v)) / (dot(u,v)*dot(u,v) - dot(u,u)*dot(v,v));
    
    //if (t == 0.0f || s == 0.0f || s+t == 1.0f)
    //    return 0.5f;
    
    if (s >= 0.0f && t >= 0.0f && s+t <=1.0f)
        return 1.0f;
    
    return 0.0f;
}

float DistancePointFace3D(my_float3 p0, my_float3 p1, my_float3 p2, my_float3 p3, my_float3 n, bool approx) {
    my_float3 center = (p1 + p2 + p3) * (1.0f/3.0f);
    if (approx) {
        float d0 = sqrt(dot(p0-center, p0-center));
        float d1 = sqrt(dot(p0-p1, p0-p1));
        float d2 = sqrt(dot(p0-p2, p0-p2));
        float d3 = sqrt(dot(p0-p3, p0-p3));
        return min(d0, min(d1, min(d2,d3)));
    }
    
    // a. Project point onto the plane of the triangle
    my_float3 p1p0 = p0 - p1;
    float dot_prod = dot(p1p0, n);
    my_float3 proj = p0 - n*dot_prod;
    
    //p1p2p3
    my_float3 cross_p1p2p3 = cross(p2-p1, p3-p1);
    float area = norm(cross_p1p2p3)/2.0f;
    if (area < 1.0e-12) {
        return 1.0e32;
    }
    
    // b. Test if projection is inside the triangle
    my_float3 C;
    
    // edge 0 = p1p2
    my_float3 edge0 = p2 - p1;
    my_float3 vp0 = proj - p1;
    C = cross(edge0, vp0);
    float w = (norm(C) / 2.0f) / area;
    if (dot(n,C) < 0.0f) {
        // P is on the right side of edge0
        // compute distance point to segment
        float curr_dist;
        my_float3 base = edge0 * (1.0f/norm(edge0));
        float Dt = dot(base, vp0);
        if (Dt < 0.0f) {
            curr_dist = norm(p0 - p1);
        } else if (Dt > norm(edge0)) {
            curr_dist = norm(p0 - p2);
        } else {
            curr_dist = norm(p0 - (p1 + base * Dt));
        }
        return curr_dist;
    }
     
    // edge 1 = p2p3
    my_float3 edge1 = p3 - p2;
    my_float3 vp1 = proj - p2;
    C = cross(edge1, vp1);
    float u = (norm(C) / 2.0f) / area;
    if (dot(n,C) < 0.0f) {
        // P is on the right side of edge1
        // compute distance point to segment
        // compute distance point to segment
        float curr_dist;
        my_float3 base = edge1 * (1.0f/norm(edge1));
        float Dt = dot(base, vp1);
        if (Dt < 0.0f) {
            curr_dist = norm(p0 - p2);
        } else if (Dt > norm(edge1)) {
            curr_dist = norm(p0 - p3);
        } else {
            curr_dist = norm(p0 - (p2 + base * Dt));
        }
        return curr_dist;
    }
     
    // edge 2 = p3p1
    my_float3 edge2 = p1 - p3;
    my_float3 vp2 = proj - p3;
    C = cross(edge2, vp2);
    float v = (norm(C) / 2.0f) / area;
    if (dot(n,C) < 0.0f) {
        // P is on the right side of edge 2;
        float curr_dist;
        my_float3 base = edge2 * (1.0f/norm(edge2));
        float Dt = dot(base, vp2);
        if (Dt < 0.0f) {
            curr_dist = norm(p0 - p3);
        } else if (Dt > norm(edge2)) {
            curr_dist = norm(p0 - p1);
        } else {
            curr_dist = norm(p0 - (p3 + base * Dt));
        }
        return curr_dist;
    }
    
    if (u <= 1.00001f && v <= 1.00001f && w <= 1.00001f) {
        return sqrt(dot(p0-proj, p0-proj));
    } else {
        cout << "intersection case case not handled!: " << u << ", " << v << ", " << w << " (l. 146 -> Utilities.cpp)" << endl;
        return 1.0e32;
    }
    
    return 1.0e32;
}

void SkinWeightsFromFace3D(float *s_w, float *w1, float *w2, float *w3, my_float3 p0, my_float3 p1, my_float3 p2, my_float3 p3, my_float3 n) {
    // a. Project point onto the plane of the triangle
    my_float3 p1p0 = p0 - p1;
    float dot_prod = dot(p1p0, n);
    my_float3 proj = p0 - n*dot_prod;
    
    //p1p2p3
    my_float3 cross_p1p2p3 = cross(p2-p1, p3-p1);
    float area = norm(cross_p1p2p3)/2.0f;
    if (area < 1.0e-12) {
        for (int i = 0; i < 24; i++)
            s_w[i] = 0.0f;
        return;
    }
    
    // b. Test if projection is inside the triangle
    my_float3 C;
    
    // edge 0 = p1p2
    my_float3 edge0 = p2 - p1;
    my_float3 vp0 = proj - p1;
    C = cross(edge0, vp0);
    float w = (norm(C) / 2.0f) / area;
    if (dot(n,C) < 0.0f) {
        // P is on the right side of edge0
        // compute distance point to segment
        my_float3 base = edge0 * (1.0f/norm(edge0));
        float Dt = dot(base, vp0);
        if (Dt < 0.0f) {
            for (int i = 0; i < 24; i++)
                s_w[i] = w1[i];
        } else if (Dt > norm(edge0)) {
            for (int i = 0; i < 24; i++)
                s_w[i] = w2[i];
        } else {
            for (int i = 0; i < 24; i++)
                s_w[i] = (1.0f - Dt/norm(edge0)) * w1[i] + (Dt/norm(edge0)) * w2[i];
        }
        return;
    }
     
    // edge 1 = p2p3
    my_float3 edge1 = p3 - p2;
    my_float3 vp1 = proj - p2;
    C = cross(edge1, vp1);
    float u = (norm(C) / 2.0f) / area;
    if (dot(n,C) < 0.0f) {
        // P is on the right side of edge1
        // compute distance point to segment
        // compute distance point to segment
        float curr_dist;
        my_float3 base = edge1 * (1.0f/norm(edge1));
        float Dt = dot(base, vp1);
        if (Dt < 0.0f) {
            for (int i = 0; i < 24; i++)
                s_w[i] = w2[i];
        } else if (Dt > norm(edge1)) {
            for (int i = 0; i < 24; i++)
                s_w[i] = w3[i];
        } else {
            for (int i = 0; i < 24; i++)
                s_w[i] = (1.0f - Dt/norm(edge1)) * w2[i] + (Dt/norm(edge1)) * w3[i];
        }
        return;
    }
     
    // edge 2 = p3p1
    my_float3 edge2 = p1 - p3;
    my_float3 vp2 = proj - p3;
    C = cross(edge2, vp2);
    float v = (norm(C) / 2.0f) / area;
    if (dot(n,C) < 0.0f) {
        // P is on the right side of edge 2;
        float curr_dist;
        my_float3 base = edge2 * (1.0f/norm(edge2));
        float Dt = dot(base, vp2);
        if (Dt < 0.0f) {
            for (int i = 0; i < 24; i++)
                s_w[i] = w3[i];
        } else if (Dt > norm(edge2)) {
            for (int i = 0; i < 24; i++)
                s_w[i] = w1[i];
        } else {
            for (int i = 0; i < 24; i++)
                s_w[i] = (1.0f - Dt/norm(edge2)) * w3[i] + (Dt/norm(edge2)) * w1[i];
        }
        return;
    }
    
    if (u <= 1.0f && v <= 1.0f && w <= 1.0f) {
        for (int i = 0; i < 24; i++)
            s_w[i] = (u*w1[i] + v*w2[i] + w*w3[i])/(u+v+w);
    } else {
        cout << "intersection case case not handled! (l. 212 -> Utilities.cpp)" << endl;
    }
    
    return;
}

bool isOnSurface(float ***sdf, float iso, int i, int j, int k, float shift_x, float shift_y,  float shift_z) {
    // look at 0.5 distance
    
    if (shift_x == 0.0f && shift_y == 0.0f && shift_z == 0.0f) {
        float sdf_center1 = (sdf[i][j][k] + sdf[i+1][j][k] + sdf[i+1][j+1][k] + sdf[i+1][j+1][k+1] +
                                sdf[i][j+1][k] + sdf[i][j+1][k+1] + sdf[i][j][k+1] + sdf[i+1][j][k+1])/8.0f;
        
        float sdf_center2 = (sdf[i-1][j][k] + sdf[i][j][k] + sdf[i][j+1][k] + sdf[i][j+1][k+1] +
                            sdf[i-1][j+1][k] + sdf[i-1][j+1][k+1] + sdf[i-1][j][k+1] + sdf[i][j][k+1])/8.0f;
        
        float sdf_center3 = (sdf[i][j-1][k] + sdf[i+1][j-1][k] + sdf[i+1][j][k] + sdf[i+1][j][k+1] +
                            sdf[i][j][k] + sdf[i][j][k+1] + sdf[i][j-1][k+1] + sdf[i+1][j-1][k+1])/8.0f;
        
        float sdf_center4 = (sdf[i][j][k-1] + sdf[i+1][j][k-1] + sdf[i+1][j+1][k-1] + sdf[i+1][j+1][k] +
                            sdf[i][j+1][k-1] + sdf[i][j+1][k] + sdf[i][j][k] + sdf[i+1][j][k])/8.0f;
        
        float sdf_center5 = (sdf[i-1][j-1][k] + sdf[i][j-1][k] + sdf[i][j][k] + sdf[i][j][k+1] +
                            sdf[i-1][j][k] + sdf[i-1][j][k+1] + sdf[i-1][j-1][k+1] + sdf[i][j-1][k+1])/8.0f;
        
        float sdf_center6 = (sdf[i-1][j][k-1] + sdf[i][j][k-1] + sdf[i][j+1][k-1] + sdf[i][j+1][k] +
                            sdf[i-1][j+1][k-1] + sdf[i-1][j+1][k] + sdf[i-1][j][k] + sdf[i][j][k])/8.0f;
        
        float sdf_center7 = (sdf[i][j-1][k-1] + sdf[i+1][j-1][k-1] + sdf[i+1][j][k-1] + sdf[i+1][j][k] +
                            sdf[i][j][k-1] + sdf[i][j][k] + sdf[i][j-1][k] + sdf[i+1][j-1][k])/8.0f;
        
        float sdf_center8 = (sdf[i-1][j-1][k-1] + sdf[i][j-1][k-1] + sdf[i][j][k-1] + sdf[i][j][k] +
                            sdf[i-1][j][k-1] + sdf[i-1][j][k] + sdf[i-1][j-1][k] + sdf[i][j-1][k])/8.0f;
        
        return (sdf[i][j][k] <= iso &&
                (sdf_center1> iso || sdf_center2 > iso || sdf_center3 > iso ||
                 sdf_center4 > iso || sdf_center5 > iso || sdf_center6 > iso || sdf_center7 > iso || sdf_center8 > iso));
    } else {
        float sdf_center = (sdf[i][j][k] + sdf[i+1][j][k] + sdf[i+1][j+1][k] + sdf[i+1][j+1][k+1] +
                            sdf[i][j+1][k] + sdf[i][j+1][k+1] + sdf[i][j][k+1] + sdf[i+1][j][k+1])/8.0f;
        
        return (sdf_center<= iso &&
                (sdf[i][j][k] > iso || sdf[i+1][j][k] > iso || sdf[i][j+1][k] > iso || sdf[i][j][k+1] > iso ||
                 sdf[i+1][j+1][k] > iso || sdf[i+1][j+1][k+11] > iso || sdf[i][j+1][k+1] > iso || sdf[i+1][j][k+1] > iso));
    }
}

float VolumeTetra(float *Nodes, int *Tetra, int tet) {
    // get the four summits
    my_float3 s1 = make_my_float3(Nodes[3*Tetra[4*tet]], Nodes[3*Tetra[4*tet]+1], Nodes[3*Tetra[4*tet]+2]);
    my_float3 s2 = make_my_float3(Nodes[3*Tetra[4*tet+1]], Nodes[3*Tetra[4*tet+1]+1], Nodes[3*Tetra[4*tet+1]+2]);
    my_float3 s3 = make_my_float3(Nodes[3*Tetra[4*tet+2]], Nodes[3*Tetra[4*tet+2]+1], Nodes[3*Tetra[4*tet+2]+2]);
    my_float3 s4 = make_my_float3(Nodes[3*Tetra[4*tet+3]], Nodes[3*Tetra[4*tet+3]+1], Nodes[3*Tetra[4*tet+3]+2]);
    
    // compute area of base s1,s2,s3
    my_float3 n_vec = cross(s2-s1, s3-s1);
    float area_base = norm(n_vec)/2.0f;
    
    // compute height
    // project s4 on the base plan
    my_float3 n_unit = n_vec * (1.0f/area_base);
    float dot_prod = dot(s4-s1, n_unit);
    
    // for test
    //my_float3 proj = s4 - n_unit*dot_prod;
    //cout << dot(proj - s1, n_unit) << endl;
    
    return (area_base * fabs(dot_prod))/3.0f;
}

float AreaFace(float *Nodes, int *Faces, int face) {
    // get the three summits
    my_float3 s1 = make_my_float3(Nodes[3*Faces[3*face]], Nodes[3*Faces[3*face]+1], Nodes[3*Faces[3*face]+2]);
    my_float3 s2 = make_my_float3(Nodes[3*Faces[3*face+1]], Nodes[3*Faces[3*face+1]+1], Nodes[3*Faces[3*face+1]+2]);
    my_float3 s3 = make_my_float3(Nodes[3*Faces[3*face+2]], Nodes[3*Faces[3*face+2]+1], Nodes[3*Faces[3*face+2]+2]);
    
    // compute area of base s1,s2,s3
    my_float3 n_vec = cross(s2-s1, s3-s1);
    return norm(n_vec)/2.0f;
}

float AreaFace(float *Nodes, int i1, int i2, int i3) {
    // get the three summits
    my_float3 s1 = make_my_float3(Nodes[3*i1], Nodes[3*i1+1], Nodes[3*i1+2]);
    my_float3 s2 = make_my_float3(Nodes[3*i2], Nodes[3*i2+1], Nodes[3*i2+2]);
    my_float3 s3 = make_my_float3(Nodes[3*i3], Nodes[3*i3+1], Nodes[3*i3+2]);
    
    // compute area of base s1,s2,s3
    my_float3 n_vec = cross(s2-s1, s3-s1);
    return norm(n_vec)/2.0f;
}

Quaternion InvQ (Quaternion a) {
    float norm = a.value.x*a.value.x + a.value.y*a.value.y + a.value.z*a.value.z + a.value.w*a.value.w;
    if (norm == 0.0f)
        return Quaternion(0.0,0.0,0.0,0.0);
    return Quaternion(-a.value.x/norm, -a.value.y/norm, -a.value.z/norm, a.value.w/norm);
}

Eigen::Matrix4f Interpolate(Eigen::Matrix4f A, Eigen::Matrix4f B, float lambda) {
    DualQuaternion dqA = DualQuaternion(A);
    DualQuaternion dqB = DualQuaternion(B);

    DualQuaternion resQ = (dqA * (1.0f-lambda)) + (dqB * lambda);
    resQ.Normalize();
    Eigen::Matrix4f result = resQ.DualQuaternionToMatrix();
    return result;

}

Eigen::Matrix4f InverseTransfo(Eigen::Matrix4f A) {
    Eigen::Matrix3f rot;
    rot << A(0,0), A(0,1), A(0,2),
            A(1,0), A(1,1), A(1,2),
            A(2,0), A(2,1), A(2,2);
    Eigen::Vector3f trans;
    trans << A(0,3), A(1,3), A(2,3);
    
    Eigen::Matrix3f rot_inv = rot.inverse();
    Eigen::Vector3f trans_inv = - rot_inv * trans;
    
    Eigen::Matrix4f res;
    res << rot_inv(0,0), rot_inv(0,1), rot_inv(0,2), trans_inv(0),
            rot_inv(1,0), rot_inv(1,1), rot_inv(1,2), trans_inv(1),
            rot_inv(2,0), rot_inv(2,1), rot_inv(2,2), trans_inv(2),
            0.0f, 0.0f, 0.0f, 1.0f;
    
    return res;
}

void matrix2quaternion(Eigen::Matrix3f currR, float *res) {
    //float t = currR(0,0)+currR(1,1)+currR(2,2);
    if (currR(0,0) > currR(1,1) && currR(0,0) > currR(2,2)) {
        float r = sqrt(1+currR(0,0)-currR(1,1)-currR(2,2));
        float s = 0.5f/r;
        float w = (currR(2,1)-currR(1,2))*s;
        float x = 0.5f*r;
        float y = (currR(0,1)+currR(1,0))*s;
        float z = (currR(2,0)+currR(0,2))*s;
        
        res[0] = w;
        res[1] = x;
        res[2] = y;
        res[3] = z;
        return;
    }

    if (currR(1,1) > currR(0,0) && currR(1,1) > currR(2,2)) {
        float r = sqrt(1+currR(1,1)-currR(0,0)-currR(2,2));
        float s = 0.5f/r;
        float w = (currR(2,0)-currR(0,2))*s;
        float x = (currR(0,1)+currR(1,0))*s;
        float y = 0.5f*r;
        float z = (currR(2,1)+currR(1,2))*s;
        
        res[0] = w;
        res[1] = x;
        res[2] = y;
        res[3] = z;
        return;
    }

    if (currR(2,2) > currR(0,0) && currR(2,2) > currR(1,1)) {
        float r = sqrt(1+currR(2,2)-currR(0,0)-currR(1,1));
        float s = 0.5f/r;
        float w = (currR(1,0)-currR(0,1))*s;
        float x = (currR(0,2)+currR(2,0))*s;
        float y = (currR(2,1)+currR(1,2))*s;
        float z = 0.5f*r;
        
        res[0] = w;
        res[1] = x;
        res[2] = y;
        res[3] = z;
        return;
    }


    float qw = sqrt(1.0f + currR(0,0) + currR(1,1) + currR(2,2)) / 2.0;
    float qx = (currR(2,1) - currR(1,2))/( 4.0f * qw);
    float qy = (currR(0,2) - currR(2,0))/( 4.0f * qw);
    float qz = (currR(1,0) - currR(0,1))/( 4.0f * qw);

    res[0] = qw;
    res[1] = qx;
    res[2] = qy;
    res[3] = qz;
}
//q = q_i i  + q_j j + q_k k + q_r
Eigen::Matrix4f quaternion2matrix(float *q) {

    float nq = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
    if (nq < 1.0e-6){
        Eigen::Matrix4f resMat;
        resMat << 1.0, 0.0, 0.0, q[4],
                0.0, 1.0, 0.0, q[5],
                0.0, 0.0, 1.0, q[6],
                0.0, 0.0, 0.0, 1.0;
        return resMat;
    }
    
    for (int i = 0; i < 4; i++)
        q[i] = q[i] * sqrt(2.0f / nq);
    
    float q_tmp[16];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            q_tmp[4*i+j] = q[i]*q[j];
        }
    }
    
    /*res[0] = 1.0-q_tmp[1*4+1]-q_tmp[2*4+2];        res[4] = q_tmp[0*4+1]-q_tmp[2*4+3];            res[8] = q_tmp[0*4+2]+q_tmp[1*4+3];            res[12] = q[4];
    res[1] = q_tmp[0*4+1]+q_tmp[2*4+3];            res[5] = 1.0-q_tmp[0*4+0]-q_tmp[2*4+2];        res[9] = q_tmp[1*4+2]-q_tmp[0*4+3];            res[13] = q[5];
    res[2] = q_tmp[0*4+2]-q_tmp[1*4+3];            res[6] = q_tmp[1*4+2]+q_tmp[0*4+3];            res[10] = 1.0-q_tmp[0*4+0]-q_tmp[1*4+1];    res[14] = q[6];
    res[3] = 0.0;                                res[7] = 0.0;                                res[11] = 0.0;                                res[15] = 1.0;*/
    Eigen::Matrix4f resMat;
    resMat << 1.0f-q_tmp[1*4+1]-q_tmp[2*4+2], q_tmp[0*4+1]-q_tmp[2*4+3], q_tmp[0*4+2]+q_tmp[1*4+3], q[4],
                q_tmp[0*4+1]+q_tmp[2*4+3], 1.0-q_tmp[0*4+0]-q_tmp[2*4+2], q_tmp[1*4+2]-q_tmp[0*4+3], q[5],
            q_tmp[0*4+2]-q_tmp[1*4+3], q_tmp[1*4+2]+q_tmp[0*4+3], 1.0f-q_tmp[0*4+0]-q_tmp[1*4+1], q[6],
            0.0f, 0.0f, 0.0f, 1.0f;
    return resMat;
    
}


void quaternion2matrix(float *q, double *res) {

    double nq = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
    if (nq < 1.0e-6){
        res[0] = 1.0; res[4] = 0.0; res[8] = 0.0; res[12] = 0.0;
        res[1] = 0.0; res[5] = 1.0; res[9] = 0.0; res[13] = 0.0;
        res[2] = 0.0; res[6] = 0.0; res[10] = 1.0; res[14] = 0.0;
        res[3] = 0.0; res[7] = 0.0; res[11] = 0.0; res[15] = 1.0;
        return;
    }
    
    for (int i = 0; i < 4; i++)
        q[i] = q[i] * float(sqrt(2.0f / nq));
    
    double q_tmp[16];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            q_tmp[4*i+j] = q[i]*q[j];
        }
    }
    
    res[0] = 1.0-q_tmp[1*4+1]-q_tmp[2*4+2];        res[4] = q_tmp[0*4+1]-q_tmp[2*4+3];            res[8] = q_tmp[0*4+2]+q_tmp[1*4+3];            res[12] = q[4];
    res[1] = q_tmp[0*4+1]+q_tmp[2*4+3];            res[5] = 1.0-q_tmp[0*4+0]-q_tmp[2*4+2];        res[9] = q_tmp[1*4+2]-q_tmp[0*4+3];            res[13] = q[5];
    res[2] = q_tmp[0*4+2]-q_tmp[1*4+3];            res[6] = q_tmp[1*4+2]+q_tmp[0*4+3];            res[10] = 1.0-q_tmp[0*4+0]-q_tmp[1*4+1];    res[14] = q[6];
    res[3] = 0.0;                                res[7] = 0.0;                                res[11] = 0.0;                                res[15] = 1.0;
    
}

void quaternion2matrix(double *q, double *res) {

    double nq = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
    if (nq < 1.0e-6){
        res[0] = 1.0; res[4] = 0.0; res[8] = 0.0; res[12] = 0.0;
        res[1] = 0.0; res[5] = 1.0; res[9] = 0.0; res[13] = 0.0;
        res[2] = 0.0; res[6] = 0.0; res[10] = 1.0; res[14] = 0.0;
        res[3] = 0.0; res[7] = 0.0; res[11] = 0.0; res[15] = 1.0;
        return;
    }
    
    for (int i = 0; i < 4; i++)
        q[i] = q[i] * sqrt(2.0 / nq);
    
    double q_tmp[16];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            q_tmp[4*i+j] = q[i]*q[j];
        }
    }
    
    res[0] = 1.0-q_tmp[1*4+1]-q_tmp[2*4+2];        res[4] = q_tmp[0*4+1]-q_tmp[2*4+3];            res[8] = q_tmp[0*4+2]+q_tmp[1*4+3];            res[12] = q[4];
    res[1] = q_tmp[0*4+1]+q_tmp[2*4+3];            res[5] = 1.0-q_tmp[0*4+0]-q_tmp[2*4+2];        res[9] = q_tmp[1*4+2]-q_tmp[0*4+3];            res[13] = q[5];
    res[2] = q_tmp[0*4+2]-q_tmp[1*4+3];            res[6] = q_tmp[1*4+2]+q_tmp[0*4+3];            res[10] = 1.0-q_tmp[0*4+0]-q_tmp[1*4+1];    res[14] = q[6];
    res[3] = 0.0;                                res[7] = 0.0;                                res[11] = 0.0;                                res[15] = 1.0;
    
}

Eigen::Matrix4f euler2matrix(float Rx, float Ry, float Rz) {
    Eigen::Matrix4f RotX;
    RotX << 1.0, 0.0, 0.0, 0.0,
            0.0, cos(Rx), -sin(Rx), 0.0,
            0.0, sin(Rx), cos(Rx), 0.0,
            0.0, 0.0, 0.0, 1.0;
            
    Eigen::Matrix4f RotY;
    RotY << cos(Ry), 0.0, sin(Ry), 0.0,
            0.0, 1.0, 0.0, 0.0,
            -sin(Ry), 0.0, cos(Ry), 0.0,
            0.0, 0.0, 0.0, 1.0;
            
    Eigen::Matrix4f RotZ;
    RotZ << cos(Rz), -sin(Rz),0.0, 0.0,
            sin(Rz), cos(Rz), 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0;

    return RotZ * RotY * RotX;
}


Eigen::Matrix3f rodrigues2matrix(float *Rodrigues) {
    Eigen::Matrix3f Rotation = Eigen::MatrixXf::Identity(3, 3);
    
    float theta = sqrt(Rodrigues[0]*Rodrigues[0] + Rodrigues[1]*Rodrigues[1] + Rodrigues[2]*Rodrigues[2]);
    if (theta == 0.0f)
        return Rotation;
    
    Eigen::Vector3f u;
    u << Rodrigues[0]/theta, Rodrigues[1]/theta, Rodrigues[2]/theta;

    Eigen::Matrix3f u_skew;
    u_skew << 0.0f, -u(2), u(1),
                u(2), 0.0f, -u(0),
                -u(1), u(0), 0.0f;

    Rotation = cos(theta) * Eigen::MatrixXf::Identity(3, 3) + (1.0f - cos(theta))*u*u.transpose() + sin(theta) * u_skew;

    return Rotation;

    /*for (int i = 0; i<3; i++) {
        for (int i = 0; i<3; i++) {
            float p_k = i != 0 && j != 0 ? 0 : 1;
            if (i != 2 && j != 2)
                p_k = 2;
            Rotation(i,j) = i==j ? (1.0f/(1.0f + p_l*p_l)) * ( (1.0f-p_l*p_l) + 2*0f*Rodriguez[i]*Rodriguez[j] + 2.0f*eps[i,j,k]*Rodriguez[p_k] ) :
                                    (1.0f/(1.0f + p_l*p_l)) * ( 2*0f*Rodriguez[i]*Rodriguez[j] + 2.0f*eps[i,j,k]*Rodriguez[p_k] );
        }
    }

    return Rotation;*/
}


// Function to get cofactor of A[p][q] in temp[][]. n is current
// dimension of A[][]
void getCofactor(float * A, float * temp, int p, int q, int n)
{
    int i = 0, j = 0;
  
    // Looping for each element of the matrix
    for (int row = 0; row < n; row++)
    {
        for (int col = 0; col < n; col++)
        {
            //  Copying into temporary matrix only those element
            //  which are not in given row and column
            if (row != p && col != q)
            {
                temp[i*(n-1) + j++] = A[row*n + col];
  
                // Row is filled, so increase row index and
                // reset col index
                if (j == n - 1)
                {
                    j = 0;
                    i++;
                }
            }
        }
    }
}
  
/* Recursive function for finding determinant of matrix.
   n is current dimension of A[][].
 The method is the Cofactor expansion*/
float determinant(float * A, int n)
{
    float D = 0.0f; // Initialize result
  
    //  Base case : if matrix contains single element
    if (n == 1)
        return A[0];
  
    //int temp[N][N]; // To store cofactors
    float *temp = new float[(n-1)*(n-1)]; // To store cofactors
  
    float sign = 1.0f;  // To store sign multiplier
  
     // Iterate for each element of first row
    for (int f = 0; f < n; f++)
    {
        // Getting Cofactor of A[0][f]
        getCofactor(A, temp, 0, f, n);
        D += sign * A[f] * determinant(temp, n - 1);
  
        // terms are to be added with alternate sign
        sign = -sign;
    }
    delete[] temp;
  
    return D;
}
  
// Function to get adjoint of A[N][N] in adj[N][N].
void adjoint(float * A, float *adj, int N)
{
    if (N == 1)
    {
        adj[0] = 1.0f;
        return;
    }
  
    // temp is used to store cofactors of A[][]
    float sign = 1.0f;
    float *temp = new float[N*N];
    //int sign = 1, temp[N][N];
  
    for (int i=0; i<N; i++)
    {
        for (int j=0; j<N; j++)
        {
            // Get cofactor of A[i][j]
            getCofactor(A, temp, i, j, N);
  
            // sign of adj[j][i] positive if sum of row
            // and column indexes is even.
            sign = ((i+j)%2==0.0f)? 1.0f: -1.0f;
  
            // Interchanging rows and columns to get the
            // transpose of the cofactor matrix
            adj[j*N+i] = (sign)*(determinant(temp, N-1));
        }
    }
    delete[] temp;
}
  
// Function to calculate and store inverse, returns false if
// matrix is singular
bool inverse(float * A, float *inverse, int N)
{
    // Find determinant of A[][]
    float det = determinant(A, N);
    if (det == 0.0f)
    {
        cout << "Singular matrix, can't find its inverse";
        return false;
    }
  
    // Find adjoint
    float * adj = new float[N*N];
    adjoint(A, adj, N);
  
    // Find Inverse using formula "inverse(A) = adj(A)/det(A)"
    for (int i=0; i<N; i++)
        for (int j=0; j<N; j++)
            inverse[i*N + j] = adj[i*N + j]/det;
  
    delete[] adj;
    return true;
}
  
// Generic function to display the matrix.  We use it to display
// both adjoin and inverse. adjoin is integer matrix and inverse
// is a float.
void display(float * A, int N)
{
    for (int i=0; i<N; i++)
    {
        for (int j=0; j<N; j++)
            cout << A[i*N + j] << " ";
        cout << endl;
    }
}


my_float3 BicubiInterpolation(my_float3 vertex, float ***tsdf_grid, my_float3 ***grad_grid, my_int3 dim_grid, my_float3 center, float res, float iso) {
    
    my_int3 vox = make_my_int3(floorf((vertex.x-center.x)/res) + dim_grid.x/2.0f, floorf((vertex.y-center.y)/res) + dim_grid.y/2.0f, floorf((vertex.z-center.z)/res) + dim_grid.z/2.0f);
    
    //vox.print();
    if (vox.x < 0 || vox.x >= dim_grid.x || vox.y < 0 || vox.y >= dim_grid.y || vox.z < 0 || vox.z >= dim_grid.z) {
        cout << "out" << endl;
        return make_my_float3(0.0f);
    }
    
    if (vox.x == dim_grid.x-1 || vox.x == dim_grid.x-2)
        vox.x = dim_grid.x-3;
    
    if (vox.y == dim_grid.y-1 || vox.y == dim_grid.y-2)
        vox.y = dim_grid.y-3;
    
    if (vox.z == dim_grid.z-1 || vox.z == dim_grid.z-2)
        vox.z = dim_grid.z-3;
    
    if (vox.x == 0 )
        vox.x = 1;
    
    if (vox.y == 0 )
        vox.y = 1;
    
    if (vox.z == 0 )
        vox.z = 1;
    
    
    /*float norm_grad = norm(grad_grid[vox.x][vox.y][vox.z]);
    if (norm_grad > 0.0f)
        return grad_grid[vox.x][vox.y][vox.z] * (tsdf_grid[vox.x][vox.y][vox.z]/norm_grad);
    return make_my_float3(0.0f);*/
    
    float f1 = exp(-100.f*norm(vertex - make_my_float3(center.x + (float(vox.x) - float(dim_grid.x)/2.0f)*res,
                                            center.y + (float(vox.y) - float(dim_grid.y)/2.0f)*res,
                                            center.z + (float(vox.z) - float(dim_grid.z)/2.0f)*res)));
    float f2 = exp(-100.f*norm(vertex - make_my_float3(center.x + (float(vox.x) - float(dim_grid.x)/2.0f)*res,
                                            center.y + (float(vox.y+1) - float(dim_grid.y)/2.0f)*res,
                                            center.z + (float(vox.z) - float(dim_grid.z)/2.0f)*res)));
    float f3 = exp(-100.f*norm(vertex - make_my_float3(center.x + (float(vox.x) - float(dim_grid.x)/2.0f)*res,
                                            center.y + (float(vox.y+1) - float(dim_grid.y)/2.0f)*res,
                                            center.z + (float(vox.z+1) - float(dim_grid.z)/2.0f)*res)));
    float f4 = exp(-100.f*norm(vertex - make_my_float3(center.x + (float(vox.x) - float(dim_grid.x)/2.0f)*res,
                                            center.y + (float(vox.y) - float(dim_grid.y)/2.0f)*res,
                                            center.z + (float(vox.z+1) - float(dim_grid.z)/2.0f)*res)));
    float f5 = exp(-100.f*norm(vertex - make_my_float3(center.x + (float(vox.x+1) - float(dim_grid.x)/2.0f)*res,
                                            center.y + (float(vox.y) - float(dim_grid.y)/2.0f)*res,
                                            center.z + (float(vox.z) - float(dim_grid.z)/2.0f)*res)));
    float f6 = exp(-100.f*norm(vertex - make_my_float3(center.x + (float(vox.x+1) - float(dim_grid.x)/2.0f)*res,
                                            center.y + (float(vox.y+1) - float(dim_grid.y)/2.0f)*res,
                                            center.z + (float(vox.z) - float(dim_grid.z)/2.0f)*res)));
    float f7 = exp(-100.f*norm(vertex - make_my_float3(center.x + (float(vox.x+1) - float(dim_grid.x)/2.0f)*res,
                                            center.y + (float(vox.y+1) - float(dim_grid.y)/2.0f)*res,
                                            center.z + (float(vox.z+1) - float(dim_grid.z)/2.0f)*res)));
    float f8 = exp(-100.f*norm(vertex - make_my_float3(center.x + (float(vox.x+1) - float(dim_grid.x)/2.0f)*res,
                                            center.y + (float(vox.y) - float(dim_grid.y)/2.0f)*res,
                                            center.z + (float(vox.z+1) - float(dim_grid.z)/2.0f)*res)));
    
    my_float3 grad_val = (grad_grid[vox.x][vox.y][vox.z]*f1 + grad_grid[vox.x][vox.y+1][vox.z]*f2 + grad_grid[vox.x][vox.y+1][vox.z+1]*f3 + grad_grid[vox.x][vox.y][vox.z+1]*f4 +
                          grad_grid[vox.x+1][vox.y][vox.z]*f5 + grad_grid[vox.x+1][vox.y+1][vox.z]*f6 + grad_grid[vox.x+1][vox.y+1][vox.z+1]*f7 + grad_grid[vox.x+1][vox.y][vox.z+1]*f8) * (1.0f/(f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8));
    
    float norm_grad = norm(grad_val);
    
    float tsdf_val = min( 1.0f, (tsdf_grid[vox.x][vox.y][vox.z]*f1 + tsdf_grid[vox.x][vox.y+1][vox.z]*f2 + tsdf_grid[vox.x][vox.y+1][vox.z+1]*f3 + tsdf_grid[vox.x][vox.y][vox.z+1]*f4 +
                      tsdf_grid[vox.x+1][vox.y][vox.z]*f5 + tsdf_grid[vox.x+1][vox.y+1][vox.z]*f6 + tsdf_grid[vox.x+1][vox.y+1][vox.z+1]*f7 + tsdf_grid[vox.x+1][vox.y][vox.z+1]*f8) * (1.0f/(f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8)) - iso );
        
    if (norm_grad > 0.0f /*&& fabs(tsdf_val) > 0.05f*/)
        return grad_val * (tsdf_val/norm_grad);
    return make_my_float3(0.0f);
    
}

my_float3 ***VolumetricGrad(float ***tsdf_grid, my_int3 dim_grid) {
    
    my_float3 ***grad_grid;
    grad_grid = new my_float3 **[dim_grid.x];
    for (int i = 1; i < dim_grid.x-1; i++) {
        grad_grid[i] = new my_float3 *[dim_grid.y];
        for (int j = 1; j < dim_grid.y-1; j++) {
            grad_grid[i][j] = new my_float3[dim_grid.z];
            for (int k = 1; k < dim_grid.z-1; k++) {
                // Compute normals at grid vertices
                grad_grid[i][j][k] = make_my_float3(tsdf_grid[i-1][j][k] - tsdf_grid[i+1][j][k],
                                                   tsdf_grid[i][j-1][k] - tsdf_grid[i][j+1][k],
                                                   tsdf_grid[i][j][k-1] - tsdf_grid[i][j][k+1]);
            }
        }
    }
    return grad_grid;
}


float ScTP(my_float3 a, my_float3 b, my_float3 c)
{
    // computes scalar triple product
    return dot(a, cross(b, c));
}

