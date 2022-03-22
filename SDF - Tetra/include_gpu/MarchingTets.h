//
//  MarchingTets.h
//  DEEPANIM
//
//  Created by Diego Thomas on 2021/01/20.
//

#ifndef MarchingTets_h
#define MarchingTets_h

int MT_Table[6 * 16] = { -1,-1,-1,-1,-1,-1, //0 no triangle
                    0,1,2,-1,-1,-1,     //1 (ab,ac,ad)
                    0,4,3,-1,-1,-1,   //2 (ab,bd,bc)
                    1,3,2,2,3,4,        //5 (ac,bc,ad/ad,bc,bd)
                    3,1,5,-1,-1,-1,     //3 (bc,ac,cd)
                    0,3,2,2,3,5,        //6 (ab,bc,ad / ad,bc,cd)
                    0,1,5,0,5,4,        //7 (ab,ac,cd / ab,cd,bd)
                    2,4,5,-1,-1,-1,     //-4 (ad,bd,cd)
                    2,5,4,-1,-1,-1,     //-4 (ad,cd,bd)
                    0,2,5,0,5,3,        //-7 (ab,cd,ac / ab,bd,cd)
                    0,2,3,2,5,3,        //-6 (ab,ad,bc / ad,cd,bc)
                    3,5,1,-1,-1,-1,     //-3 (bc,cd,ac)
                    1,2,3,2,4,3,        //-5 (ac,ad,bc/ad,bd,bc)
                    0,3,4,-1,-1,-1,   //-2 (ab,bc,bd)
                    0,1,2,-1,-1,-1,     //-1 (ab,ad,ac)
                    -1,-1,-1,-1,-1,-1 };


void GenerateFaces(float *TSDF, int *Edges_row_ptr, int *Edges_columns, int *Tetra, int nb_tets, int *Faces, float m_iso = 0.0) {
    
    for (int voxid = 0; voxid < nb_tets; voxid++) {
        float a, b, c, d; //4 summits if the tetrahedra voxel

        //Value of the TSDF
        a = TSDF[Tetra[voxid * 4]];
        b = TSDF[Tetra[voxid * 4 + 1]];
        c = TSDF[Tetra[voxid * 4 + 2]];
        d = TSDF[Tetra[voxid * 4 + 3]];

        //printf("TSDF : %f, %f, %f, %f\n", a.w, b.w, c.w, d.w);

        int count = 0;
        if (a >= m_iso)
            count += 1;
        if (b >= m_iso)
            count += 1;
        if (c >= m_iso)
            count += 1;
        if (d >= m_iso)
            count += 1;

        /*if (fabs(a) == 1.0f || fabs(b) == 1.0f || fabs(c) == 1.0f || fabs(d) == 1.0f)
            count = 0;*/

        if (count == 0 || count == 4) //return;
        {
            Faces[6 * (voxid)+0] = 0;
            Faces[6 * (voxid)+1] = 0;
            Faces[6 * (voxid)+2] = 0;

            Faces[6 * (voxid)+3] = 0;
            Faces[6 * (voxid)+4] = 0;
            Faces[6 * (voxid)+5] = 0;
        }

        //! Three vertices are inside the volume
        else if (count == 3) {
            my_int2 list[6] = { make_my_int2(0,1), make_my_int2(0,2), make_my_int2(0,3), make_my_int2(1,2), make_my_int2(1,3), make_my_int2(2,3) };
            //! Make sure that fourth value lies outside
            if (d < m_iso)
            {
            }
            else if (c < m_iso)
            {
                list[0] = make_my_int2(0, 3);
                list[1] = make_my_int2(0, 1);
                list[2] = make_my_int2(0, 2);
                list[3] = make_my_int2(1, 3);
                list[4] = make_my_int2(2, 3);
                list[5] = make_my_int2(1, 2);
            }
            else if (b < m_iso)
            {
                list[0] = make_my_int2(0, 2);
                list[1] = make_my_int2(0, 3);
                list[2] = make_my_int2(0, 1);
                list[3] = make_my_int2(2, 3);
                list[4] = make_my_int2(1, 2);
                list[5] = make_my_int2(1, 3);
            }
            else
            {
                list[0] = make_my_int2(1, 3);
                list[1] = make_my_int2(1, 2);
                list[2] = make_my_int2(0, 1);
                list[3] = make_my_int2(2, 3);
                list[4] = make_my_int2(0, 3);
                list[5] = make_my_int2(0, 2);
            }
            //ad
            int sum1 = Tetra[voxid * 4 + list[2].x] < Tetra[voxid * 4 + list[2].y] ? Tetra[voxid * 4 + list[2].x] : Tetra[voxid * 4 + list[2].y];
            int sum2 = Tetra[voxid * 4 + list[2].x] < Tetra[voxid * 4 + list[2].y] ? Tetra[voxid * 4 + list[2].y] : Tetra[voxid * 4 + list[2].x];
            int idx_ad = 0;
            for (int k = Edges_row_ptr[sum1]; k < Edges_row_ptr[sum1 + 1]; k++) {
                if (Edges_columns[k] == sum2) {
                    idx_ad = k;
                    break;
                }
            }

            //bd
            sum1 = Tetra[voxid * 4 + list[4].x] < Tetra[voxid * 4 + list[4].y] ? Tetra[voxid * 4 + list[4].x] : Tetra[voxid * 4 + list[4].y];
            sum2 = Tetra[voxid * 4 + list[4].x] < Tetra[voxid * 4 + list[4].y] ? Tetra[voxid * 4 + list[4].y] : Tetra[voxid * 4 + list[4].x];
            int idx_bd = 0;
            for (int k = Edges_row_ptr[sum1]; k < Edges_row_ptr[sum1 + 1]; k++) {
                if (Edges_columns[k] == sum2) {
                    idx_bd = k;
                    break;
                }
            }

            //cd
            sum1 = Tetra[voxid * 4 + list[5].x] < Tetra[voxid * 4 + list[5].y] ? Tetra[voxid * 4 + list[5].x] : Tetra[voxid * 4 + list[5].y];
            sum2 = Tetra[voxid * 4 + list[5].x] < Tetra[voxid * 4 + list[5].y] ? Tetra[voxid * 4 + list[5].y] : Tetra[voxid * 4 + list[5].x];
            int idx_cd = 0;
            for (int k = Edges_row_ptr[sum1]; k < Edges_row_ptr[sum1 + 1]; k++) {
                if (Edges_columns[k] == sum2) {
                    idx_cd = k;
                    break;
                }
            }

            Faces[6 * (voxid)+0] = idx_ad;
            Faces[6 * (voxid)+1] = idx_cd;
            Faces[6 * (voxid)+2] = idx_bd;
            if (idx_ad == idx_cd || idx_ad == idx_bd || idx_cd == idx_bd) {
                cout << "!! : " << voxid << endl;
            }

            Faces[6 * (voxid)+3] = 0;
            Faces[6 * (voxid)+4] = 0;
            Faces[6 * (voxid)+5] = 0;

        }

        //! Two vertices are inside the volume
        else if (count == 2) {
            //! Make sure that the last two points lie outside
            my_int2 list[6] = { make_my_int2(0,1), make_my_int2(0,2), make_my_int2(0,3), make_my_int2(1,2), make_my_int2(1,3), make_my_int2(2,3) };
            if (a >= m_iso && b >= m_iso)
            {
            }
            else if (a >= m_iso && c >= m_iso)
            {
                list[0] = make_my_int2(0, 2);
                list[1] = make_my_int2(0, 3);
                list[2] = make_my_int2(0, 1);
                list[3] = make_my_int2(2, 3);
                list[4] = make_my_int2(1, 2);
                list[5] = make_my_int2(1, 3);
            }
            else if (a >= m_iso && d >= m_iso)
            {
                list[0] = make_my_int2(0, 3);
                list[1] = make_my_int2(0, 1);
                list[2] = make_my_int2(0, 2);
                list[3] = make_my_int2(1, 3);
                list[4] = make_my_int2(2, 3);
                list[5] = make_my_int2(1, 2);
            }
            else if (b >= m_iso && c >= m_iso)
            {
                list[0] = make_my_int2(1, 2);
                list[1] = make_my_int2(0, 1);
                list[2] = make_my_int2(1, 3);
                list[3] = make_my_int2(0, 2);
                list[4] = make_my_int2(2, 3);
                list[5] = make_my_int2(0, 3);
            }
            else if (b >= m_iso && d >= m_iso)
            {
                list[0] = make_my_int2(1, 3);
                list[1] = make_my_int2(1, 2);
                list[2] = make_my_int2(0, 1);
                list[3] = make_my_int2(2, 3);
                list[4] = make_my_int2(0, 3);
                list[5] = make_my_int2(0, 2);
            }
            else //c && d > m_iso
            {
                list[0] = make_my_int2(2, 3);
                list[1] = make_my_int2(0, 2);
                list[2] = make_my_int2(1, 2);
                list[3] = make_my_int2(0, 3);
                list[4] = make_my_int2(1, 3);
                list[5] = make_my_int2(0, 1);
            }

            //ac
            int sum1 = Tetra[voxid * 4 + list[1].x] < Tetra[voxid * 4 + list[1].y] ? Tetra[voxid * 4 + list[1].x] : Tetra[voxid * 4 + list[1].y];
            int sum2 = Tetra[voxid * 4 + list[1].x] < Tetra[voxid * 4 + list[1].y] ? Tetra[voxid * 4 + list[1].y] : Tetra[voxid * 4 + list[1].x];
            int idx_ac = 0;
            for (int k = Edges_row_ptr[sum1]; k < Edges_row_ptr[sum1 + 1]; k++) {
                if (Edges_columns[k] == sum2) {
                    idx_ac = k;
                    break;
                }
            }

            //ad
            sum1 = Tetra[voxid * 4 + list[2].x] < Tetra[voxid * 4 + list[2].y] ? Tetra[voxid * 4 + list[2].x] : Tetra[voxid * 4 + list[2].y];
            sum2 = Tetra[voxid * 4 + list[2].x] < Tetra[voxid * 4 + list[2].y] ? Tetra[voxid * 4 + list[2].y] : Tetra[voxid * 4 + list[2].x];
            int idx_ad = 0;
            for (int k = Edges_row_ptr[sum1]; k < Edges_row_ptr[sum1 + 1]; k++) {
                if (Edges_columns[k] == sum2) {
                    idx_ad = k;
                    break;
                }
            }

            //bc
            sum1 = Tetra[voxid * 4 + list[3].x] < Tetra[voxid * 4 + list[3].y] ? Tetra[voxid * 4 + list[3].x] : Tetra[voxid * 4 + list[3].y];
            sum2 = Tetra[voxid * 4 + list[3].x] < Tetra[voxid * 4 + list[3].y] ? Tetra[voxid * 4 + list[3].y] : Tetra[voxid * 4 + list[3].x];
            int idx_bc = 0;
            for (int k = Edges_row_ptr[sum1]; k < Edges_row_ptr[sum1 + 1]; k++) {
                if (Edges_columns[k] == sum2) {
                    idx_bc = k;
                    break;
                }
            }

            //bd
            sum1 = Tetra[voxid * 4 + list[4].x] < Tetra[voxid * 4 + list[4].y] ? Tetra[voxid * 4 + list[4].x] : Tetra[voxid * 4 + list[4].y];
            sum2 = Tetra[voxid * 4 + list[4].x] < Tetra[voxid * 4 + list[4].y] ? Tetra[voxid * 4 + list[4].y] : Tetra[voxid * 4 + list[4].x];
            int idx_bd = 0;
            for (int k = Edges_row_ptr[sum1]; k < Edges_row_ptr[sum1 + 1]; k++) {
                if (Edges_columns[k] == sum2) {
                    idx_bd = k;
                    break;
                }
            }

            // storeTriangle(ac,bc,ad);
            Faces[6 * (voxid)+0] = idx_ac;
            Faces[6 * (voxid)+1] = idx_bc;
            Faces[6 * (voxid)+2] = idx_ad;
            if (idx_ac == idx_bc || idx_ac == idx_ad || idx_bc == idx_ad) {
                cout << "!! : " << voxid << ", " << idx_ac << ", " << idx_bc << ", " << idx_ad << endl;
                cout << "tetra : " << Tetra[voxid * 4] << ", " << Tetra[voxid * 4+1] << ", " << Tetra[voxid * 4+2] << ", " << Tetra[voxid * 4+3] << endl;
                cout << "list : " << list[0].x << ", " << list[1].x << ", " << list[2].x << ", " << list[3].x << ", " << list[4].x << endl;
                cout << "list : " << list[0].y << ", " << list[1].y << ", " << list[2].y << ", " << list[3].y << ", " << list[4].y << endl;
            }

            //storeTriangle(bc,bd,ad);
            Faces[6 * (voxid)+3] = idx_bc;
            Faces[6 * (voxid)+4] = idx_bd;
            Faces[6 * (voxid)+5] = idx_ad;
            if (idx_bc == idx_bd || idx_bc == idx_ad || idx_bd == idx_ad) {
                cout << "!! : " << idx_bc << ", " << idx_bd << ", " << idx_ad << endl;
                cout << "tetra : " << Tetra[voxid * 4] << ", " << Tetra[voxid * 4+1] << ", " << Tetra[voxid * 4+2] << ", " << Tetra[voxid * 4+3] << endl;
                cout << "list : " << list[0].x << ", " << list[1].x << ", " << list[2].x << ", " << list[3].x << ", " << list[4].x << endl;
                cout << "list : " << list[0].y << ", " << list[1].y << ", " << list[2].y << ", " << list[3].y << ", " << list[4].y << endl;
            }
        }
        //! One vertex is inside the volume
        else if (count == 1) {
            //! Make sure that the last three points lie outside
            my_int2 list[6] = { make_my_int2(0,1), make_my_int2(0,2), make_my_int2(0,3), make_my_int2(1,2), make_my_int2(1,3), make_my_int2(2,3) };
            if (a >= m_iso)
            {
            }
            else if (b >= m_iso)
            {
                list[0] = make_my_int2(1, 2);
                list[1] = make_my_int2(0, 1);
                list[2] = make_my_int2(1, 3);
                list[3] = make_my_int2(0, 2);
                list[4] = make_my_int2(2, 3);
                list[5] = make_my_int2(0, 3);
            }
            else if (c >= m_iso)
            {
                list[0] = make_my_int2(0, 2);
                list[1] = make_my_int2(1, 2);
                list[2] = make_my_int2(2, 3);
                list[3] = make_my_int2(0, 1);
                list[4] = make_my_int2(0, 3);
                list[5] = make_my_int2(1, 3);
            }
            else // d > m_iso
            {
                list[0] = make_my_int2(2, 3);
                list[1] = make_my_int2(1, 3);
                list[2] = make_my_int2(0, 3);
                list[3] = make_my_int2(1, 2);
                list[4] = make_my_int2(0, 2);
                list[5] = make_my_int2(0, 1);
            }

            //ab
            int sum1 = Tetra[voxid * 4 + list[0].x] < Tetra[voxid * 4 + list[0].y] ? Tetra[voxid * 4 + list[0].x] : Tetra[voxid * 4 + list[0].y];
            int sum2 = Tetra[voxid * 4 + list[0].x] < Tetra[voxid * 4 + list[0].y] ? Tetra[voxid * 4 + list[0].y] : Tetra[voxid * 4 + list[0].x];
            int idx_ab = 0;
            for (int k = Edges_row_ptr[sum1]; k < Edges_row_ptr[sum1 + 1]; k++) {
                if (Edges_columns[k] == sum2) {
                    idx_ab = k;
                    break;
                }
            }

            //ac
            sum1 = Tetra[voxid * 4 + list[1].x] < Tetra[voxid * 4 + list[1].y] ? Tetra[voxid * 4 + list[1].x] : Tetra[voxid * 4 + list[1].y];
            sum2 = Tetra[voxid * 4 + list[1].x] < Tetra[voxid * 4 + list[1].y] ? Tetra[voxid * 4 + list[1].y] : Tetra[voxid * 4 + list[1].x];
            int idx_ac = 0;
            for (int k = Edges_row_ptr[sum1]; k < Edges_row_ptr[sum1 + 1]; k++) {
                if (Edges_columns[k] == sum2) {
                    idx_ac = k;
                    break;
                }
            }

            //ad
            sum1 = Tetra[voxid * 4 + list[2].x] < Tetra[voxid * 4 + list[2].y] ? Tetra[voxid * 4 + list[2].x] : Tetra[voxid * 4 + list[2].y];
            sum2 = Tetra[voxid * 4 + list[2].x] < Tetra[voxid * 4 + list[2].y] ? Tetra[voxid * 4 + list[2].y] : Tetra[voxid * 4 + list[2].x];
            int idx_ad = 0;
            for (int k = Edges_row_ptr[sum1]; k < Edges_row_ptr[sum1 + 1]; k++) {
                if (Edges_columns[k] == sum2) {
                    idx_ad = k;
                    break;
                }
            }

            //storeTriangle(ab,ad,ac);
            Faces[6 * (voxid)+0] = idx_ab;
            Faces[6 * (voxid)+1] = idx_ad;
            Faces[6 * (voxid)+2] = idx_ac;
            if (idx_ab == idx_ad || idx_ab == idx_ac || idx_ad == idx_ac) {
                cout << "!! : " << voxid << endl;
            }

            Faces[6 * (voxid)+3] = 0;
            Faces[6 * (voxid)+4] = 0;
            Faces[6 * (voxid)+5] = 0;
        }
    }
    return;
}

void ComputeVertices(float *TSDF, float *volume_weights, int *Edges_row_ptr, int *Edges_columns, float *Nodes, int nb_Nodes, float *Vertices, float *skin_weights, float m_iso = 0.0) {
    
    for (int idx = 0; idx < nb_Nodes; idx++) {
        float tsdf1 = TSDF[idx];
        //float weight1 = fabs(tsdf1) < 1.0f ? 1.0f - fabs(tsdf1) : 0.00001f;
        float weight1 = 1.0f / (0.00001f + fabs(tsdf1));
        float weight2;

        bool cross;
        float tsdf2;
        
        // Go through all neighborhing summits
        for (int k = Edges_row_ptr[idx]; k < Edges_row_ptr[idx + 1]; k++) {
            int sum2 = Edges_columns[k];
            tsdf2 = TSDF[sum2];
            cross = tsdf1 * tsdf2 < 0.0f;

            if (!cross /*|| fabs(tsdf1) == 1.0f || fabs(tsdf2) == 1.0f*/) {
                Vertices[3 * k] = 0.0f;
                Vertices[3 * k + 1] = 0.0f;
                Vertices[3 * k + 2] = 0.0f;
                
                /*for (int s = 0; s < 24; s++) {
                    skin_weights[24 * k + s] = 0.0f;
                }*/
                continue;
            }

            //weight2 = fabs(TSDF[sum2]) < 1.0f ? 1.0f - fabs(TSDF[sum2]) : 0.00001f;
            weight2 = 1.0f / (0.00001f + fabs(TSDF[sum2]));

            Vertices[3 * k] = (weight1 * Nodes[3 * idx] + weight2 * Nodes[3 * sum2]) / (weight1+weight2); //frac1 * Nodes[3 * idx] + frac2 * Nodes[3 * sum2];
            Vertices[3 * k + 1] = (weight1 * Nodes[3 * idx + 1] + weight2 * Nodes[3 * sum2 + 1]) / (weight1+weight2); //frac1 * Nodes[3 * idx + 1] + frac2 * Nodes[3 * sum2 + 1];
            Vertices[3 * k + 2] = (weight1 * Nodes[3 * idx + 2] + weight2 * Nodes[3 * sum2 + 2]) / (weight1+weight2); //frac1 * Nodes[3 * idx + 2] + frac2 * Nodes[3 * sum2 + 2];
            
            /*for (int s = 0; s < 24; s++) {
                skin_weights[24 * k + s] = (weight1 * volume_weights[24 * idx + s] + weight2 * volume_weights[24 * sum2 + s]) / (weight1+weight2);
            }*/
        }
    }
}

int *MarchingTetrahedra(float *TSDF, float *volume_weights, float *Vertices, float *skin_weights, int *Faces, float *Nodes, int *Tetra, int *Edges_row_ptr, int *Edges_columns, int nb_nodes, int nb_tets, int nb_edges, float m_iso = 0.0) {
    // 1. Compute vertices
    ComputeVertices(TSDF, volume_weights, Edges_row_ptr, Edges_columns, Nodes, nb_nodes, Vertices, skin_weights, m_iso);
    
    // 2. Create the triangular faces
    GenerateFaces(TSDF, Edges_row_ptr, Edges_columns, Tetra, nb_tets, Faces, m_iso);
        
    int *res = new int[2];
    res[0] = nb_edges;
    res[1] = 2*nb_tets;
    return res;
    
    // 3. Remove null faces and vertices
    // Merge nodes
    //int *res = new int[2];
    int *Index = new int[nb_edges];
    vector<float> merged_vtx;
    vector<float> merged_weights;
    int count_merge = 0;
    for (int idx = 0; idx < nb_edges; idx++) {
        if (Vertices[3*idx] != 0.0f || Vertices[3*idx+1] != 0.0f || Vertices[3*idx+2] != 0.0f) {
            merged_vtx.push_back(Vertices[3*idx]);
            merged_vtx.push_back(Vertices[3*idx+1]);
            merged_vtx.push_back(Vertices[3*idx+2]);
            for (int s = 0; s < 24; s++) {
                merged_weights.push_back(skin_weights[24*idx + s]);
            }
            Index[idx] = count_merge;
            count_merge++;
        } else {
            Index[idx] = 0;
        }
    }
    cout << "vertices merged: " << count_merge << "; nb edges: " << nb_edges << endl;
    
    res[0] = merged_vtx.size()/3;
    memcpy(Vertices, merged_vtx.data(), merged_vtx.size()*sizeof(float));
    merged_vtx.clear();
    
    memcpy(skin_weights, merged_weights.data(), merged_weights.size()*sizeof(float));
    merged_weights.clear();
        
    // Reorganize faces
    vector<int> merged_faces;
    for (int idx = 0; idx < 2*nb_tets; idx++) {
        if ((Index[Faces[3*idx]] != 0 || Index[Faces[3*idx+1]] != 0 || Index[Faces[3*idx+2]]) &&
            (Index[Faces[3*idx]] != Index[Faces[3*idx+1]] && Index[Faces[3*idx]] != Index[Faces[3*idx+2]] &&
             Index[Faces[3*idx+1]] != Index[Faces[3*idx+2]])) {
            merged_faces.push_back(Index[Faces[3*idx]]);
            merged_faces.push_back(Index[Faces[3*idx+1]]);
            merged_faces.push_back(Index[Faces[3*idx+2]]);
        }
    }
    
    res[1] = merged_faces.size()/3;
    memcpy(Faces, merged_faces.data(), merged_faces.size()*sizeof(int));
    merged_faces.clear();
    
    return res;
}

int *MarchingTetrahedraNaive(float *TSDF, float *Vertices, int *Faces, float *Nodes, int *Tetra, int nb_nodes, int nb_tets, float m_iso = 0.0) {
    for (int voxid = 0; voxid < nb_tets; voxid++) {
       my_float4 a,b,c,d; //4 summits if the tetrahedra voxel

       int node_index_a,node_index_b,node_index_c,node_index_d;
       int sorted_node_a, sorted_node_b, sorted_node_c, sorted_node_d;

       a.x = Nodes[3*Tetra[voxid*4]] ;
       a.y = Nodes[3*Tetra[voxid*4] + 1] ;
       a.z = Nodes[3*Tetra[voxid*4] + 2] ;
       node_index_a = 5*Tetra[voxid*4];

       b.x = Nodes[3*Tetra[voxid*4+1]] ;
       b.y = Nodes[3*Tetra[voxid*4+1] + 1] ;
       b.z = Nodes[3*Tetra[voxid*4+1] + 2] ;
       node_index_b =5*Tetra[voxid*4+1];

       c.x = Nodes[3*Tetra[voxid*4+2]] ;
       c.y = Nodes[3*Tetra[voxid*4+2] + 1] ;
       c.z = Nodes[3*Tetra[voxid*4+2] + 2] ;
       node_index_c = 5*Tetra[voxid*4+2];

       d.x = Nodes[3*Tetra[voxid*4+3]] ;
       d.y = Nodes[3*Tetra[voxid*4+3] + 1] ;
       d.z = Nodes[3*Tetra[voxid*4+3] + 2] ;
       node_index_d = 5*Tetra[voxid*4+3];

       //Value of the TSDF
       a.w = TSDF[Tetra[voxid*4]];
       b.w = TSDF[Tetra[voxid*4+1]];
       c.w = TSDF[Tetra[voxid*4+2]];
       d.w = TSDF[Tetra[voxid*4+3]];

       //printf("_TSDF_Weights : %f, %f, %f, %f\n", a.w, b.w, c.w, d.w);

       float m_iso = 0.0 ;
       int count = 0;
       if (a.w >= m_iso)
           count += 1;
       if (b.w >= m_iso)
           count += 1;
       if (c.w >= m_iso)
           count += 1;
       if (d.w >= m_iso)
           count += 1;

       if (fabs(a.w) == 1.0f || fabs(b.w) == 1.0f || fabs(c.w) == 1.0f || fabs(d.w) == 1.0f)
           count = 0;

       //Three vertices of each triangle
       my_float4 sort_a,sort_b,sort_c,sort_d; //sorted summits if the tetrahedra
       int ad_index,ac_index,ab_index,bc_index,bd_index,cd_index;
       //Caseler
   
       if(count==0 || count==4) //return;
       {
       //There is no triangles to be stroed
           
           Faces[6*(voxid)+0] = 0;
           Faces[6*(voxid)+1] = 0;
           Faces[6*(voxid)+2] = 0;

           Faces[6*(voxid)+3] = 0;
           Faces[6*(voxid)+4] = 0;
           Faces[6*(voxid)+5] = 0;
       }

       //! Three _VMap are inside the volume
       else if (count == 3) {
           int list[6] = {0, 1, 2, 3, 4, 5};
           //! Make sure that fourth value lies outside
           if (d.w < m_iso)
           {
               sort_a = a;
               sort_b = b;
               sort_c = c;
               sort_d = d;
               //Todo index sirasini da tasi
               sorted_node_a = node_index_a ;
               sorted_node_b = node_index_b ;
               sorted_node_c = node_index_c ;
               sorted_node_d = node_index_d ;
           }
           else if (c.w < m_iso)
           {
               sort_a = a;
               sort_b = d;
               sort_c = b;
               sort_d = c;
               list[0] = 2;
               list[1] = 0;
               list[2] = 1;
               list[3] = 4;
               list[4] = 5;
               list[5] = 3;
               sorted_node_a = node_index_a ;
               sorted_node_b = node_index_d ;
               sorted_node_c = node_index_b ;
               sorted_node_d = node_index_c ;
           }
           else if (b.w  < m_iso)
           {
               sort_a = a;
               sort_b = c;
               sort_c = d;
               sort_d = b;
               list[0] = 1;
               list[1] = 2;
               list[2] = 0;
               list[3] = 5;
               list[4] = 3;
               list[5] = 4;
               sorted_node_a = node_index_a;
               sorted_node_b = node_index_c ;
               sorted_node_c = node_index_d ;
               sorted_node_d = node_index_b ;
           }
           else
           {
               sort_a = b;
               sort_b = d;
               sort_c = c;
               sort_d = a;
               list[0] = 4;
               list[1] = 3;
               list[2] = 0;
               list[3] = 5;
               list[4] = 2;
               list[5] = 1;
               sorted_node_a = node_index_b ;
               sorted_node_b = node_index_d ;
               sorted_node_c = node_index_c ;
               sorted_node_d = node_index_a ;
           }
           // CASE 1
           sort_a.w = 1.0f/(0.00001f + fabs(sort_a.w));
           sort_b.w = 1.0f/(0.00001f + fabs(sort_b.w));
           sort_c.w = 1.0f/(0.00001f + fabs(sort_c.w));
           sort_d.w = 1.0f/(0.00001f + fabs(sort_d.w));

           float da_frac = (sort_a.w) / (sort_a.w + sort_d.w);
           float ad_frac = 1.0-da_frac;//(sort_d.w) / (sort_a.w + sort_d.w);
           float db_frac = (sort_b.w) / (sort_b.w + sort_d.w);
           float bd_frac = 1.0-db_frac;//(sort_d.w) / (sort_b.w + sort_d.w);
           float dc_frac = (sort_c.w) / (sort_c.w + sort_d.w);
           float cd_frac = 1.0-dc_frac;//(sort_d.w) / (sort_c.w + sort_d.w);
           //! Compute the vertices of the intersections of the isosurface with the tetrahedron
           my_float3 ad,bd,cd;
       
           ad.x = da_frac*(sort_a.x) + (ad_frac)*sort_d.x;
           ad.y = da_frac*(sort_a.y) + (ad_frac)*sort_d.y;
           ad.z = da_frac*(sort_a.z) + (ad_frac)*sort_d.z;

           bd.x = db_frac*(sort_b.x) + (bd_frac)*sort_d.x;
           bd.y = db_frac*(sort_b.y) + (bd_frac)*sort_d.y;
           bd.z = db_frac*(sort_b.z) + (bd_frac)*sort_d.z;

           cd.x = dc_frac*(sort_c.x) + (cd_frac)*sort_d.x;
           cd.y = dc_frac*(sort_c.y) + (cd_frac)*sort_d.y;
           cd.z = dc_frac*(sort_c.z) + (cd_frac)*sort_d.z;

           //Check for fraction
           if(da_frac > ad_frac ) //Aya yakin
               ad_index= sorted_node_a;
           else //Dye yakin
               ad_index = sorted_node_d;

           if(db_frac>bd_frac) //Closer to B
               bd_index = sorted_node_b;
           else //closer to D
               bd_index  = sorted_node_d;

           if(dc_frac>cd_frac) //Cye yakin
               cd_index = sorted_node_c;
           else // Closer to D
               cd_index = sorted_node_d;

           //ad
           Vertices[3*(6*(voxid)+list[2])+0] = ad.x;
           Vertices[3*(6*(voxid)+list[2])+1] = ad.y;
           Vertices[3*(6*(voxid)+list[2])+2] = ad.z;
       
           //bd
           Vertices[3*(6*(voxid)+list[4])+0] = bd.x;
           Vertices[3*(6*(voxid)+list[4])+1] = bd.y;
           Vertices[3*(6*(voxid)+list[4])+2] = bd.z;
              
           //cd
           Vertices[3*(6*(voxid)+list[5])+0] = cd.x;
           Vertices[3*(6*(voxid)+list[5])+1] = cd.y;
           Vertices[3*(6*(voxid)+list[5])+2] = cd.z;

           Faces[6*(voxid)+0] = 6*(voxid)+list[2];
           Faces[6*(voxid)+1] = 6*(voxid)+list[5];
           Faces[6*(voxid)+2] = 6*(voxid)+list[4];

           Faces[6*(voxid)+3] = 0;
           Faces[6*(voxid)+4] = 0;
           Faces[6*(voxid)+5] = 0;

       }
       
       //! Two _VMap are inside the volume
       else if (count == 2) {
           //! Make sure that the last two points lie outside
           int list[6] = {0, 1, 2, 3, 4, 5};
           if (a.w >= m_iso && b.w >= m_iso)
           {
               sort_a = a;
               sort_b = b;
               sort_c = c;
               sort_d = d;
               sorted_node_a = node_index_a ;
               sorted_node_b = node_index_b ;
               sorted_node_c = node_index_c ;
               sorted_node_d = node_index_d ;
           }
           else if (a.w >= m_iso && c.w >= m_iso)
           {
               sort_a = a;
               sort_b = c;
               sort_c = d;
               sort_d = b;
               list[0] = 1;
               list[1] = 2;
               list[2] = 0;
               list[3] = 5;
               list[4] = 3;
               list[5] = 4;
               sorted_node_a = node_index_a ;
               sorted_node_b = node_index_c ;
               sorted_node_c = node_index_d ;
               sorted_node_d = node_index_b ;
           }
           else if (a.w >= m_iso && d.w >= m_iso)
           {
               sort_a = a;
               sort_b = d;
               sort_c = b;
               sort_d = c;
               list[0] = 2;
               list[1] = 0;
               list[2] = 1;
               list[3] = 4;
               list[4] = 5;
               list[5] = 3;
               sorted_node_a = node_index_a ;
               sorted_node_b = node_index_d ;
               sorted_node_c = node_index_b ;
               sorted_node_d = node_index_c ;
           }
           else if (b.w >= m_iso && c.w >= m_iso)
           {
               sort_a = b;
               sort_b = c;
               sort_c = a;
               sort_d = d;
               list[0] = 3;
               list[1] = 0;
               list[2] = 4;
               list[3] = 1;
               list[4] = 5;
               list[5] = 2;
               sorted_node_a = node_index_b ;
               sorted_node_b = node_index_c ;
               sorted_node_c = node_index_a ;
               sorted_node_d = node_index_d ;
           }
           else if (b.w >= m_iso && d.w >= m_iso)
           {
               sort_a = b;
               sort_b = d;
               sort_c = c;
               sort_d = a;
               list[0] = 4;
               list[1] = 3;
               list[2] = 0;
               list[3] = 5;
               list[4] = 2;
               list[5] = 1;
               sorted_node_a = node_index_b ;
               sorted_node_b = node_index_d ;
               sorted_node_c = node_index_c ;
               sorted_node_d = node_index_a ;
           }
           else //c && d > m_iso
           {
               sort_a = c;
               sort_b = d;
               sort_c = a;
               sort_d = b;
               list[0] = 5;
               list[1] = 1;
               list[2] = 3;
               list[3] = 2;
               list[4] = 4;
               list[5] = 0;
               sorted_node_a = node_index_c ;
               sorted_node_b = node_index_d ;
               sorted_node_c = node_index_a ;
               sorted_node_d = node_index_b ;
           }
           //CASE 2
           sort_a.w = 1.0f/(0.00001f + fabs(sort_a.w));
           sort_b.w = 1.0f/(0.00001f + fabs(sort_b.w));
           sort_c.w = 1.0f/(0.00001f + fabs(sort_c.w));
           sort_d.w = 1.0f/(0.00001f + fabs(sort_d.w));

           float ac_frac = (sort_c.w) / (sort_a.w + sort_c.w);
           float ca_frac = 1.0-ac_frac;//(sort_a.w) / (sort_a.w + sort_c.w);
           float ad_frac = (sort_d.w) / (sort_a.w + sort_d.w);
           float da_frac = 1.0-ad_frac;//(sort_a.w) / (sort_a.w + sort_d.w);
           float bc_frac = (sort_c.w) / (sort_b.w + sort_c.w);
           float cb_frac = 1.0-bc_frac;//(sort_b.w) / (sort_b.w + sort_c.w);
           float bd_frac = (sort_d.w) / (sort_b.w + sort_d.w);
           float db_frac = 1.0-bd_frac;//(sort_b.w) / (sort_b.w + sort_d.w);
       
           //! Compute the _VMap of the intersections of the isosurface with the tetrahedron
           my_float3 ac,ad,bd,bc;
       
           ad.x = da_frac*(sort_a.x) + (ad_frac)*sort_d.x;
           ad.y = da_frac*(sort_a.y) + (ad_frac)*sort_d.y;
           ad.z = da_frac*(sort_a.z) + (ad_frac)*sort_d.z;
           bd.x = db_frac*(sort_b.x) + (bd_frac)*sort_d.x;
           bd.y = db_frac*(sort_b.y) + (bd_frac)*sort_d.y;
           bd.z = db_frac*(sort_b.z) + (bd_frac)*sort_d.z;
           ac.x = ca_frac*(sort_a.x) + (ac_frac)*sort_c.x;
           ac.y = ca_frac*(sort_a.y) + (ac_frac)*sort_c.y;
           ac.z = ca_frac*(sort_a.z) + (ac_frac)*sort_c.z;
           bc.x = cb_frac*(sort_b.x) + (bc_frac)*sort_c.x;
           bc.y = cb_frac*(sort_b.y) + (bc_frac)*sort_c.y;
           bc.z = cb_frac*(sort_b.z) + (bc_frac)*sort_c.z;

           //Check for fraction
           if(da_frac > ad_frac ) //Aya yakin
               ad_index= sorted_node_a;
           else //Dye yakin
               ad_index = sorted_node_d;
           if(db_frac>bd_frac) //Closer to B
               bd_index = sorted_node_b;
           else //closer to D
               bd_index  = sorted_node_d;

           if(ca_frac>ac_frac) //Aye yakin
               ac_index = sorted_node_a;
           else // Closer to D
               ac_index = sorted_node_c;

           if(bc_frac>cb_frac) //Cye yakin
               bc_index = sorted_node_c;
           else // Closer to B
               bc_index = sorted_node_b;

           //ac
           Vertices[3*(6*(voxid)+list[1])+0] = ac.x;
           Vertices[3*(6*(voxid)+list[1])+1] = ac.y;
           Vertices[3*(6*(voxid)+list[1])+2] = ac.z;

           //ad
           Vertices[3*(6*(voxid)+list[2])+0] = ad.x;
           Vertices[3*(6*(voxid)+list[2])+1] = ad.y;
           Vertices[3*(6*(voxid)+list[2])+2] = ad.z;
       
           //bc
           Vertices[3*(6*(voxid)+list[3])+0] = bc.x;
           Vertices[3*(6*(voxid)+list[3])+1] = bc.y;
           Vertices[3*(6*(voxid)+list[3])+2] = bc.z;

           //bd
           Vertices[3*(6*(voxid)+list[4])+0] = bd.x;
           Vertices[3*(6*(voxid)+list[4])+1] = bd.y;
           Vertices[3*(6*(voxid)+list[4])+2] = bd.z;
       
           // storeTriangle(ac,bc,ad);
           Faces[6*(voxid)+0] = 6*(voxid)+list[1];
           Faces[6*(voxid)+1] = 6*(voxid)+list[3];
           Faces[6*(voxid)+2] = 6*(voxid)+list[2];

           //storeTriangle(bc,bd,ad);
           Faces[6*(voxid)+3] = 6*(voxid)+list[3];
           Faces[6*(voxid)+4] = 6*(voxid)+list[4];
           Faces[6*(voxid)+5] = 6*(voxid)+list[2];

       }
       //! One vertex is inside the volume
       else if (count == 1) {
           //! Make sure that the last three points lie outside
           int list[6] = {0, 1, 2, 3, 4, 5};
           if (a.w >= m_iso)
           {
               sort_a = a;
               sort_b = b;
               sort_c = c;
               sort_d = d;
               sorted_node_a = node_index_a ;
               sorted_node_b = node_index_b ;
               sorted_node_c = node_index_c ;
               sorted_node_d = node_index_d ;
           }
           else if (b.w >= m_iso)
           {
               sort_a = b;
               sort_b = c;
               sort_c = a;
               sort_d = d;
               list[0] = 3;
               list[1] = 0;
               list[2] = 4;
               list[3] = 1;
               list[4] = 5;
               list[5] = 2;
               sorted_node_a = node_index_b ;
               sorted_node_b = node_index_c ;
               sorted_node_c = node_index_a ;
               sorted_node_d = node_index_d ;
           }
           else if (c.w >= m_iso)
           {
               sort_a = c;
               sort_b = a;
               sort_c = b;
               sort_d = d;
               list[0] = 1;
               list[1] = 3;
               list[2] = 5;
               list[3] = 0;
               list[4] = 2;
               list[5] = 4;
               sorted_node_a = node_index_c ;
               sorted_node_b = node_index_a ;
               sorted_node_c = node_index_b ;
               sorted_node_d = node_index_d ;
           }
           else // d > m_iso
           {
               sort_a = d;
               sort_b = c;
               sort_c = b;
               sort_d = a;
               list[0] = 5;
               list[1] = 4;
               list[2] = 2;
               list[3] = 3;
               list[4] = 1;
               list[5] = 0;
               sorted_node_a = node_index_d ;
               sorted_node_b = node_index_c ;
               sorted_node_c = node_index_b ;
               sorted_node_d = node_index_a ;
           }
           //CASE 3
       
           sort_a.w = 1.0f/(0.00001f + fabs(sort_a.w));
           sort_b.w = 1.0f/(0.00001f + fabs(sort_b.w));
           sort_c.w = 1.0f/(0.00001f + fabs(sort_c.w));
           sort_d.w = 1.0f/(0.00001f + fabs(sort_d.w));

           float ab_frac = (sort_b.w) / (sort_a.w + sort_b.w);
           float ba_frac = 1.0-ab_frac;//(sort_a.w) / (sort_a.w + sort_b.w);
           float ac_frac = (sort_c.w) / (sort_a.w + sort_c.w);
           float ca_frac = 1.0-ac_frac;//(sort_a.w) / (sort_a.w + sort_c.w);
           float ad_frac = (sort_d.w) / (sort_a.w + sort_d.w);
           float da_frac = 1.0-ad_frac;//(sort_a.w) / (sort_a.w + sort_d.w);

       
           my_float3 ab,ac,ad ;
           ab.x = ba_frac*(sort_a.x) + (ab_frac)*sort_b.x;
           ab.y = ba_frac*(sort_a.y) + (ab_frac)*sort_b.y;
           ab.z = ba_frac*(sort_a.z) + (ab_frac)*sort_b.z;

           ad.x = da_frac*(sort_a.x) + (ad_frac)*sort_d.x;
           ad.y = da_frac*(sort_a.y) + (ad_frac)*sort_d.y;
           ad.z = da_frac*(sort_a.z) + (ad_frac)*sort_d.z;

           ac.x = ca_frac*(sort_a.x) + (ac_frac)*sort_c.x;
           ac.y = ca_frac*(sort_a.y) + (ac_frac)*sort_c.y;
           ac.z = ca_frac*(sort_a.z) + (ac_frac)*sort_c.z;

           //Check for fraction
           if(da_frac > ad_frac ) //Aya yakin
               ad_index= sorted_node_a;
           else //Dye yakin
               ad_index = sorted_node_d;

           if(ba_frac>ab_frac) //Closer to A
               ab_index = sorted_node_a;
           else //closer to B
               ab_index  = sorted_node_b;

           if(ca_frac>ac_frac) //Aye yakin
               ac_index = sorted_node_a;
           else // Closer to C
               ac_index = sorted_node_c;
           //ab
           Vertices[3*(6*(voxid)+list[0])+0] = ab.x;
           Vertices[3*(6*(voxid)+list[0])+1] = ab.y;
           Vertices[3*(6*(voxid)+list[0])+2] = ab.z;
       
           //ac
           Vertices[3*(6*(voxid)+list[1])+0] = ac.x;
           Vertices[3*(6*(voxid)+list[1])+1] = ac.y;
           Vertices[3*(6*(voxid)+list[1])+2] = ac.z;

           //ad
           Vertices[3*(6*(voxid)+list[2])+0] = ad.x;
           Vertices[3*(6*(voxid)+list[2])+1] = ad.y;
           Vertices[3*(6*(voxid)+list[2])+2] = ad.z;

           //storeTriangle(ab,ad,ac);
           Faces[6*(voxid)+0] = 6*(voxid)+list[0];
           Faces[6*(voxid)+1] = 6*(voxid)+list[2];
           Faces[6*(voxid)+2] = 6*(voxid)+list[1];

           Faces[6*(voxid)+3] = 0;
           Faces[6*(voxid)+4] = 0;
           Faces[6*(voxid)+5] = 0;
       }
    }
        
    int *res = new int[2];
    res[0] = 6*nb_tets;
    res[1] = 2*nb_tets;
    return res;
}

#endif /* MarchingTets_h */
