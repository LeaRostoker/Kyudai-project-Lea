//
//  CubicMesh.cpp
//  deepanim
//
//  Created by Diego Thomas on 2021/02/12.
//

#include <stdio.h>
#include "include_gpu/CubicMesh.h"

void OptimizeCorners(my_float3 ***voxels, my_float3 ***voxels_copy, my_float3 ***voxels_prev, my_int3 Dim, float delta_elastic) {
    
    { // 1 vertex 0,0,0
        my_float3 curr_vox = voxels[0][0][0];
        my_float3 prev_vox = voxels_prev[0][0][0];
        // Compute the force applied to the voxel
        my_float3 force = make_my_float3(0.0f);
        // Go through all edges from the current voxel
        
        my_float3 curr_edge = curr_vox - voxels[1][0][0];
        float curr_edge_length = norm(curr_edge);
        float prev_edge_length = norm(prev_vox - voxels_prev[1][0][0]);
        force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));

        curr_edge = curr_vox - voxels[0][1][0];
        curr_edge_length = norm(curr_edge);
        prev_edge_length = norm(prev_vox - voxels_prev[0][1][0]);
        force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));

        curr_edge = curr_vox - voxels[0][0][1];
        curr_edge_length = norm(curr_edge);
        prev_edge_length = norm(prev_vox - voxels_prev[0][0][1]);
        force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
                
        //Apply force to the voxel
        voxels_copy[0][0][0] = curr_vox + force * delta_elastic;
    }
    
    { // 2 vertex _Dim.x-1,0,0
        my_float3 curr_vox = voxels[Dim.x-1][0][0];
        my_float3 prev_vox = voxels_prev[Dim.x-1][0][0];
        // Compute the force applied to the voxel
        my_float3 force = make_my_float3(0.0f);
        // Go through all edges from the current voxel
        
        my_float3 curr_edge = curr_vox - voxels[Dim.x-2][0][0];
        float curr_edge_length = norm(curr_edge);
        float prev_edge_length = norm(prev_vox - voxels_prev[Dim.x-2][0][0]);
        force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));

        curr_edge = curr_vox - voxels[Dim.x-1][1][0];
        curr_edge_length = norm(curr_edge);
        prev_edge_length = norm(prev_vox - voxels_prev[Dim.x-1][1][0]);
        force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));

        curr_edge = curr_vox - voxels[Dim.x-1][0][1];
        curr_edge_length = norm(curr_edge);
        prev_edge_length = norm(prev_vox - voxels_prev[Dim.x-1][0][1]);
        force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
                
        //Apply force to the voxel
        voxels_copy[Dim.x-1][0][0] = curr_vox + force * delta_elastic;
    }
    
    { // 3 vertex Dim.x-1, Dim.y-1,0
        my_float3 curr_vox = voxels[Dim.x-1][Dim.y-1][0];
        my_float3 prev_vox = voxels_prev[Dim.x-1][Dim.y-1][0];
        // Compute the force applied to the voxel
        my_float3 force = make_my_float3(0.0f);
        // Go through all edges from the current voxel
        
        my_float3 curr_edge = curr_vox - voxels[Dim.x-2][Dim.y-1][0];
        float curr_edge_length = norm(curr_edge);
        float prev_edge_length = norm(prev_vox - voxels_prev[Dim.x-2][Dim.y-1][0]);
        force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));

        curr_edge = curr_vox - voxels[Dim.x-1][Dim.y-2][0];
        curr_edge_length = norm(curr_edge);
        prev_edge_length = norm(prev_vox - voxels_prev[Dim.x-1][Dim.y-2][0]);
        force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));

        curr_edge = curr_vox - voxels[Dim.x-1][Dim.y-1][1];
        curr_edge_length = norm(curr_edge);
        prev_edge_length = norm(prev_vox - voxels_prev[Dim.x-1][Dim.y-1][1]);
        force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
                
        //Apply force to the voxel
        voxels_copy[Dim.x-1][Dim.y-1][0] = curr_vox + force * delta_elastic;
    }
    
    { // 4 vertex 0, Dim.y-1,0
        my_float3 curr_vox = voxels[0][Dim.y-1][0];
        my_float3 prev_vox = voxels_prev[0][Dim.y-1][0];
        // Compute the force applied to the voxel
        my_float3 force = make_my_float3(0.0f);
        // Go through all edges from the current voxel
        
        my_float3 curr_edge = curr_vox - voxels[1][Dim.y-1][0];
        float curr_edge_length = norm(curr_edge);
        float prev_edge_length = norm(prev_vox - voxels_prev[1][Dim.y-1][0]);
        force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));

        curr_edge = curr_vox - voxels[0][Dim.y-2][0];
        curr_edge_length = norm(curr_edge);
        prev_edge_length = norm(prev_vox - voxels_prev[0][Dim.y-2][0]);
        force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));

        curr_edge = curr_vox - voxels[0][Dim.y-1][1];
        curr_edge_length = norm(curr_edge);
        prev_edge_length = norm(prev_vox - voxels_prev[0][Dim.y-1][1]);
        force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
                
        //Apply force to the voxel
        voxels_copy[0][Dim.y-1][0] = curr_vox + force * delta_elastic;
    }
    
    { // 5 vertex 0, 0, Dim.z-1
        my_float3 curr_vox = voxels[0][0][Dim.z-1];
        my_float3 prev_vox = voxels_copy[0][0][Dim.z-1];
        // Compute the force applied to the voxel
        my_float3 force = make_my_float3(0.0f);
        // Go through all edges from the current voxel
        
        my_float3 curr_edge = curr_vox - voxels[1][0][Dim.z-1];
        float curr_edge_length = norm(curr_edge);
        float prev_edge_length = norm(prev_vox - voxels_prev[1][0][Dim.z-1]);
        force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));

        curr_edge = curr_vox - voxels[0][1][Dim.z-1];
        curr_edge_length = norm(curr_edge);
        prev_edge_length = norm(prev_vox - voxels_prev[0][1][Dim.z-1]);
        force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));

        curr_edge = curr_vox - voxels[0][0][Dim.z-2];
        curr_edge_length = norm(curr_edge);
        prev_edge_length = norm(prev_vox - voxels_prev[0][0][Dim.z-2]);
        force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
                
        //Apply force to the voxel
        voxels_copy[0][0][Dim.z-1] = curr_vox + force * delta_elastic;
    }
    
    { //6 vertex Dim.x-1, 0, Dim.z-1
        my_float3 curr_vox = voxels[Dim.x-1][0][Dim.z-1];
        my_float3 prev_vox = voxels_prev[Dim.x-1][0][Dim.z-1];
        // Compute the force applied to the voxel
        my_float3 force = make_my_float3(0.0f);
        // Go through all edges from the current voxel
        
        my_float3 curr_edge = curr_vox - voxels[Dim.x-2][0][Dim.z-1];
        float curr_edge_length = norm(curr_edge);
        float prev_edge_length = norm(prev_vox - voxels_prev[Dim.x-2][0][Dim.z-1]);
        force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));

        curr_edge = curr_vox - voxels[Dim.x-1][1][Dim.z-1];
        curr_edge_length = norm(curr_edge);
        prev_edge_length = norm(prev_vox - voxels_prev[Dim.x-1][1][Dim.z-1]);
        force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));

        curr_edge = curr_vox - voxels[Dim.x-1][0][Dim.z-2];
        curr_edge_length = norm(curr_edge);
        prev_edge_length = norm(prev_vox - voxels_prev[Dim.x-1][0][Dim.z-2]);
        force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
                
        //Apply force to the voxel
        voxels_copy[Dim.x-1][0][Dim.z-1] = curr_vox + force * delta_elastic;
    }
    
    { // 7 vertex Dim.x-1, Dim.y-1, Dim.z-1
        my_float3 curr_vox = voxels[Dim.x-1][Dim.y-1][Dim.z-1];
        my_float3 prev_vox = voxels_prev[Dim.x-1][Dim.y-1][Dim.z-1];
        // Compute the force applied to the voxel
        my_float3 force = make_my_float3(0.0f);
        // Go through all edges from the current voxel
        
        my_float3 curr_edge = curr_vox - voxels[Dim.x-2][Dim.y-1][Dim.z-1];
        float curr_edge_length = norm(curr_edge);
        float prev_edge_length = norm(prev_vox - voxels_prev[Dim.x-2][Dim.y-1][Dim.z-1]);
        force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));

        curr_edge = curr_vox - voxels[Dim.x-1][Dim.y-2][Dim.z-1];
        curr_edge_length = norm(curr_edge);
        prev_edge_length = norm(prev_vox - voxels_prev[Dim.x-1][Dim.y-2][Dim.z-1]);
        force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));

        curr_edge = curr_vox - voxels[Dim.x-1][Dim.y-1][Dim.z-2];
        curr_edge_length = norm(curr_edge);
        prev_edge_length = norm(prev_vox - voxels_prev[Dim.x-1][Dim.y-1][Dim.z-2]);
        force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
                
        //Apply force to the voxel
        voxels_copy[Dim.x-1][Dim.y-1][Dim.z-1] = curr_vox + force * delta_elastic;
    }
    
    { // 8 vertex 0, Dim.y-1, Dim.z-1
        my_float3 curr_vox = voxels[0][Dim.y-1][Dim.z-1];
        my_float3 prev_vox = voxels_prev[0][Dim.y-1][Dim.z-1];
        // Compute the force applied to the voxel
        my_float3 force = make_my_float3(0.0f);
        // Go through all edges from the current voxel
        
        my_float3 curr_edge = curr_vox - voxels[1][Dim.y-1][Dim.z-1];
        float curr_edge_length = norm(curr_edge);
        float prev_edge_length = norm(prev_vox - voxels_prev[1][Dim.y-1][Dim.z-1]);
        force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));

        curr_edge = curr_vox - voxels[0][Dim.y-2][Dim.z-1];
        curr_edge_length = norm(curr_edge);
        prev_edge_length = norm(prev_vox - voxels_prev[0][Dim.y-2][Dim.z-1]);
        force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));

        curr_edge = curr_vox - voxels[0][Dim.y-1][Dim.z-2];
        curr_edge_length = norm(curr_edge);
        prev_edge_length = norm(prev_vox - voxels_prev[0][Dim.y-1][Dim.z-2]);
        force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
                
        //Apply force to the voxel
        voxels_copy[0][Dim.y-1][Dim.z-1] = curr_vox + force * delta_elastic;
    }
}

void OptimizeEdges(my_float3 ***voxels, my_float3 ***voxels_copy, my_float3 ***voxels_prev, my_int3 Dim, float delta_elastic) {
    float energy = 0.0f;
    
    { //Edge 1
        for (int i = 1; i < Dim.x-1; i++) {
            my_float3 curr_vox = voxels[i][0][0];
            my_float3 prev_vox = voxels_prev[i][0][0];
            // Compute the force applied to the voxel
            my_float3 force = make_my_float3(0.0f);
            // Go through all edges from the current voxel
            for (int x = -1; x < 2; x++) {
                if (x == 0)
                    continue;
                my_float3 curr_edge = curr_vox - voxels[i+x][0][0];
                float curr_edge_length = norm(curr_edge);
                float prev_edge_length = norm(prev_vox - voxels_prev[i+x][0][0]);
                force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
                energy += fabs(prev_edge_length - curr_edge_length);
            }
            
            my_float3 curr_edge = curr_vox - voxels[i][1][0];
            float curr_edge_length = norm(curr_edge);
            float prev_edge_length = norm(prev_vox - voxels_prev[i][1][0]);
            force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
            energy += fabs(prev_edge_length - curr_edge_length);
            
            curr_edge = curr_vox - voxels[i][0][1];
            curr_edge_length = norm(curr_edge);
            prev_edge_length = norm(prev_vox - voxels_prev[i][0][1]);
            force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
            energy += fabs(prev_edge_length - curr_edge_length);
            
            //Apply force to the voxel
            voxels_copy[i][0][0] = curr_vox + force * delta_elastic;
        }
    }
    
    { //Edge 2
        for (int j = 1; j < Dim.y-1; j++) {
            my_float3 curr_vox = voxels[Dim.x-1][j][0];
            my_float3 prev_vox = voxels_prev[Dim.x-1][j][0];
            // Compute the force applied to the voxel
            my_float3 force = make_my_float3(0.0f);
            // Go through all edges from the current voxel
            for (int y = -1; y < 2; y++) {
                if (y == 0)
                    continue;
                my_float3 curr_edge = curr_vox - voxels[Dim.x-1][j+y][0];
                float curr_edge_length = norm(curr_edge);
                float prev_edge_length = norm(prev_vox - voxels_prev[Dim.x-1][j+y][0]);
                force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
                energy += fabs(prev_edge_length - curr_edge_length);
            }
            
            my_float3 curr_edge = curr_vox - voxels[Dim.x-2][j][0];
            float curr_edge_length = norm(curr_edge);
            float prev_edge_length = norm(prev_vox - voxels_prev[Dim.x-2][j][0]);
            force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
            energy += fabs(prev_edge_length - curr_edge_length);
            
            curr_edge = curr_vox - voxels[Dim.x-1][j][1];
            curr_edge_length = norm(curr_edge);
            prev_edge_length = norm(prev_vox - voxels_prev[Dim.x-1][j][1]);
            force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
            energy += fabs(prev_edge_length - curr_edge_length);
            
            //Apply force to the voxel
            voxels_copy[Dim.x-1][j][0] = curr_vox + force * delta_elastic;
        }
    }
        
    { //Edge 3
        for (int i = 1; i < Dim.x-1; i++) {
            my_float3 curr_vox = voxels[i][Dim.y-1][0];
            my_float3 prev_vox = voxels_prev[Dim.y-1][0][0];
            // Compute the force applied to the voxel
            my_float3 force = make_my_float3(0.0f);
            // Go through all edges from the current voxel
            for (int x = -1; x < 2; x++) {
                if (x == 0)
                    continue;
                my_float3 curr_edge = curr_vox - voxels[i+x][Dim.y-1][0];
                float curr_edge_length = norm(curr_edge);
                float prev_edge_length = norm(prev_vox - voxels_prev[i+x][Dim.y-1][0]);
                force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
                energy += fabs(prev_edge_length - curr_edge_length);
            }
            
            my_float3 curr_edge = curr_vox - voxels[i][Dim.y-2][0];
            float curr_edge_length = norm(curr_edge);
            float prev_edge_length = norm(prev_vox - voxels_prev[i][Dim.y-2][0]);
            force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
            energy += fabs(prev_edge_length - curr_edge_length);
            
            curr_edge = curr_vox - voxels[i][Dim.y-1][1];
            curr_edge_length = norm(curr_edge);
            prev_edge_length = norm(prev_vox - voxels_prev[i][Dim.y-1][1]);
            force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
            energy += fabs(prev_edge_length - curr_edge_length);
            
            //Apply force to the voxel
            voxels_copy[i][Dim.y-1][0] = curr_vox + force * delta_elastic;
        }
    }
    
    { //Edge 4
        for (int j = 1; j < Dim.y-1; j++) {
            my_float3 curr_vox = voxels[0][j][0];
            my_float3 prev_vox = voxels_prev[0][j][0];
            // Compute the force applied to the voxel
            my_float3 force = make_my_float3(0.0f);
            // Go through all edges from the current voxel
            for (int y = -1; y < 2; y++) {
                if (y == 0)
                    continue;
                my_float3 curr_edge = curr_vox - voxels[0][j+y][0];
                float curr_edge_length = norm(curr_edge);
                float prev_edge_length = norm(prev_vox - voxels_prev[0][j+y][0]);
                force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
                energy += fabs(prev_edge_length - curr_edge_length);
            }
            
            my_float3 curr_edge = curr_vox - voxels[1][j][0];
            float curr_edge_length = norm(curr_edge);
            float prev_edge_length = norm(prev_vox - voxels_prev[1][j][0]);
            force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
            energy += fabs(prev_edge_length - curr_edge_length);
            
            curr_edge = curr_vox - voxels[0][j][1];
            curr_edge_length = norm(curr_edge);
            prev_edge_length = norm(prev_vox - voxels_prev[0][j][1]);
            force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
            energy += fabs(prev_edge_length - curr_edge_length);
            
            //Apply force to the voxel
            voxels_copy[0][j][0] = curr_vox + force * delta_elastic;
        }
    }
    
    { //Edge 5
        for (int k = 1; k < Dim.z-1; k++) {
            my_float3 curr_vox = voxels[0][0][k];
            my_float3 prev_vox = voxels_prev[0][0][k];
            // Compute the force applied to the voxel
            my_float3 force = make_my_float3(0.0f);
            // Go through all edges from the current voxel
            for (int z = -1; z < 2; z++) {
                if (z == 0)
                    continue;
                my_float3 curr_edge = curr_vox - voxels[0][0][k+z];
                float curr_edge_length = norm(curr_edge);
                float prev_edge_length = norm(prev_vox - voxels_prev[0][0][k+z]);
                force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
                energy += fabs(prev_edge_length - curr_edge_length);
            }
            
            my_float3 curr_edge = curr_vox - voxels[1][0][k];
            float curr_edge_length = norm(curr_edge);
            float prev_edge_length = norm(prev_vox - voxels_prev[1][0][k]);
            force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
            energy += fabs(prev_edge_length - curr_edge_length);
            
            curr_edge = curr_vox - voxels[0][1][k];
            curr_edge_length = norm(curr_edge);
            prev_edge_length = norm(prev_vox - voxels_prev[0][1][k]);
            force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
            energy += fabs(prev_edge_length - curr_edge_length);
            
            //Apply force to the voxel
            voxels_copy[0][0][k] = curr_vox + force * delta_elastic;
        }
    }
    
    { //Edge 6
        for (int k = 1; k < Dim.z-1; k++) {
            my_float3 curr_vox = voxels[Dim.x-1][0][k];
            my_float3 prev_vox = voxels_prev[Dim.x-1][0][k];
            // Compute the force applied to the voxel
            my_float3 force = make_my_float3(0.0f);
            // Go through all edges from the current voxel
            for (int z = -1; z < 2; z++) {
                if (z == 0)
                    continue;
                my_float3 curr_edge = curr_vox - voxels[Dim.x-1][0][k+z];
                float curr_edge_length = norm(curr_edge);
                float prev_edge_length = norm(prev_vox - voxels_prev[Dim.x-1][0][k+z]);
                force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
                energy += fabs(prev_edge_length - curr_edge_length);
            }
            
            my_float3 curr_edge = curr_vox - voxels[Dim.x-2][0][k];
            float curr_edge_length = norm(curr_edge);
            float prev_edge_length = norm(prev_vox - voxels_prev[Dim.x-2][0][k]);
            force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
            energy += fabs(prev_edge_length - curr_edge_length);
            
            curr_edge = curr_vox - voxels[Dim.x-1][1][k];
            curr_edge_length = norm(curr_edge);
            prev_edge_length = norm(prev_vox - voxels_prev[Dim.x-1][1][k]);
            force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
            energy += fabs(prev_edge_length - curr_edge_length);
            
            //Apply force to the voxel
            voxels_copy[Dim.x-1][0][k] = curr_vox + force * delta_elastic;
        }
    }
    
    { //Edge 7
        for (int k = 1; k < Dim.z-1; k++) {
            my_float3 curr_vox = voxels[Dim.x-1][Dim.y-1][k];
            my_float3 prev_vox = voxels_prev[Dim.x-1][Dim.y-1][k];
            // Compute the force applied to the voxel
            my_float3 force = make_my_float3(0.0f);
            // Go through all edges from the current voxel
            for (int z = -1; z < 2; z++) {
                if (z == 0)
                    continue;
                my_float3 curr_edge = curr_vox - voxels[Dim.x-1][Dim.y-1][k+z];
                float curr_edge_length = norm(curr_edge);
                float prev_edge_length = norm(prev_vox - voxels_prev[Dim.x-1][Dim.y-1][k+z]);
                force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
                energy += fabs(prev_edge_length - curr_edge_length);
            }
            
            my_float3 curr_edge = curr_vox - voxels[Dim.x-2][Dim.y-1][k];
            float curr_edge_length = norm(curr_edge);
            float prev_edge_length = norm(prev_vox - voxels_prev[Dim.x-2][Dim.y-1][k]);
            force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
            energy += fabs(prev_edge_length - curr_edge_length);
            
            curr_edge = curr_vox - voxels[Dim.x-1][Dim.y-2][k];
            curr_edge_length = norm(curr_edge);
            prev_edge_length = norm(prev_vox - voxels_prev[Dim.x-1][Dim.y-2][k]);
            force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
            energy += fabs(prev_edge_length - curr_edge_length);
            
            //Apply force to the voxel
            voxels_copy[Dim.x-1][Dim.y-1][k] = curr_vox + force * delta_elastic;
        }
    }
    
    { //Edge 8
        for (int k = 1; k < Dim.z-1; k++) {
            my_float3 curr_vox = voxels[0][Dim.y-1][k];
            my_float3 prev_vox = voxels_prev[0][Dim.y-1][k];
            // Compute the force applied to the voxel
            my_float3 force = make_my_float3(0.0f);
            // Go through all edges from the current voxel
            for (int z = -1; z < 2; z++) {
                if (z == 0)
                    continue;
                my_float3 curr_edge = curr_vox - voxels[0][Dim.y-1][k+z];
                float curr_edge_length = norm(curr_edge);
                float prev_edge_length = norm(prev_vox - voxels_prev[0][Dim.y-1][k+z]);
                force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
                energy += fabs(prev_edge_length - curr_edge_length);
            }
            
            my_float3 curr_edge = curr_vox - voxels[1][Dim.y-1][k];
            float curr_edge_length = norm(curr_edge);
            float prev_edge_length = norm(prev_vox - voxels_prev[1][Dim.y-1][k]);
            force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
            energy += fabs(prev_edge_length - curr_edge_length);
            
            curr_edge = curr_vox - voxels[0][Dim.y-2][k];
            curr_edge_length = norm(curr_edge);
            prev_edge_length = norm(prev_vox - voxels_prev[0][Dim.y-2][k]);
            force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
            energy += fabs(prev_edge_length - curr_edge_length);
            
            //Apply force to the voxel
            voxels_copy[0][Dim.y-1][k] = curr_vox + force * delta_elastic;
        }
    }
    
    { //Edge 9
        for (int i = 1; i < Dim.x-1; i++) {
            my_float3 curr_vox = voxels[i][0][Dim.z-1];
            my_float3 prev_vox = voxels_prev[i][0][Dim.z-1];
            // Compute the force applied to the voxel
            my_float3 force = make_my_float3(0.0f);
            // Go through all edges from the current voxel
            for (int x = -1; x < 2; x++) {
                if (x == 0)
                    continue;
                my_float3 curr_edge = curr_vox - voxels[i+x][0][Dim.z-1];
                float curr_edge_length = norm(curr_edge);
                float prev_edge_length = norm(prev_vox - voxels_prev[i+x][0][Dim.z-1]);
                force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
                energy += fabs(prev_edge_length - curr_edge_length);
            }
            
            my_float3 curr_edge = curr_vox - voxels[i][1][Dim.z-1];
            float curr_edge_length = norm(curr_edge);
            float prev_edge_length = norm(prev_vox - voxels_prev[i][1][Dim.z-1]);
            force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
            energy += fabs(prev_edge_length - curr_edge_length);
            
            curr_edge = curr_vox - voxels[i][0][Dim.z-2];
            curr_edge_length = norm(curr_edge);
            prev_edge_length = norm(prev_vox - voxels_prev[i][0][Dim.z-2]);
            force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
            energy += fabs(prev_edge_length - curr_edge_length);
            
            //Apply force to the voxel
            voxels_copy[i][0][Dim.z-1] = curr_vox + force * delta_elastic;
        }
    }
    
    { //Edge 10
        for (int j = 1; j < Dim.y-1; j++) {
            my_float3 curr_vox = voxels[Dim.x-1][j][Dim.z-1];
            my_float3 prev_vox = voxels_prev[Dim.x-1][j][Dim.z-1];
            // Compute the force applied to the voxel
            my_float3 force = make_my_float3(0.0f);
            // Go through all edges from the current voxel
            for (int y = -1; y < 2; y++) {
                if (y == 0)
                    continue;
                my_float3 curr_edge = curr_vox - voxels[Dim.x-1][j+y][Dim.z-1];
                float curr_edge_length = norm(curr_edge);
                float prev_edge_length = norm(prev_vox - voxels_prev[Dim.x-1][j+y][Dim.z-1]);
                force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
                energy += fabs(prev_edge_length - curr_edge_length);
            }
            
            my_float3 curr_edge = curr_vox - voxels[Dim.x-2][j][Dim.z-1];
            float curr_edge_length = norm(curr_edge);
            float prev_edge_length = norm(prev_vox - voxels_prev[Dim.x-2][j][Dim.z-1]);
            force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
            energy += fabs(prev_edge_length - curr_edge_length);
            
            curr_edge = curr_vox - voxels[Dim.x-1][j][Dim.z-2];
            curr_edge_length = norm(curr_edge);
            prev_edge_length = norm(prev_vox - voxels_prev[Dim.x-1][j][Dim.z-2]);
            force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
            energy += fabs(prev_edge_length - curr_edge_length);
            
            //Apply force to the voxel
            voxels_copy[Dim.x-1][j][Dim.z-1] = curr_vox + force * delta_elastic;
        }
    }
        
    { //Edge 11
        for (int i = 1; i < Dim.x-1; i++) {
            my_float3 curr_vox = voxels[i][Dim.y-1][Dim.z-1];
            my_float3 prev_vox = voxels_prev[Dim.y-1][0][Dim.z-1];
            // Compute the force applied to the voxel
            my_float3 force = make_my_float3(0.0f);
            // Go through all edges from the current voxel
            for (int x = -1; x < 2; x++) {
                if (x == 0)
                    continue;
                my_float3 curr_edge = curr_vox - voxels[i+x][Dim.y-1][Dim.z-1];
                float curr_edge_length = norm(curr_edge);
                float prev_edge_length = norm(prev_vox - voxels_prev[i+x][Dim.y-1][Dim.z-1]);
                force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
                energy += fabs(prev_edge_length - curr_edge_length);
            }
            
            my_float3 curr_edge = curr_vox - voxels[i][Dim.y-2][Dim.z-1];
            float curr_edge_length = norm(curr_edge);
            float prev_edge_length = norm(prev_vox - voxels_prev[i][Dim.y-2][Dim.z-1]);
            force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
            energy += fabs(prev_edge_length - curr_edge_length);
            
            curr_edge = curr_vox - voxels[i][Dim.y-1][Dim.z-2];
            curr_edge_length = norm(curr_edge);
            prev_edge_length = norm(prev_vox - voxels_prev[i][Dim.y-1][Dim.z-2]);
            force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
            energy += fabs(prev_edge_length - curr_edge_length);
            
            //Apply force to the voxel
            voxels_copy[i][Dim.y-1][Dim.z-1] = curr_vox + force * delta_elastic;
        }
    }
    
    { //Edge 12
        for (int j = 1; j < Dim.y-1; j++) {
            my_float3 curr_vox = voxels[0][j][Dim.z-1];
            my_float3 prev_vox = voxels_prev[0][j][Dim.z-1];
            // Compute the force applied to the voxel
            my_float3 force = make_my_float3(0.0f);
            // Go through all edges from the current voxel
            for (int y = -1; y < 2; y++) {
                if (y == 0)
                    continue;
                my_float3 curr_edge = curr_vox - voxels[0][j+y][Dim.z-1];
                float curr_edge_length = norm(curr_edge);
                float prev_edge_length = norm(prev_vox - voxels_prev[0][j+y][Dim.z-1]);
                force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
                energy += fabs(prev_edge_length - curr_edge_length);
            }
            
            my_float3 curr_edge = curr_vox - voxels[1][j][Dim.z-1];
            float curr_edge_length = norm(curr_edge);
            float prev_edge_length = norm(prev_vox - voxels_prev[1][j][Dim.z-1]);
            force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
            energy += fabs(prev_edge_length - curr_edge_length);
            
            curr_edge = curr_vox - voxels[0][j][Dim.z-2];
            curr_edge_length = norm(curr_edge);
            prev_edge_length = norm(prev_vox - voxels_prev[0][j][Dim.z-2]);
            force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
            energy += fabs(prev_edge_length - curr_edge_length);
            
            //Apply force to the voxel
            voxels_copy[0][j][Dim.z-1] = curr_vox + force * delta_elastic;
        }
    }
    //cout << "Energy: " << energy << endl;
    
}

void OptimizeFaces(my_float3 ***voxels, my_float3 ***voxels_copy, my_float3 ***voxels_prev, my_int3 Dim, float delta_elastic) {
    
    float energy = 0.0f;
    
    { //Face 1
        for (int i = 1; i < Dim.x-1; i++) {
            for (int j = 1; j < Dim.y-1; j++) {
                my_float3 curr_vox = voxels[i][j][0];
                my_float3 prev_vox = voxels_prev[i][j][0];
                // Compute the force applied to the voxel
                my_float3 force = make_my_float3(0.0f);
                // Go through all edges from the current voxel
                for (int x = -1; x < 2; x++) {
                    for (int y = -1; y < 2; y++) {
                            if (x == 0 && y == 0)
                                continue;
                            my_float3 curr_edge = curr_vox - voxels[i+x][j+y][0];
                            float curr_edge_length = norm(curr_edge);
                            float prev_edge_length = norm(prev_vox - voxels_prev[i+x][j+y][0]);
                            force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
                            energy += fabs(prev_edge_length - curr_edge_length);
                    }
                }
                
                //if (norm(force) > 0.0f)
                //    force.print();
                //Apply force to the voxel
                voxels_copy[i][j][0] = curr_vox + force * delta_elastic;
            }
        }
    }
    
    { //Face 2
        for (int j = 1; j < Dim.y-1; j++) {
            for (int k = 1; k < Dim.z-1; k++) {
                my_float3 curr_vox = voxels[Dim.x-1][j][k];
                my_float3 prev_vox = voxels_prev[Dim.x-1][j][k];
                // Compute the force applied to the voxel
                my_float3 force = make_my_float3(0.0f);
                // Go through all edges from the current voxel
                for (int z = -1; z < 2; z++) {
                    for (int y = -1; y < 2; y++) {
                            if (z == 0 && y == 0)
                                continue;
                            my_float3 curr_edge = curr_vox - voxels[Dim.x-1][j+y][k+z];
                            float curr_edge_length = norm(curr_edge);
                            float prev_edge_length = norm(prev_vox - voxels_prev[Dim.x-1][j+y][k+z]);
                            force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
                            energy += fabs(prev_edge_length - curr_edge_length);
                    }
                }
                
                //Apply force to the voxel
                voxels_copy[Dim.x-1][j][k] = curr_vox + force * delta_elastic;
            }
        }
    }
    
    { //Face 3
        for (int i = 1; i < Dim.x-1; i++) {
            for (int k = 1; k < Dim.z-1; k++) {
                my_float3 curr_vox = voxels[i][Dim.y-1][k];
                my_float3 prev_vox = voxels_prev[i][Dim.y-1][k];
                // Compute the force applied to the voxel
                my_float3 force = make_my_float3(0.0f);
                // Go through all edges from the current voxel
                for (int x = -1; x < 2; x++) {
                    for (int z = -1; z < 2; z++) {
                            if (x == 0 && z == 0)
                                continue;
                            my_float3 curr_edge = curr_vox - voxels[i+x][Dim.y-1][k+z];
                            float curr_edge_length = norm(curr_edge);
                            float prev_edge_length = norm(prev_vox - voxels_prev[i+x][Dim.y-1][k+z]);
                            force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
                            energy += fabs(prev_edge_length - curr_edge_length);
                    }
                }
                
                //Apply force to the voxel
                voxels_copy[i][Dim.y-1][k] = curr_vox + force * delta_elastic;
            }
        }
    }
    
    { //Face 4
        for (int j = 1; j < Dim.y-1; j++) {
            for (int k = 1; k < Dim.z-1; k++) {
                my_float3 curr_vox = voxels[0][j][k];
                my_float3 prev_vox = voxels_prev[0][j][k];
                // Compute the force applied to the voxel
                my_float3 force = make_my_float3(0.0f);
                // Go through all edges from the current voxel
                for (int z = -1; z < 2; z++) {
                    for (int y = -1; y < 2; y++) {
                            if (z == 0 && y == 0)
                                continue;
                            my_float3 curr_edge = curr_vox - voxels[0][j+y][k+z];
                            float curr_edge_length = norm(curr_edge);
                            float prev_edge_length = norm(prev_vox - voxels_prev[0][j+y][k+z]);
                            force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
                            energy += fabs(prev_edge_length - curr_edge_length);
                    }
                }
                
                //Apply force to the voxel
                voxels_copy[0][j][k] = curr_vox + force * delta_elastic;
            }
        }
    }
    
    { //Face 5
        for (int i = 1; i < Dim.x-1; i++) {
            for (int k = 1; k < Dim.z-1; k++) {
                my_float3 curr_vox = voxels[i][0][k];
                my_float3 prev_vox = voxels_prev[i][0][k];
                // Compute the force applied to the voxel
                my_float3 force = make_my_float3(0.0f);
                // Go through all edges from the current voxel
                for (int x = -1; x < 2; x++) {
                    for (int z = -1; z < 2; z++) {
                            if (x == 0 && z == 0)
                                continue;
                            my_float3 curr_edge = curr_vox - voxels[i+x][0][k+z];
                            float curr_edge_length = norm(curr_edge);
                            float prev_edge_length = norm(prev_vox - voxels_prev[i+x][0][k+z]);
                            force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
                            energy += fabs(prev_edge_length - curr_edge_length);
                    }
                }
                
                //Apply force to the voxel
                voxels_copy[i][0][k] = curr_vox + force * delta_elastic;
            }
        }
    }
    
    { //Face 6
        for (int i = 1; i < Dim.x-1; i++) {
            for (int j = 1; j < Dim.y-1; j++) {
                my_float3 curr_vox = voxels[i][j][Dim.z-1];
                my_float3 prev_vox = voxels_prev[i][j][Dim.z-1];
                // Compute the force applied to the voxel
                my_float3 force = make_my_float3(0.0f);
                // Go through all edges from the current voxel
                for (int x = -1; x < 2; x++) {
                    for (int y = -1; y < 2; y++) {
                            if (x == 0 && y == 0)
                                continue;
                            my_float3 curr_edge = curr_vox - voxels[i+x][j+y][Dim.z-1];
                            float curr_edge_length = norm(curr_edge);
                            float prev_edge_length = norm(prev_vox - voxels_prev[i+x][j+y][Dim.z-1]);
                            force = force + curr_edge * ((1.0f/curr_edge_length) * (prev_edge_length - curr_edge_length));
                            energy += fabs(prev_edge_length - curr_edge_length);
                    }
                }
                
                //Apply force to the voxel
                voxels_copy[i][j][Dim.z-1] = curr_vox + force * delta_elastic;
            }
        }
    }
    cout << "Energy: " << energy << endl;
}


void CubicMesh::FitToTSDF(float ***tsdf_grid, my_int3 dim_grid, my_float3 center, float res, int outerloop_maxiter, int innerloop_maxiter, float delta, float delta_elastic){
    
    // Compute gradient of tsdf
    my_float3 ***grad_grid = VolumetricGrad(tsdf_grid, dim_grid);
    
    // Copy of vertex position
    my_float3 ***voxels_copy;
    voxels_copy = new my_float3 **[_Dim.x];
    for (int i = 0; i < _Dim.x; i++) {
        voxels_copy[i] = new my_float3 *[_Dim.y];
        for (int j = 0; j < _Dim.y; j++) {
            voxels_copy[i][j] = new my_float3[_Dim.z];
            memcpy(voxels_copy[i][j], _voxels[i][j], _Dim.z*sizeof(my_float3));
        }
    }
    
    // Prev of vertex position
    my_float3 ***voxels_prev;
    voxels_prev = new my_float3 **[_Dim.x];
    for (int i = 0; i < _Dim.x; i++) {
        voxels_prev[i] = new my_float3 *[_Dim.y];
        for (int j = 0; j < _Dim.y; j++) {
            voxels_prev[i][j] = new my_float3[_Dim.z];
            memcpy(voxels_prev[i][j], _voxels[i][j], _Dim.z*sizeof(my_float3));
        }
    }
    
    //############## Outer loop ##################
    for (int OuterIter = 0; OuterIter < outerloop_maxiter; OuterIter++) {
        // Copy current state of voxels
        for (int i = 0; i < _Dim.x; i++) {
            for (int j = 0; j < _Dim.y; j++) {
                memcpy(voxels_prev[i][j], _voxels[i][j], _Dim.z*sizeof(my_float3));
                memcpy(voxels_copy[i][j], _voxels[i][j], _Dim.z*sizeof(my_float3));
            }
        }
        
        // Increment deplacement of surface vertices
        
        //face 1
        for (int i = 0; i < _Dim.x; i++) {
            for (int j = 0; j < _Dim.y; j++) {
                my_float3 vertex = voxels_copy[i][j][0];
                my_float3 grad = BicubiInterpolation(vertex, tsdf_grid, grad_grid, dim_grid, center, res);
                _voxels[i][j][0] = vertex + grad * delta;
                /*if (norm(grad) > 0.0f) {
                    grad = grad * (1.0f/norm(grad));
                    //_voxels[i][j][0] = (tsdf_grid[i][j][0] > 0.0f) ? vertex + grad * (delta/float(OuterIter+1)) : vertex - grad * (delta/float(OuterIter+1));
                    _voxels[i][j][0] = vertex + grad * tsdf_grid[i][j][0] * delta;
                    //_voxels[i][j][0].print();
                    //grad.print();
                }*/
            }
        }
        //continue;
        
        //face 2
        for (int i = 0; i < _Dim.x; i++) {
            for (int j = 0; j < _Dim.y; j++) {
                my_float3 vertex = voxels_copy[i][j][_Dim.z-1];
                my_float3 grad = BicubiInterpolation(vertex, tsdf_grid, grad_grid, dim_grid, center, res);
                _voxels[i][j][_Dim.z-1] = vertex + grad * delta;
            }
        }
        
        //face 3
        for (int i = 0; i < _Dim.x; i++) {
            for (int k = 0; k < _Dim.z; k++) {
                my_float3 vertex = voxels_copy[i][0][k];
                my_float3 grad = BicubiInterpolation(vertex, tsdf_grid, grad_grid, dim_grid, center, res);
                _voxels[i][0][k] = vertex + grad * delta;
            }
        }
        
        //face 4
        for (int i = 0; i < _Dim.x; i++) {
            for (int k = 0; k < _Dim.z; k++) {
                my_float3 vertex = voxels_copy[i][_Dim.y-1][k];
                my_float3 grad = BicubiInterpolation(vertex, tsdf_grid, grad_grid, dim_grid, center, res);
                _voxels[i][_Dim.y-1][k] = vertex + grad * delta;
            }
        }
        
        //face 5
        for (int j = 0; j < _Dim.y; j++) {
            for (int k = 0; k < _Dim.z; k++) {
                my_float3 vertex = voxels_copy[0][j][k];
                my_float3 grad = BicubiInterpolation(vertex, tsdf_grid, grad_grid, dim_grid, center, res);
                _voxels[0][j][k] = vertex + grad * delta;
            }
        }
        
        //face 6
        for (int j = 0; j < _Dim.y; j++) {
            for (int k = 0; k < _Dim.z; k++) {
                my_float3 vertex = voxels_copy[_Dim.x-1][j][k];
                my_float3 grad = BicubiInterpolation(vertex, tsdf_grid, grad_grid, dim_grid, center, res);
                _voxels[_Dim.x-1][j][k] = vertex + grad * delta;
            }
        }
        
        cout << "Inner loop" << endl;
        // Elastic optimization for surface points
        for (int InnerIter = 1; InnerIter < innerloop_maxiter; InnerIter++) {
            // Do mass-spring elastic propagation for each outer voxel
            
            // do all corners
            OptimizeCorners(_voxels, voxels_copy, voxels_prev, _Dim, 0.0f);
            
            // do all edges
            OptimizeEdges(_voxels, voxels_copy, voxels_prev, _Dim, 0.0f);
            
            // do all faces
            OptimizeFaces(_voxels, voxels_copy, voxels_prev, _Dim, delta_elastic);
            
            for (int i = 0; i < _Dim.x; i++) {
                for (int j = 0; j < _Dim.y; j++) {
                    memcpy(_voxels[i][j], voxels_copy[i][j], _Dim.z*sizeof(my_float3));
                }
            }
            
        }
        
        //############## Inner loop ##################
        /*for (int InnerIter = 0; InnerIter < innerloop_maxiter; InnerIter++) {
            float energy = 0.0f;
            // Do mass-spring elastic propagation for each inner voxel
            for (int i = 1; i < _Dim.x-1; i++) {
                for (int j = 1; j < _Dim.y-1; j++) {
                    for (int k = 1; k < _Dim.z-1; k++) {
                        my_float3 curr_vox = _voxels[i][j][k];
                        my_float3 prev_vox = voxels_copy[i][j][k];
                        // Compute the force applied to the voxel
                        my_float3 force = make_my_float3(0.0f);
                        // Go through all edges from the current voxel
                        for (int x = -1; x < 2; x++) {
                            for (int y = -1; y < 2; y++) {
                                for (int z = -1; z < 2; z++) {
                                    my_float3 curr_edge = curr_vox - _voxels[i+x][j+y][k+z];
                                    float curr_edge_length = norm(curr_edge);
                                    float prev_edge_length = norm(prev_vox - voxels_copy[i+x][j+y][k+z]);
                                    energy += fabs(prev_edge_length - curr_edge_length);
                                    
                                    force = force + curr_edge * ((1.0f/curr_edge_length) * kappa * (prev_edge_length - curr_edge_length));
                                }
                            }
                        }
                        
                        //Apply force to the voxel
                        _voxels[i][j][k] = curr_vox + force * delta_elastic;
                    }
                }
                cout << "Energy: " << energy << endl;
            }
        }*/
    }
    
    
    for (int i = 1; i < dim_grid.x-1; i++) {
        for (int j = 1; j < dim_grid.y-1; j++) {
            delete []grad_grid[i][j];
        }
        delete []grad_grid[i];
    }
    delete []grad_grid;
    
    for (int i = 0; i < _Dim.x; i++) {
        for (int j = 0; j < _Dim.y; j++) {
            delete []voxels_copy[i][j];
            delete []voxels_prev[i][j];
        }
        delete []voxels_copy[i];
        delete []voxels_prev[i];
    }
    delete []voxels_copy;
    delete []voxels_prev;
}
