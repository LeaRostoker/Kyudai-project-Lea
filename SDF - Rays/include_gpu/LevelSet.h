//
//  LevelSet.h
//  DEEPANIM
//
//  Created by Diego Thomas on 2021/01/14.
//  Updated by Lea Rostoker on 2022/03/22.
//

#ifndef LevelSet_h
#define LevelSet_h

#define PI 3.14159265f

void AddAvatarSDF(float** bbox, float* sdf, int* labels, int3 size_grid, float3 center_grid, float res_x, float res_y, float res_z, float delta) {

    cout << "----- Adding the avatar's sdf to the scene's sdf -----" << endl;
    
    int count_in = 0;

    for (int i = 0; i < size_grid.x; i++) {
        for (int j = 0; j < size_grid.y; j++) {
            for (int k = 0; k < size_grid.z; k++) {
                unsigned int idx = i * size_grid.y * size_grid.z + j * size_grid.z + k;
                // Get the 3D coordinate
                float3 p_curr;
                p_curr.x = (float(i) - float(size_grid.x) / 2.0f) * res_x + center_grid.x;
                p_curr.y = (float(j) - float(size_grid.y) / 2.0f) * res_y + center_grid.y;
                p_curr.z = (float(k) - float(size_grid.z) / 2.0f) * res_z + center_grid.z;

                if (   (p_curr.x >= bbox[0][0] && p_curr.x <= bbox[0][1])
                    && (p_curr.y >= bbox[1][0] && p_curr.y <= bbox[1][1])
                    && (p_curr.z >= bbox[2][0] && p_curr.z <= bbox[2][1])
                    ) 
                {
                    count_in++;
                    if (sdf[idx] >= 10e-6) {
                        sdf[idx] = - sdf[idx];
                        labels[idx] = 40;
                    }

                } 

            }
        }
    }

    cout << "count_in : " << count_in << endl;

}

#endif /* LevelSet_h */
