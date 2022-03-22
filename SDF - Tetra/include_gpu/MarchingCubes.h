//
//  MarchingCubes.h
//  DEEPANIM
//
//  Created by Diego Thomas on 2021/01/14.
//  Updated by Lea Rostoker on 2022/02/17
//

#ifndef MarchingCubes_h
#define MarchingCubes_h

//    create the precalculated 256 possible polygon configuration (128 + symmetries)
int Config[128][4][3] = { {{0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}}, //0
    {{0, 7, 3}, {0,0,0}, {0,0,0}, {0,0,0}}, // v0 1
    {{0, 1, 4}, {0,0,0}, {0,0,0}, {0,0,0}}, // v1 2
    {{1, 7, 3}, {1, 4, 7}, {0,0,0}, {0,0,0}}, // v0|v1 3
    {{1, 2, 5}, {0,0,0}, {0,0,0}, {0,0,0}}, // v2 4
    {{0, 7, 3}, {1, 2, 5}, {0,0,0}, {0,0,0}}, //v0|v2 5
    {{0, 2 ,4}, {2, 5, 4}, {0,0,0}, {0,0,0}}, //v1|v2 6
    {{3, 2, 7}, {2, 5, 7}, {5, 4, 7}, {0,0,0}}, //v0|v1|v2 7
    {{2, 3, 6}, {0,0,0}, {0,0,0}, {0,0,0}}, // v3 8
    {{0, 7, 2}, {7, 6, 2}, {0,0,0}, {0,0,0}}, //v0|v3 9
    {{0, 1, 4}, {3, 6, 2}, {0,0,0}, {0,0,0}}, //v1|v3 10
    {{1, 4, 2}, {2, 4, 6}, {4, 7, 6}, {0,0,0}}, //v0|v1|v3 11
    {{3, 5, 1}, {3, 6, 5}, {0,0,0}, {0,0,0}}, //v2|v3 12
    {{0, 7, 1}, {1, 7, 5}, {7, 6, 5}, {0,0,0}}, //v0|v2|v3 13
    {{0, 3, 4}, {3, 6, 4}, {6, 5, 4}, {0,0,0}}, //v1|v2|v3 14
    {{7, 6, 5}, {4, 7, 5}, {0,0,0}, {0,0,0}}, //v0|v1|v2|v3 15
    {{7, 8, 11}, {0,0,0}, {0,0,0}, {0,0,0}}, // v4 16
    {{0, 8, 3}, {3, 8, 11}, {0,0,0}, {0,0,0}}, //v0|v4 17
    {{0, 1, 4}, {7, 8, 11}, {0,0,0}, {0,0,0}}, //v1|v4 18
    {{1, 4, 8}, {1, 8, 11}, {1, 11, 3}, {0,0,0}}, //v0|v1|v4 19
    {{7, 8, 11}, {1, 2, 5}, {0,0,0}, {0,0,0}}, //v2|v4 20
    {{0, 8, 3}, {3, 8, 11}, {1, 2, 5}, {0,0,0}}, //v0|v2|v4 21
    {{0, 2 ,4}, {2, 5, 4}, {7, 8, 11}, {0,0,0}}, //v1|v2|v4 22
    {{4, 2, 5}, {2, 4, 11}, {4, 8, 11}, {2, 11, 3}}, //v0|v1|v2|v4 23
    {{2, 3, 6}, {7, 8, 11}, {0,0,0}, {0,0,0}}, //v3|v4 24
    {{6, 8, 11}, {2, 8, 6}, {0, 8, 2}, {0,0,0}}, //v0|v3|v4 25
    {{0, 1, 4}, {2, 3, 6}, {7, 8, 11}, {0,0,0}}, //v1|v3|v4 26
    {{6, 8, 11}, {8, 6, 2}, {4, 8, 2}, {4, 2, 1}}, //v0|v1|v3|v4 27
    {{7, 8, 11}, {3, 5, 1}, {3, 6, 5}, {0,0,0}}, //v2|v3|v4 28
    {{0, 8 ,1}, {1, 8, 6}, {1, 6, 5}, {6, 8, 11}}, //v0|v2|v3|v4 29
    {{7, 8, 11}, {0, 3, 4}, {3, 6, 4}, {6, 5, 4}}, //v1|v2|v3|v4 30
    {{6, 8, 11}, {6, 4, 8}, {5, 4, 6}, {0,0,0}}, //v0|v1|v2|v3|v4 31 =                              v5|v6|v7   //////////////////////////
    {{4, 9, 8}, {0,0,0}, {0,0,0}, {0,0,0}}, // v5 32
    {{0, 7, 3}, {4, 9, 8}, {0,0,0}, {0,0,0}}, //v0|v5 33
    {{0, 1, 8}, {1, 9, 8}, {0,0,0}, {0,0,0}}, //v1|v5 34
    {{1, 9, 3}, {3, 9, 7}, {7, 9, 8}, {0,0,0}}, //v0|v1|v5 35
    {{4, 9, 8}, {1, 2, 5}, {0,0,0}, {0,0,0}}, //v2|v5 36
    {{4, 9, 8}, {1, 2, 5}, {0, 7, 3}, {0,0,0}}, //v0|v2|v5 37
    {{0, 2 ,8}, {2, 5, 8}, {8, 5, 9}, {0,0,0}}, //v1|v2|v5 38
    {{7, 9, 8}, {3, 9, 7}, {3, 5, 9}, {2, 5, 3}}, //v0|v1|v2|v5 39
    {{4, 9, 8}, {2, 3, 6}, {0,0,0}, {0,0,0}}, //v3|v5 40
    {{4, 9, 8}, {0, 7, 2}, {7, 6, 2}, {0,0,0}}, //v0|v3|v5 41
    {{2, 3, 6}, {0, 1, 8}, {1, 9, 8}, {0,0,0}}, //v1|v3|v5 42
    {{1, 9, 2}, {2, 9, 7}, {7, 6, 9}, {7, 9, 8}}, //v0|v1|v3|v5 43
    {{4, 9, 8}, {3, 5, 1}, {3, 6, 5}, {0,0,0}}, //v2|v3|v5 44
    {{4, 9, 8}, {0, 7, 1}, {1, 7, 5}, {7, 6, 5}}, //v0|v2|v3|v5 45
    {{5, 9, 8}, {0, 3, 8}, {3, 5, 8}, {3, 6, 5}}, //v1|v2|v3|v5 46
    {{5, 7, 6}, {8, 5, 9}, {7, 5, 8}, {0,0,0}}, //v0|v1|v2|v3|v5 47                                     = v4 | v6 | v7  ////////////////////
    {{4, 9, 7}, {7, 9, 11}, {0,0,0}, {0,0,0}}, //v4|v5 48
    {{3, 9, 11}, {0, 4, 3}, {4, 9, 3}, {0,0,0}}, //v0|v4|v5 49
    {{1, 9, 11}, {0, 11, 7}, {0, 1, 11}, {0,0,0}}, //v1|v4|v5 50
    {{1, 9, 11}, {1, 11, 3}, {0,0,0}, {0,0,0}}, //v0|v1|v4|v5 51
    {{1, 2, 5}, {4, 9, 7}, {7, 9, 11}, {0,0,0}}, //v2|v4|v5 52
    {{1, 2, 5}, {3, 9, 11}, {0, 4, 3}, {4, 9, 3}}, //v0|v2|v4|v5 53
    {{0, 2, 7}, {2, 5, 9}, {2, 9, 7}, {7, 9, 11}}, //v1|v2|v4|v5 54
    {{11, 3, 9}, {3, 2, 5}, {9, 3, 5}, {0,0,0}}, //v0|v1|v2|v4|v5 55                                          = v3 v6 v7 //////////////////////////
    {{2, 3, 6}, {4, 9, 7}, {7, 9, 11}, {0,0,0}}, //v3|v4|v5 56
    {{2, 0, 4}, {6, 2, 11}, {2, 4, 11}, {4, 9, 11}}, //v0|v3|v4|v5 57
    {{2, 3, 6}, {1, 2, 5}, {4, 9, 7}, {7, 9, 11}}, //v1|v3|v4|v5 58
    {{11, 1, 9}, {2, 1, 6}, {6, 1, 11}, {0,0,0}}, //v0|v1|v3|v4|v5 59                                                               = v2 v6 v7 //////////////////
    {{7, 4, 11}, {4, 9, 11}, {3, 6, 1}, {1, 6, 5}}, //v2|v3|v4|v5 60
    {{1, 0, 4}, {11, 6, 9}, {9, 6, 5}, {0,0,0}}, //v0|v2|v3|v4|v5 61                                                                  = v1 v6 v7
    {{3, 0, 7}, {11, 6, 9}, {9, 6, 5}, {0,0,0}}, //v1|v2|v3|v4|v5 62                                                                 = v0 v6 v7
    {{11, 6, 9}, {9, 6, 5}, {0,0,0}, {0,0,0}}, //v0|v1|v2|v3|v4|v5 63                                                                 = v6 v7
    {{5, 10, 9}, {0,0,0}, {0,0,0}, {0,0,0}}, //v6 64
    {{5, 10, 9}, {0, 7, 3}, {0,0,0}, {0,0,0}}, //v0|v6 65
    {{5, 10, 9}, {0, 1, 4}, {0,0,0}, {0,0,0}}, //v1|v6 66
    {{5, 10, 9}, {1, 3, 7}, {1, 7, 4}, {0,0,0}}, //v0|v1|v6 67
    {{1, 2, 9}, {2, 10, 9}, {0,0,0}, {0,0,0}}, //v2|v6 68
    {{1, 2, 9}, {2, 10, 9}, {0, 7, 3}, {0,0,0}}, //v0|v2|v6 69
    {{0, 2, 10}, {4, 10, 9}, {0, 10, 4}, {0,0,0}}, //v1|v2|v6 70
    {{2, 10, 3}, {4, 10, 9}, {4, 3, 10}, {3, 4, 7}}, //v0|v1|v2|v6 71
    {{5, 10, 9}, {2, 3, 6}, {0,0,0}, {0,0,0}}, //v3|v6 72
    {{5, 10, 9}, {0, 7, 2}, {7, 6, 2}, {0,0,0}}, //v0|v3|v6 73
    {{5, 10, 9}, {0, 1, 4}, {2, 3, 6}, {0,0,0}}, //v1|v3|v6 74
    {{5, 10, 9}, {1, 4, 2}, {2, 4, 6}, {4, 6, 7}}, //v0|v1|v3|v6 75
    {{1, 3, 9}, {6, 10, 9}, {3, 6, 9}, {0,0,0}}, //v2|v3|v6 76
    {{0, 7, 6}, {6, 10, 9}, {0, 6, 9}, {0, 9, 1}}, //v0|v2|v3|v6 77
    {{6, 10, 9}, {3, 6, 9}, {3, 9, 4}, {0, 3, 4}}, //v1|v2|v3|v6 78
    {{4, 7, 6}, {4, 10, 9}, {4, 6, 10}, {0,0,0}}, //v0|v1|v2|v3|v6 79    v4 v5 v7   ////////////////////////////
    {{5, 10, 9}, {7, 8, 11}, {0,0,0}, {0,0,0}}, //v4|v6 80
    {{5, 10, 9}, {0, 8, 3}, {3, 8, 11}, {0,0,0}}, //v0|v4|v6 81
    {{0, 1, 4}, {7, 8, 11}, {5, 10, 9}, {0,0,0}}, //v1|v4|v6 82
    {{5, 10, 9}, {1, 4, 8}, {1, 8, 11}, {1, 11, 3}}, //v0|v1|v4|v6 83
    {{1, 2, 9}, {2, 10, 9}, {7, 8, 11}, {0,0,0}}, //v2|v4|v6 84
    {{1, 2, 9}, {2, 10, 9}, {0, 8, 3}, {3, 8, 11}}, //v0|v2|v4|v6 85
    {{7, 8, 11}, {0, 2, 10}, {4, 10, 9}, {0, 10, 4}}, //v1|v2|v4|v6 86
    {{4, 8, 9}, {3, 2, 11}, {2, 10, 11}, {0,0,0}}, //v0|v1|v2|v4|v6 87                      = v3 v5 v7
    {{2, 3, 6}, {7, 8, 11}, {5, 10, 9}, {0,0,0}}, //v3|v4|v6 88
    {{5, 10, 9}, {6, 8, 11}, {2, 8, 6}, {0, 8, 2}}, //v0|v3|v4|v6 89
    {{0, 1, 4}, {2, 3, 6}, {7, 8, 11}, {5, 10, 9}}, //v1|v3|v4|v6 90
    {{2, 1, 5}, {9, 4, 8}, {11, 6, 10}, {0,0,0}}, //v0|v1|v3|v4|v6 91     = v2 v5 v7   //////////////////////
    {{7, 8, 11}, {1, 3, 9}, {6, 10, 9}, {3, 6, 9}}, //v2|v3|v4|v6 92
    {{11, 6, 10}, {1, 0, 8}, {9, 1, 8}, {0,0,0}}, //v0|v2|v3|v4|v6 93                              = v1 v5 v7 //////////
    {{0, 3, 7}, {9, 4, 8}, {10, 5, 9}, {0,0,0}}, //v1|v2|v3|v4|v6 94                                      = v0 v5 v7  ////////////
    {{11, 6, 10}, {9, 4, 8}, {0,0,0}, {0,0,0}}, //v0|v1|v2|v3|v4|v6 95                                               = v5 v7 ////////////
    {{4, 5, 8}, {8, 5, 10}, {0,0,0}, {0,0,0}}, //v5|v6 96
    {{0, 7, 3}, {4, 5, 8}, {8, 5, 10}, {0,0,0}}, //v0|v5|v6 97
    {{0, 10, 8}, {1, 5, 10}, {0, 1, 10}, {0,0,0}}, //v1|v5|v6 98
    {{1, 5, 10}, {8, 7, 10}, {1, 10, 7}, {3, 1, 7}}, //v0|v1|v5|v6 99
    {{2, 10, 8}, {1, 8, 4}, {1, 2, 8}, {0,0,0}}, //v2|v5|v6 100
    {{2, 10, 8}, {1, 8, 4}, {1, 2, 8}, {0, 7, 3}}, //v0|v2|v5|v6 101
    {{0, 10, 8}, {0, 2, 10}, {0,0,0}, {0,0,0}}, //v1|v2|v5|v6 102
    {{8, 2, 10}, {3, 8, 7}, {3, 2, 8}, {0,0,0}}, //v0|v1|v2|v5|v6 103                                                         = v3 v4 v7 //////////////
    {{2, 3, 6}, {4, 5, 8}, {8, 5, 10}, {0,0,0}}, //v3|v5|v6 104
    {{4, 5, 8}, {8, 5, 10}, {0, 7, 2}, {7, 6, 2}}, //v0|v3|v5|v6 105
    {{2, 3, 6}, {0, 10, 8}, {1, 5, 10}, {0, 1, 10}}, //v1|v3|v5|v6 106
    {{1, 5, 2}, {7, 6, 8}, {6, 10, 8}, {0,0,0}}, //v0|v1|v3|v5|v6 107                                                 =v2 v4 v7 ////////////////////////
    {{3, 6, 10}, {1, 3, 4}, {4, 3, 10}, {4, 10, 8}}, //v2|v3|v5|v6 108
    {{1, 0, 4}, {7, 6, 8}, {6, 10, 8}, {0,0,0}}, //v0|v2|v3|v5|v6 109                                                 = v1 v4 v7 //////////
    {{0, 10, 8}, {0, 3, 6}, {0, 6, 10}, {0,0,0}}, //v1|v2|v3|v5|v6 110                                                = v0 v4 v7 //////////
    {{7, 6, 8}, {6, 10, 8}, {0,0,0}, {0,0,0}}, //v0|v1|v2|v3|v5|v6 111                                                = v4 v7 //////////
    {{4, 5, 7}, {7, 10, 11}, {7, 5, 10}, {0,0,0}}, //v4|v5|v6 112
    {{5, 10, 11}, {0, 4, 5}, {0, 5, 11}, {0, 11, 3}}, //v0|v4|v5|v6 113
    {{0, 1, 5}, {0, 5, 10}, {0, 10, 7}, {7, 10, 11}}, //v1|v4|v5|v6 114
    {{3, 1, 11}, {1, 5, 10}, {1, 10, 11}, {0,0,0}}, //v0|v1|v4|v5|v6 115                                             = v2 v3 v7 ////////////
    {{7, 10, 11}, {4, 11, 7}, {4, 10, 11}, {1, 10, 4}}, //v2|v4|v5|v6 116
    {{0, 4, 1}, {3, 2, 11}, {2, 10, 11}, {0,0,0}}, //v0|v2|v4|v5|v6 117                                            = v1 v3 v7 ////////////
    {{0, 2, 10}, {0, 11, 7}, {0, 10, 11}, {0,0,0}}, //v1|v2|v4|v5|v6 118                                            = v0 v3 v7 ////////////
    {{3, 2, 11}, {2, 10, 11}, {0,0,0}, {0,0,0}}, //v0|v1|v2|v4|v5|v6 119                                           = v3 v7 ////////////
    {{2, 3, 6}, {4, 5, 7}, {7, 10, 11}, {7, 5, 10}}, //v3|v4|v5|v6 120
    {{6, 10, 11}, {0, 4, 2}, {2, 4, 5}, {0,0,0}}, //v0|v3|v4|v5|v6 121                                          = v1 v2 v7 ////////////
    {{3, 0, 7}, {2, 1, 5}, {11, 6, 10}, {0,0,0}}, //v1|v3|v4|v5|v6 122                                          = v0 v2 v7 ////////////
    {{2, 1, 5}, {11, 6, 10}, {0,0,0}, {0,0,0}}, //v0|v1|v3|v4|v5|v6 123                                          = v2 v7 ////////////
    {{3, 1, 7}, {7, 1, 4}, {11, 6, 10}, {0,0,0}}, //v2|v3|v4|v5|v6 124                                         = v0 v1 v7 ////////////
    {{1, 0, 4}, {11, 6, 10}, {0,0,0}, {0,0,0}}, //v0|v2|v3|v4|v5|v6 125                                         = v1 v7 ////////////
    {{3, 0, 7}, {11, 6, 10}, {0,0,0}, {0,0,0}}, //v1|v2|v3|v4|v5|v6 126                                         = v0 v7 ////////////
    {{11, 6, 10}, {0,0,0}, {0,0,0}, {0,0,0}} //v0|v1|v2|v3|v4|v5|v6 127                                         = v7 ////////////
};

int ConfigCount[128] = { 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 2, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 1, 2, 2, 3,
    2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 2, 3, 3, 2, 3, 4, 4, 3, 3, 4, 4, 3, 4, 3, 3, 2, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4,
    3, 2, 3, 3, 4, 3, 4, 4, 3, 3, 4, 4, 3, 4, 3, 3, 2, 2, 3, 3, 4, 3, 4, 2, 3, 3, 4, 4, 3, 4, 3, 3, 2, 3, 4, 4, 3, 4, 3, 3, 2, 4, 3,
    3, 2, 3, 2, 2, 1 };


int Indexing(float ***sdf, int ***IndexVal, int ***Offset, my_int3 size_grid, float iso) {
    
    int faces_counter = 0;
    for (int x = 1; x < size_grid.x-2; x++) {
        for (int y = 1; y < size_grid.y-2; y++) {
            for (int z = 1; z < size_grid.z-2; z++) {
            
                int s[8][3] = {{x, y, z}, {x+1,y , z}, {x+1, y+1, z}, {x, y + 1, z},
                    {x, y, z+1}, {x+1, y, z+1}, {x+1, y+1, z+1}, {x, y+1, z+1}};
                float vals[8] = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
                
                // get the values of the implicit function at the summits
                // [val_0 ... val_7]
                bool valid = true;
                for (int k=0; k < 8; k++) {
                    vals[k] = sdf[s[k][0]][s[k][1]][s[k][2]];
                    // Next code should be used only for truncated signed distance function
                    /*if (fabs(vals[k]) >= 1.0f) {
                        IndexVal[x][y][z] = 0;
                        valid = false;
                        break;
                    }*/
                }
                
                if (!valid)
                    continue;
                
                int index_val = 0;
                // get the index value corresponding to the implicit function
                if (vals[7] <= iso) {
                    index_val = (int)(vals[0] > iso) +
                    (int)(vals[1] > iso)*2 +
                    (int)(vals[2] > iso)*4 +
                    (int)(vals[3] > iso)*8 +
                    (int)(vals[4] > iso)*16 +
                    (int)(vals[5] > iso)*32 +
                    (int)(vals[6] > iso)*64;
                    IndexVal[x][y][z] = index_val;
                } else{
                    index_val = (int)(vals[0] <= iso) +
                    (int)(vals[1] <= iso)*2 +
                    (int)(vals[2] <= iso)*4 +
                    (int)(vals[3] <= iso)*8 +
                    (int)(vals[4] <= iso)*16 +
                    (int)(vals[5] <= iso)*32 +
                    (int)(vals[6] <= iso)*64;
                    IndexVal[x][y][z] = -index_val;
                }
                
                // get the corresponding configuration
                if (index_val == 0)
                    continue;
                
                Offset[x][y][z] = faces_counter;
                faces_counter += ConfigCount[index_val];
            }
        }
    }
    return faces_counter;
}

void InnerLoop_MC(float *vertices, float* colors, float *normals, int *faces, int x, float ***sdf, int ***sdf_l, int ***IndexVal, int ***Offset, my_int3 size_grid, my_float3 center_grid, float res, float iso) {
    
    for (int y = 1; y < size_grid.y-2; y++) {
        if (x == 1)
            std::cout << 100.0f*float(y)/float(size_grid.y) << endl;
        for (int z = 1; z < size_grid.z-2; z++) {
            int index = IndexVal[x][y][z];
            
            if (index == 0)
                continue;
            
            float s[8][3] = {{0.0f,0.0f, 0.0f}, {0.0f,0.0f, 0.0f}, {0.0f,0.0f, 0.0f}, {0.0f,0.0f, 0.0f},
                {0.0f,0.0f, 0.0f}, {0.0f,0.0f, 0.0f}, {0.0f,0.0f, 0.0f}, {0.0f,0.0f, 0.0f}};
            
            float nmle[8][3] = {{0.0f,0.0f, 0.0f}, {0.0f,0.0f, 0.0f}, {0.0f,0.0f, 0.0f}, {0.0f,0.0f, 0.0f},
                {0.0f,0.0f, 0.0f}, {0.0f,0.0f, 0.0f}, {0.0f,0.0f, 0.0f}, {0.0f,0.0f, 0.0f}};
            
            float v[12][3] = {{0.0f,0.0f, 0.0f}, {0.0f,0.0f, 0.0f}, {0.0f,0.0f, 0.0f}, {0.0f,0.0f, 0.0f},
                {0.0f,0.0f, 0.0f}, {0.0f,0.0f, 0.0f}, {0.0f,0.0f, 0.0f}, {0.0f,0.0f, 0.0f},
                {0.0f,0.0f, 0.0f}, {0.0f,0.0f, 0.0f}, {0.0f,0.0f, 0.0f}, {0.0f,0.0f, 0.0f}};
            
            float n[12][3] = {{0.0f,0.0f, 0.0f}, {0.0f,0.0f, 0.0f}, {0.0f,0.0f, 0.0f}, {0.0f,0.0f, 0.0f},
                {0.0f,0.0f, 0.0f}, {0.0f,0.0f, 0.0f}, {0.0f,0.0f, 0.0f}, {0.0f,0.0f, 0.0f},
                {0.0f,0.0f, 0.0f}, {0.0f,0.0f, 0.0f}, {0.0f,0.0f, 0.0f}, {0.0f,0.0f, 0.0f}};

            int l[12] = { 0,0,0,0,0,0,0,0,0,0,0,0 };
            
            float vals[8] = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
            
            // get the 8  current summits
            s[0][0] = center_grid.x + (float(x) - float(size_grid.x)/2.0f)*res;
            s[0][1] = center_grid.y + (float(y) - float(size_grid.y)/2.0f)*res;
            s[1][0] = center_grid.x + (float(x+1) - float(size_grid.x)/2.0f)*res;
            s[1][1] = center_grid.y + (float(y) - float(size_grid.y)/2.0f)*res;
            s[2][0] = center_grid.x + (float(x+1) - float(size_grid.x)/2.0f)*res;
            s[2][1] = center_grid.y + (float(y+1) - float(size_grid.y)/2.0f)*res;
            s[3][0] = center_grid.x + (float(x) - float(size_grid.x)/2.0f)*res;
            s[3][1] = center_grid.y + (float(y+1) - float(size_grid.y)/2.0f)*res;
            s[4][0] = center_grid.x + (float(x) - float(size_grid.x)/2.0f)*res;
            s[4][1] = center_grid.y + (float(y) - float(size_grid.y)/2.0f)*res;
            s[5][0] = center_grid.x + (float(x+1) - float(size_grid.x)/2.0f)*res;
            s[5][1] = center_grid.y + (float(y) - float(size_grid.y)/2.0f)*res;
            s[6][0] = center_grid.x + (float(x+1) - float(size_grid.x)/2.0f)*res;
            s[6][1] = center_grid.y + (float(y+1) - float(size_grid.y)/2.0f)*res;
            s[7][0] = center_grid.x + (float(x) - float(size_grid.x)/2.0f)*res;
            s[7][1] = center_grid.y + (float(y+1) - float(size_grid.y)/2.0f)*res;
            
            // get the 8  current summits
            s[0][2] = center_grid.z + (float(z) - float(size_grid.z)/2.0f)*res;
            s[1][2] = center_grid.z + (float(z) - float(size_grid.z)/2.0f)*res;
            s[2][2] = center_grid.z + (float(z) - float(size_grid.z)/2.0f)*res;
            s[3][2] = center_grid.z + (float(z) - float(size_grid.z)/2.0f)*res;
            s[4][2] = center_grid.z + (float(z+1) - float(size_grid.z)/2.0f)*res;
            s[5][2] = center_grid.z + (float(z+1) - float(size_grid.z)/2.0f)*res;
            s[6][2] = center_grid.z + (float(z+1) - float(size_grid.z)/2.0f)*res;
            s[7][2] = center_grid.z + (float(z+1) - float(size_grid.z)/2.0f)*res;
            
            bool reverse = false;
            if (index < 0) {
                reverse = true;
                index = -index;
            }
            
            //convert TSDF to float
            float tsdf0 = sdf[x][y][z];
            float tsdf1 = sdf[x+1][y][z];
            float tsdf2 = sdf[x+1][y+1][z];
            float tsdf3 = sdf[x][y+1][z];
            float tsdf4 = sdf[x][y][z+1];
            float tsdf5 = sdf[x+1][y][z+1];
            float tsdf6 = sdf[x+1][y+1][z+1];
            float tsdf7 = sdf[x][y+1][z+1];

            int label0 = sdf_l[x][y][z];
            int label1 = sdf_l[x + 1][y][z];
            int label2 = sdf_l[x + 1][y + 1][z];
            int label3 = sdf_l[x][y + 1][z];
            int label4 = sdf_l[x][y][z + 1];
            int label5 = sdf_l[x + 1][y][z + 1];
            int label6 = sdf_l[x + 1][y + 1][z + 1];
            int label7 = sdf_l[x][y + 1][z + 1];
            
            // Compute normals at grid vertices
            nmle[0][0] = sdf[x-1][y][z] - sdf[x+1][y][z];
            nmle[0][1] = sdf[x][y-1][z] - sdf[x][y+1][z];
            nmle[0][2] = sdf[x][y][z-1] - sdf[x][y][z+1];
            
            float mag = sqrt(nmle[0][0]*nmle[0][0] + nmle[0][1]*nmle[0][1] + nmle[0][2]*nmle[0][2]);
            if (mag > 0.0f) {
                nmle[0][0] = nmle[0][0]/mag; nmle[0][1] = nmle[0][1]/mag; nmle[0][2] = nmle[0][2]/mag;
            }
            
            nmle[1][0] = sdf[x][y][z] - sdf[x+2][y][z];
            nmle[1][1] = sdf[x+1][y-1][z] - sdf[x+1][y+1][z];
            nmle[1][2] = sdf[x+1][y][z-1] - sdf[x+1][y][z+1];
            
            mag = sqrt(nmle[1][0]*nmle[1][0] + nmle[1][1]*nmle[1][1] + nmle[1][2]*nmle[1][2]);
            if (mag > 0.0f) {
                nmle[1][0] = nmle[1][0]/mag; nmle[1][1] = nmle[1][1]/mag; nmle[1][2] = nmle[1][2]/mag;
            }
            
            nmle[2][0] = sdf[x][y+1][z] - sdf[x+2][y+1][z];
            nmle[2][1] = sdf[x+1][y][z] - sdf[x+1][y+2][z];
            nmle[2][2] = sdf[x+1][y+1][z] - sdf[x+1][y+1][z+1];
            
            mag = sqrt(nmle[2][0]*nmle[2][0] + nmle[2][1]*nmle[2][1] + nmle[2][2]*nmle[2][2]);
            if (mag > 0.0f) {
                nmle[2][0] = nmle[2][0]/mag; nmle[2][1] = nmle[2][1]/mag; nmle[2][2] = nmle[2][2]/mag;
            }
            
            nmle[3][0] = sdf[x-1][y+1][z] - sdf[x+1][y+1][z];
            nmle[3][1] = sdf[x][y][z] - sdf[x][y+2][z];
            nmle[3][2] = sdf[x][y+1][z-1] - sdf[x][y+1][z+1];
            
            mag = sqrt(nmle[3][0]*nmle[3][0] + nmle[3][1]*nmle[3][1] + nmle[3][2]*nmle[3][2]);
            if (mag > 0.0f) {
                nmle[3][0] = nmle[3][0]/mag; nmle[3][1] = nmle[3][1]/mag; nmle[3][2] = nmle[3][2]/mag;
            }
            
            nmle[4][0] = sdf[x-1][y][z+1] - sdf[x+1][y][z+1];
            nmle[4][1] = sdf[x][y-1][z+1] - sdf[x][y+1][z+1];
            nmle[4][2] = sdf[x][y][z] - sdf[x][y][z+2];
            
            mag = sqrt(nmle[4][0]*nmle[4][0] + nmle[4][1]*nmle[4][1] + nmle[4][2]*nmle[4][2]);
            if (mag > 0.0f) {
                nmle[4][0] = nmle[4][0]/mag; nmle[4][1] = nmle[4][1]/mag; nmle[4][2] = nmle[4][2]/mag;
            }
            
            nmle[5][0] = sdf[x][y][z+1] - sdf[x+2][y][z+1];
            nmle[5][1] = sdf[x+1][y-1][z+1] - sdf[x+1][y+1][z+1];
            nmle[5][2] = sdf[x+1][y][z] - sdf[x+1][y][z+2];
            
            mag = sqrt(nmle[5][0]*nmle[5][0] + nmle[5][1]*nmle[5][1] + nmle[5][2]*nmle[5][2]);
            if (mag > 0.0f) {
                nmle[5][0] = nmle[5][0]/mag; nmle[5][1] = nmle[5][1]/mag; nmle[5][2] = nmle[5][2]/mag;
            }
            
            nmle[6][0] = sdf[x][y+1][z+1] - sdf[x+2][y+1][z+1];
            nmle[6][1] = sdf[x+1][y][z+1] - sdf[x+1][y+2][z+1];
            nmle[6][2] = sdf[x+1][y+1][z] - sdf[x+1][y+1][z+2];
            
            mag = sqrt(nmle[6][0]*nmle[6][0] + nmle[6][1]*nmle[6][1] + nmle[6][2]*nmle[6][2]);
            if (mag > 0.0f) {
                nmle[6][0] = nmle[6][0]/mag; nmle[6][1] = nmle[6][1]/mag; nmle[6][2] = nmle[6][2]/mag;
            }
            
            nmle[7][0] = sdf[x-1][y+1][z+1] - sdf[x+1][y+1][z+1];
            nmle[7][1] = sdf[x][y][z+1] - sdf[x][y+2][z+1];
            nmle[7][2] = sdf[x][y+1][z] - sdf[x][y+1][z+2];
            
            mag = sqrt(nmle[7][0]*nmle[7][0] + nmle[7][1]*nmle[7][1] + nmle[7][2]*nmle[7][2]);
            if (mag > 0.0f) {
                nmle[7][0] = nmle[7][0]/mag; nmle[7][1] = nmle[7][1]/mag; nmle[7][2] = nmle[7][2]/mag;
            }
                        
            // get the values of the implicit function at the summits
            // [val_0 ... val_7]
            vals[0] = fabs(tsdf0) < 1.0f ? 1.0f - fabs(tsdf0) : 0.00001f; //1.0f/(0.00001f + fabs(tsdf0)*fabs(tsdf0));
            vals[1] = fabs(tsdf1) < 1.0f ? 1.0f - fabs(tsdf1) : 0.00001f; //1.0f/(0.00001f + fabs(tsdf1)*fabs(tsdf1));
            vals[2] = fabs(tsdf2) < 1.0f ? 1.0f - fabs(tsdf2) : 0.00001f; //1.0f/(0.00001f + fabs(tsdf2)*fabs(tsdf2));
            vals[3] = fabs(tsdf3) < 1.0f ? 1.0f - fabs(tsdf3) : 0.00001f; //1.0f/(0.00001f + fabs(tsdf3)*fabs(tsdf3));
            vals[4] = fabs(tsdf4) < 1.0f ? 1.0f - fabs(tsdf4) : 0.00001f; //1.0f/(0.00001f + fabs(tsdf4)*fabs(tsdf4));
            vals[5] = fabs(tsdf5) < 1.0f ? 1.0f - fabs(tsdf5) : 0.00001f; //1.0f/(0.00001f + fabs(tsdf5)*fabs(tsdf5));
            vals[6] = fabs(tsdf6) < 1.0f ? 1.0f - fabs(tsdf6) : 0.00001f; //1.0f/(0.00001f + fabs(tsdf6)*fabs(tsdf6));
            vals[7] = fabs(tsdf7) < 1.0f ? 1.0f - fabs(tsdf7) : 0.00001f; //1.0f/(0.00001f + fabs(tsdf7)*fabs(tsdf7));
            
            /*vals[0] = 1.0f/(0.00001f + fabs(tsdf0)*fabs(tsdf0));
            vals[1] = 1.0f/(0.00001f + fabs(tsdf1)*fabs(tsdf1));
            vals[2] = 1.0f/(0.00001f + fabs(tsdf2)*fabs(tsdf2));
            vals[3] = 1.0f/(0.00001f + fabs(tsdf3)*fabs(tsdf3));
            vals[4] = 1.0f/(0.00001f + fabs(tsdf4)*fabs(tsdf4));
            vals[5] = 1.0f/(0.00001f + fabs(tsdf5)*fabs(tsdf5));
            vals[6] = 1.0f/(0.00001f + fabs(tsdf6)*fabs(tsdf6));
            vals[7] = 1.0f/(0.00001f + fabs(tsdf7)*fabs(tsdf7));
            */
            
            int nb_faces = ConfigCount[index];
            int offset = Offset[x][y][z];
            
            //Compute the position on the edge
            v[0][0] = (vals[0]*s[0][0] + vals[1]*s[1][0])/(vals[0]+vals[1]); v[0][1] = (vals[0]*s[0][1] + vals[1]*s[1][1])/(vals[0]+vals[1]);
            v[1][0] = (vals[1]*s[1][0] + vals[2]*s[2][0])/(vals[1]+vals[2]); v[1][1] = (vals[1]*s[1][1] + vals[2]*s[2][1])/(vals[1]+vals[2]);
            v[2][0] = (vals[2]*s[2][0] + vals[3]*s[3][0])/(vals[2]+vals[3]); v[2][1] = (vals[2]*s[2][1] + vals[3]*s[3][1])/(vals[2]+vals[3]);
            v[3][0] = (vals[0]*s[0][0] + vals[3]*s[3][0])/(vals[0]+vals[3]); v[3][1] = (vals[0]*s[0][1] + vals[3]*s[3][1])/(vals[0]+vals[3]);
            v[4][0] = (vals[1]*s[1][0] + vals[5]*s[5][0])/(vals[1]+vals[5]); v[4][1] = (vals[1]*s[1][1] + vals[5]*s[5][1])/(vals[1]+vals[5]);
            v[5][0] = (vals[2]*s[2][0] + vals[6]*s[6][0])/(vals[2]+vals[6]); v[5][1] = (vals[2]*s[2][1] + vals[6]*s[6][1])/(vals[2]+vals[6]);
            v[6][0] = (vals[3]*s[3][0] + vals[7]*s[7][0])/(vals[3]+vals[7]); v[6][1] = (vals[3]*s[3][1] + vals[7]*s[7][1])/(vals[3]+vals[7]);
            v[7][0] = (vals[0]*s[0][0] + vals[4]*s[4][0])/(vals[0]+vals[4]); v[7][1] = (vals[0]*s[0][1] + vals[4]*s[4][1])/(vals[0]+vals[4]);
            v[8][0] = (vals[4]*s[4][0] + vals[5]*s[5][0])/(vals[4]+vals[5]); v[8][1] = (vals[4]*s[4][1] + vals[5]*s[5][1])/(vals[4]+vals[5]);
            v[9][0] = (vals[5]*s[5][0] + vals[6]*s[6][0])/(vals[5]+vals[6]); v[9][1] = (vals[5]*s[5][1] + vals[6]*s[6][1])/(vals[5]+vals[6]);
            v[10][0] = (vals[6]*s[6][0] + vals[7]*s[7][0])/(vals[6]+vals[7]); v[10][1] = (vals[6]*s[6][1] + vals[7]*s[7][1])/(vals[6]+vals[7]);
            v[11][0] = (vals[4]*s[4][0] + vals[7]*s[7][0])/(vals[4]+vals[7]); v[11][1] = (vals[4]*s[4][1] + vals[7]*s[7][1])/(vals[4]+vals[7]);
            
            v[0][2] = (vals[0]*s[0][2] + vals[1]*s[1][2])/(vals[0]+vals[1]);
            v[1][2] = (vals[1]*s[1][2] + vals[2]*s[2][2])/(vals[1]+vals[2]);
            v[2][2] = (vals[2]*s[2][2] + vals[3]*s[3][2])/(vals[2]+vals[3]);
            v[3][2] = (vals[0]*s[0][2] + vals[3]*s[3][2])/(vals[0]+vals[3]);
            v[4][2] = (vals[1]*s[1][2] + vals[5]*s[5][2])/(vals[1]+vals[5]);
            v[5][2] = (vals[2]*s[2][2] + vals[6]*s[6][2])/(vals[2]+vals[6]);
            v[6][2] = (vals[3]*s[3][2] + vals[7]*s[7][2])/(vals[3]+vals[7]);
            v[7][2] = (vals[0]*s[0][2] + vals[4]*s[4][2])/(vals[0]+vals[4]);
            v[8][2] = (vals[4]*s[4][2] + vals[5]*s[5][2])/(vals[4]+vals[5]);
            v[9][2] = (vals[5]*s[5][2] + vals[6]*s[6][2])/(vals[5]+vals[6]);
            v[10][2] = (vals[6]*s[6][2] + vals[7]*s[7][2])/(vals[6]+vals[7]);
            v[11][2] = (vals[4]*s[4][2] + vals[7]*s[7][2])/(vals[4]+vals[7]);

            l[0] = vals[0] > vals[1] ? label0 : label1;
            l[1] = vals[1] > vals[2] ? label1 : label2;
            l[2] = vals[2] > vals[3] ? label2 : label3;
            l[3] = vals[0] > vals[3] ? label0 : label3;
            l[4] = vals[1] > vals[5] ? label0 : label5; 
            l[5] = vals[2] > vals[6] ? label2 : label6;
            l[6] = vals[3] > vals[7] ? label3 : label7;
            l[7] = vals[0] > vals[4] ? label0 : label4;
            l[8] = vals[4] > vals[5] ? label4 : label5;
            l[9] = vals[5] > vals[6] ? label5 : label6; 
            l[10] = vals[6] > vals[7] ? label6 : label7;
            l[11] = vals[4] > vals[7] ? label4 : label7;
            
            //Compute the normals
            n[0][0] = (vals[0]*nmle[0][0] + vals[1]*nmle[1][0])/(vals[0]+vals[1]);
            n[0][1] = (vals[0]*nmle[0][1] + vals[1]*nmle[1][1])/(vals[0]+vals[1]);
            n[1][0] = (vals[1]*nmle[1][0] + vals[2]*nmle[2][0])/(vals[1]+vals[2]);
            n[1][1] = (vals[1]*nmle[1][1] + vals[2]*nmle[2][1])/(vals[1]+vals[2]);
            n[2][0] = (vals[2]*nmle[2][0] + vals[3]*nmle[3][0])/(vals[2]+vals[3]);
            n[2][1] = (vals[2]*nmle[2][1] + vals[3]*nmle[3][1])/(vals[2]+vals[3]);
            n[3][0] = (vals[0]*nmle[0][0] + vals[3]*nmle[3][0])/(vals[0]+vals[3]);
            n[3][1] = (vals[0]*nmle[0][1] + vals[3]*nmle[3][1])/(vals[0]+vals[3]);
            n[4][0] = (vals[1]*nmle[1][0] + vals[5]*nmle[5][0])/(vals[1]+vals[5]);
            n[4][1] = (vals[1]*nmle[1][1] + vals[5]*nmle[5][1])/(vals[1]+vals[5]);
            n[5][0] = (vals[2]*nmle[2][0] + vals[6]*nmle[6][0])/(vals[2]+vals[6]);
            n[5][1] = (vals[2]*nmle[2][1] + vals[6]*nmle[6][1])/(vals[2]+vals[6]);
            n[6][0] = (vals[3]*nmle[3][0] + vals[7]*nmle[7][0])/(vals[3]+vals[7]);
            n[6][1] = (vals[3]*nmle[3][1] + vals[7]*nmle[7][1])/(vals[3]+vals[7]);
            n[7][0] = (vals[0]*nmle[0][0] + vals[4]*nmle[4][0])/(vals[0]+vals[4]);
            n[7][1] = (vals[0]*nmle[0][1] + vals[4]*nmle[4][1])/(vals[0]+vals[4]);
            n[8][0] = (vals[4]*nmle[4][0] + vals[5]*nmle[5][0])/(vals[4]+vals[5]);
            n[8][1] = (vals[4]*nmle[4][1] + vals[5]*nmle[5][1])/(vals[4]+vals[5]);
            n[9][0] = (vals[5]*nmle[5][0] + vals[6]*nmle[6][0])/(vals[5]+vals[6]);
            n[9][1] = (vals[5]*nmle[5][1] + vals[6]*nmle[6][1])/(vals[5]+vals[6]);
            n[10][0] = (vals[6]*nmle[6][0] + vals[7]*nmle[7][0])/(vals[6]+vals[7]);
            n[10][1] = (vals[6]*nmle[6][1] + vals[7]*nmle[7][1])/(vals[6]+vals[7]);
            n[11][0] = (vals[4]*nmle[4][0] + vals[7]*nmle[7][0])/(vals[4]+vals[7]);
            n[11][1] = (vals[4]*nmle[4][1] + vals[7]*nmle[7][1])/(vals[4]+vals[7]);
            
            n[0][2] = (vals[0]*nmle[0][2] + vals[1]*nmle[1][2])/(vals[0]+vals[1]);
            n[1][2] = (vals[1]*nmle[1][2] + vals[2]*nmle[2][2])/(vals[1]+vals[2]);
            n[2][2] = (vals[2]*nmle[2][2] + vals[3]*nmle[3][2])/(vals[2]+vals[3]);
            n[3][2] = (vals[0]*nmle[0][2] + vals[3]*nmle[3][2])/(vals[0]+vals[3]);
            n[4][2] = (vals[1]*nmle[1][2] + vals[5]*nmle[5][2])/(vals[1]+vals[5]);
            n[5][2] = (vals[2]*nmle[2][2] + vals[6]*nmle[6][2])/(vals[2]+vals[6]);
            n[6][2] = (vals[3]*nmle[3][2] + vals[7]*nmle[7][2])/(vals[3]+vals[7]);
            n[7][2] = (vals[0]*nmle[0][2] + vals[4]*nmle[4][2])/(vals[0]+vals[4]);
            n[8][2] = (vals[4]*nmle[4][2] + vals[5]*nmle[5][2])/(vals[4]+vals[5]);
            n[9][2] = (vals[5]*nmle[5][2] + vals[6]*nmle[6][2])/(vals[5]+vals[6]);
            n[10][2] = (vals[6]*nmle[6][2] + vals[7]*nmle[7][2])/(vals[6]+vals[7]);
            n[11][2] = (vals[4]*nmle[4][2] + vals[7]*nmle[7][2])/(vals[4]+vals[7]);
            
            for (int i = 0; i < 12; i++) {
                mag = sqrt(n[i][0]*n[i][0] + n[i][1]*n[i][1] + n[i][2]*n[i][2]);
                if (mag > 0.0f) {
                    n[i][0] = n[i][0]/mag; n[i][1] = n[i][1]/mag; n[i][2] = n[i][2]/mag;
                }
            }
            
            // add new faces in the list
            int f = 0;
            for ( f = 0; f < nb_faces; f++) {
                if (reverse) {
                    faces[3*(offset+f)] = 3*(offset+f)+2;
                    faces[3*(offset+f) +1] = 3*(offset+f)+1;
                    faces[3*(offset+f) + 2] = 3*(offset+f);
                } else {
                    faces[3*(offset+f)] = 3*(offset+f);
                    faces[3*(offset+f) +1] = 3*(offset+f)+1;
                    faces[3*(offset+f) + 2] = 3*(offset+f)+2;
                }
                
                int id_f_0 = Config[index][f][0];
                int id_f_1 = Config[index][f][1];
                int id_f_2 = Config[index][f][2];
                
                vertices[9*(offset+f)] = v[id_f_0][0];
                vertices[9*(offset+f)+1] = v[id_f_0][1];
                vertices[9*(offset+f)+2] = v[id_f_0][2];
                
                vertices[9*(offset+f)+3] = v[id_f_1][0];
                vertices[9*(offset+f)+4] = v[id_f_1][1];
                vertices[9*(offset+f)+5] = v[id_f_1][2];
                
                vertices[9*(offset+f)+6] = v[id_f_2][0];
                vertices[9*(offset+f)+7] = v[id_f_2][1];
                vertices[9*(offset+f)+8] = v[id_f_2][2];

                colors[9 * (offset + f)] = float(l[id_f_0]*777 % 255)/255.0f;
                colors[9 * (offset + f) + 1] = float(l[id_f_0] * 5677 % 255) / 255.0f;
                colors[9 * (offset + f) + 2] = float(l[id_f_0] * 11246 % 255) / 255.0f;

                colors[9 * (offset + f) + 3] = float(l[id_f_1] * 777 % 255) / 255.0f;
                colors[9 * (offset + f) + 4] = float(l[id_f_1] * 5677 % 255) / 255.0f; 
                colors[9 * (offset + f) + 5] = float(l[id_f_1] * 11246 % 255) / 255.0f; 

                colors[9 * (offset + f) + 6] = float(l[id_f_2] * 777 % 255) / 255.0f; 
                colors[9 * (offset + f) + 7] = float(l[id_f_2] * 5677 % 255) / 255.0f; 
                colors[9 * (offset + f) + 8] = float(l[id_f_2] * 11246 % 255) / 255.0f; 
                
                normals[9*(offset+f)] = -n[id_f_0][0];
                normals[9*(offset+f)+1] = -n[id_f_0][1];
                normals[9*(offset+f)+2] = -n[id_f_0][2];
                
                normals[9*(offset+f)+3] = -n[id_f_1][0];
                normals[9*(offset+f)+4] = -n[id_f_1][1];
                normals[9*(offset+f)+5] = -n[id_f_1][2];
                
                normals[9*(offset+f)+6] = -n[id_f_2][0];
                normals[9*(offset+f)+7] = -n[id_f_2][1];
                normals[9*(offset+f)+8] = -n[id_f_2][2];
            }
        }
    }
}

/**
 This function generates the 3D mesh that corresponds to the iso level set
 */
int *MarchingCubes(float ***sdf, int*** sdf_l, float **vertices, float** colors, float **normals, int **faces, my_int3 size_grid, my_float3 center_grid, float res, float iso) {
    
    int *res_out = new int[2];
    
    // Indexing edges
    // Should return the number of faces
    int ***IndexVal = new int **[size_grid.x];
    int ***Offset = new int **[size_grid.x];
    for (int i = 0; i < size_grid.x; i++) {
        IndexVal[i] = new int *[size_grid.y];
        Offset[i] = new int *[size_grid.y];
        for (int j = 0; j < size_grid.y; j++) {
            IndexVal[i][j] = new int[size_grid.z];
            Offset[i][j] = new int[size_grid.z];
            for (int k = 0; k < size_grid.z; k++) {
                IndexVal[i][j][k] = 0;
                Offset[i][j][k] = 0;
            }
        }
    }
    
    int nb_faces = Indexing(sdf, IndexVal, Offset, size_grid, iso);
    cout << nb_faces << " faces will be generated" << endl;
    res_out[0] = 3*nb_faces;
    res_out[1] = nb_faces;
        
    // Generate vertices, normals and faces
    *vertices = new float [9*nb_faces];
    *colors = new float[9 * nb_faces];
    *normals = new float [9*nb_faces];
    *faces = new int [3*nb_faces];
    
    std::vector< std::thread > my_threads;
    for (int x = 1; x < size_grid.x-2; x++) {
        //InnerLoop(volume, i, vertices, faces, normals, nb_faces, size_grid, res);
        my_threads.push_back( std::thread(InnerLoop_MC, *vertices, *colors , *normals, *faces, x, sdf, sdf_l, IndexVal, Offset, size_grid, center_grid, res, iso) );
    }
    std::for_each(my_threads.begin(), my_threads.end(), std::mem_fn(&std::thread::join));

    return res_out;
    
    // 3. Merge nodes
    int *Index = new int[3*nb_faces];
    for (int idx = 0; idx < 3*nb_faces; idx++) {
        Index[idx] = -1;
    }
    vector<float> merged_vtx;
    vector<float> merged_nmls;
    int count_merge = 0;
    for (int idx = 0; idx < 3*nb_faces; idx++) {
        if (Index[idx] == -1) {
            merged_vtx.push_back((*vertices)[3*idx]);
            merged_vtx.push_back((*vertices)[3*idx+1]);
            merged_vtx.push_back((*vertices)[3*idx+2]);
            merged_nmls.push_back((*normals)[3*idx]);
            merged_nmls.push_back((*normals)[3*idx+1]);
            merged_nmls.push_back((*normals)[3*idx+2]);
            Index[idx] = count_merge;
            
            my_float3 curr_v = make_my_float3((*vertices)[3*idx], (*vertices)[3*idx+1], (*vertices)[3*idx+2]);
            for (int match = idx; match < 3*nb_faces; match++) {
                if (Index[match] == -1) {
                    my_float3 match_v = make_my_float3((*vertices)[3*match], (*vertices)[3*match+1], (*vertices)[3*match+2]);
                    if (norm(curr_v - match_v) == 0.0f) {
                        Index[match] = count_merge;
                    }
                }
            }
            
            count_merge++;
        }
    }
    cout << "vertices merged: " << count_merge << endl;
    
    res_out[0] = count_merge;
    delete[] *vertices;
    delete[] *normals;
    *vertices = new float [merged_vtx.size()];
    *normals = new float [merged_nmls.size()];
    memcpy(*vertices, merged_vtx.data(), merged_vtx.size()*sizeof(float));
    memcpy(*normals, merged_nmls.data(), merged_nmls.size()*sizeof(float));
    merged_vtx.clear();
    merged_nmls.clear();
        
    // Reorganize faces
    vector<int> merged_faces;
    for (int idx = 0; idx < nb_faces; idx++) {
        //cout << idx << ", " << (*faces)[3*idx] << endl;
        //cout << Index[(*faces)[3*idx]] << endl;
        merged_faces.push_back(Index[(*faces)[3*idx]]);
        merged_faces.push_back(Index[(*faces)[3*idx+1]]);
        merged_faces.push_back(Index[(*faces)[3*idx+2]]);
    }
    
    delete[] *faces;
    *faces = new int [merged_faces.size()];
    memcpy((*faces), merged_faces.data(), merged_faces.size()*sizeof(int));
    merged_faces.clear();
    
    for (int i = 0; i < size_grid.x; i++) {
        for (int j = 0; j < size_grid.y; j++) {
            delete []IndexVal[i][j];
            delete []Offset[i][j];
        }
        delete []IndexVal[i];
        delete []Offset[i];
    }
    delete []IndexVal;
    delete []Offset;
    
    return res_out;
}

#endif /* MarchingCubes_h */
