#include "include_gpu/LevelSet_gpu.cuh"

//////////////////////////////////////////
///**** Device Function definitions ****/
/////////////////////////////////////////

__device__ __forceinline__ float norm(float3 a) {
    return sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}

__device__ __forceinline__ float3 make_ray(float x, float y, float z) {

    float3 ray = make_float3(x, y, z);
    ray = ray / sqrt(x * x + y * y + z * z);
    return ray;
}

__device__ __forceinline__ int IsInterectingRayTriangle3D_gpu(float3 ray, float3 p0, float3 p1, float3 p2, float3 p3, float3 n) { 
    float den = dot(ray, n);
    if (fabs(den) < 1.0e-6f) {
        if (dot(p1 - p0, n) == 0.0f) {
            return 0;
        }
        return 0;
    }

    float fact = (dot(p1 - p0, n) / den);
    if (fact < 1.0e-6f)
        return 0;

    float3 proj = p0 + ray * fact;
    // Compute if proj is inside the triangle
    // V = p1 + s(p2-p1) + t(p3-p1)
    // find s and t
    float3 u = p2 - p1;
    float3 v = p3 - p1;
    float3 w = proj - p1;

    float s = (dot(u, v) * dot(w, v) - dot(v, v) * dot(w, u)) / (dot(u, v) * dot(u, v) - dot(u, u) * dot(v, v));
    float t = (dot(u, v) * dot(w, u) - dot(u, u) * dot(w, v)) / (dot(u, v) * dot(u, v) - dot(u, u) * dot(v, v));


    if (s >= 0.0f && t >= 0.0f && s + t <= 1.0f) {
        float3 t_12 = cross(u, w);
        if (norm(t_12) < 1.0e-6f) {
            return 0;
        }

        float3 t_13 = cross(v, w);
        if (norm(t_13) < 1.0e-6f) {
            return 0;
        }

        float3 t_23 = cross((p3 - proj), (p3 - p2));
        if (norm(t_23) < 1.0e-6f) {
            return 0;
        }
        return 1;
    }

    //coordonnees barycentriques
    //float3 bary;

    // The area of a triangle is 
    //float areaABC = fabs(dot(n, cross((p1 - p2), (p3 - p2))));
    //float areaPBC = fabs(dot(n, cross((p1 - proj), (p3 - proj))));
    //float areaPCA = fabs(dot(n, cross((p3 - proj), (p2 - proj))));

    //bary.x = areaPBC / areaABC; // alpha
    //bary.y = areaPCA / areaABC; // beta
    //bary.z = 1.0f - bary.x - bary.y; // gamma
    //
    //if (bary.z < 0.0f && bary.z < 1.0f)
    //    return 1.0f;

    /*if (bary.z < 0.0f)
        return 0.0f;

    if ()*/
    /*if (fabs(t) < 1.0e-6f || fabs(s) < 1.0e-6f || fabs(s + t) < 1 + 1.0e-6f)
        return 0.5f;*/

    /*if (t == 0.0f && s >= 0.0f && s <= 1.0f)
        return 0.5f;

    if (s == 0.0f && t >= 0.0f && t <= 1.0f)
        return 0.5f;

    if (s+t == 1.0f && s >= 0.0f && t >= 0.0f)
        return 0.5f;

    if (s >= 0.0f && t >= 0.0f && s + t <= 1.0f )
        return 1.0f;*/

    return 0;
}

__device__ __forceinline__ float DistancePointFace3D_gpu(float3 p0, float3 p1, float3 p2, float3 p3, float3 n, bool approx = false) {
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
    float area = norm(cross_p1p2p3) / 2.0f;
    if (area < 1.0e-12) {
        return 1.0e32;
    }

    // b. Test if projection is inside the triangle
    float3 C;

    // edge 0 = p1p2
    float3 edge0 = p2 - p1;
    float3 vp0 = proj - p1;
    C = cross(edge0, vp0);
    float w = (norm(C) / 2.0f) / area;
    if (dot(n, C) < 0.0f) {
        // P is on the right side of edge0
        // compute distance point to segment
        float curr_dist;
        float3 base = edge0 * (1.0f / norm(edge0));
        float Dt = dot(base, vp0);
        if (Dt < 0.0f) {
            curr_dist = norm(p0 - p1);
        }
        else if (Dt > norm(edge0)) {
            curr_dist = norm(p0 - p2);
        }
        else {
            curr_dist = norm(p0 - (p1 + base * Dt));
        }
        return curr_dist;
    }

    // edge 1 = p2p3
    float3 edge1 = p3 - p2;
    float3 vp1 = proj - p2;
    C = cross(edge1, vp1);
    float u = (norm(C) / 2.0f) / area;
    if (dot(n, C) < 0.0f) {
        // P is on the right side of edge1
        // compute distance point to segment
        float curr_dist;
        float3 base = edge1 * (1.0f / norm(edge1));
        float Dt = dot(base, vp1);
        if (Dt < 0.0f) {
            curr_dist = norm(p0 - p2);
        }
        else if (Dt > norm(edge1)) {
            curr_dist = norm(p0 - p3);
        }
        else {
            curr_dist = norm(p0 - (p2 + base * Dt));
        }
        return curr_dist;
    }

    // edge 2 = p3p1
    float3 edge2 = p1 - p3;
    float3 vp2 = proj - p3;
    C = cross(edge2, vp2);
    float v = (norm(C) / 2.0f) / area;
    if (dot(n, C) < 0.0f) {
        // P is on the right side of edge 2;
        float curr_dist;
        float3 base = edge2 * (1.0f / norm(edge2));
        float Dt = dot(base, vp2);
        if (Dt < 0.0f) {
            curr_dist = norm(p0 - p3);
        }
        else if (Dt > norm(edge2)) {
            curr_dist = norm(p0 - p1);
        }
        else {
            curr_dist = norm(p0 - (p3 + base * Dt));
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

__device__ __forceinline__ void LevelSetSemProcess(float* volume, int* volume_l, float* vertices, int* labels, int* faces, float* normals, int nb_faces, int3 size_grid, float3 center_grid, float res_x, float res_y, float res_z, float disp)
{
    unsigned int i = threadIdx.x + blockIdx.x * THREAD_SIZE_X; // cols
    unsigned int j = threadIdx.y + blockIdx.y * THREAD_SIZE_Y; // rows
    unsigned int k = threadIdx.z + blockIdx.z * THREAD_SIZE_Z; // rows
    unsigned int idx = i * size_grid.y * size_grid.z + j * size_grid.z + k;

    if (i > size_grid.x - 1 || j > size_grid.y - 1 || k > size_grid.z - 1)
        return;

    // Get the 3D coordinate
    float3 p0;
    p0.x = (float(i) - float(size_grid.x) / 2.0f) * res_x + center_grid.x;
    p0.y = (float(j) - float(size_grid.y) / 2.0f) * res_y + center_grid.y;
    p0.z = (float(k) - float(size_grid.z) / 2.0f) * res_z + center_grid.z;

    float3 ray1 = make_ray(0.0f, 0.0f, 1.0f);
    float3 ray2 = make_ray(0.0f, 1.0f, 0.0f);
    float3 ray3 = make_ray(1.0f, 0.0f, 0.0f);
    float3 ray4 = make_ray(1.0f, 0.0f, 1.0f);
    float3 ray5 = make_ray(0.0f, 1.0f, 1.0f);
    float3 ray6 = make_ray(1.0f, 1.0f, 0.0f);
    float3 ray7 = make_ray(1.0f, 1.0f, 1.0f);

    // Compute the smallest distance to the faces
    float min_dist = 1.0e32f;
    float sdf = 1.0f;
    int lbl = 0;
    int intersections1 = 0;
    int intersections2 = 0;
    int intersections3 = 0;
    int intersections4 = 0;
    int intersections5 = 0;
    int intersections6 = 0;
    int intersections7 = 0;

    for (int f = 0; f < nb_faces; f++) {
        // Compute distance point to face
        float3 n = make_float3(normals[3 * f], normals[3 * f + 1], normals[3 * f + 2]);

        float3 p1 = make_float3(vertices[3 * faces[3 * f + 0]], vertices[3 * faces[3 * f + 0] + 1], vertices[3 * faces[3 * f + 0] + 2]);
        float3 p2 = make_float3(vertices[3 * faces[3 * f + 1]], vertices[3 * faces[3 * f + 1] + 1], vertices[3 * faces[3 * f + 1] + 2]);
        float3 p3 = make_float3(vertices[3 * faces[3 * f + 2]], vertices[3 * faces[3 * f + 2] + 1], vertices[3 * faces[3 * f + 2] + 2]);

        // Compute line plane intersection
        intersections1 += IsInterectingRayTriangle3D_gpu(ray1, p0, p1, p2, p3, n);
        intersections2 += IsInterectingRayTriangle3D_gpu(ray2, p0, p1, p2, p3, n);
        intersections3 += IsInterectingRayTriangle3D_gpu(ray3, p0, p1, p2, p3, n);
        intersections4 += IsInterectingRayTriangle3D_gpu(ray4, p0, p1, p2, p3, n);
        intersections5 += IsInterectingRayTriangle3D_gpu(ray5, p0, p1, p2, p3, n);
        intersections6 += IsInterectingRayTriangle3D_gpu(ray6, p0, p1, p2, p3, n);
        intersections7 += IsInterectingRayTriangle3D_gpu(ray7, p0, p1, p2, p3, n);

        // Compute point to face distance
        float curr_dist = DistancePointFace3D_gpu(p0, p1, p2, p3, n);

        if((curr_dist < min_dist) || fabs(curr_dist - min_dist) < 10e-6)
        {
            min_dist = curr_dist;
            sdf = curr_dist;
            lbl = labels[faces[3 * f]];
        }
                        
    }

    int countOut = 0;

    countOut += intersections1 % 2 == 0 ? 1 : 0;
    countOut += intersections2 % 2 == 0 ? 1 : 0;  
    countOut += intersections3 % 2 == 0 ? 1 : 0;
    countOut += intersections4 % 2 == 0 ? 1 : 0;
    countOut += intersections5 % 2 == 0 ? 1 : 0;
    countOut += intersections6 % 2 == 0 ? 1 : 0;
    countOut += intersections7 % 2 == 0 ? 1 : 0;

    if (countOut >= 4)
        volume[idx] = sdf;
    else
        volume[idx] = - sdf;

    volume_l[idx] = lbl;

}

__global__ void LevelSetSemKernel(float* volume, int* volume_l, float* vertices, int* labels, int* faces, float* normals, int nb_faces, int3 size_grid, float3 center_grid, float res_x, float res_y, float res_z, float disp)
{
	LevelSetSemProcess(volume, volume_l, vertices, labels, faces, normals, nb_faces, size_grid, center_grid, res_x, res_y, res_z, disp);
}


//////////////////////////////////////////
///******* Function definitions *********/
//////////////////////////////////////////

pair<float***, int***> LevelSet_gpu(float* vertices, int *labels, int* faces, float* normals, int nb_vertices, int nb_faces, int3 size_grid, float3 center_grid, float res_x, float res_y, float res_z, float disp) {
	// Allocate data
	float*** volume = new float** [size_grid.x];
	for (int i = 0; i < size_grid.x; i++) {
		volume[i] = new float* [size_grid.y];
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
                volume_l[i][j][k] = 2;
            }
        }
    }

	float* volume_gpu;
	checkCudaErrors(cudaMalloc((void**)&volume_gpu, size_grid.x * size_grid.y * size_grid.z * sizeof(float)));

    int* volume_l_gpu;
    checkCudaErrors(cudaMalloc((void**)&volume_l_gpu, size_grid.x * size_grid.y * size_grid.z * sizeof(int)));
    
	for (int i = 0; i < size_grid.x; i++) {
		for (int j = 0; j < size_grid.y; j++) {
			checkCudaErrors(cudaMemcpy((void*)&volume_gpu[i * size_grid.y * size_grid.z + j * size_grid.z], (void*)volume[i][j], size_grid.z * sizeof(float), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemset((void*)&volume_l_gpu[i * size_grid.y * size_grid.z + j * size_grid.z], 0, size_grid.z * sizeof(int)));
		}
	}

	float* vertices_gpu;
	checkCudaErrors(cudaMalloc((void**)&vertices_gpu, 3 * nb_vertices * sizeof(float)));
	checkCudaErrors(cudaMemcpy((void*)vertices_gpu, (void*)vertices, 3 * nb_vertices * sizeof(float), cudaMemcpyHostToDevice));

    int* labels_gpu;
    checkCudaErrors(cudaMalloc((void**)&labels_gpu,  nb_vertices * sizeof(int)));
    checkCudaErrors(cudaMemcpy((void*)labels_gpu, (void*)labels,  nb_vertices * sizeof(int), cudaMemcpyHostToDevice));

	int* faces_gpu;
	checkCudaErrors(cudaMalloc((void**)&faces_gpu, 3 * nb_faces * sizeof(int)));
	checkCudaErrors(cudaMemcpy((void*)faces_gpu, (void*)faces, 3 * nb_faces * sizeof(int), cudaMemcpyHostToDevice));

	float* normals_gpu;
	checkCudaErrors(cudaMalloc((void**)&normals_gpu, 3 * nb_faces * sizeof(float)));
	checkCudaErrors(cudaMemcpy((void*)normals_gpu, (void*)normals, 3 * nb_faces * sizeof(float), cudaMemcpyHostToDevice));


	dim3 dimBlock(THREAD_SIZE_X, THREAD_SIZE_Y, THREAD_SIZE_Z);
	dim3 dimGrid(1, 1, 1);
	dimGrid.x = divUp(size_grid.x, dimBlock.x); // #cols
	dimGrid.y = divUp(size_grid.y, dimBlock.y); // # rows
	dimGrid.z = divUp(size_grid.z, dimBlock.z); // # rows

    std::cout << "Start level set on GPU" << std::endl;

	LevelSetSemKernel << <dimGrid, dimBlock >> > (volume_gpu, volume_l_gpu, vertices_gpu, labels_gpu, faces_gpu, normals_gpu, nb_faces, size_grid, center_grid, res_x, res_y, res_z, disp);
   
	checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "End level set on GPU" << std::endl;

	for (int i = 0; i < size_grid.x; i++) {
		for (int j = 0; j < size_grid.y; j++) {
			checkCudaErrors(cudaMemcpy((void*)volume[i][j], (void*)&volume_gpu[i * size_grid.y * size_grid.z + j * size_grid.z], size_grid.z * sizeof(float), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy((void*)volume_l[i][j], (void*)&volume_l_gpu[i * size_grid.y * size_grid.z + j * size_grid.z], size_grid.z * sizeof(int), cudaMemcpyDeviceToHost));
            
		}
	}

	checkCudaErrors(cudaFree(volume_gpu));
	checkCudaErrors(cudaFree(vertices_gpu));
    checkCudaErrors(cudaFree(labels_gpu));
	checkCudaErrors(cudaFree(faces_gpu));
	checkCudaErrors(cudaFree(normals_gpu));

	return pair<float***, int***>(volume, volume_l);
}
