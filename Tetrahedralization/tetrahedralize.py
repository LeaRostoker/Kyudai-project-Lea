#PyMesh ile tetra yapma denemeleri yasasin ppymesh olleeyy olleeeyy
from argparse import ArgumentParser
import pymesh
import numpy as np
import tkinter as tk
import tkinter.filedialog as tkFileDialog
import os
from os.path import join, exists, abspath, dirname, basename, splitext

import utils

from tqdm import tqdm

def main(tetra_path, filename):
    print ("######### Tetrahedralize", filename, "#########")
    mesh = pymesh.load_mesh(tetra_path)
    mesh.add_attribute("vertex_normal")
    mesh.get_attribute("vertex_normal")
    mesh.get_vertex_attribute("vertex_normal")

    #mesh, info = pymesh.remove_duplicated_vertices_raw(mesh.vertices, mesh.faces)
    #mesh, info = pymesh.remove_duplicated_faces(mesh)
    #Self intersection cleanup

    self_mesh = pymesh.resolve_self_intersection(mesh, engine='auto')

    self_mesh.add_attribute("vertex_normal")
    self_mesh.get_attribute("vertex_normal")
    self_mesh.get_vertex_attribute("vertex_normal")
    

    print("Starting Tetgen..............................")
    tetgen = pymesh.tetgen()
    tetgen.points = self_mesh.vertices
    tetgen.triangles = self_mesh.faces
    tetgen.max_radius_edge_ratio = 0.01
    #tetgen.coarsening = True
    #tetgen.optimization_level = 5
    #tetgen.max_tet_volume = 0.00000001
    tetgen.verbosity = 0
    tetgen.run()

    #outmesh = tetgen.mesh
    outmesh = tetgen.mesh
    print(outmesh.num_vertices, outmesh.num_faces,outmesh.num_voxels)
    print(outmesh.dim, outmesh.vertex_per_face,outmesh.vertex_per_voxel)

    # Save 
    pymesh.meshio.save_mesh("tetrahedralized/" + filename, outmesh, ascii=True)


if __name__ == '__main__':
    
    directory = 'blenderOut'
    # iterate over files in directory
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        plypath = os.path.join(directory, filename)
        main(plypath, filename)
        

