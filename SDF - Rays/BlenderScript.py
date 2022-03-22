# Script to render a sequence of ply files with Blender. First, setup your scene
# and material for the imported meshes. Scene must be in "OBJECT"-mode.
# Fill in variables in options.
# References.
# See: http://blender.stackexchange.com/questions/24133/modify-obj-after-import-using-python
# and: http://blenderartists.org/forum/showthread.php?320309-How-to-import-ply-files-from-script
import bpy

# Options.
meshFolder = "/Users/diegothomas/Documents/Projects/POSA-main/dataKyudai/meshes/" # Folder without ending "\\".
AmountOfNumbers = 4  # Amount of numbers in filepath, e.g., 000010.ply

def MeshPath(folder = "", name = "", fileEnding = "obj"):
    return folder + name + "." + fileEnding

def ExtractMeshes():
    # deselect all objects
    bpy.ops.object.select_all(action='DESELECT')
    
    # Get the imported object.
    scene = bpy.context.scene
    for ob in scene.objects:
        # make the current object active and select it
        ob.select_set(True)

        # make sure that we only export meshes
        if ob.type == 'MESH':
            # export the currently selected object to its own file based on its name
            fullPathToMesh = MeshPath(folder = meshFolder, name = ob.name)
            bpy.ops.export_scene.obj(
                    filepath=fullPathToMesh,
                    use_selection=True,
                    )
            
        # deselect the object and move on to another if any more are left
        ob.select_set(False)
        
# Run the script.
ExtractMeshes()
