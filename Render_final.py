import bpy
import os
import numpy as np
import json
from numpy import random


# = = = = FUNCTIONS

def generate_model(model, loc_x, loc_y, loc_z, rot_x, rot_y, rot_z):
    ### GENERATE SATELLITE   (dovrebbe essere la funzione che genera il modello del satellite)
    path = "C:/Users/corra/OneDrive/Desktop/Blender/"
    file_path = path + model
    # append all objects starting with 'house'
    with bpy.data.libraries.load(file_path) as (data_from, data_to):
        data_to.objects = [name for name in data_from.objects]
    # link them to scene
    scene = bpy.context.scene
    for obj in data_to.objects:
        if obj is not None:
            scene.collection.objects.link(obj)
    ### GENERATE TARGET AND MODIFY PROPERTY
    bpy.ops.object.select_by_type(type='MESH')
    ov = bpy.context.copy()
    ov['area'] = [a for a in bpy.context.screen.areas if a.type == "VIEW_3D"][0]
    bpy.ops.transform.resize(value=(1 / 1000, 1 / 1000, 1 / 1000), orient_type='GLOBAL',
                             orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                             orient_matrix_type='GLOBAL', mirror=False, use_proportional_edit=False,
                             proportional_edit_falloff='SMOOTH', proportional_size=1.61051,
                             use_proportional_connected=False, use_proportional_projected=False)
    bpy.ops.transform.rotate(ov, value=rot_x, orient_axis='X')
    bpy.ops.transform.rotate(ov, value=rot_y, orient_axis='Y')
    bpy.ops.transform.rotate(ov, value=rot_z, orient_axis='Z')
    bpy.context.scene.cursor.location = (loc_x, loc_y, loc_z)
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            override = bpy.context.copy()
            override['area'] = area
            override['region'] = area.regions[4]
            bpy.ops.view3d.snap_selected_to_cursor(override, use_offset=True)
    bpy.ops.object.select_all(action='DESELECT')

def move(loc_sat, rot_sat, loc_cam, loc_E):
    # MOVE SATELLITE non mi serve muoverlo dato che e sempre in 0 0 0, solo rotazione
    #bpy.context.scene.cursor.location = (loc_sat[0], loc_sat[1], loc_sat[2])
    bpy.ops.object.select_by_type(type='MESH')
    bpy.data.objects['Camera'].select_set(False)
    bpy.data.objects['Earth'].select_set(False)
    bpy.data.objects['Sun'].select_set(False)
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            override = bpy.context.copy()
            override['area'] = area
            override['region'] = area.regions[4]
            bpy.ops.view3d.snap_selected_to_cursor(override, use_offset=True)
    ov = bpy.context.copy()
    ov['area'] = [a for a in bpy.context.screen.areas if a.type == "VIEW_3D"][0]
    bpy.ops.transform.rotate(ov, value=rot_sat[0], orient_axis='X')
    bpy.ops.transform.rotate(ov, value=rot_sat[1], orient_axis='Y')
    bpy.ops.transform.rotate(ov, value=rot_sat[2], orient_axis='Z')
    bpy.ops.view3d.snap_selected_to_cursor(override, use_offset=True)
    bpy.ops.object.select_all(action='DESELECT')
    # MOVE CAMERA
    bpy.data.objects['Camera'].select_set(True)
    bpy.context.selected_objects[0].location = (loc_cam[0], loc_cam[1], loc_cam[2])
    bpy.data.objects['Camera'].select_set(False)
    # MOVE EARTH
    #bpy.data.objects['Earth'].select_set(True)
    #bpy.context.selected_objects[0].location = (loc_E[0], loc_E[1], loc_E[2])

def camera(cam_x, cam_y, cam_z):
    bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(cam_x, cam_y, cam_z),
                              rotation=(1.09607, 1.98354e-07, 0.649263), scale=(1, 1, 1))
    bpy.context.object.data.lens = 100
    bpy.context.object.data.clip_start = 1 / 1000
    bpy.context.object.data.clip_end = 10000
    bpy.ops.object.constraint_add(type='TRACK_TO')
    bpy.context.object.constraints["Track To"].target = bpy.data.objects["Body"]
    bpy.context.scene.camera = bpy.data.objects['Camera']

def param(mode, id, image_prefix):
    #BASE_DIR = os.path.dirname(os.path.abspath('C:/Users/corra/OneDrive/Desktop/Blender'))  # the directory of this file
    BASE_DIR = 'C:/Users/corra/OneDrive/Desktop/Blender' # the directory of this file
    DATASET_DIR = BASE_DIR + '/coco'
    masks_dir = '/{}_masks/'.format(mode)  # ???? no need to prepend the full path since it is already specified in the blender node
    image_dir = DATASET_DIR + '/{}/'.format(mode)
    ANN_DIR = DATASET_DIR + '/annotations'
    image_id = id  # MI SERVE PER DIFFERENZIARE LE IMMAGINI AD OGNI ITERAZIONE
    basename = image_prefix + "{0:04}".format(image_id)
    IMAGE_NAME = basename
    #basedir = "C:/Users/corra/OneDrive/Desktop/Blender/Mask"
    basedir = DATASET_DIR
    return DATASET_DIR, masks_dir, image_dir, basename, IMAGE_NAME, basedir

def index_obj(label_names):
    i = 1
    for obj in [n for n in bpy.data.objects if n.type == 'MESH']:
        label = obj["label"] if ("label" in obj) else None
        if label in label_names:
            obj.pass_index = i
            i += 1
            continue
        elif label is not None:
            print('Invalid label "{}" for object "{}"'.format(label, obj.name))
    print(i)

def nodes(basedir, image_dir, IMAGE_NAME, basename, masks_dir):
    nodesField = scene.node_tree
    for currentNode in nodesField.nodes:
        nodesField.nodes.remove(currentNode)
    # = = = NEW TREE
    tree = scene.node_tree
    links = tree.links
    input_node = tree.nodes.new(type='CompositorNodeRLayers')
    input_node.location = (0, 0)
    input_node.scene = scene
    indexob_output = None
    for output in input_node.outputs:
        if output.name == 'IndexOB':
            indexob_output = output
            break
    output_node = tree.nodes.new(type='CompositorNodeOutputFile')
    output_node.location = (1300, 0)
    output_node.base_path = basedir
    output_node.format.color_mode = 'RGB'
    #output_node.format.color_mode = 'BW'
    output_node.file_slots[0].path = os.path.basename(os.path.normpath(image_dir)) + '/' + IMAGE_NAME
    pass_num = 1  # necessary to make the script robust to mislabeled items, can't use enumerate id because it would
    # cause problems in adding the output node entry if some item is skipped
    for ob in [n for n in bpy.data.objects if n.type == 'MESH']:
        label = ob["label"] if ("label" in ob) else None
        if label in label_names:
            ANNOTATION_IMG = masks_dir + basename + '_' + str(pass_num) + "_{}".format(label)
            output_node.file_slots.new(ANNOTATION_IMG)
            output_node.file_slots[pass_num].use_node_format = False
            output_node.file_slots[pass_num].path = ANNOTATION_IMG
            output_node.file_slots[pass_num].format.color_mode = 'RGB'
            #output_node.file_slots[pass_num].format.color_mode = 'BW'
            # MASK NODE
            mask_node = tree.nodes.new(type='CompositorNodeIDMask')
            mask_node.location = (350, -300 * (pass_num - 1))
            mask_node.index = ob.pass_index
            mask_node.use_antialiasing = True
            # VIEW NODE
            view_node = tree.nodes.new(type="CompositorNodeViewer")
            view_node.location = (550, 100 - 300 * (pass_num - 1))
            # LINK NODES
            links.new(input_node.outputs["Image"], output_node.inputs[0])
            links.new(indexob_output, mask_node.inputs[0])
            links.new(mask_node.outputs[0], output_node.inputs[pass_num])
            links.new(mask_node.outputs[0], view_node.inputs[0])
            pass_num += 1
        elif label is not None:
            print('invalid label')

def Earth():
    path = "C:/Users/corra/OneDrive/Desktop/Blender/"
    file_path = path + "Earth.blend"
    inner_path = "Object"
    object_name = "Earth"
    bpy.ops.wm.append(filepath=os.path.join(file_path, inner_path, object_name),
                      directory=os.path.join(file_path, inner_path), filename=object_name)
    ov = bpy.context.copy()
    ov['area'] = [a for a in bpy.context.screen.areas if a.type == "VIEW_3D"][0]
    bpy.ops.transform.rotate(ov, value=0.9, orient_axis='X')
    bpy.ops.transform.rotate(ov, value=2, orient_axis='Y')
    bpy.ops.transform.rotate(ov, value=1.7, orient_axis='Z')

def suppress_frames(paths):
    if isinstance(paths, list):
        pass
    else:
        quit('The input must be a list (even if it counts only one path).')
    for path in paths:
        for file in os.listdir(path):
            if file.endswith(".png"):
                filename = os.path.splitext(file)[0]
                try:
                    float(filename[-4:])
                    newname = filename[:-4] + '.png'
                except ValueError:
                    newname = file
                # if filename[-4:] == '0000':
                #     newname = filename[:-4] + '.png'
                # else:
                #     newname = file
                os.rename(os.path.join(path, file), os.path.join(path, newname))

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = =  = #
# = = = = = = = = = = = = = = MAIN = = = = = = = = = = = = = = #
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = =  = #

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

bpy.context.scene.unit_settings.length_unit = 'KILOMETERS'
bpy.context.scene.unit_settings.scale_length = 1000

# = = = IMPORT POSITION in [km]
target_mat = np.genfromtxt(r'C:/Users/corra/OneDrive/Desktop/Blender/Orbit/Orbit_target.txt', delimiter=' ')
chaser_mat = np.genfromtxt(r'C:/Users/corra/OneDrive/Desktop/Blender/Orbit/Orbit_chaser.txt', delimiter=' ')

# = = = RENDER SETTINGS
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.render.image_settings.file_format = 'PNG'
#bpy.context.scene.render.image_settings.color_depth = '16'
scene.render.resolution_x, scene.render.resolution_y = (700, 700)
scene.cycles.samples = 150
scene.render.image_settings.color_mode = 'BW'
scene.use_nodes = True
if len(bpy.context.scene.view_layers) != 1:
    print('Warning - more than one render layer in scene. Output may be wrong.')

bpy.context.scene.view_layers[0].use_pass_object_index = True

# = = = PARAMETERS
mode = 'train'
id = 1
image_prefix = 'sat4clr_'
DATASET_DIR, masks_dir, image_dir, basename, IMAGE_NAME, basedir = param(mode, id, image_prefix)

# = = = SELECT MODEL
model = "satellite4.blend"  # model of the satellite we want to sue
label_names = ['antenna', 'body', 'solarPanel', 'earth']     # object properties --> custom properties
v_target = target_mat[0, :]  # location of the cursor (""center of the satellite"")
loc = (0, 0, 0)
rot = (0, 0, 0)  # rotation of satellite around X,Y,Z axis (for attitude?)
generate_model(model, loc[0], loc[1], loc[2], rot[0], rot[1], rot[2])

# = = = GENERATE EARTH
Earth()
#loc_E = target_mat[0,0:3]
loc_E = (0,-7000,0)
bpy.data.objects["Earth"].location = loc_E

# = = = ADD INDEX TO THE OBJECT
index_obj(label_names)  # assign index to each part of the satellite

# = = = NODES
nodes(basedir, image_dir, IMAGE_NAME, basename, masks_dir)

# = = = CAMERA
v_camera = chaser_mat[0, :]
cam_loc = v_camera - v_target
camera(cam_loc[0], cam_loc[1], cam_loc[2])

#Sun_loc = np.genfromtxt(r'C:/Users/corra/OneDrive/Desktop/Blender/Orbit/Sun_position.txt',delimiter = ' ')
#Sun_loc = (Sun_loc/np.linalg.norm(Sun_loc))*10000


# = = = GENERATE SUN
Sun_loc = (10000, 10000, 10000)
name = 'Sun'
# create light datablock, set attributes
light_data = bpy.data.lights.new(name, type='SUN')
light_data.energy = 13.5
# create new object with our light datablock
light_object = bpy.data.objects.new(name, object_data=light_data)
# link light object
bpy.context.collection.objects.link(light_object)
# make it active
bpy.context.view_layer.objects.active = light_object
#change location
light_object.location = Sun_loc

bpy.ops.object.constraint_add(type='TRACK_TO')
bpy.context.object.constraints["Track To"].target = bpy.data.objects["Earth"]

info_list=[]

#bpy.ops.render.render(write_still=False)

# = = =  GENERATE  IMAGE + MASKS
val = int(input("Numbers of images to be generated: "))
i = 1
rot = (0,0,0)
while i <= val:
    cam = chaser_mat[i-1,:]-target_mat[i-1,:]
    rot = random.randint(1,4,size=3)
    #rot = (i*np.pi/300,0,-i*np.pi/300)
    print(rot)
    DATASET_DIR, masks_dir, image_dir, basename, IMAGE_NAME, basedir = param (mode,i,image_prefix)
    nodes(basedir,image_dir,IMAGE_NAME,basename,masks_dir)
    #loc_E = target_mat[i-1,0:3]
    loc_E = (0,-7000, 0)
    move(loc,rot,cam, loc_E)
    bpy.ops.render.render(write_still= False)
    #entry = {'filename': basename,
    #         'target position/velocity': list(target_mat[i-1,:]) ,
    #         'camera position/velocity': list(chaser_mat[i-1,:]) ,
    #                     }
    #info_list = []
    #info_list.append(entry)
    #with open(os.path.join(image_dir, '{}.json'.format(IMAGE_NAME)), 'w') as file:
    #    json.dump(info_list, file, indent=4)
    i += 1

suppress_frames([DATASET_DIR + masks_dir])
suppress_frames([image_dir])




