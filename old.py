import bpy
import bpy_extras
from mathutils import Matrix
from mathutils import Vector
import numpy
import cv2

#---------------------------------------------------------------
# 3x4 P matrix from Blender camera
#---------------------------------------------------------------

# Build intrinsic camera parameters from Blender camera data
#
# See notes on this in
# blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
def get_calibration_matrix_K_from_blender(camd):
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == 'VERTICAL'):
        # the sensor height is fixed (sensor fit is horizontal),
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else: # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal),
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm


    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((alpha_u, skew,    u_0),
         (    0  , alpha_v, v_0),
         (    0  , 0,        1 )))
    return K

# Returns camera rotation and translation matrices from Blender.
#
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates
#         used in digital images)
#       - right-handed: positive z look-at direction
def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
         (0, -1, 0),
         (0, 0, -1)))

    # Transpose since the rotation is object rotation,
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1*R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision camera
    # NOTE: Use * instead of @ here for older versions of Blender
    # TODO: detect Blender version
    R_world2cv = R_bcam2cv@R_world2bcam
    T_world2cv = R_bcam2cv@T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
    ))
    return RT

def get_3x4_P_matrix_from_blender(cam):
    K = get_calibration_matrix_K_from_blender(cam.data)
    RT = get_3x4_RT_matrix_from_blender(cam)
    return K@RT, K, RT

# ----------------------------------------------------------
# Alternate 3D coordinates to 2D pixel coordinate projection code
# adapted from https://blender.stackexchange.com/questions/882/how-to-find-image-coordinates-of-the-rendered-vertex?lq=1
# to have the y axes pointing up and origin at the top-left corner
def project_by_object_utils(cam, point):
    scene = bpy.context.scene
    co_2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, point)
    render_scale = scene.render.resolution_percentage / 100
    render_size = (
        int(scene.render.resolution_x * render_scale),
        int(scene.render.resolution_y * render_scale),
    )
    return Vector((co_2d.x * render_size[0], render_size[1] - co_2d.y * render_size[1]))

# ----------------------------------------------------------



def get_bone_pos(bone_name):
    body_name = "Armature"
    R = bpy.data.objects[body_name].matrix_world.to_3x3()
    R = numpy.array(R)

    t = bpy.data.objects[body_name].matrix_world.translation
    t = numpy.array(t)

    #print(f"R = {R.shape}\n{R}")
    #print(f"t = {t.shape}\n{t}")

    head_local_location = bpy.data.objects[body_name].data.bones[bone_name].head_local
    tail_local_location = bpy.data.objects[body_name].data.bones[bone_name].tail_local

    head_local_location = numpy.array(head_local_location)
    tail_local_location = numpy.array(tail_local_location)
    #print(f"local position = {local_location.shape}\n{local_location}")

    head_loc = numpy.dot(R, head_local_location) + t
    tail_loc = numpy.dot(R, tail_local_location) + t
    print(f"final loc = {loc.shape}\n{loc}")

    return [loc[0], loc[1], loc[2]]

def draw_bones(matrix, img):
    body = bpy.data.objects["Armature"]

    if body:
        for bone in body.data.bones:
            get_bone_pos(bone_name)
            tail = body.matrix_world @ bone.tail_local
            head = body.matrix_world @ bone.head_local
            print(f"Bone: {bone.name}, Head: {head}, Tail: {tail}")

            p1_3d = tail.to_4d()
            p1_2d = matrix @ p1_3d
            p1_2d /= p1_2d[2]
            p1_2d = p1_2d.to_2d().to_tuple(0)
            p1_2d = tuple(int(num) for num in p1_2d)

            p2_3d = head.to_4d()
            p2_2d = matrix @ p2_3d
            p2_2d /= p2_2d[2]
            p2_2d = p2_2d.to_2d().to_tuple(0)
            p2_2d = tuple(int(num) for num in p2_2d)

            print(p1_2d, p2_2d)
            img = cv2.line(img, p1_2d, p2_2d, (0, 0, 255))

            #print(f"Abosulute Head: {head}, Abosulute Tail: {tail}")
            print()
    else:
        print("Nessuna armatura trovata.")

    return img

def draw_bones2(matrix, img):
    body = bpy.data.objects["Armature"]

    if body:
        for bone in body.data.bones:
            tail = body.matrix_world @ bone.tail_local
            head = body.matrix_world @ bone.head_local
            print(f"Bone: {bone.name}, Head: {head}, Tail: {tail}")

            p1_3d = tail.to_4d()
            p1_2d = matrix @ p1_3d
            p1_2d /= p1_2d[2]
            p1_2d = p1_2d.to_2d().to_tuple(0)
            p1_2d = tuple(int(num) for num in p1_2d)

            p2_3d = head.to_4d()
            p2_2d = matrix @ p2_3d
            p2_2d /= p2_2d[2]
            p2_2d = p2_2d.to_2d().to_tuple(0)
            p2_2d = tuple(int(num) for num in p2_2d)

            print(p1_2d, p2_2d)
            img = cv2.line(img, p1_2d, p2_2d, (0, 0, 255))

            #print(f"Abosulute Head: {head}, Abosulute Tail: {tail}")
            print()
    else:
        print("Nessuna armatura trovata.")

    return img


if __name__ == "__main__":
    # Insert your camera name here
    cam = bpy.data.objects['Camera']
    P, K, RT = get_3x4_P_matrix_from_blender(cam)
    print("K")
    print(K)
    print("RT")
    print(RT)
    print("P")
    print(P)

    print("==== Tests ====")
    mano  = Vector((-4.6235, -0.96058,    9.0911, 1))
    pM = P @ mano
    pM /= pM[2]
    print(project_by_object_utils(cam, Vector(mano[0:3])))

    cc_base_pelvis  = Vector((0.035273, 0.063744, 9.164755, 1))
    pV = P @ cc_base_pelvis
    pV /= pV[2]
    print(project_by_object_utils(cam, Vector(cc_base_pelvis[0:3])))
    print("#########")
    print()

    # Bonus code: save the 3x4 P matrix into a plain text file
    # Don't forget to import numpy for this
    nP = numpy.matrix(P)
    print(nP)
    numpy.savetxt("D:\Download\matrice.txt", nP)  # to select precision, use e.g. fmt='%.2f'


    # Try to draw all bones on final image
    print("PRINTING BONES ON IMAGE")
    img = cv2.imread(r"D:\Download\baseline_new.png")
    #cv2.imshow('image',img)


    # DISPLAY CUBES
    for point in bpy.data.collections["Punti"].all_objects:
        point2d = P @ point.location.to_4d()
        point2d /= point2d[2]

        print(point2d.to_2d().to_tuple(0))
        p2d_a = point2d.to_2d().to_tuple(0)
        p2d_a = tuple(int(num) for num in p2d_a)
        print(p2d_a)
        cv2.circle(img, p2d_a, 20, (255, 0, 255))

    img = draw_bones2(P, img)
    cv2.imshow('output',img)

    filename = r"D:\Download\output.png"
    cv2.imwrite(filename, img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()