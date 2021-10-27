import sys
from os.path import join
sys.path.append("/home/idocz/repos/3D_Graph_Renderer/code/")
from classes.scene_airmspi import *
from classes.camera import AirMSPICamera
from classes.visual import *
from cuda_utils import *
import pickle

cuda.select_device(0)
def get_intersection_with_bbox(point, direction, bbox):
    t1 = (bbox[0, 0] - point[0]) / direction[0]
    t2 = (bbox[0, 1] - point[0]) / direction[0]
    t3 = (bbox[1, 0] - point[1]) / direction[1]
    t4 = (bbox[1, 1] - point[1]) / direction[1]
    t5 = (bbox[2, 0] - point[2]) / direction[2]
    t6 = (bbox[2, 1] - point[2]) / direction[2]
    # print(t1,t2,t3,t4,t5,t6)
    tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6))
    tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6))
    # print(tmin, tmax)
    # ray (line) is intersecting AABB, but the whole AABB is behind us or the ray does not intersect
    if tmax < 0 or tmin>tmax:
        return False, tmin, tmax


    return True, tmin, tmax


a_file = open(join("data", "airmspi_data.pkl"), "rb")
airmspi_data = pickle.load(a_file)
grid = Grid(airmspi_data["bbox"], airmspi_data["grid_shape"])
print(grid.bbox)

# sun_angles = np.array([-150, 50]) * (np.pi/180)
# sun_direction = theta_phi_to_direction(*sun_angles)
zenith = airmspi_data["sun_zenith"]
azimuth = airmspi_data["sun_azimuth"]
dir_x = -(np.sin(zenith)*np.cos(azimuth))
dir_y = -(np.sin(zenith)*np.sin(azimuth))
dir_z = -np.cos(zenith)
sun_direction = np.array([dir_x, dir_y, dir_z])

sun_temp = sun_direction.copy()
sun_temp[0] = 0
sun_temp /= np.linalg.norm(sun_temp)
sun_y = np.arccos(np.dot(sun_temp, np.array([0,0,-1])))
sun_temp = sun_direction.copy()
sun_temp[1] = 0
sun_temp /= np.linalg.norm(sun_temp)
sun_x = np.arccos(np.dot(sun_temp, np.array([0,0,-1])))

if sun_direction[0] < 0:
    sign_x = 1
else:
    sign_x = -1
delta_x = sign_x * np.tan(sun_x) * grid.bbox_size[2]

if sun_direction[1] < 0:
    sign_y = 1
else:
    sign_y = -1
delta_y = sign_y * np.tan(sun_y) * grid.bbox_size[2]

print(sun_direction)
print(sun_x*(180/np.pi), sun_y*(180/np.pi))


visual = Visual_wrapper(grid)

visual.create_grid()
N = 8

# delta_x *= 0
# delta_y *= 0
if delta_x >= 0:
    dots_x = np.linspace(grid.bbox[0,0], grid.bbox[0,1] + delta_x, N)
else:
    dots_x = np.linspace(grid.bbox[0, 0] + delta_x, N, grid.bbox[0, 1])

if delta_y >= 0:
    dots_y = np.linspace(grid.bbox[1, 0], grid.bbox[1, 1] + delta_y, N)
else:
    dots_y = np.linspace(grid.bbox[1, 0] + delta_y, N, grid.bbox[1, 1])

for dot_x in dots_x:
    for dot_y in dots_y:
        point = np.array([dot_x, dot_y, grid.bbox[2,1]])
        # if dot_x < grid.bbox[0,0] or dot_x > grid.bbox[0,1] or dot_y < grid.bbox[1,0] or dot_y > grid.bbox[1,1]:
        #     visual.ax.scatter(*point, color="red")
        #
        #
        # else:
        #     visual.ax.scatter(*point, color="yellow")
        if dot_x < grid.bbox[0, 0] or dot_x > grid.bbox[0, 1] or dot_y < grid.bbox[1, 0] or dot_y > grid.bbox[1, 1]:
            visual.ax.quiver(dot_x, dot_y, grid.bbox[2, 1], *sun_direction * grid.bbox_size[2] * 2, color="blue",
                             arrow_length_ratio=0)
            t = -point[2] / sun_direction[2]
            groun_point = point + t*sun_direction
            visual.ax.scatter(*groun_point, color="green")

        intersected, tmin, tmax = get_intersection_with_bbox(point, sun_direction, grid.bbox)
        if not intersected:
            if point[0] >= grid.bbox[0,1] and point[1] >= grid.bbox[1,1]:
                print(True)
            else:
                print(False)



    # visual.ax.scatter(dot_x,0,0, color="blue")
visual.ax.quiver(0,0,0, 1*grid.bbox_size[0],0,0, arrow_length_ratio=0, color="blue")
visual.ax.quiver(0,grid.bbox_size[1],0, 1*grid.bbox_size[0],0,0, arrow_length_ratio=0, color="blue")

visual.ax.quiver(0,0,0, 0,1*grid.bbox_size[1],0, arrow_length_ratio=0, color="purple")
visual.ax.quiver(grid.bbox_size[0],0,0, 0,1*grid.bbox_size[1],0, arrow_length_ratio=0, color="purple")

visual.ax.quiver(0,0,grid.bbox_size[2], 1*grid.bbox_size[0],0,0, arrow_length_ratio=0, color="blue")
visual.ax.quiver(0,grid.bbox_size[1],grid.bbox_size[2], 1*grid.bbox_size[0],0,0, arrow_length_ratio=0, color="blue")

visual.ax.quiver(0,0,grid.bbox_size[2], 0,1*grid.bbox_size[1],0, arrow_length_ratio=0, color="purple")
visual.ax.quiver(grid.bbox_size[0],0,grid.bbox_size[2], 0,1*grid.bbox_size[1],0, arrow_length_ratio=0, color="purple")

plt.show()

print(dots_x[0], dots_x[-1])
print(dots_y[0], dots_y[-1])