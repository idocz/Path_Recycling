from classes.visual import *
from classes.camera import *

bbox = np.array([[0,1],
                 [0,1],
                 [0,1]])

shape = np.array((5,5,5))
grid = Grid(bbox, shape)
beta_cloud = np.ones(shape.tolist())
beta_air = 1
volume = Volume(grid, beta_cloud, beta_air)
sun_angles = [-np.pi/2, 0]

t = np.array([0.5, 0.1, 0.5])
focal_length = 20e-3
sensor_size = np.array((32e-3, 32e-3))
pixels = np.array((50, 50))
euler_angles = np.array((-90, 30, 0))
camera = Camera(t, euler_angles, focal_length, sensor_size, pixels)
cameras = [camera]

scene = Scene(volume, cameras, sun_angles)
visual = Visual_wrapper(scene)
visual.create_grid()
visual.plot_cameras()

X_W = np.array([[-1.5, -1. , -0.5, -0.5, -1.5, -1. , -0.5, -0.5, -1.5, -1. , -0.5,
         0.5,  1. ,  0.5,  1.5,  0.5,  1.5,  0.5,  1.5,  0.5,  1. ,  0. ],
       [ 5. ,  5. ,  5. ,  5. ,  5. ,  5. ,  5. ,  5. ,  5. ,  5. ,  5. ,
         7. ,  7. ,  7. ,  7. ,  7. ,  7. ,  7. ,  7. ,  7. ,  7. ,  6. ],
       [ 1. ,  1. ,  1. ,  0.5,  0. ,  0. ,  0. , -0.5, -1. , -1. , -1. ,
         1. ,  1. ,  0.5,  0.5,  0. ,  0. , -0.5, -0.5, -1. , -1. ,  0. ]])

X_W = (X_W - X_W.min()) / (X_W.max() - X_W.min())
X_W[0] += 0.4
X_W[1] -= 0.2
X_W[2] += 0.4
visual.ax.scatter(X_W[0,:],X_W[1,:],X_W[2,:],alpha=1)
plt.show()

img = np.zeros(pixels, dtype=np.uint8)
for point in X_W.T:
    pixel = camera.project_point(point)

    if (pixel != -1).all():
        print(pixel)
        img[pixel[1], pixel[0]] = 255

plt.imshow(img, cmap="gray")
plt.show()


