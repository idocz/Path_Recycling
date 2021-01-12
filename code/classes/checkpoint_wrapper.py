from utils import *
from classes.scene_gpu import SceneGPU

class CheckpointWrapper(object):
    def __init__(self, scene, optimizer, Np_gt, Np_start, Ns, resample_freq, step_size, iterations, tensorboard_freq, train_id):
        self.optimizer = optimizer
        # Scene
        self.Ns = Ns
        self.volume = scene.volume
        self.sun_angles = scene.sun_angles
        self.sun_direction = scene.sun_direction
        self.cameras = scene.cameras
        self.g_cloud = scene.g_cloud
        self.g_air = scene.g_air
        self.N_cams = scene.N_cams
        self.N_pixels = scene.N_pixels
        self.is_camera_in_medium = scene.is_camera_in_medium
        self.scene_str = str(scene)

        self.Np_gt = Np_gt
        self.Np_start = Np_start
        self.resample_freq = resample_freq
        self.step_size = step_size
        self.iterations = iterations
        self.tensorboard_freq = tensorboard_freq
        self.train_id = train_id


    def recreate_scene(self):
        return SceneGPU(self.volume, self.cameras, self.sun_angles, self.g_cloud, self.g_air, self.Ns)

    def __str__(self):
        opti_desc = f"Simualtion Parameters:  \nNp_gt={self.Np_gt:.2e}  \nNp_start={self.Np_start}  \nNs={self.Ns}" \
                    f"  \niterations={self.iterations}  \n  nstep_size={self.step_size:.2e}  \nresample_freq={self.resample_freq}" \
                    f"  \ntensorboard_freq={self.tensorboard_freq}  \nOptimizer:  \n{str(self.optimizer)}"
        return str(self.scene_str) + opti_desc