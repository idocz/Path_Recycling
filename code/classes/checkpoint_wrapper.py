from utils import *
from classes.scene_gpu import SceneGPU

class CheckpointWrapper(object):
    def __init__(self, scene, Np_gt, Nps, Ns, resample_freqs, step_sizes, iter_phase, iterations, tensorboard_freq, train_id):
        # Scene
        self.Ns = Ns
        self.volume = scene.volume
        self.sun_angles = scene.sun_angles
        self.sun_direction = scene.sun_direction
        self.cameras = scene.cameras
        self.g = scene.g
        self.N_cams = scene.N_cams
        self.N_pixels = scene.N_pixels
        self.is_camera_in_medium = scene.is_camera_in_medium
        self.scene_str = str(scene)

        self.Np_gt = Np_gt
        self.Nps = Nps
        self.resample_freqs = resample_freqs
        self.step_sizes = step_sizes
        self.iter_phase = iter_phase
        self.iterations = iterations
        self.tensorboard_freq = tensorboard_freq
        self.train_id = train_id


    def recreate_scene(self):
        return SceneGPU(self.volume, self.cameras, self.sun_angles, self.g, self.Ns)

    def __str__(self):
        Np_string = [f"{x:.2E}" for x in self.Nps]
        step_string = [f"{x:.2E}" for x in self.step_sizes]
        opti_desc = f"Simualtion Parameters:  \nNp_gt={self.Np_gt:.2E}  \nNps={Np_string}  \nNs={self.Ns}  \niterations={self.iterations}  \n" \
                    f"iter_phases={self.iter_phase}  \nstep_sizes={step_string}  \nresample_freq={self.resample_freqs}" \
                    f"  \ntensorboard_freq={self.tensorboard_freq}"
        return str(self.scene_str) + opti_desc