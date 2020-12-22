


class CheckpointWrapper(object):
    def __init__(self, scene, Np_gt, Nps, Ns, resample_freqs, step_sizes, iter_phase, iterations, tensorboard_freq, train_id):
        self.scene = scene
        self.Np_gt = Np_gt
        self.Nps = Nps
        self.Ns = Ns
        self.resample_freqs = resample_freqs
        self.step_sizes = step_sizes
        self.iter_phase = iter_phase
        self.iterations = iterations
        self.tensorboard_freq = tensorboard_freq
        self.scene = scene
        self.train_id = train_id


    def __str__(self):
        Np_string = [f"{x:.2E}" for x in self.Nps]
        step_string = [f"{x:.2E}" for x in self.step_sizes]
        opti_desc = f"Simualtion Parameters:  \nNp_gt={self.Np_gt:.2E}  \nNps={Np_string}  \nNs={self.Ns}  \niterations={self.iterations}  \n" \
                    f"iter_phases={self.iter_phase}  \nstep_sizes={step_string}  \nresample_freq={self.resample_freqs}" \
                    f"  \ntensorboard_freq={self.tensorboard_freq}"
        return str(self.scene) + opti_desc