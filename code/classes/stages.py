

class Stages(object):
    def __init__(self, N_stages, Np_start, Np_gt, volume_downfactor, images_downfactor, step_size_start, step_size_end,
                  volume):
        self.N_stages = N_stages
        self.Np_start = Np_start
        self.Np_gt = Np_gt
        self.volume_downfactor = volume_downfactor
        self.images_downfactor = images_downfactor
        self.step_size_start = step_size_start
        self.step_size_start_end = step_size_end
