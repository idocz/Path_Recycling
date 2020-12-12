import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
from os.path import join
from datetime import datetime

class TensorBoardWrapper(object):
    def __init__(self, I_gt, betas_gt, scene_descr):
        self.max_val = np.max(I_gt, axis=(1,2)).reshape(-1,1,1)
        self.min_val = np.min(I_gt, axis=(1,2)).reshape(-1,1,1)
        time = datetime.now().strftime("%d%m-%H%M-%S")
        self.train_id = time
        self.writer = SummaryWriter(log_dir=f"checkpoints/{self.train_id}")
        os.mkdir(join("checkpoints", self.train_id, "data"))
        np.savez(join("checkpoints",self.train_id,"data","gt"), betas=betas_gt, images=I_gt,
                 min_val=self.min_val, max_val=self.max_val)
        I_gt_norm = transform(I_gt, self.min_val, self.max_val)
        for i in range(I_gt.shape[0]):
            self.writer.add_image(f"ground_truth/{i}", I_gt_norm[i][None, :, :])
        self.writer.add_text("scene", scene_descr)


    def update(self, beta_opt, I_opt, loss, mean_dist, max_dist, rel_dist, iter):
        np.savez(join("checkpoints", self.train_id, "data", f"opt_{iter}"), betas=beta_opt, images=I_opt)
        I_opt_norm = transform(I_opt, self.min_val, self.max_val)
        for i in range(I_opt.shape[0]):
            self.writer.add_image(f"simulated_images/{i}", I_opt_norm[i][None, :, :],
                             global_step=iter)
        self.writer.add_scalar("loss", loss, global_step=iter)
        self.writer.add_scalar("mean_dist", mean_dist, global_step=iter)
        self.writer.add_scalar("max_dist", max_dist, global_step=iter)
        self.writer.add_scalar("relative_dist", rel_dist, global_step=iter)



def transform(I_opt, min_val, max_val):
    I_opt_norm = (I_opt - min_val) / (max_val - min_val)
    I_opt_norm *= 255
    return I_opt_norm.astype("uint8")
