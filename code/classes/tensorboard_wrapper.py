import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
from os.path import join
from datetime import datetime
import pickle

class TensorBoardWrapper(object):
    def __init__(self, I_gt, betas_gt, title=None):
        self.max_val = np.max(I_gt, axis=(1,2)).reshape(-1,1,1)
        self.min_val = np.min(I_gt, axis=(1,2)).reshape(-1,1,1)
        if title is None:
            self.train_id = datetime.now().strftime("%d%m-%H%M-%S")
        else:
            self.train_id = title

        self.writer = SummaryWriter(log_dir=f"checkpoints/{self.train_id}")
        self.folder = join("checkpoints", self.train_id)
        os.mkdir(join("checkpoints", self.train_id, "data"))

        np.savez(join("checkpoints",self.train_id,"data","gt"), betas=betas_gt, images=I_gt,
                 min_val=self.min_val, max_val=self.max_val)
        I_gt_norm = transform(np.copy(I_gt), self.min_val, self.max_val)
        for i in range(I_gt.shape[0]):
            self.writer.add_image(f"ground_truth/{i}", I_gt_norm[i][None, :, :])



    def add_scene_text(self, text):
        self.writer.add_text("scene", text)


    def update(self, beta_opt, I_opt, loss, mean_dist, max_dist, rel_dist1, rel_dist2, grad_norm, iter):
        np.savez(join("checkpoints", self.train_id, "data", f"opt_{iter}"), betas=beta_opt, images=I_opt)
        I_opt_norm = transform(np.copy(I_opt), self.min_val, self.max_val)
        for i in range(I_opt.shape[0]):
            self.writer.add_image(f"simulated_images/{i}", I_opt_norm[i][None, :, :],
                             global_step=iter)
        self.writer.add_scalar("loss", loss, global_step=iter)
        self.writer.add_scalar("mean_dist", mean_dist, global_step=iter)
        self.writer.add_scalar("max_dist", max_dist, global_step=iter)
        self.writer.add_scalar("relative_dist1", rel_dist1, global_step=iter)
        self.writer.add_scalar("relative_dist", rel_dist2, global_step=iter)
        self.writer.add_scalar("grad_norm", grad_norm, global_step=iter)
        # self.writer.add_scalar("win_grad_norm", win_grad_norm, global_step=iter)




def transform(I_opt, min_val, max_val):
    I_opt_norm = (I_opt - min_val) / (max_val - min_val)
    I_opt_norm *= 255
    return I_opt_norm.astype("uint8")
