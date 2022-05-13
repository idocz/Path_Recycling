import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
from os.path import join
from datetime import datetime
from utils import relative_distance, relative_bias
import pickle

class TensorBoardWrapper(object):
    def __init__(self, I_gt, betas_gt, title=None):
        self.max_val = np.max(I_gt, axis=(1,2)).reshape(-1,1,1)
        self.min_val = np.min(I_gt, axis=(1,2)).reshape(-1,1,1)
        self.gt_counter = 1
        if title is None:
            self.train_id = datetime.now().strftime("%d%m-%H%M-%S")
        else:
            self.train_id = title

        self.writer = SummaryWriter(log_dir=f"checkpoints/{self.train_id}")
        self.folder = join("checkpoints", self.train_id)
        os.mkdir(join("checkpoints", self.train_id, "data"))
        if betas_gt is not None:
            np.savez(join("checkpoints",self.train_id,"data","gt"), betas=betas_gt, images=I_gt,
                     min_val=self.min_val, max_val=self.max_val)
        # I_gt_norm = transform(np.copy(I_gt), self.min_val, self.max_val)
        I_gt_norm = transform(np.copy(I_gt), self.min_val.min(), self.max_val.max())
        for i in range(I_gt.shape[0]):
            self.writer.add_image(f"ground_truth/{i}", I_gt_norm[i][None, :, :], global_step=0)



    def add_scene_text(self, text):
        self.writer.add_text("scene", text)


    def update(self, beta_opt, I_opt, loss, max_dist, rel_dist1, Np, iter, time):
        time = np.array(time)
        if iter % (10) ==0:
            np.savez(join("checkpoints", self.train_id, "data", f"opt_{iter}"), betas=beta_opt, time=time)#, images=I_opt)
        # I_opt_norm = transform(np.copy(I_opt), self.min_val, self.max_val)
        # I_opt_norm[I_opt_norm<0]=0
        I_opt_norm = transform(np.copy(I_opt), I_opt.min(), I_opt.max())
        for i in range(I_opt.shape[0]):
            self.writer.add_image(f"simulated_images/{i}", I_opt_norm[i][None, :, :],
                             global_step=iter)
        self.writer.add_scalar("loss", loss, global_step=iter)
        if max_dist is not None:
            self.writer.add_scalar("max_dist", max_dist, global_step=iter)
        if rel_dist1 is not None:
            self.writer.add_scalar("relative_dist1", rel_dist1, global_step=iter)
        self.writer.add_scalar("Np", Np, global_step=iter)

    def update_gt(self, I_gt):
        self.max_val = np.max(I_gt, axis=(1,2)).reshape(-1,1,1)
        self.min_val = np.min(I_gt, axis=(1,2)).reshape(-1,1,1)
        I_gt_norm = transform(np.copy(I_gt), self.min_val, self.max_val)
        for i in range(I_gt.shape[0]):
            self.writer.add_image(f"ground_truth/{i}", I_gt_norm[i][None, :, :], global_step=self.gt_counter)
        self.gt_counter += 1

    def add_scatter_plot(self, I_gt, I_opt, step):
        # fig = plt.figure()
        fig = None
        ax = fig.add_subplot(111)
        X = I_gt.reshape(-1)
        Y = I_opt.reshape(-1)
        mask = X != 0
        X = X[mask]
        Y = Y[mask]
        N = 0.1
        N = int(Y.shape[0] * N)
        rand_inds = np.random.randint(0, X.shape[0], N)
        max_val = np.max([X.max(), Y.max()])
        min_val = np.min([X.min(), Y.min()])
        rel_err = relative_distance(X, Y)
        rel_bias = relative_bias(X, Y)
        ax.scatter(X[rand_inds], Y[rand_inds])
        # ax.scatter(X, Y)
        ax.plot([min_val, max_val], [min_val, max_val], color="red")
        fig.tight_layout()
        self.writer.add_figure("I_gt vs I_opt", fig, global_step=step, close=True)
        self.writer.add_scalar("rel_dist_img", rel_err, global_step=step)
        self.writer.add_scalar("rel_bias_img", rel_bias, global_step=step)


def transform(I_opt, min_val, max_val):
    I_opt_norm = (I_opt - min_val) / max_val - min_val
    I_opt_norm *= 255
    return I_opt_norm.astype("uint8")
