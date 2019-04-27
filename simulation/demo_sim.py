import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from theory.units import *
from simulation.astrometry_sim import QuasarSim

class DemoSim:
    def __init__(self, theta_x_lims, theta_y_lims, n_x, n_y, n_dens):

        self.theta_x_lims = theta_x_lims
        self.theta_y_lims = theta_y_lims

        self.theta_x, self.theta_y = np.meshgrid(
            np.linspace(theta_x_lims[0], theta_x_lims[1], n_x),
            np.linspace(theta_y_lims[0], theta_y_lims[1], n_y),
        )

        self.pix_area = (theta_x_lims[1] - theta_x_lims[0]) / n_x * \
                        (theta_y_lims[1] - theta_y_lims[0]) / n_y

        self.roi_area = (theta_x_lims[1] - theta_x_lims[0]) * \
                        (theta_y_lims[1] - theta_y_lims[0])

        self.n_total = np.random.poisson(n_dens * self.roi_area)

    def animation(self, dt, v_l, D_l):

        self.n_total += 2

        self.sources = np.zeros(self.n_total, dtype=[("theta_x", float, 1),
                                           ("theta_y", float, 1),
                                           ("theta_x_0", float, 1),
                                           ("theta_y_0", float, 1),
                                           ("mu", float, 1)])

        self.lenses = np.zeros(1, dtype=[("theta_x", float, 1),
                                    ("theta_y", float, 1),
                                    ("velocity", float, 1)])

        self.sources["theta_x_0"] = np.array(list(np.random.uniform(*self.theta_x_lims, self.n_total - 2)) + [-0.5, -0.4])
        self.sources["theta_y_0"] = np.array(list(np.random.uniform(*self.theta_x_lims, self.n_total - 2)) + [0.001, -0.002])

        self.sources["theta_x"] = self.sources["theta_x_0"]
        self.sources["theta_y"] = self.sources["theta_y_0"]

        self.lenses["theta_x"] = -1
        self.lenses["theta_y"] = 0


        fig = plt.figure(figsize=(16,9))
        ax = plt.axes(xlim=self.theta_x_lims,ylim=self.theta_y_lims)
        ax.set_facecolor('black')

        self.scatter=ax.scatter(self.sources["theta_x"], self.sources["theta_y"], marker='*', color='gold', s=30);
        self.scatter2=ax.scatter(self.lenses["theta_x"], self.lenses["theta_y"], color='black');

        anim = FuncAnimation(fig, self.update, interval=10, frames=1000, fargs=[dt, v_l, D_l])

        return anim

    def update(self, frame_number, dt, v_l, D_l):
        b_ary = np.transpose([self.sources["theta_x"] - self.lenses["theta_x"], self.sources["theta_y"] - self.lenses["theta_y"]]) * asctorad
        #     mu_s = np.array([QuasarSim.mu_ext(b, v_l / D_l, 0.5 * pc, 10 ** 6 * M_s, D_l) for b in b_ary])
        theta_s = np.array([QuasarSim.theta_ext(b, v_l / D_l, 0.005 * pc, 5 * 10 ** 5 * M_s, D_l) for b in b_ary])

        mu_l = (v_l / D_l) / (Year ** -1) * radtoasc

        self.lenses["theta_x"] = self.lenses["theta_x"] + mu_l[0] * dt
        self.lenses["theta_x"] = self.lenses["theta_x"] + mu_l[1] * dt

        self.sources["theta_x"] = self.sources["theta_x_0"] + theta_s[:, 0]
        self.sources["theta_y"] = self.sources["theta_y_0"] + theta_s[:, 1]

        self.scatter.set_offsets(np.transpose([self.sources["theta_x"], self.sources["theta_y"]]))
        self.scatter2.set_offsets(np.transpose([self.lenses["theta_x"], self.lenses["theta_y"]]))

        return self.scatter,
