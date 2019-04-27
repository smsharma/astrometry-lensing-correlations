import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from theory.units import *
from simulation.astrometry_sim import QuasarSim

from theory.profiles import Profiles

class DemoSim:
    def __init__(self, theta_x_lims, theta_y_lims, n_dens, source_pos="random", custom_source_pos=[]):

        self.theta_x_lims = theta_x_lims
        self.theta_y_lims = theta_y_lims

        self.roi_area = (theta_x_lims[1] - theta_x_lims[0]) * \
                        (theta_y_lims[1] - theta_y_lims[0])

        self.n_total = np.random.poisson(n_dens * self.roi_area)

        self.n_custom = len(custom_source_pos)

        self.sources = np.zeros(self.n_total, dtype=[("theta_x", float, 1),
                                                     ("theta_y", float, 1),
                                                     ("theta_x_0", float, 1),
                                                     ("theta_y_0", float, 1),
                                                     ("mu", float, 1)])

        if source_pos == "random":
            self.sources["theta_x_0"] = np.array(
                list(np.random.uniform(*self.theta_x_lims, self.n_total - self.n_custom)) + list(custom_source_pos[:0]))
            self.sources["theta_y_0"] = np.array(
                list(np.random.uniform(*self.theta_x_lims, self.n_total - self.n_custom)) + list(custom_source_pos[:0]))
        elif source_pos == "uniform":
            xy_ratio = (theta_y_lims[1] - theta_y_lims[0]) / (theta_x_lims[1] - theta_x_lims[0])
            x_pos = np.linspace(theta_x_lims[0], theta_x_lims[1], np.round(np.sqrt(self.n_total / xy_ratio)))
            y_pos = np.linspace(theta_y_lims[0], theta_y_lims[1], np.round(np.sqrt(self.n_total * xy_ratio)))

            self.n_total = len(np.meshgrid(x_pos, y_pos)[0].flatten())

            self.sources = np.zeros(self.n_total, dtype=[("theta_x", float, 1),
                                                         ("theta_y", float, 1),
                                                         ("theta_x_0", float, 1),
                                                         ("theta_y_0", float, 1),
                                                         ("mu", float, 1)])

            self.sources["theta_x_0"] = np.meshgrid(x_pos, y_pos)[0].flatten()
            self.sources["theta_y_0"] = np.meshgrid(x_pos, y_pos)[1].flatten()

    def animation(self, dt, v_l, D_l, R_0, n_x=1000, n_y=1000):


        self.lenses = np.zeros(1, dtype=[("theta_x", float, 1),
                                         ("theta_y", float, 1),
                                         ("velocity", float, 1)])

        self.sources["theta_x"] = self.sources["theta_x_0"]
        self.sources["theta_y"] = self.sources["theta_y_0"]

        self.lenses["theta_x"] = -1
        self.lenses["theta_y"] = 0

        fig = plt.figure(figsize=(16, 9))
        ax = plt.axes(xlim=self.theta_x_lims, ylim=self.theta_y_lims)
        ax.set_facecolor('black')

        self.scatter = ax.scatter(self.sources["theta_x"], self.sources["theta_y"], marker='*', color='gold', s=50);

        self.x_coords = np.linspace(self.theta_x_lims[0], self.theta_x_lims[1], n_x)
        self.y_coords = np.linspace(self.theta_y_lims[0], self.theta_y_lims[1], n_y)

        self.x_grid, self.y_grid = np.meshgrid(self.x_coords - self.lenses["theta_x"], self.y_coords - self.lenses["theta_y"])
        r_grid = np.sqrt(self.x_grid ** 2 + self.y_grid ** 2)
        self.imshow = ax.imshow(Profiles.MdMdb_Gauss(r_grid, R_0 / D_l, 5 * 10 ** 5 * M_s)[0], origin='lower', cmap='Greys',
                                extent= [*self.theta_x_lims, *self.theta_y_lims])

        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.get_xaxis().set_ticklabels([])
        ax.get_yaxis().set_ticklabels([])

        anim = FuncAnimation(fig, self.update, interval=10, frames=100, fargs=[dt, v_l, D_l, R_0])

        return anim

    def update(self, frame_number, dt, v_l, D_l, R_0):
        b_ary = np.transpose([self.sources["theta_x"] - self.lenses["theta_x"],
                              self.sources["theta_y"] - self.lenses["theta_y"]]) * asctorad
        #     mu_s = np.array([QuasarSim.mu_ext(b, v_l / D_l, 0.5 * pc, 10 ** 6 * M_s, D_l) for b in b_ary])
        theta_s = np.array([QuasarSim.theta_ext(b, v_l / D_l, 0.005 * pc, 5 * 10 ** 5 * M_s, D_l) for b in b_ary])

        mu_l = (v_l / D_l) / (Year ** -1) * radtoasc

        self.lenses["theta_x"] = self.lenses["theta_x"] + mu_l[0] * dt
        self.lenses["theta_x"] = self.lenses["theta_x"] + mu_l[1] * dt

        self.sources["theta_x"] = self.sources["theta_x_0"] + theta_s[:, 0]
        self.sources["theta_y"] = self.sources["theta_y_0"] + theta_s[:, 1]

        self.scatter.set_offsets(np.transpose([self.sources["theta_x"], self.sources["theta_y"]]))

        self.x_grid, self.y_grid = np.meshgrid(self.x_coords - self.lenses["theta_x"], self.y_coords - self.lenses["theta_y"])
        r_grid = np.sqrt(self.x_grid ** 2 + self.y_grid ** 2)

        self.imshow.set_array(Profiles.MdMdb_Gauss(r_grid, R_0/D_l, 5 * 10 ** 5 * M_s)[0])
