import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Arrow

from theory.units import *
from simulation.astrometry_sim import QuasarSim

from theory.profiles import Profiles

class DemoSim:
    def __init__(self, theta_x_lims, theta_y_lims, n_dens, source_pos="random", custom_source_pos=None):

        self.theta_x_lims = theta_x_lims
        self.theta_y_lims = theta_y_lims

        self.roi_area = (theta_x_lims[1] - theta_x_lims[0]) * \
                        (theta_y_lims[1] - theta_y_lims[0])

        self.n_total = np.random.poisson(n_dens * self.roi_area)



        self.sources = np.zeros(self.n_total, dtype=[("theta_x", float, 1),
                                                     ("theta_y", float, 1),
                                                     ("theta_x_0", float, 1),
                                                     ("theta_y_0", float, 1),
                                                     ("mu", float, 1)])


        if source_pos == "random":
            if custom_source_pos is None:
                self.sources["theta_x_0"] = np.array(
                    list(np.random.uniform(*self.theta_x_lims, self.n_total)))
                self.sources["theta_y_0"] = np.array(
                    list(np.random.uniform(*self.theta_x_lims, self.n_total)))
            else:
                self.n_custom = len(custom_source_pos)
                self.sources["theta_x_0"] = np.array(
                    list(np.random.uniform(*self.theta_x_lims, self.n_total - self.n_custom)) + list(custom_source_pos[:,0]))
                self.sources["theta_y_0"] = np.array(
                    list(np.random.uniform(*self.theta_x_lims, self.n_total - self.n_custom)) + list(custom_source_pos[:,1]))

        elif source_pos == "uniform":
            xy_ratio = (theta_y_lims[1] - theta_y_lims[0]) / (theta_x_lims[1] - theta_x_lims[0])
            x_pos = np.linspace(theta_x_lims[0], theta_x_lims[1], np.round(np.sqrt(self.n_total / xy_ratio)))
            y_pos = np.linspace(theta_y_lims[0], theta_y_lims[1], np.round(np.sqrt(self.n_total * xy_ratio)))

            self.n_total = len(np.meshgrid(x_pos, y_pos)[0].flatten())

            self.sources = np.zeros(self.n_total, dtype=[("theta_x", float, 1),
                                                         ("theta_y", float, 1),
                                                         ("theta_x_0", float, 1),
                                                         ("theta_y_0", float, 1)])

            self.sources["theta_x_0"] = np.meshgrid(x_pos, y_pos)[0].flatten()
            self.sources["theta_y_0"] = np.meshgrid(x_pos, y_pos)[1].flatten()

    def animation(self, dt, pos_l, M_l, R_l, v_l, D_l, n_x=200, n_y=200, interval=10, frames=100, mult=2000, animate=True,
                  show_lens=True, show_sources=True, show_orig=False, show_vel_arrows=True, star_kwargs={}, star_orig_kwargs={}, arrow_kwargs={}):

        self.mult = mult
        self.show_lens = show_lens
        self.show_orig = show_orig
        self.show_vel_arrow = show_vel_arrows
        self.show_sources = show_sources

        assert len(pos_l) == len(v_l), "Lens position and velocity arrays must be the same size!"

        self.n_lens = len(pos_l)

        self.lenses = np.zeros(self.n_lens, dtype=[("theta_x", float, 1),
                                         ("theta_y", float, 1),
                                         ("M_0", float, 1),
                                         ("R_0", float, 1),
                                         ("v_x", float, 1),
                                         ("v_y", float, 1)])

        self.sources["theta_x"] = self.sources["theta_x_0"]
        self.sources["theta_y"] = self.sources["theta_y_0"]

        self.lenses["theta_x"] = pos_l[:, 0]
        self.lenses["theta_y"] = pos_l[:, 1]

        self.lenses["v_x"] = v_l[:, 0]
        self.lenses["v_y"] = v_l[:, 1]

        self.lenses["M_0"] = M_l
        self.lenses["R_0"] = R_l

        self.star_kwargs = star_kwargs
        self.star_orig_kwargs = star_orig_kwargs
        self.arrow_kwargs = arrow_kwargs

        fig = plt.figure(figsize=(16, 9))
        self.ax = plt.axes(xlim=self.theta_x_lims, ylim=self.theta_y_lims)
        self.ax.set_facecolor('black')

        if self.show_orig:
            self.scatter = self.ax.scatter(self.sources["theta_x"], self.sources["theta_y"], **self.star_orig_kwargs);
        if self.show_lens:
            self.x_coords = np.linspace(self.theta_x_lims[0], self.theta_x_lims[1], n_x)
            self.y_coords = np.linspace(self.theta_y_lims[0], self.theta_y_lims[1], n_y)

            im = np.zeros((n_x, n_y))

            for i_lens in range(self.n_lens):
                self.x_grid, self.y_grid = np.meshgrid(self.x_coords - self.lenses["theta_x"][i_lens], self.y_coords - self.lenses["theta_y"][i_lens])
                r_grid = np.sqrt(self.x_grid ** 2 + self.y_grid ** 2)
                im += Profiles.MdMdb_Gauss(r_grid, self.lenses["R_0"][i_lens] / D_l * radtoasc, self.lenses["M_0"][i_lens])[0]

            self.imshow = self.ax.imshow(im, origin='lower', cmap='Greys',
                                    extent= [*self.theta_x_lims, *self.theta_y_lims])

        mu_s = np.zeros((self.n_total, 2))
        theta_s = np.zeros((self.n_total, 2))

        for i_lens in range(self.n_lens):
            b_ary = np.transpose([self.sources["theta_x"] - self.lenses["theta_x"][i_lens],
                                  self.sources["theta_y"] - self.lenses["theta_y"][i_lens]]) * asctorad

            vel_l = np.array([self.lenses["v_x"][i_lens], self.lenses["v_y"][i_lens]])

            for i_source in range(self.n_total):

                mu_s[i_source] += QuasarSim.mu_ext(b_ary[i_source], vel_l / D_l, self.lenses["R_0"][i_lens], self.lenses["M_0"][i_lens], D_l)
                theta_s[i_source] += QuasarSim.theta_ext(b_ary[i_source], vel_l / D_l, self.lenses["R_0"][i_lens], self.lenses["M_0"][i_lens], D_l)

        self.sources["theta_x"] = self.sources["theta_x_0"] + theta_s[:, 0]
        self.sources["theta_y"] = self.sources["theta_y_0"] + theta_s[:, 1]

        if self.show_sources:
            self.scatter = self.ax.scatter(self.sources["theta_x"], self.sources["theta_y"], **self.star_kwargs);

        if self.show_vel_arrow:
            self.arrows = []

            for i_source in range(self.n_total):
                self.arrows.append(self.ax.add_patch(Arrow(self.sources["theta_x"][i_source],
                                                      self.sources["theta_y"][i_source],
                                                      mu_s[i_source, 0] * self.mult,
                                                      mu_s[i_source, 1] * self.mult,
                                                           **self.arrow_kwargs)))

        self.ax.get_xaxis().set_ticks([])
        self.ax.get_yaxis().set_ticks([])
        self.ax.get_xaxis().set_ticklabels([])
        self.ax.get_yaxis().set_ticklabels([])


        if animate:
            anim = FuncAnimation(fig, self.update, interval=interval, frames=frames, fargs=[dt, D_l])

            return anim

    def update(self, frame_number, dt, D_l):

        mu_s = np.zeros((self.n_total, 2))
        theta_s = np.zeros((self.n_total, 2))

        for i_lens in range(self.n_lens):
            b_ary = np.transpose([self.sources["theta_x"] - self.lenses["theta_x"][i_lens],
                                  self.sources["theta_y"] - self.lenses["theta_y"][i_lens]]) * asctorad

            vel_l = np.array([self.lenses["v_x"][i_lens], self.lenses["v_y"][i_lens]])
            for i_source in range(self.n_total):
                mu_s[i_source] += QuasarSim.mu_ext(b_ary[i_source], vel_l / D_l, self.lenses["R_0"][i_lens], self.lenses["M_0"][i_lens], D_l)
                theta_s[i_source] += QuasarSim.theta_ext(b_ary[i_source], vel_l / D_l, self.lenses["R_0"][i_lens], self.lenses["M_0"][i_lens], D_l)

            mu_l = (vel_l / D_l) / (Year ** -1) * radtoasc

            self.lenses["theta_x"][i_lens] = self.lenses["theta_x"][i_lens] + mu_l[0] * dt
            self.lenses["theta_y"][i_lens] = self.lenses["theta_y"][i_lens] + mu_l[1] * dt

        self.sources["theta_x"] = self.sources["theta_x_0"] + theta_s[:, 0]
        self.sources["theta_y"] = self.sources["theta_y_0"] + theta_s[:, 1]

        self.scatter.set_offsets(np.transpose([self.sources["theta_x"], self.sources["theta_y"]]))

        if self.show_lens:
            im = np.zeros_like(self.x_grid)

            for i_lens in range(self.n_lens):
                self.x_grid, self.y_grid = np.meshgrid(self.x_coords - self.lenses["theta_x"][i_lens],
                                                       self.y_coords - self.lenses["theta_y"][i_lens])
                r_grid = np.sqrt(self.x_grid ** 2 + self.y_grid ** 2)
                im += Profiles.MdMdb_Gauss(r_grid, self.lenses["R_0"][i_lens] / D_l * radtoasc, self.lenses["M_0"][i_lens])[0]

                self.imshow.set_array(im)

        if self.show_vel_arrow:
            for i_source in range(len(mu_s)):
                self.arrows[i_source].remove()
                self.arrows[i_source] = self.ax.add_patch(Arrow(self.sources["theta_x"][i_source],
                                                      self.sources["theta_y"][i_source],
                                                      mu_s[i_source, 0] * self.mult,
                                                      mu_s[i_source, 1] * self.mult,
                                                               **self.arrow_kwargs))

