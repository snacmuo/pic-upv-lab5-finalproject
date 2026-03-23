import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.pyplot import cm

import gplugins.tidy3d.materials as mat

from .geometry import waveguide, waveguide_array
from .modes import get_TETM, mode_overlap, propagate_modes, slice_mode
from .plotting import my_plot_mode


class MMI_EME:
    def __init__(
        self,
        name="MMI EME",
        dim="2D",
        wvl=1.55,
        mat_core=mat.sin(1.55),
        mat_cladd=mat.sio2(1.55),
        polarization="TE",
        n_IN=2,
        IN_WVG_positions=(-1, 1),
        n_OUT=2,
        OUT_WVG_positions=(-1, 1),
        power_IN=(1, 0),
        wg_width=1.0,
        wg_num_modes=2,
        MMI_width=6.0,
        MMI_num_modes=20,
        dz=0.05,
        slices_np=1024,
        VERBOSE=False,
        ENABLE_MODE_PLOTS=False,
        ENABLE_MMI_PLOTS=False,
    ):
        self.name = name
        self.dim = dim
        self.wvl = wvl
        self.mat_core = mat_core
        self.mat_cladd = mat_cladd
        self.polarization = polarization
        self.n_IN = n_IN
        self.IN_WVG_positions = list(IN_WVG_positions)
        self.n_OUT = n_OUT
        self.OUT_WVG_positions = list(OUT_WVG_positions)
        self.power_IN_config = list(power_IN)
        self.wg_width = wg_width
        self.wg_num_modes = wg_num_modes
        self.MMI_width = MMI_width
        self.MMI_num_modes = MMI_num_modes
        self.dz = dz
        self.slices_np = slices_np
        self.wg_width_dw = 0.0
        self.dL_MMI = 0.0
        self.dx_IO = 0.0

        self.VERBOSE = VERBOSE
        self.ENABLE_MODE_PLOTS = ENABLE_MODE_PLOTS
        self.ENABLE_MMI_PLOTS = ENABLE_MMI_PLOTS

        self.MMI_modes = []
        self.MMI_basis = []
        self.XY = []
        self.MMI_modes_pol = []
        self.n_MODES = 0
        self.slices = []
        self.mode_IN_wvg = []
        self.mode_OUT_wvg = []

        self.L_pi_2D = 0.0
        self.L_MMI = 0.0
        self.IO_x_c = 0.0

        self.ovl_IN = []
        self.ovl_OUT = []
        self.z = []
        self.x = []
        self.intensity_z = []
        self.propag = []

        self.field_OUT = []
        self.power_OUT = []
        self.phase_OUT = []
        self.power_IN = 0.0
        self.tot_power_OUT = 0.0
        self.EL = 0.0
        self.ratio_OUT = 0.0

        self.io_wvg_mode = []
        self.io_wvg_mode_1D_slice = []
        self.io_wvg_mode_1D_x = []

        self.IN_WVG_slices_1D = []
        self.OUT_WVG_slices_1D = []

    def find_modes(self):
        x0 = -4 - self.MMI_width / 2.0
        x1 = -x0
        y0 = -2
        y1 = -y0
        self.XY = [x0, y0, x1, y1]

        self.MMI_modes, self.MMI_basis = waveguide(
            wg_width=self.MMI_width,
            wvl=self.wvl,
            num_modes=self.MMI_num_modes,
            XY=self.XY,
            ENABLE_PLOTS=self.ENABLE_MODE_PLOTS,
        )

        if self.VERBOSE:
            print(self.MMI_modes)
            print("Wavelength", self.wvl, " µm")
            print("---------------------------------------------")
            print("Core refractive index:", self.mat_core)
            for i, mode in enumerate(self.MMI_modes):
                print("--- Mode #" + str(i), " --------------------------------")
                print(f"Effective refractive index: {mode.n_eff:.4f}")
                print(f"TE fraction: {mode.te_fraction:.4f}")
                print(f"TM fraction: {mode.tm_fraction:.4f}")
                confinement = mode.calculate_confinement_factor(elements="core")
                print(f"Confinement factor: {confinement:.4f}")
            print("---------------------------------------------")
            print("Cladding refractive index:", self.mat_cladd)

        if self.ENABLE_MMI_PLOTS:
            for i, mode in enumerate(self.MMI_modes):
                my_plot_mode(mode, i, xlim=[x0, x1])

        MMI_modes_TE, MMI_modes_TM = get_TETM(modes=self.MMI_modes)
        n_TE = np.size(MMI_modes_TE)
        n_TM = np.size(MMI_modes_TM)
        n_all = np.size(self.MMI_modes)
        if self.VERBOSE:
            print(f"TE number: {n_TE}, TM number: {n_TM}, n_all: {n_all}")

        self.slice_modes_1D(MMI_modes_TE, MMI_modes_TM, n_TE, n_TM, x0, x1)
        self.L_pi_2D = 0.5 * self.wvl / np.real(self.MMI_modes_pol[0].n_eff - self.MMI_modes_pol[1].n_eff)

        if self.VERBOSE:
            print("n_MODES=", self.n_MODES)
            print(f"Lπ 2D = {self.L_pi_2D:.2f} µm")
            print(self.MMI_modes)

    def get_L_pi(self):
        return self.L_pi_2D

    def slice_modes_1D(self, MMI_modes_TE, MMI_modes_TM, n_TE, n_TM, x0, x1):
        if self.ENABLE_MMI_PLOTS:
            fig, axs = plt.subplots(2, 1)

        color = cm.rainbow(np.linspace(0, 1, n_TE))
        s_TE = np.zeros((n_TE, self.slices_np), dtype=np.complex128)
        for i, mode in enumerate(MMI_modes_TE):
            x, s = slice_mode(mode=mode, num_points=self.slices_np, x0=x0, x1=x1, polarization="TE")
            s_TE[i] = s
            if self.ENABLE_MMI_PLOTS:
                axs[0].plot(x, s_TE[i], color=color[i], label="TE" + str(i))

        color = cm.rainbow(np.linspace(0, 1, n_TM))
        s_TM = np.zeros((n_TM, self.slices_np), dtype=np.complex128)
        for i, mode in enumerate(MMI_modes_TM):
            x, s = slice_mode(mode=mode, num_points=self.slices_np, x0=x0, x1=x1, polarization="TM")
            s_TM[i] = s
            if self.ENABLE_MMI_PLOTS:
                axs[1].plot(x, s_TM[i], color=color[i], label="TM" + str(i))

        self.x = x

        if self.ENABLE_MMI_PLOTS:
            for ax in axs:
                ax.grid(True)
                ax.set_xlabel("x [µm]")
                ax.legend(loc="upper right")

        if self.polarization == "TE":
            self.MMI_modes_pol = MMI_modes_TE
            self.n_MODES = n_TE
            self.slices = s_TE
        else:
            self.MMI_modes_pol = MMI_modes_TM
            self.n_MODES = n_TM
            self.slices = s_TM

    def set_wg_width(self, w):
        self.wg_width = w

    def set_wg_width_dw(self, dw):
        self.wg_width_dw = dw

    def set_dL_MMI(self, dL):
        self.dL_MMI = dL

    def set_dx_IO(self, dx):
        self.dx_IO = dx

    def set_dz(self, dz):
        self.dz = dz

    def set_wg_num_modes(self, num_modes):
        self.wg_num_modes = num_modes

    def io_overlaps(self):
        self.ovl_IN = np.zeros((self.n_IN, self.n_MODES), dtype=np.complex128)
        for i, mode in enumerate(self.mode_IN_wvg):
            self.ovl_IN[i] = mode_overlap(mode_single=mode, modes_multiple=self.MMI_modes_pol)

        self.ovl_OUT = np.zeros((self.n_OUT, self.n_MODES), dtype=np.complex128)
        for i, mode in enumerate(self.mode_OUT_wvg):
            self.ovl_OUT[i] = mode_overlap(mode_single=mode, modes_multiple=self.MMI_modes_pol)

    def propagate(self):
        self.z, self.intensity_z, self.propag = propagate_modes(
            wvl=self.wvl,
            dz=self.dz,
            L=self.L_MMI + self.dL_MMI,
            ovl_z_0=self.ovl_IN[0],
            modes=self.MMI_modes_pol,
            slices=self.slices,
        )

    def output_transfer(self):
        self.field_OUT = np.zeros((self.n_OUT), dtype=np.complex128)
        self.power_OUT = np.zeros((self.n_OUT))
        self.phase_OUT = np.zeros((self.n_OUT))

        for i, ovl in enumerate(self.ovl_OUT):
            o = np.dot(self.propag, ovl)
            self.field_OUT[i] = o
            self.power_OUT[i] = np.abs(o) ** 2
            self.phase_OUT[i] = (180 / np.pi) * np.angle(o)

        self.power_IN = np.sum(np.abs(self.ovl_IN[0]) ** 2)
        self.tot_power_OUT = np.sum(self.power_OUT)
        self.EL = 10 * np.log10(1.0 / self.tot_power_OUT)
        self.ratio_OUT = self.power_OUT / self.tot_power_OUT

        print("------- Pameters -------")
        print("MMI length", f"{self.L_MMI:.4f}")
        print("MMI length increment", f"{self.dL_MMI:.4f}")
        print("IO wg width", f"{self.wg_width:.4f}")
        print("IO wg width increment", f"{self.wg_width_dw:.4f}")
        print("------------------------")
        print("Total power IN coupled", f"{self.power_IN:.4f}")
        print("Total OUT power:", f"{self.tot_power_OUT:.4f}")
        print("Excess loss [dB] = ", f"{self.EL:.4f}")
        print("------------------------")
        print("Power over OUTs: ", [f"{num:.4f}" for num in self.power_OUT])
        print("Ratio over OUTs", [f"{num:.4f}" for num in self.ratio_OUT])

    def plot_propagation(self, AspectRatioOne=True):
        fig, ax = plt.subplots()
        if AspectRatioOne:
            ax.set_aspect(1)

        x_m, z_m = np.meshgrid(self.x, self.z)
        ax.pcolormesh(z_m, x_m, self.intensity_z, cmap="hsv")
        ax.add_patch(
            Rectangle(
                (0, -self.MMI_width / 2.0),
                self.L_MMI + self.dL_MMI,
                self.MMI_width,
                edgecolor="white",
                facecolor="white",
                alpha=0.25,
                lw=4,
            )
        )

        for x_c_i in self.IN_WVG_positions:
            ax.add_patch(
                Rectangle(
                    (-5, x_c_i - (self.wg_width + self.wg_width_dw) / 2.0),
                    5,
                    self.wg_width + self.wg_width_dw,
                    edgecolor="white",
                    facecolor="grey",
                    alpha=0.25,
                    lw=4,
                )
            )
        for x_c_i in self.OUT_WVG_positions:
            ax.add_patch(
                Rectangle(
                    (self.L_MMI + self.dL_MMI, x_c_i - (self.wg_width + self.wg_width_dw) / 2.0),
                    5,
                    self.wg_width + self.wg_width_dw,
                    edgecolor="white",
                    facecolor="grey",
                    alpha=0.25,
                    lw=4,
                )
            )

        ax.set_xlabel("z [µm]")
        ax.set_ylabel("x [µm]")
        ax.set_title("EME propagation")
        return fig, ax

    def io_waveguides_modes(self):
        IWVG_0_modes, _ = waveguide(
            wg_width=self.wg_width + self.wg_width_dw,
            x_c=0.0,
            wvl=self.wvl,
            mat_core=self.mat_core,
            mat_cladd=self.mat_cladd,
            num_modes=self.wg_num_modes,
            XY=self.XY,
            ENABLE_PLOTS=self.ENABLE_MODE_PLOTS,
        )

        IWVG_0_modes_TE, IWVG_0_modes_TM = get_TETM(IWVG_0_modes)
        if self.polarization == "TE":
            self.mode_IN_wvg.append(IWVG_0_modes_TE[0])
        else:
            self.mode_IN_wvg.append(IWVG_0_modes_TM[0])

    def io_waveguide_mode_1D(self):
        IOWVG_mode, _ = waveguide(
            wg_width=self.wg_width + self.wg_width_dw,
            x_c=0.0,
            wvl=self.wvl,
            mat_core=self.mat_core,
            mat_cladd=self.mat_cladd,
            num_modes=self.wg_num_modes,
            XY=self.XY,
            ENABLE_PLOTS=self.ENABLE_MODE_PLOTS,
        )

        m_TE, m_TM = get_TETM(IOWVG_mode)

        x0 = self.XY[0]
        x1 = self.XY[2]
        if self.polarization == "TE" and m_TE:
            self.io_wvg_mode_1D = m_TE[0]
            x, s = slice_mode(mode=m_TE[0], num_points=self.slices_np, x0=x0, x1=x1, polarization="TE")
            self.io_wvg_mode_1D_slice = s
            self.io_wvg_mode_1D_x = x

        if self.polarization == "TM" and m_TM:
            self.io_wvg_mode_1D = m_TM[0]
            x, s = slice_mode(mode=m_TM[0], num_points=self.slices_np, x0=x0, x1=x1, polarization="TM")
            self.io_wvg_mode_1D_slice = s
            self.io_wvg_mode_1D_x = x

    def shift_IO_wvg_slices(self):
        self.IN_WVG_slices_1D = np.zeros((self.n_IN, self.slices_np), dtype=np.complex128)
        self.OUT_WVG_slices_1D = np.zeros((self.n_OUT, self.slices_np), dtype=np.complex128)
        x = self.x
        s = self.io_wvg_mode_1D_slice

        for i in range(self.n_IN):
            dx = self.IN_WVG_positions[i]
            dx_idx = int(dx / (x[1] - x[0]))
            self.IN_WVG_slices_1D[i] = np.roll(s, dx_idx)

        for i in range(self.n_OUT):
            dx = self.OUT_WVG_positions[i]
            dx_idx = int(dx / (x[1] - x[0]))
            self.OUT_WVG_slices_1D[i] = np.roll(s, dx_idx)

    def normalize_slice_1D(self):
        s = self.io_wvg_mode_1D_slice
        integral = np.trapezoid(np.abs(s) ** 2, self.x)
        self.io_wvg_mode_1D_slice = s / np.sqrt(integral)

    def normalize_MMI_slices_1D(self):
        for i, s in enumerate(self.slices):
            integral = np.trapezoid(np.abs(s) ** 2, self.x)
            self.slices[i] = s / np.sqrt(integral)

    def overlap_integral(self, s1, s2, x):
        return np.trapezoid(s1 * s2, x)

    def IO_overlap_1D(self):
        self.normalize_MMI_slices_1D()
        self.normalize_slice_1D()
        self.shift_IO_wvg_slices()

        x = self.x
        self.ovl_IN = np.zeros((self.n_IN, self.n_MODES), dtype=np.complex128)
        for i, s1 in enumerate(self.IN_WVG_slices_1D):
            for j, s2 in enumerate(self.slices):
                self.ovl_IN[i][j] = self.overlap_integral(s1, s2, x)

        self.ovl_OUT = np.zeros((self.n_OUT, self.n_MODES), dtype=np.complex128)
        for i, s1 in enumerate(self.OUT_WVG_slices_1D):
            for j, s2 in enumerate(self.slices):
                self.ovl_OUT[i][j] = self.overlap_integral(s1, s2, x)

    def plot_mode_slices(self):
        x = self.x
        s_in = self.IN_WVG_slices_1D
        s_MMI = self.slices
        s_out = self.OUT_WVG_slices_1D
        pol = self.polarization

        fig, ax = plt.subplots(3, 1)
        ax[0].plot(x, s_in.T, label=pol + "0, in0")
        ax[0].grid(True)
        ax[0].legend(fontsize=8)

        for i, s in enumerate(s_MMI):
            ax[1].plot(x, s.T, label=pol + f"{i}" + ", MMI")

        ax[1].grid(True)
        ax[1].legend(fontsize=8)

        for i, s in enumerate(s_out):
            ax[2].plot(x, s.T, label=pol + "0, out" + f"{i}")
        ax[2].grid(True)
        ax[2].legend(fontsize=8)
        ax[2].set_xlabel("x [µm]")
        return fig, ax

    def run_full(self):
        self.find_modes()
        self.io_waveguides_modes()
        self.io_overlaps()
        self.propagate()
        self.output_transfer()

    def run_1D_full(self):
        self.find_modes()
        self.io_waveguides_modes()
        self.io_waveguide_mode_1D()
        self.IO_overlap_1D()
        self.propagate()
        self.output_transfer()

    def run(self):
        self.io_waveguides_modes()
        self.io_waveguide_mode_1D()
        self.IO_overlap_1D()
        self.propagate()
        self.output_transfer()

    def find_all_modes(self):
        self.find_modes()
        self.io_waveguides_modes()
        self.io_waveguide_mode_1D()

    def propagation(self):
        self.IO_overlap_1D()
        self.propagate()
        self.output_transfer()
        self.plot_propagation()


class DC_EME(MMI_EME):
    def __init__(
        self,
        name="Directional Coupler",
        DC_N_waveguides=2,
        DC_wg_width=1.0,
        DC_wg_gap=0.8,
    ):
        super().__init__(name=name)
        self.DC_N_waveguides = DC_N_waveguides
        self.DC_wg_width = DC_wg_width
        self.DC_wg_gap = DC_wg_gap
        self.centers = []

    def find_modes(self):
        x0 = -4 - self.MMI_width / 2.0
        x1 = -x0
        y0 = -2
        y1 = -y0
        self.XY = [x0, y0, x1, y1]

        self.MMI_modes, self.MMI_basis, XY, centers = waveguide_array(
            wg_width=self.DC_wg_width,
            wvl=self.wvl,
            num_modes=self.MMI_num_modes,
            XY=self.XY,
            ENABLE_PLOTS=self.ENABLE_MODE_PLOTS,
            wg_N=self.DC_N_waveguides,
            wg_gap=self.DC_wg_gap,
        )

        self.centers = centers
        self.OUT_WVG_positions = list(centers)
        self.XY = XY
        x0, y0, x1, y1 = self.XY

        if self.VERBOSE:
            print(self.MMI_modes)
            print("Wavelength", self.wvl, " µm")
            print("---------------------------------------------")
            print("Core refractive index:", self.mat_core)
            for i, mode in enumerate(self.MMI_modes):
                print("--- Mode #" + str(i), " --------------------------------")
                print(f"Effective refractive index: {mode.n_eff:.4f}")
                print(f"TE fraction: {mode.te_fraction:.4f}")
                print(f"TM fraction: {mode.tm_fraction:.4f}")
                confinement = mode.calculate_confinement_factor(elements="core")
                print(f"Confinement factor: {confinement:.4f}")
            print("---------------------------------------------")
            print("Cladding refractive index:", self.mat_cladd)

        if self.ENABLE_MMI_PLOTS:
            for i, mode in enumerate(self.MMI_modes):
                my_plot_mode(mode, i, xlim=[x0, x1])

        MMI_modes_TE, MMI_modes_TM = get_TETM(modes=self.MMI_modes)
        n_TE = np.size(MMI_modes_TE)
        n_TM = np.size(MMI_modes_TM)
        n_all = np.size(self.MMI_modes)
        if self.VERBOSE:
            print(f"TE number: {n_TE}, TM number: {n_TM}, n_all: {n_all}")

        self.slice_modes_1D(MMI_modes_TE, MMI_modes_TM, n_TE, n_TM, x0, x1)
        self.L_pi_2D = 0.5 * self.wvl / np.real(self.MMI_modes_pol[0].n_eff - self.MMI_modes_pol[1].n_eff)

        if self.VERBOSE:
            print("n_MODES=", self.n_MODES)
            print(f"Lπ 2D = {self.L_pi_2D:.2f} µm")
            print(self.MMI_modes)

    def plot_propagation(self, AspectRatioOne=True):
        fig, ax = plt.subplots()
        if AspectRatioOne:
            ax.set_aspect(1)

        x_m, z_m = np.meshgrid(self.x, self.z)
        ax.pcolormesh(z_m, x_m, self.intensity_z, cmap="hsv")

        for x_c_i in self.centers:
            ax.add_patch(
                Rectangle(
                    (0, x_c_i - self.wg_width / 2.0),
                    self.L_MMI + self.dL_MMI,
                    self.wg_width,
                    edgecolor="white",
                    facecolor="white",
                    alpha=0.25,
                    lw=4,
                )
            )

        for x_c_i in self.IN_WVG_positions:
            ax.add_patch(
                Rectangle(
                    (-5, x_c_i - (self.wg_width + self.wg_width_dw) / 2.0),
                    5,
                    self.wg_width + self.wg_width_dw,
                    edgecolor="white",
                    facecolor="grey",
                    alpha=0.25,
                    lw=4,
                )
            )
        for x_c_i in self.OUT_WVG_positions:
            ax.add_patch(
                Rectangle(
                    (self.L_MMI + self.dL_MMI, x_c_i - (self.wg_width + self.wg_width_dw) / 2.0),
                    5,
                    self.wg_width + self.wg_width_dw,
                    edgecolor="white",
                    facecolor="grey",
                    alpha=0.25,
                    lw=4,
                )
            )

        ax.set_xlabel("z [µm]")
        ax.set_ylabel("x [µm]")
        ax.set_title("EME propagation")
        return fig, ax
