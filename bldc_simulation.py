import numpy as np
import scipy.constants
import scipy.integrate


class Coil:
    """
    Represents a coil of wire in the motor.
    """

    def __init__(
        self: "Coil",
        x: float,
        y: float,
        z: float,
        n_windings: int,
        resistance: float,
        theta: float,
        winding_radius: float,
        winding_depth: float,
    ):
        self.x = x
        self.y = y
        self.z = z
        self.n_windings = n_windings
        self.current = 0
        self.rotation_matrix = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        self.theta = theta
        self.winding_radius = winding_radius
        self.winding_depth = winding_depth
        self.Vmu_integrated = 0
        self.inductance = (
            scipy.constants.mu_0
            * n_windings**2
            * np.pi
            * winding_radius**2
            / winding_depth
        )
        self.tau = resistance / self.inductance

    def _l(self, t: float) -> np.array:
        return self.rotation_matrix @ np.array(
            [
                self.x
                + self.winding_radius
                * np.cos(2 * np.pi * self.n_windings * t / self.winding_depth),
                self.y + t,
                self.x
                + self.winding_radius
                * np.sin(2 * np.pi * self.n_windings * t / self.winding_depth),
            ]
        )

    def _l_dot(self, t: float) -> np.array:
        return self.rotation_matrix @ np.array(
            [
                -2
                * np.pi
                * self.n_windings
                * self.winding_radius
                * np.sin(2 * np.pi * self.n_windings * t / self.winding_depth)
                / self.winding_depth,
                1,
                2
                * np.pi
                * self.n_windings
                * self.winding_radius
                * np.cos(2 * np.pi * self.n_windings * t / self.winding_depth)
                / self.winding_depth,
            ]
        )

    def apply_voltage(self, v: float, t: float, dt: float):
        self.Vmu_integrated += (np.exp(-self.tau * t) * v / self.inductance) * dt
        self.current = self.Vmu_integrated / np.exp(self.tau * t)

    def B_field(self, x: float, y: float, z: float) -> np.array:
        """
        Uses the Biot-Savart law to calculate the magnetic field due to this
        coil at a point (x, y, z).
        """
        p = np.array([x, y, z])
        return (
            self.current
            * scipy.constants.mu_0
            * scipy.integrate.quad_vec(
                lambda t: np.cross(self._l_dot(t), p - self._l(t))
                / np.linalg.norm(self._l(t) - p) ** 3,
                0,
                self.winding_depth,
            )[0]
            / (4 * np.pi)
        )


class Motor:
    """
    Represents a simplified model of a three-phase brushless DC motor with the
    rotor centered on the origin and the stator composed of coils of wire
    surrounding it embedded in the x-y plane.
    """

    def __init__(
        self: "Motor",
        coils_per_phase: int,
        inner_radius: float,
        rotor_magnetic_moment: float,
        n_windings: int,
        coil_resistance: float,
        rotor_moment_of_inertia: float,
    ):
        # We need three dummy coils to model the currents for a given voltage
        # over time. Right here, though, we're just using them to determine the
        # magnetic field at the rotor.
        self.dummy_coils = [
            Coil(
                0,
                inner_radius,
                0,
                n_windings,
                coil_resistance,
                0,
                0.5e-2,
                1e-2,
            )
            for _ in range(3)
        ]
        self.dummy_coils[0].current = 15
        max_unidirectional_B = self.dummy_coils[0].B_field(0, 0, 0)
        rotation_matrix = np.array(
            [
                [
                    np.cos(2 * np.pi / (3 * coils_per_phase)),
                    -np.sin(2 * np.pi / (3 * coils_per_phase)),
                    0,
                ],
                [
                    np.sin(2 * np.pi / (3 * coils_per_phase)),
                    np.cos(2 * np.pi / (3 * coils_per_phase)),
                    0,
                ],
                [0, 0, 1],
            ]
        )
        self.max_Bs = [
            np.linalg.matrix_power(rotation_matrix, i) @ max_unidirectional_B
            for i in range(3 * coils_per_phase)
        ]
        self.coils_per_phase = coils_per_phase
        self.coil_resistance = coil_resistance
        self.rotor_magnetic_moment = rotor_magnetic_moment
        self.rotor_moment_of_inertia = rotor_moment_of_inertia
        self.t = 0
        self.states = []
        self.currents = []
        self.controller = None
        self.last_t = 0

    def _rhs(self: "Motor", t: float, state: np.array) -> np.array:
        """
        RHS of the internal ODE system.
        """
        dipole_moment = (
            np.array([np.cos(state[0]), np.sin(state[0]), 0])
            * self.rotor_magnetic_moment
        )
        voltages = self.controller.control(t, state)
        dt = t - self.last_t
        for coil, voltage in zip(self.dummy_coils, voltages):
            coil.apply_voltage(voltage, t, dt)
        self.last_t = t
        input_ = np.array([coil.current for coil in self.dummy_coils])
        B = sum(
            [
                B_at_origin * input_[i // self.coils_per_phase]
                for i, B_at_origin in enumerate(self.max_Bs)
            ]
        )
        torque = np.cross(dipole_moment, B) - 0.01 * state[1]  # added friction term
        return np.array([state[1], torque[2] / self.rotor_moment_of_inertia])

    def solve(self: "Motor", t_max: float, dt: float):
        """
        Solve the ODE system.
        """
        if self.controller is None:
            raise ValueError("Controller not set.")
        solver = scipy.integrate.ode(self._rhs)
        solver.set_integrator("vode", method="bdf", order=5)
        solver.set_initial_value(
            self.states[-1] if self.states else np.zeros(2), self.t
        )
        while solver.successful() and solver.t < t_max:
            print(solver.t)
            solver.integrate(solver.t + dt)
            self.states.append(solver.y)
        self.t = t_max
