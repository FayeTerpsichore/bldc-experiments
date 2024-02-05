import scipy
import scipy.integrate
import scipy.constants
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
import seaborn as sns
import sympy
import typing


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
        current: float,
        resistance: float,
        theta: float,
        winding_radius: float,
        winding_depth: float,
    ):
        self.x = x
        self.y = y
        self.z = z
        self.n_windings = n_windings
        self.current = current
        self.resistance = resistance
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

    def _l(self, t: float):
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

    def _l_dot(self, t: float):
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

    def B_field(self, x: float, y: float, z: float):
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
        max_current: float,
        coil_resistance: float,
        rotor_moment_of_inertia: float,
    ):
        max_unidirectional_B = Coil(
            0,
            inner_radius,
            0,
            n_windings,
            max_current,
            coil_resistance,
            0,
            0.5e-2,
            1e-2,
        ).B_field(0, 0, 0)
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
        self.controller: "SpeedController" | "PositionController" = None

    def _rhs(self: "Motor", t: float, state: np.array) -> np.array:
        """
        RHS of the internal ODE system.
        """
        dipole_moment = (
            np.array([np.cos(state[0]), np.sin(state[0]), 0])
            * self.rotor_magnetic_moment
        )
        input_ = self.controller.control(t, state) / (
            self.coils_per_phase * self.coil_resistance
        )
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
        solver.set_integrator("vode")
        solver.set_initial_value(
            self.states[-1] if self.states else np.zeros(2), self.t
        )
        while solver.successful() and solver.t < t_max:
            solver.integrate(solver.t + dt)
            self.controller.target_position = np.sqrt(200 * solver.t)
            self.states.append(solver.y)
        self.t = t_max


class SpeedController:
    """
    Represents a proportional controller for the motor intended to keep the rotor
    at a constant speed.
    """

    def __init__(
        self: "SpeedController",
        motor: Motor,
        target_speed: float,
    ):
        self.max_Bs = motor.max_Bs
        self.motor_coils_per_phase = motor.coils_per_phase
        self.motor_coil_resistance = motor.coil_resistance
        self.rotor_magnetic_moment = motor.rotor_magnetic_moment
        self.rotor_moment_of_inertia = motor.rotor_moment_of_inertia
        self.target_speed = target_speed
        self.last_phases = np.zeros(2)
        self.torque_optimizer = self.find_torque_optimizer()
        self.past_error = 0
        self.gain = 10
        self.motor = motor

    def find_torque_optimizer(
        self: "SpeedController",
    ) -> typing.Callable[[float, float, float, float], float]:
        """
        Derives the expression relating phase 2's voltage and phase 1's voltage for a
        given torque, rotor angle, and rotor angular velocity.
        """
        phi_1, phi_2 = sympy.symbols("phi_1 phi_2", real=True)
        phi_3 = -phi_1 - phi_2
        t = sympy.Symbol("t", positive=True, real=True)
        theta = sympy.Function("theta")(t)
        dipole_moment = self.rotor_magnetic_moment * sympy.Matrix(
            [[sympy.cos(theta)], [sympy.sin(theta)], [0]]
        )
        input_ = (
            self.motor_coils_per_phase * self.motor_coil_resistance
        ) ** -1 * sympy.Matrix([phi_1, phi_2, phi_3])
        B = sympy.Matrix([0, 0, 0])
        for i, B_at_origin in enumerate(self.max_Bs):
            B += sympy.Matrix(B_at_origin) * input_[i // self.motor_coils_per_phase]
        B = B.transpose()
        torque = dipole_moment.cross(B) - 0.01 * theta.diff(t) * sympy.Matrix([1, 1, 1])
        tau = sympy.Symbol("tau", real=True)
        # print(sympy.simplify(torque[2]))
        return sympy.lambdify(
            (tau, phi_2, theta, theta.diff(t)), sympy.solve(torque[2] - tau, phi_1)[0]
        )

    def control(self: "SpeedController", t: float, state: np.array) -> np.array:
        """
        An MPC controller that tries to keep the rotor at a constant speed.
        """
        error = self.target_speed - state[1]
        desired_torque = error * self.rotor_moment_of_inertia * self.gain
        phase_2 = scipy.optimize.minimize(
            lambda phases: np.linalg.norm(
                [
                    self.torque_optimizer(
                        desired_torque, phases[1], state[0], state[1]
                    ),
                    phases[0],
                ]
            ),
            self.last_phases,
            bounds=[(-15, 15)] * 2,
            method="L-BFGS-B",
            options={"ftol": 1e-2},
        ).x[1]
        phase_1 = self.torque_optimizer(desired_torque, phase_2, state[0], state[1])
        self.last_phases = np.array([phase_1, phase_2])
        return np.array([phase_1, phase_2, -phase_1 - phase_2])


class PositionController:
    def __init__(self: "PositionController", motor: Motor, target_position: float):
        self.speed_controller = SpeedController(motor, 0)
        self.target_position = target_position
        self.gain = 1

    def control(self: "PositionController", t: float, state: np.array) -> np.array:
        """
        A simple proportional controller that tries to keep the rotor at a constant
        position.
        """
        error = self.target_position - state[0]
        desired_speed = error * self.gain
        self.speed_controller.target_speed = desired_speed
        # print(self.target_position)
        return self.speed_controller.control(t, state)


motor = Motor(3, 1e-2, 7e-2, 2000, 1, 0.1, 1e-3)
# motor.controller = SpeedController(motor, -10)
motor.controller = PositionController(motor, np.pi - 0.00000000001)

motor.solve(10, 1 / 60)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

sns.set(
    "paper",
    "whitegrid",
    "dark",
    font_scale=1.5,
    rc={"lines.linewidth": 2, "grid.linestyle": "--"},
)


def animate(i):
    data = motor.states[(0 if i < 200 else i - 200) : i]  # select data range
    ax.clear()
    ax.set_title("Phase plane")
    ax.set_aspect("equal")
    ax.set_xlabel("angle")
    ax.set_ylabel("angular velocity")
    ax.set_xlim(-16 * np.pi, 16 * np.pi)
    ax.set_ylim(-30, 30)
    plt.plot(
        [point[0] for point in data],
        [point[1] for point in data],
        "b",
    )


ani = matplotlib.animation.FuncAnimation(
    fig, animate, frames=len(motor.states), repeat=True, interval=1000 / 60
)
writer = matplotlib.animation.FFMpegWriter(fps=60, metadata={"author": "Juniper Mills"})
ani.save("bldc_simple_controlled.mp4", writer=writer)
