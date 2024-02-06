import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
import seaborn as sns
import sympy
import typing
from bldc_simulation import *


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
        self.torque_finder = self.find_torque()
        self.past_error = 0
        self.gain = 5
        self.motor = motor
        self.last_params = np.zeros(4)
        self.psis = []
        self.torques = []

    def find_torque(
        self: "SpeedController",
    ) -> typing.Callable[[float, float, float, float], float]:
        """
        Derives the expression relating phase 2's voltage and phase 1's voltage for a
        given torque, rotor angle, and rotor angular velocity.
        """
        psi_1, psi_2 = sympy.symbols("psi_1 psi_2", real=True)
        alpha_1, alpha_2 = sympy.symbols("alpha_1 alpha_2", real=True)
        t = sympy.Symbol("t", positive=True, real=True)
        phi_1 = sympy.sin(2 * sympy.pi * (t * self.target_speed + psi_1)) * alpha_1
        phi_2 = sympy.sin(2 * sympy.pi * (t * self.target_speed + psi_2)) * alpha_2
        phi_3 = -phi_1 - phi_2
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
        torque = dipole_moment.cross(B) - 0.01 * theta.diff(t) * sympy.Matrix(
            [[1], [1], [1]]
        )
        return sympy.lambdify(
            (t, alpha_1, alpha_2, psi_1, psi_2, theta, theta.diff(t)), torque[2]
        )

    def control(self: "SpeedController", t: float, state: np.array) -> np.array:
        """
        An MPC controller that tries to keep the rotor at a constant speed.
        """
        error = self.target_speed - state[1]
        desired_torque = error * self.rotor_moment_of_inertia * self.gain
        alpha_1 = 7.5
        alpha_2 = 7.5
        psi_1, psi_2 = scipy.optimize.minimize(
            lambda x: (
                self.torque_finder(t, alpha_1, alpha_2, x[0], x[1], state[0], state[1])
                - desired_torque
            )
            ** 2,
            np.zeros(2),
            bounds=[(-np.pi, np.pi)] * 2,
            method="L-BFGS-B",
            options={"ftol": 1e-14},
        ).x
        print(t)
        phase_1 = alpha_1 * np.sin(2 * np.pi * (t * self.target_speed + psi_1))
        phase_2 = alpha_2 * np.sin(2 * np.pi * (t * self.target_speed + psi_2))
        phase_3 = -phase_1 - phase_2
        self.torques.append(
            self.torque_finder(t, alpha_1, alpha_2, psi_1, psi_2, state[0], state[1])
        )
        # print(self.torque_finder(desired_torque, phase_2, state[0], state[1]))
        # phase_2 = scipy.optimize.minimize(
        #     lambda phases: np.linalg.norm(
        #         [
        #             self.torque_optimizer(
        #                 desired_torque, phases[1], state[0], state[1]
        #             ),
        #             phases[0],
        #         ]
        #     ),
        #     self.last_phases,
        #     bounds=[(-15, 15)] * 2,
        #     method="L-BFGS-B",
        #     options={"ftol": 1e-2},
        # ).x[1]
        # phase_1 = self.torque_optimizer(desired_torque, phase_2, state[0], state[1])
        self.last_params = np.array([alpha_1, alpha_2, psi_1, psi_2])
        self.psis.append(np.array([psi_1, psi_2]))
        return np.array([phase_1, phase_2, phase_3])


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
motor.controller = SpeedController(motor, -10)
# motor.controller = PositionController(motor, np.pi - 0.00000000001)

motor.solve(10, 1 / 60)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

sns.set(
    "paper",
    "whitegrid",
    "dark",
    font_scale=1.5,
    rc={"lines.linewidth": 2, "grid.linestyle": "--"},
)

ax.clear()
ax.set_title("Control signals")
ax.set_xlabel("timestep")
ax.set_ylabel("torque (N-m)")
ax.plot(motor.controller.torques)
fig.savefig("bldc_simple_torques.png")


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
