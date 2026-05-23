import gymnasium as gym
from gymnasium import spaces
import numpy as np

try:
    from .simulation.simulation import SwarmBallSimulation
except ImportError:
    from simulation.simulation import SwarmBallSimulation


class SwarmBall(gym.Env):
    def __init__(self, acc_factor=0.25, number_of_clusters=3, v_max=10, **kwargs):
        super().__init__()  # [ZMIANA] Wymagane przez nowe API Gymnasium

        self.sim = SwarmBallSimulation(number_of_clusters, **kwargs)
        self.cluster_count = number_of_clusters
        self.thresh_vel = np.zeros(number_of_clusters, dtype=np.float32)
        self.v_max = v_max
        self.acc_factor = acc_factor

        # [ZMIANA] W Gymnasium musimy jawnie zdefiniować przestrzenie (spaces)
        # Akcje to ciągłe wartości dla każdego klastra
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(number_of_clusters,), dtype=np.float32)

        # Obserwacje to słownik (obraz + pozycje klastrów)
        screen_size = kwargs.get('screen_size', (1800, 840))
        self.observation_space = spaces.Dict({
            'thresholds': spaces.Box(low=-np.inf, high=np.inf, shape=(number_of_clusters,), dtype=np.float32)
        })

    def reward(self):
        points = self.sim._goal_object.body.position[0] - self.goal_prev_pos
        self.goal_prev_pos = self.sim._goal_object.body.position[0]
        return float(points)  # [ZMIANA] Upewniamy się, że nagroda to float

    def step(self, action):
        self.thresh_vel = self.thresh_vel + (2 * action - 1) * self.acc_factor
        self.thresh_vel = np.clip(self.thresh_vel, -self.v_max, self.v_max)
        for i in range(self.cluster_count):
            self.sim.update_thresholds_position(
                i, self.sim.threshold_positions()[i] + self.thresh_vel[i])
        self.sim.step()

        observations = {
            'thresholds': np.array(self.sim.threshold_positions(), dtype=np.float32) -
                          self.sim._goal_object.body.position[0]
        }

        reward = self.reward()

        # [ZMIANA] Rozbicie 'done' na 'terminated' (koniec gry) i 'truncated' (limit czasu)
        terminated = bool(self.sim._enemy_position >= self.sim._goal_object.body.position[0])
        truncated = False
        info = {'message': 'You look great today cutiepie!'}

        return observations, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # [ZMIANA] Inicjalizacja seed z Gymnasium
        super().reset(seed=seed)

        self.thresh_vel = np.zeros(self.cluster_count, dtype=np.float32)
        self.sim.reset()
        self.goal_prev_pos = self.sim._goal_object.body.position[0]
        self.initial_goal_position = self.sim._goal_object.body.position[0]

        observations = {
            'thresholds': np.array(self.sim.threshold_positions(), dtype=np.float32) -
                          self.sim._goal_object.body.position[0]
        }
        info = {}

        # [ZMIANA] Reset w Gymnasium zwraca zawsze (obs, info)
        return observations, info

    def render(self):
        self.sim.redraw()

    def close(self):
        """
            Zamknięcie środowiska.
        """
        pass