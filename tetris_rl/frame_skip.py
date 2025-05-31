import gym
from typing import Tuple

class FrameSkip(gym.Wrapper):
    """
    Execute the given action once and then repeat the *last* action
    for (k-1) additional frames, returning the observation/info from
    the final step.  Env rewards are ignored because the project uses
    shaped_reward() externally.
    """
    def __init__(self, env: gym.Env, k: int = 4):
        super().__init__(env)
        assert k >= 1
        self.k = k

    def step(self, action) -> Tuple:
        obs, reward, done, info = self.env.step(action)
        for _ in range(self.k - 1):
            if done:
                break
            obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info
