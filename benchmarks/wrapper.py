import gym
from stable_baselines3.common.atari_wrappers import ClipRewardEnv
from stable_baselines3.common.atari_wrappers import EpisodicLifeEnv
from stable_baselines3.common.atari_wrappers import FireResetEnv
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.atari_wrappers import WarpFrame


class AtariWrapper(gym.Wrapper):
    """
    Atari 2600 preprocessings

    Specifically:

    * NoopReset: obtain initial state by taking random number of no-ops on reset.
    * Frame skipping: 4 by default
    * Max-pooling: most recent two observations
    * Termination signal when a life is lost.
    * Resize to a square image: 84x84 by default
    * Grayscale observation
    * Clip reward to {-1, 0, 1}

    :param env: gym environment
    :param noop_max: max number of no-ops
    :param frame_skip: the frequency at which the agent experiences the game.
    :param screen_size: resize Atari frame
    :param terminal_on_life_loss: if True, then step() returns done=True whenever a life is lost.
    :param clip_reward: If True (default), the reward is clip to {-1, 0, 1} depending on its sign.
    """

    def __init__(
        self,
        env: gym.Env,
        frame_skip: int = 4,
        screen_size: int = 84,
        terminal_on_life_loss: bool = True,
        clip_reward: bool = True,
    ):
        env = MaxAndSkipEnv(env, skip=frame_skip)
        if terminal_on_life_loss:
            env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = WarpFrame(env, width=screen_size, height=screen_size)
        if clip_reward:
            env = ClipRewardEnv(env)

        super(AtariWrapper, self).__init__(env)
