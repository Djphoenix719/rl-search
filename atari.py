from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.utils import get_device
from search_policy import CnnSearchPolicy

def main():
    # 0 no output, 1 info, 2 debug
    verbose = 2
    device = get_device('cuda')
    policy_kwargs = dict(
        features_extractor_class=CnnSearchPolicy,
        features_extractor_kwargs=dict(features_dim=128),
    )

    env_wrapper_args = dict(
        noop_max = 30,
        frame_skip = 4,
        screen_size = 84,
        terminal_on_life_loss = True,
        clip_reward = True,
    )
    env = make_atari_env('Pong-v0', n_envs=1, seed=42, wrapper_kwargs=env_wrapper_args)
    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=verbose, device=device)
    model.learn(1)


if __name__ == '__main__':
    main()