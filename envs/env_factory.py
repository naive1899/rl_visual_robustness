import gymnasium as gym
from typing import Optional
from miniworld.params import DEFAULT_PARAMS # Импортируем дефолтные параметры из либы

class MiniWorldEnvFactory:
    AVAILABLE_ENVS = {
        'maze': 'MiniWorld-Maze-v0',
        'oneroom': 'MiniWorld-OneRoom-v0',
        'hallway': 'MiniWorld-Hallway-v0',
        'tworooms': 'MiniWorld-TwoRooms-v0',
    }

    @classmethod
    def create_env(
        cls,
        env_name: str = 'maze',
        #domain_rand=False по дефолту 
        max_episode_steps: int = 500,
        seed: Optional[int] = 0,
        obs_width: int = 64,
        obs_height: int = 64,
        forward_step: Optional[float] = 0.225,   # длина шага агента (default 0.15)
        turn_step: Optional[float] = 22.5,       # угол поворота агента в градусах (default 15)
        fov: int = 75,                           # угол обзора камеры дефолт 60
        **kwargs
    ):
        env_id = cls.AVAILABLE_ENVS.get(env_name)
        if env_id is None:
            raise ValueError(f"Unknown env: {env_name}")

        # Работаем с объектом параметров MiniWorld
        # Делаем глубокую копию, чтобы не испортить глобальные настройки
        custom_params = DEFAULT_PARAMS.copy()  # Синглтон

        if forward_step is not None:
            custom_params.set("forward_step", default=forward_step)

        if turn_step is not None:
            custom_params.set("turn_step", default=turn_step)
       
        if fov is not None:
            custom_params.set("cam_fov_y", default=fov)

        # Чтобы сохранить точность позиционирования камеры при поворотах,
        # нужно проверить/установить параметр высоты и наклона:
        custom_params.set("cam_pitch", default=0)

        print(f"[Factory] Creating: {env_id} with forward_step={forward_step}, turn_step={turn_step}, fov={fov}")

        
        env = gym.make(
            env_id,
            max_episode_steps=max_episode_steps,
            obs_width=obs_width,
            obs_height=obs_height, 
            params=custom_params,
            **kwargs
        )

        if seed is not None:
            env.reset(seed=seed)

        return env
