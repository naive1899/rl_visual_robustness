import numpy as np
import gymnasium as gym
from gymnasium import ObservationWrapper

class GrayscaleWrapper(ObservationWrapper):
    """Преобразует RGB-изображение в grayscale (один канал)."""
    def __init__(self, env):
        super().__init__(env)
        # Ожидаем, что наблюдение – dict с ключом 'image' либо просто массив
        obs = self.observation_space
        if isinstance(obs, gym.spaces.Dict):
            img_space = obs['image']
            # У нас в MultiModalObservationWrapper изображение становится channel‑first (C,H,W).
            # Но на этом этапе (до MultiModal) оно ещё (H,W,C). 
            # Мы вставим обёртку до MultiModalObservationWrapper.
            new_shape = list(img_space.shape)
            if len(new_shape) == 3:
                # (H, W, 3) -> (H, W, 1)
                new_shape[-1] = 1
            elif len(new_shape) == 4:
                # (C, H, W) -> (1, H, W)
                new_shape[0] = 1
            obs['image'] = gym.spaces.Box(0, 255, shape=tuple(new_shape), dtype=np.uint8)
        else:
            # Если наблюдение – просто массив
            shape = list(obs.shape)
            if len(shape) == 3:
                shape[-1] = 1
            self.observation_space = gym.spaces.Box(0, 255, shape=tuple(shape), dtype=np.uint8)

    def observation(self, obs):
        if isinstance(obs, dict):
            img = obs['image']
            # img сейчас (H, W, 3)
            gray = np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
            obs['image'] = np.expand_dims(gray, axis=-1)   # (H, W, 1)
        else:
            gray = np.dot(obs[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
            obs = np.expand_dims(gray, axis=-1) if obs.ndim == 3 else gray
        return obs
