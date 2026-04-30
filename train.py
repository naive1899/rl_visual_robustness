#!/usr/bin/env python3
"""
Обучение QR-DQN с Frame Stacking (4 кадра) для MiniWorld.
QR-DQN (без LSTM, память через стек кадров).
Оптимизированная версия с Curriculum Learning и индивидуальными порогами успеха.

Теперь используется MultiInputPolicy с наблюдением Dict['image', 'vector'].
Добавлена поддержка RayCastingWrapper (лучи) при конфиге ray_cast.
"""
import argparse
import os
import numpy as np
import gymnasium as gym
import miniworld
from sb3_contrib import QRDQN
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from envs.env_factory import MiniWorldEnvFactory      
from envs.wrappers import ShapedRewardWrapper, PerturbationWrapper, DilatedFrameStack, ActionRepeatWrapper, MultiModalObservationWrapper, RayCastingWrapper


class LevelMetricsCallback(BaseCallback):
    """Отслеживает success rate, награду и длину эпизода с настраиваемым окном."""

    def __init__(self, level_label="unknown", window_size=200, verbose=1):
        super().__init__(verbose)
        self.level_label = level_label
        self.window_size = window_size

        # Success rate
        self.total_episodes = 0
        self.total_successes = 0
        self.recent_successes = []

        # Награды (для recent и total)
        self.recent_rewards = []
        self.total_reward_sum = 0.0

        # Длины эпизодов (для recent и total)
        self.recent_lengths = []
        self.total_length_sum = 0

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])

        for info in infos:
            if 'episode' in info:
                self.total_episodes += 1

                # Success rate
                is_success = 0
                if 'original_reward' in info and info['original_reward'] > 0.1:
                    is_success = 1
                elif info['episode']['r'] > 10:
                    is_success = 1

                self.total_successes += is_success
                self.recent_successes.append(is_success)

                # Награда
                episode_reward = info['episode']['r']
                self.total_reward_sum += episode_reward
                self.recent_rewards.append(episode_reward)

                # Длина эпизода
                episode_length = info['episode']['l']
                self.total_length_sum += episode_length
                self.recent_lengths.append(episode_length)

                # Ограничиваем окно
                if len(self.recent_successes) > self.window_size:
                    self.recent_successes.pop(0)
                    self.recent_rewards.pop(0)
                    self.recent_lengths.pop(0)

                # Логирование каждые 1000 эпизодов
                if self.total_episodes % 1000 == 0:
                    self._log_stats()

        return True

    def _log_stats(self):
        total_success_rate = (self.total_successes / max(1, self.total_episodes)) * 100
        recent_success_rate = np.mean(self.recent_successes) * 100 if self.recent_successes else 0

        recent_reward_mean = np.mean(self.recent_rewards) if self.recent_rewards else 0
        total_reward_mean = self.total_reward_sum / max(1, self.total_episodes)

        recent_length_mean = np.mean(self.recent_lengths) if self.recent_lengths else 0
        total_length_mean = self.total_length_sum / max(1, self.total_episodes)

        # Запись в TensorBoard
        self.logger.record(f"level_{self.level_label}/01_success_rate_total_%", total_success_rate)
        self.logger.record(f"level_{self.level_label}/02_success_rate_recent_%", recent_success_rate)
        self.logger.record(f"level_{self.level_label}/03_total_len_mean", total_length_mean)
        self.logger.record(f"level_{self.level_label}/04_total_reward_mean", total_reward_mean)
        self.logger.record(f"level_{self.level_label}/05_recent_reward_mean", recent_reward_mean)
        self.logger.record(f"level_{self.level_label}/06_recent_len_mean", recent_length_mean)

        if self.verbose > 0:
            print(f"\n📊 Уровень {self.level_label} | "
                  f"Всего: {self.total_episodes} эп | "
                  f"Success Rate: {total_success_rate:.1f}% | "
                  f"Последние {self.window_size}: {recent_success_rate:.1f}%")

    def _on_training_end(self) -> None:
        total_success_rate = (self.total_successes / max(1, self.total_episodes)) * 100
        total_reward_mean = self.total_reward_sum / max(1, self.total_episodes)
        total_length_mean = self.total_length_sum / max(1, self.total_episodes)

        print(f"\n🏆 [{self.level_label}] ИТОГО:")
        print(f"   Success Rate: {total_success_rate:.1f}%")
        print(f"   Avg Reward: {total_reward_mean:.2f}")
        print(f"   Avg Length: {total_length_mean:.1f}")

    def reset_level(self, new_label):
        """Сброс для нового уровня."""
        self.level_label = new_label
        self.total_episodes = 0
        self.total_successes = 0
        self.total_reward_sum = 0.0
        self.total_length_sum = 0
        self.recent_successes = []
        self.recent_rewards = []
        self.recent_lengths = []

    def get_recent_success_rate(self):
        """Возвращает success rate за последние window_size эпизодов."""
        return np.mean(self.recent_successes) * 100 if self.recent_successes else 0.0


# Curriculum для Maze с индивидуальными порогами и максимальными шагами
MAZE_CURRICULUM = [
    { # 0
        'grid_size': (2, 2),
        'max_timesteps': 1_500_000,    # 1.5 млн шагов
        'max_episode_steps': 150,
        'label': '2x2',
        'target_success_rate': 97.0,   # 97% за последние 200 эпизодов
        'window_size': 200,
    },
    { # 1
        'grid_size': (3, 3),
        'max_timesteps': 3_500_000,    
        'max_episode_steps': 250,
        'label': '3x3',
        'target_success_rate': 95.0,
        'window_size': 200,
    },
    { # 2
        'grid_size': (4, 4),
        'max_timesteps': 5_000_000,    
        'max_episode_steps': 550,
        'label': '4x4',
        'target_success_rate': 95.0,
        'window_size': 200,
    },
    {  # 3
        'grid_size': (5, 4),
        'max_timesteps': 5_000_000,    
        'max_episode_steps': 550,
        'label': '5x4',
        'target_success_rate': 95.0,
        'window_size': 200,
    },
    { # 4
        'grid_size': (4, 5),
        'max_timesteps': 5_000_000,    
        'max_episode_steps': 550,
        'label': '4x5',
        'target_success_rate': 95.0,
        'window_size': 200,
    },
    { # 5
        'grid_size': (5, 5),
        'max_timesteps': 7_000_000,    
        'max_episode_steps': 800,
        'label': '5x5',
        'target_success_rate': 93.0,
        'window_size': 200,
    },
]


def make_env(env_name, rank, seed, config,  grid_size=None, max_episode_steps=None, 
             use_dilated_stack=True, dilation=2, n_stack=4, action_repeat=1):
    """Фабрика среды 64×64 с MultiModalObservationWrapper и опциональным RayCasting."""
    def _init():
        max_steps = max_episode_steps or (250 if env_name in ['oneroom', 'hallway'] else 700)

        obs_width = 64
        obs_height = 64

        # Дополнительные параметры для Maze
        extra_kwargs = {}
        if env_name == 'maze' and grid_size is not None:
            rows, cols = grid_size
            extra_kwargs['num_rows'] = rows
            extra_kwargs['num_cols'] = cols

        # Единый вызов фабрики
        env = MiniWorldEnvFactory.create_env(
            env_name=env_name,                     
            max_episode_steps=max_steps,
            obs_width=obs_width,
            obs_height=obs_height,
            seed=seed + rank,                      # передаём seed
            **extra_kwargs
        )

        if hasattr(env.unwrapped, 'max_episode_steps'):
            env.unwrapped.max_episode_steps = max_steps

        # Параметры shaped reward
        reward_kwargs = {
            'time_penalty': -0.01,
            'distance_reward_coef': 3.0,
            'goal_bonus': 10.0,
            'use_pbrs': True,
            'gamma': 0.99,
            'use_novelty_reward': False,
            'novelty_bonus': 0.05,
            'use_room_reward': True,
            'room_bonus': 0.3,   
            'forward_bonus': 0.0,
            'spin_penalty': 0.0,
            'spin_threshold': 7,
            'use_bfs_distance': True,
            'grid_resolution': 0.5,
            'use_stagnation_penalty': True,
            'stagnation_penalty': -0.2,
            'stagnation_threshold':  3,  
            'stagnation_precision': 0.2,
        }
        
        env = ShapedRewardWrapper(env, **reward_kwargs)

        # Action Repeat (если >1)
        if action_repeat > 1:
            env = ActionRepeatWrapper(env, repeat=action_repeat)

        # Визуальные помехи (опционально)
        if config in ["progressive_dr", "ray_cast"]:
            mode = "progressive"
            env = PerturbationWrapper(
                env,
                mode=mode,
                severity=0.6,
                enable_domain_rand=True  
            )

        # Dilated Frame Stack (только изображение)
        if use_dilated_stack:
            env = DilatedFrameStack(env, n_stack=n_stack, dilation=dilation)
            print(f" DilatedFrameStack: {n_stack} кадров (шаг {dilation})")

        # MultiModalObservationWrapper - добавляет вектор и формирует Dict
        env = MultiModalObservationWrapper(env)

        # Ray Casting если конфиг ray_cast
        if config == "ray_cast":
            env = RayCastingWrapper(env, num_rays=8, max_dist=10.0, fov=75.0)
            print(f"  RayCastingWrapper: 8 лучей, FOV=75°, max_dist=10")

        log_dir = f"models/{env_name}_{config}_seed_{seed}/logs"
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, filename=os.path.join(log_dir, f"env_{rank}"))

        return env

    return _init


def create_vec_env(env_name, num_envs, seed, config, grid_size=None, max_episode_steps=None, 
                   use_dilated_stack=True, dilation=2, n_stack=4, action_repeat=1):
    """
    Создание векторизованной среды с MultiInputPolicy.
    Важно: больше не используем VecTransposeImage и VecFrameStack,
    так как наблюдение уже словарь и политика MultiInputPolicy его обрабатывает.
    """
    label = f" grid={grid_size}" if grid_size else ""
    print(f"\n Создание среды: {env_name}{label} ({num_envs} параллельных)")

    env_fns = [
        make_env(env_name, i, seed, config, grid_size=grid_size, max_episode_steps=max_episode_steps,  
                 use_dilated_stack=use_dilated_stack, dilation=dilation, n_stack=n_stack, action_repeat=action_repeat)
        for i in range(num_envs)
    ]
    # Raw Env возвращает Dict с ключами 'image' и 'vector' (и 'rays' если ray_cast)
    env = SubprocVecEnv(env_fns)

    print(f"   Observation space: {env.observation_space}")
    if use_dilated_stack:
        print(f"   🧩 Dilated Frame Stack: {n_stack} кадров (dilation={dilation})")
    print(f"   📊 MultiModal: изображение + вектор [x_norm, z_norm, sin, cos]")
    if config == "ray_cast":
        print(f"   🔦 RayCasting: добавлены лучи (8) в наблюдение")
    
    return env


class CurriculumSuccessCallback(BaseCallback):
    """
    Callback для автоматического перехода на следующий уровень curriculum.

    Переход происходит при выполнении ЛЮБОГО из условий:
    1. Success rate за последние N эпизодов >= target_success_rate
    2. Исчерпан максимальный бюджет шагов (max_timesteps)

    Также сохраняет чекпоинты каждые save_freq шагов.
    """

    def __init__(self, success_callback, target_success_rate,
                 check_freq=1000, save_freq=50_000, save_path=None,
                 name_prefix="qrdqn", verbose=1):
        super().__init__(verbose)
        self.success_callback = success_callback
        self.target_success_rate = target_success_rate
        self.check_freq = check_freq
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.level_completed = False
        self.level_steps = 0

    def _on_step(self) -> bool:
        self.level_steps += 1

        if self.level_steps % self.check_freq == 0:
            recent_sr = self.success_callback.get_recent_success_rate()

            if recent_sr >= self.target_success_rate:
                print(f"\n УРОВЕНЬ ПРОЙДЕН! Success rate (последние {self.success_callback.window_size}): {recent_sr:.1f}% >= {self.target_success_rate}%")
                self.level_completed = True
                return False  # Прерываем model.learn()

            if self.verbose > 0:
                print(f"   [Curriculum] Шаг {self.level_steps} | Recent SR: {recent_sr:.1f}% | Цель: {self.target_success_rate}%")

        if self.save_path is not None and self.level_steps % self.save_freq == 0:
            # Временно отключаем env для безопасного сохранения (избегаем pickling ошибок)
            try:
                old_env = self.model.env
                self.model.env = None
                self.model.save(os.path.join(self.save_path, f"{self.name_prefix}_{self.level_steps}_steps"))
            except Exception as e:
                print(f"⚠️ Ошибка сохранения модели: {e}")
            finally:
                self.model.env = old_env

        return True


def train_curriculum_maze(num_envs, seed, config, save_dir):
    """
    Обучение Maze с Curriculum Learning (QR-DQN версия).
    Использует MultiInputPolicy.
    """
    print(f"\n{'='*70}")
    print("🎓 CURRICULUM LEARNING для Maze (QR-DQN + MultiInputPolicy + FrameStack)")
    print("   Условия перехода: success rate за 200 эпизодов или исчерпание лимита шагов")
    print(f"{'='*70}")

    
    run_name = f"maze_curriculum_{config}_seed_{seed}"
    model_dir = os.path.join(save_dir, run_name)
    os.makedirs(model_dir, exist_ok=True)

    total_timesteps = 0
    model = None
    success_callback = None

    for level_idx, level_config in enumerate(MAZE_CURRICULUM):
        grid_size = level_config['grid_size']
        label = level_config['label']
        max_steps = level_config['max_timesteps']
        max_episode_steps = level_config['max_episode_steps']
        target_sr = level_config['target_success_rate']
        window_size = level_config['window_size']

        print(f"\n📚 Уровень {level_idx + 1}/{len(MAZE_CURRICULUM)}: {label}")
        print(f"   Максимум шагов: {max_steps:,} | Макс. длина эпизода: {max_episode_steps}")
        print(f"   Целевой success rate: {target_sr}% (окно {window_size} эпизодов)")

        # Создаём колбэк метрик для этого уровня
        success_callback = LevelMetricsCallback(level_label=label, window_size=window_size, verbose=1)

        # Среда с нужным размером лабиринта
        env = create_vec_env('maze', num_envs, seed, config, grid_size=grid_size, max_episode_steps=max_episode_steps, 
                             use_dilated_stack=True, dilation=2, n_stack=4, action_repeat=1)
        
        if model is None:
            model = QRDQN(
                "MultiInputPolicy",   # используем политику для Dict наблюдений
                env,
                learning_rate=2e-4,
                buffer_size=200_000,
                learning_starts=20_000,
                batch_size=64,
                train_freq=4,     
                gradient_steps=2, 
                tau=0.005,
                target_update_interval=1,
                gamma=0.99,
                exploration_fraction=0.5,
                exploration_final_eps=0.1,
                policy_kwargs={
                    "n_quantiles": 32,
                    "net_arch": [128, 128],
                    # "features_extractor_class": "CombinedExtractor",  # это значение по умолчанию для MultiInputPolicy
                    # "features_extractor_kwargs": {}
                },
                verbose=1,
                tensorboard_log=os.path.join(model_dir, "tensorboard"),
                device="auto",
                seed=seed
            )
        else:
            model.set_env(env)

        checkpoint_dir = os.path.join(model_dir, f"checkpoints_{label}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        curriculum_callback = CurriculumSuccessCallback(
            success_callback=success_callback,
            target_success_rate=target_sr,
            check_freq=1000,
            save_freq=50_000,
            save_path=checkpoint_dir,
            name_prefix=f"qrdqn_{label}",
            verbose=1
        )

        model.learn(
            total_timesteps=max_steps,
            callback=[curriculum_callback, success_callback],
            reset_num_timesteps=False,
            tb_log_name=f"level_{label}",
            progress_bar=True,
            log_interval=200
        )

        total_timesteps += curriculum_callback.level_steps

        level_path = os.path.join(model_dir, f"level_{label}_final")
        model.save(level_path)
        model.save_replay_buffer(os.path.join(model_dir, f"level_{label}_replay_buffer"))
        print(f"   ✅ Уровень {label} сохранен ({curriculum_callback.level_steps:,} шагов)")

        if not curriculum_callback.level_completed:
            print(f"   ⚠️  Уровень {label} завершен по таймауту (не достигнут {target_sr}% success rate)")

        env.close()

        # Если уровень не пройден по success rate — останавливаем curriculum
        if not curriculum_callback.level_completed and level_idx < len(MAZE_CURRICULUM) - 1:
            print(f"\n⛔ Остановка curriculum: уровень {label} не пройден. Переход отменен.")
            break

    final_path = os.path.join(model_dir, "final_model")
    model.save(final_path)
    print(f"\n✅ Curriculum завершен: {final_path}")

    return model


def train_standard(env_name, steps, seed, config,
                   num_envs, save_dir):
    """Стандартное обучение (QR-DQN с MultiInputPolicy)."""
    
    run_name = f"{env_name}_{config}_seed_{seed}"
    model_dir = os.path.join(save_dir, run_name)
    os.makedirs(model_dir, exist_ok=True)

    print(f"\n🎓 Обучение {env_name} (QR-DQN + MultiInputPolicy + FrameStack, {steps:,} шагов)")

    env = create_vec_env(env_name, num_envs, seed, config,  None,
                         use_dilated_stack=True, dilation=2, n_stack=4, action_repeat=1)
                
    model = QRDQN(
        "MultiInputPolicy",
        env,
        learning_rate=2e-4,
        buffer_size=200_000,
        learning_starts=50_000,
        batch_size=64,
        tau=0.05,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1,
        exploration_fraction=0.5,
        exploration_final_eps=0.1,
        policy_kwargs={
            "n_quantiles": 50,
            "net_arch": [128, 128],
        },
        verbose=1,
        tensorboard_log=os.path.join(model_dir, "tensorboard"),
        device="auto",
        seed=seed
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=os.path.join(model_dir, "checkpoints"),
        name_prefix=f"qrdqn_{env_name}"
    )

    success_callback = LevelMetricsCallback(level_label=env_name, window_size=200, verbose=1)

    model.learn(
        total_timesteps=steps,
        callback=[checkpoint_callback, success_callback],
        progress_bar=True,
        tb_log_name=f"train_{env_name}",
        log_interval=200
    )

    final_path = os.path.join(model_dir, "final_model")
    model.save(final_path)
    print(f"\n✅ Обучение завершено: {final_path}")

    env.close()
    return model


def main():
    parser = argparse.ArgumentParser(
        description='QR-DQN с MultiInputPolicy (изображение + вектор) для MiniWorld с улучшенным Curriculum'
    )
    parser.add_argument('--env', type=str, default='maze',
                        choices=['maze', 'oneroom', 'hallway', 'tworooms'])
    parser.add_argument('--steps', type=int, default=None,
                        help='Количество шагов для стандартного обучения (для maze игнорируется)')
    parser.add_argument('--config', type=str, default='baseline',
                        choices=['baseline', 'progressive_dr', 'ray_cast'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-envs', type=int, default=8,
                        help='Количество параллельных сред')
    parser.add_argument('--save-dir', type=str, default='models')

    args = parser.parse_args()

    if args.env == 'maze':
        train_curriculum_maze(
            args.num_envs, args.seed, args.config, args.save_dir)
    else:
        # Для не-maze сред устанавливаем разумные дефолтные шаги
        if args.steps is None:
            steps = 300 if args.env in ['oneroom', 'hallway'] else 600
        else:
            steps = args.steps
        train_standard(
            args.env, steps, args.seed, args.config,
            args.num_envs, args.save_dir
        )


if __name__ == "__main__":
    main()
