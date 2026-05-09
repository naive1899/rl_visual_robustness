#!/usr/bin/env python3
"""
Оценка PPO на MiniWorld Maze с параллельными средами.

ИСПРАВЛЕНИЯ:
- Убран VecTransposeImage (изображение уже в формате channel-first от MultiModalObservationWrapper)
- Убрано лишнее копирование observation_space в make_eval_env (SubprocVecEnv сам обрабатывает spaces)
- Добавлен Monitor wrapper для корректного логирования эпизодов
- Добавлена обработка случая, когда model_path не существует
"""
import argparse
import json
import os
import re
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

import gymnasium as gym
import miniworld
from envs.env_factory import MiniWorldEnvFactory
from envs.wrappers import (
    ShapedRewardWrapper, PerturbationWrapper, DilatedFrameStack,
    MultiModalObservationWrapper, RayCastingWrapper, WallPenaltyWrapper
)

# ============================================================
# КОНФИГУРАЦИЯ
# ============================================================

REWARD_KWARGS = {
    'time_penalty': -0.01,
    'distance_reward_coef': 0.12,
    'goal_bonus': 10.0,
    'use_pbrs': True,
    'gamma': 1.00,
    'use_novelty_reward': False,
    'state_precision': 0.2,
    'novelty_bonus': 0.0,
    'use_room_reward': False,
    'room_bonus': 0.3,
    'forward_bonus': 0.0,
    'spin_penalty': -0.1,
    'spin_threshold': 6,
    'use_bfs_distance': True,
    'grid_resolution': 0.5,
    'use_stagnation_penalty': True,
    'stagnation_penalty': -0.1,
    'stagnation_threshold': 2,
    'stagnation_precision': 0.4,
    'wall_collision_penalty': -0.2
}

DILATED_STACK_KWARGS = {'n_stack': 4, 'dilation': 2}

LEVEL_STEPS = {
    '4x4': 550, '5x4': 750, '4x5': 750,
    '5x5': 1150, '6x6': 1150,
}

TRAIN_CONFIGS = {
    'baseline': {'perturbation_mode': 'none', 'perturbation_severity': 0.0,
                 'enable_domain_rand': False, 'use_ray_casting': False},
    'progressive_dr': {'perturbation_mode': 'progressive', 'perturbation_severity': 0.6,
                       'enable_domain_rand': False, 'use_ray_casting': False},
    'ray_cast': {'perturbation_mode': 'progressive', 'perturbation_severity': 0.6,
                 'enable_domain_rand': False, 'use_ray_casting': True},
}

EVAL_MODES = {
    'clean': {'perturbation_mode': 'none', 'perturbation_severity': 0.0, 'enable_domain_rand': False},
    'light_dr': {'perturbation_mode': 'none', 'perturbation_severity': 0.0, 'enable_domain_rand': True},
    'sensor_stress': {'perturbation_mode': 'fixed', 'perturbation_severity': 0.6, 'enable_domain_rand': False},
    'total_chaos': {'perturbation_mode': 'naive', 'perturbation_severity': 0.6, 'enable_domain_rand': True},
}


# ============================================================
# УТИЛИТЫ
# ============================================================

def infer_from_path(model_path: str) -> Tuple[Optional[Tuple[int, int]], int]:
    basename = os.path.basename(model_path)
    match = re.search(r'(\d+)[xX](\d+)', basename)
    if match:
        grid = (int(match.group(1)), int(match.group(2)))
        steps = LEVEL_STEPS.get(f"{grid[0]}x{grid[1]}", 1150)
        return grid, steps
    return None, 1150


def make_eval_env(rank, env_name, grid_size, max_episode_steps, eval_mode, train_config, base_seed):
    def _init():
        env = MiniWorldEnvFactory.create_env(
            env_name=env_name, max_episode_steps=max_episode_steps,
            obs_width=64, obs_height=64, seed=base_seed + rank,
            num_rows=grid_size[0], num_cols=grid_size[1],
        )
        if hasattr(env.unwrapped, 'max_episode_steps'):
            env.unwrapped.max_episode_steps = max_episode_steps

        env = WallPenaltyWrapper(env)
        env = ShapedRewardWrapper(env, **REWARD_KWARGS)

        mp = EVAL_MODES[eval_mode]
        env = PerturbationWrapper(env, mode=mp['perturbation_mode'],
                                  severity=mp['perturbation_severity'],
                                  enable_domain_rand=mp['enable_domain_rand'])

        env = DilatedFrameStack(env, **DILATED_STACK_KWARGS)
        env = MultiModalObservationWrapper(env)

        tp = TRAIN_CONFIGS.get(train_config, TRAIN_CONFIGS['baseline'])
        if tp['use_ray_casting']:
            env = RayCastingWrapper(env, num_rays=8, max_dist=10.0, fov=75.0)

        # Monitor нужен для логирования episode reward/length в info
        log_dir = "/tmp/eval_logs"
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, filename=os.path.join(log_dir, f"env_{rank}"))

        return env
    return _init


# ============================================================
# ПАРАЛЛЕЛЬНАЯ ОЦЕНКА
# ============================================================

def evaluate_parallel(model_path, train_config, eval_mode='clean',
                      grid_size=None, max_episode_steps=None,
                      num_episodes=100, num_envs=8, device='auto',
                      base_seed=0, deterministic=False, verbose=True):
    """Оценка на SubprocVecEnv БЕЗ VecTransposeImage.

    MultiModalObservationWrapper уже переводит изображение в channel-first (C,H,W),
    поэтому VecTransposeImage не нужен и будет ломать наблюдения.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    inferred_grid, inferred_steps = infer_from_path(model_path)
    if grid_size is None:
        grid_size = inferred_grid or (5, 5)
    if max_episode_steps is None:
        max_episode_steps = inferred_steps

    eps_per_env = num_episodes // num_envs
    total_eps = eps_per_env * num_envs

    if verbose:
        print(f"\n{'='*60}")
        print(f"📦 {os.path.basename(model_path)} | {train_config} | {eval_mode}")
        print(f"🗺️  {grid_size[0]}x{grid_size[1]} | steps={max_episode_steps}")
        print(f"🎲 Эпизодов: {total_eps} ({eps_per_env} на {num_envs} env)")
        print(f"🎯 Deterministic: {deterministic} | Seed: {base_seed}")
        print(f"{'='*60}")

    # Создаём SubprocVecEnv
    env_fns = [
        make_eval_env(i, 'maze', grid_size, max_episode_steps,
                      eval_mode, train_config, base_seed)
        for i in range(num_envs)
    ]
    env = SubprocVecEnv(env_fns)

    # ❌ УБРАНО: VecTransposeImage — изображение уже (C,H,W) от MultiModalObservationWrapper
    # env = VecTransposeImage(env)

    # Загружаем модель — observation_space совпадёт
    model = PPO.load(model_path, env=env, device=device)

    # Собираем эпизоды
    rewards = []
    lengths = []
    successes = []

    obs = env.reset()
    eps_done = 0

    pbar = tqdm(total=total_eps, desc=eval_mode, ncols=70, disable=not verbose)

    while eps_done < total_eps:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, dones, infos = env.step(action)

        for info, done in zip(infos, dones):
            if done and 'episode' in info:
                eps_done += 1
                pbar.update(1)

                ep_reward = info['episode']['r']
                ep_len = info['episode']['l']
                is_success = 1 if info.get('original_reward', 0) > 0.1 else 0

                rewards.append(ep_reward)
                lengths.append(ep_len)
                successes.append(is_success)

                pbar.set_postfix({
                    'succ': f"{np.mean(successes)*100:.1f}%",
                    'avgR': f"{np.mean(rewards):.1f}",
                    'len': f"{np.mean(lengths):.0f}",
                })

    pbar.close()
    env.close()

    results = {
        'model': os.path.basename(model_path),
        'config': train_config,
        'mode': eval_mode,
        'grid_size': grid_size,
        'num_envs': num_envs,
        'episodes': total_eps,
        'success_rate': float(np.mean(successes) * 100),
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'mean_length': float(np.mean(lengths)),
        'deterministic': deterministic,
    }

    if verbose:
        print(f"\n✅ {eval_mode}: SR={results['success_rate']:.1f}%, "
              f"R={results['mean_reward']:.2f}±{results['std_reward']:.2f}, "
              f"L={results['mean_length']:.0f}")

    return results


def evaluate_all_modes(model_path, train_config, num_episodes=100,
                       num_envs=8, device='auto', base_seed=0,
                       deterministic=False, output_dir=None):
    """Оценка во всех режимах."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    grid_size, max_steps = infer_from_path(model_path)
    grid_size = grid_size or (5, 5)

    if output_dir is None:
        name = os.path.splitext(os.path.basename(model_path))[0]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/{name}_parallel_{num_envs}env_{ts}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"📦 ПАРАЛЛЕЛЬНАЯ ОЦЕНКА: {os.path.basename(model_path)}")
    print(f"🗺️  {grid_size[0]}x{grid_size[1]} | envs={num_envs} | eps={num_episodes}")
    print(f"{'='*70}")

    all_results = {}
    for mode in EVAL_MODES:
        try:
            res = evaluate_parallel(
                model_path, train_config, mode,
                grid_size, max_steps, num_episodes, num_envs,
                device, base_seed, deterministic, verbose=True
            )
            all_results[mode] = res
        except Exception as e:
            print(f"\n❌ Ошибка в режиме {mode}: {e}")
            import traceback
            traceback.print_exc()
            all_results[mode] = {'error': str(e)}

    print(f"\n{'='*70}")
    print("📊 СВОДНАЯ ТАБЛИЦА")
    print(f"{'='*70}")
    print(f"{'Режим':<15} {'SR %':>8} {'Reward':>10} {'Length':>8}")
    print("-" * 45)
    for mode, res in all_results.items():
        if 'error' in res:
            print(f"{mode:<15} {'ERROR':>8}")
        else:
            print(f"{mode:<15} {res['success_rate']:>7.1f}% "
                  f"{res['mean_reward']:>9.2f} {res['mean_length']:>8.0f}")

    with open(os.path.join(output_dir, "all_modes.json"), 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False,
                  default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"\n💾 Сохранено: {output_dir}")
    return all_results


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Параллельная оценка PPO на MiniWorld Maze")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--config', type=str, default='baseline',
                        choices=list(TRAIN_CONFIGS.keys()))
    parser.add_argument('--mode', type=str, default='clean',
                        choices=list(EVAL_MODES.keys()))
    parser.add_argument('--all-modes', action='store_true')
    parser.add_argument('--grid-size', type=int, nargs=2, default=None)
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--num-envs', type=int, default=8)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--output-dir', type=str, default=None)

    args = parser.parse_args()
    grid_size = tuple(args.grid_size) if args.grid_size else None

    if args.all_modes:
        evaluate_all_modes(
            args.model, args.config, args.episodes,
            args.num_envs, args.device, args.seed,
            args.deterministic, args.output_dir
        )
    else:
        evaluate_parallel(
            args.model, args.config, args.mode,
            grid_size, args.max_steps, args.episodes,
            args.num_envs, args.device, args.seed,
            args.deterministic, verbose=True
        )


if __name__ == "__main__":
    main()
