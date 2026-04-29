#!/usr/bin/env python3
"""
Оценка QR-DQN модели на Maze с выбором режима доменной рандомизации.
Режимы:
  clean         – без помех, domain_rand=False
  light_dr      – только domain_rand=True, без визуальных искажений
  sensor_stress – фиксированные искажения severity=0.6, domain_rand=False
  total_chaos   – случайные искажения severity=0.6, domain_rand=True
"""
import argparse
import json
import os
import numpy as np
from tqdm import tqdm
from sb3_contrib import QRDQN
import gymnasium as gym
import miniworld

from envs.env_factory import MiniWorldEnvFactory
from envs.wrappers import ShapedRewardWrapper, PerturbationWrapper, DilatedFrameStack

# Конфигурация наград – должна совпадать с той, что использовалась при обучении
REWARD_KWARGS = {
    'time_penalty': -0.01,
    'distance_reward_coef': 3.0,
    'goal_bonus': 10.0,
    'use_pbrs': True,
    'gamma': 0.99,
    'use_novelty_reward': False,
    'novelty_bonus': 0.0,
    'use_room_reward': True,
    'room_bonus': 0.3,
    'forward_bonus': 0.0,
    'spin_penalty': 0.0,
    'spin_threshold': 7,
    'use_bfs_distance': True,
    'grid_resolution': 0.5,
    'use_stagnation_penalty': True,
    'stagnation_penalty': -0.035,
    'stagnation_threshold': 3,
    'stagnation_precision': 0.5,
}

# Параметры Dilated Frame Stack (как в обучении)
DILATED_STACK_KWARGS = {
    'n_stack': 4,
    'dilation': 2,
}

def make_eval_env(env_name, grid_size, max_episode_steps,
                  perturbation_mode, perturbation_severity,
                  enable_domain_rand, seed):
    env = MiniWorldEnvFactory.create_env(
        env_name=env_name,
        max_episode_steps=max_episode_steps,
        obs_width=64, obs_height=64,
        seed=seed,
        num_rows=grid_size[0], num_cols=grid_size[1]
    )
    if hasattr(env.unwrapped, 'max_episode_steps'):
        env.unwrapped.max_episode_steps = max_episode_steps

    env = ShapedRewardWrapper(env, **REWARD_KWARGS)

    # Всегда используем PerturbationWrapper (даже для mode='none')
    env = PerturbationWrapper(
        env,
        mode=perturbation_mode,
        severity=perturbation_severity,
        enable_domain_rand=enable_domain_rand
    )

    env = DilatedFrameStack(env, **DILATED_STACK_KWARGS)
    return env

def evaluate_model(
    model_path: str,
    env_name: str = 'maze',
    grid_size: tuple = (5, 5),
    mode: str = 'clean',
    num_episodes: int = 100,
    device: str = 'auto',
    output_path: str = None,
    seed: int = 42
):
    """Оценка модели в одном из четырёх режимов."""
    # Определяем параметры для запрошенного режима
    if mode == 'clean':
        pert_mode = 'none'
        pert_severity = 0.0
        domain_rand = False
    elif mode == 'light_dr':
        pert_mode = 'none'
        pert_severity = 0.0
        domain_rand = True
    elif mode == 'sensor_stress':
        pert_mode = 'fixed'
        pert_severity = 0.6
        domain_rand = False
    elif mode == 'total_chaos':
        pert_mode = 'naive'
        pert_severity = 0.6
        domain_rand = True
    else:
        raise ValueError(f"Unknown mode: {mode}. Use clean/light_dr/sensor_stress/total_chaos")

    print(f"\n{'='*60}")
    print(f"Оценка модели: {model_path}")
    print(f"Режим: {mode}")
    print(f"  perturbation_mode = {pert_mode}, severity = {pert_severity}")
    print(f"  domain_rand = {domain_rand}")
    print(f"Лабиринт: {grid_size[0]}x{grid_size[1]}, эпизодов: {num_episodes}")
    print(f"{'='*60}")

    # Загружаем модель
    model = QRDQN.load(model_path, device=device)

    # Создаём среду с нужными параметрами
    env = make_eval_env(
        env_name=env_name,
        grid_size=grid_size,
        max_episode_steps=700,
        perturbation_mode=pert_mode,
        perturbation_severity=pert_severity,
        enable_domain_rand=domain_rand,
        seed=seed
    )

    rewards = []
    lengths = []
    successes = []

    # Для красивого прогресс-бара
    pbar = tqdm(range(num_episodes), desc=f"Оценка {mode}", ncols=80)
    for ep in pbar:
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        ep_len = 0
        while not done and ep_len < 1000:
            # obs уже в формате (C, H, W) благодаря DilatedFrameStack + VecTransposeImage?
            # В одиночной среде DilatedFrameStack выдаёт (H, W, C). Надо привести к (C, H, W)
            # для модели, обученной с VecTransposeImage.
            # Модель ожидает (C, H, W).
            if obs.shape[0] != 3 * DILATED_STACK_KWARGS['n_stack']:
                # Если пришло (H,W,C) – транспонируем
                obs = np.transpose(obs, (2, 0, 1))
            action, _ = model.predict(obs, deterministic=True)
            obs, raw_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # Для успеха используем original_reward (награда за достижение цели)
            ep_reward += info.get('original_reward', 0.0)
            ep_len += 1
        # Успех, если original_reward > 0.1 (как в обучении)
        success = 1.0 if info.get('original_reward', 0.0) > 0.1 else 0.0
        rewards.append(ep_reward)
        lengths.append(ep_len)
        successes.append(success)

        pbar.set_postfix({
            'succ': f"{np.mean(successes)*100:.1f}%",
            'avgR': f"{np.mean(rewards):.1f}"
        })

    env.close()

    # Статистика
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    mean_length = np.mean(lengths)
    success_rate = np.mean(successes) * 100

    results = {
        'mode': mode,
        'num_episodes': num_episodes,
        'mean_reward': float(mean_reward),
        'std_reward': float(std_reward),
        'mean_length': float(mean_length),
        'success_rate': float(success_rate),
        'details': {
            'rewards': rewards,
            'lengths': lengths,
            'successes': successes
        }
    }

    # Сохранение в JSON (опционально без деталей, но с ними для анализа)
    if output_path is None:
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        output_path = f"results/{env_name}_{grid_size[0]}x{grid_size[1]}_{mode}_{model_name}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

    print(f"\n✅ Результаты сохранены: {output_path}")
    print(f"   Success rate: {success_rate:.1f}%")
    print(f"   Средняя награда: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"   Средняя длина эпизода: {mean_length:.1f}")

    return results

def main():
    parser = argparse.ArgumentParser(description="Оценка QR-DQN на Maze с различными уровнями доменной рандомизации")
    parser.add_argument('--model', type=str, required=True, help='Путь к обученной модели (файл .zip)')
    parser.add_argument('--env', type=str, default='maze', choices=['maze'])
    parser.add_argument('--grid-size', type=int, nargs=2, default=[5, 5], metavar=('ROWS', 'COLS'),
                        help='Размер лабиринта (число комнат по вертикали и горизонтали), например 5 5')
    parser.add_argument('--mode', type=str, default='clean',
                        choices=['clean', 'light_dr', 'sensor_stress', 'total_chaos'],
                        help='Режим оценки')
    parser.add_argument('--episodes', type=int, default=100, help='Количество эпизодов')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--output', type=str, default=None, help='Путь для сохранения JSON-результатов')
    parser.add_argument('--seed', type=int, default=42, help='Сид для воспроизводимости')
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model,
        env_name=args.env,
        grid_size=tuple(args.grid_size),
        mode=args.mode,
        num_episodes=args.episodes,
        device=args.device,
        output_path=args.output,
        seed=args.seed
    )

if __name__ == "__main__":
    main()