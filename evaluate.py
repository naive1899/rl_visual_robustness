#!/usr/bin/env python3
"""
Оценка QR-DQN модели на Maze с выбором режима доменной рандомизации.
Поддерживает все конфиги обучения: baseline, progressive_dr, ray_cast.
Опциональный анализ паттернов провалов с heatmap.

Режимы оценки:
  clean         – без помех, domain_rand=False
  light_dr      – только domain_rand=True, без визуальных искажений
  sensor_stress – фиксированные искажения severity=0.6, domain_rand=False
  total_chaos   – случайные искажения severity=0.6, domain_rand=True

Использование:
  # Одиночная оценка 
  python evaluate.py --model models/maze_curriculum_baseline_seed_0/final_model --mode clean --episodes 100

  # С анализом провалов
  python evaluate.py --model ... --config baseline --mode clean --episodes 100 --analyze-failures

  # Пакетная оценка
  python evaluate.py --batch \
      --models-dir models \
      --configs baseline,progressive_dr,ray_cast \
      --modes clean,light_dr,sensor_stress,total_chaos \
      --grid-sizes "5 4,5 5" \
      --episodes 100 --analyze-failures
"""
import argparse
import json
import os
import sys
import re
import math
from datetime import datetime
from collections import Counter, defaultdict
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
from tqdm import tqdm
from sb3_contrib import QRDQN
import gymnasium as gym

# MiniWorld импорты
import miniworld
from envs.env_factory import MiniWorldEnvFactory
from envs.wrappers import (
    ShapedRewardWrapper,
    PerturbationWrapper,
    DilatedFrameStack,
    MultiModalObservationWrapper,
    RayCastingWrapper,
)

# ============================================================
# КОНФИГУРАЦИЯ — должна совпадать с обучением
# ============================================================

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

DILATED_STACK_KWARGS = {
    'n_stack': 4,
    'dilation': 2,
}

# Параметры обучения по конфигам (какие wrappers были при обучении)
TRAIN_CONFIGS = {
    'baseline': {
        'perturbation_mode': 'none',
        'perturbation_severity': 0.0,
        'enable_domain_rand': False,
        'use_ray_casting': False,
    },
    'progressive_dr': {
        'perturbation_mode': 'progressive',
        'perturbation_severity': 0.6,
        'enable_domain_rand': True,
        'use_ray_casting': False,
    },
    'ray_cast': {
        'perturbation_mode': 'progressive',
        'perturbation_severity': 0.6,
        'enable_domain_rand': True,
        'use_ray_casting': True,
    },
}

# Параметры оценки по режимам
EVAL_MODES = {
    'clean': {
        'perturbation_mode': 'none',
        'perturbation_severity': 0.0,
        'enable_domain_rand': False,
    },
    'light_dr': {
        'perturbation_mode': 'none',
        'perturbation_severity': 0.0,
        'enable_domain_rand': True,
    },
    'sensor_stress': {
        'perturbation_mode': 'fixed',
        'perturbation_severity': 0.6,
        'enable_domain_rand': False,
    },
    'total_chaos': {
        'perturbation_mode': 'naive',
        'perturbation_severity': 0.6,
        'enable_domain_rand': True,
    },
}

# ============================================================
# КЛАСС: ДЕТЕКТОР ПАТТЕРНОВ ПРОВАЛОВ
# ============================================================

class FailurePatternDetector:
    """
    Анализирует траекторию неудачного эпизода и классифицирует паттерн провала.
    """

    PATTERNS: Dict[str, Dict[str, str]] = {
        'oscillation': {
            'desc': 'Дрожание (0,2,0,2...) — агент застрял, Q-значения равны',
            'fix': 'Увеличь stagnation_threshold, убери novelty_reward',
        },
        'distance_increase': {
            'desc': 'Удаление от цели — PBRS инвертирован или gamma неверный',
            'fix': 'Проверь gamma в PBRS = model.gamma, снизь distance_reward_coef',
        },
        'room_loop': {
            'desc': 'Круги по комнате — room_bonus мешает',
            'fix': 'Убери use_room_reward или снизь room_bonus',
        },
        'wall_hug': {
            'desc': 'Висит у стены — forward_bonus или высокий exploration',
            'fix': 'Убери forward_bonus, снизь exploration_final_eps',
        },
        'timeout': {
            'desc': 'Дистанция растет к концу — target network не успевает',
            'fix': 'Уменьши target_update_interval, увеличь max_episode_steps',
        },
        'unknown': {
            'desc': 'Неопределенный паттерн',
            'fix': 'Требуется ручной анализ траектории',
        },
    }

    @classmethod
    def detect(cls, trajectory: List[Dict[str, Any]]) -> str:
        """
        Классифицирует паттерн провала по траектории.

        Args:
            trajectory: список dict с ключами 'pos', 'action', 'bfs_dist', 'reward', 'step'

        Returns:
            str: ключ паттерна из PATTERNS
        """
        if not trajectory:
            return 'unknown'

        actions = [t['action'] for t in trajectory if t.get('action') is not None]
        dists = [t['bfs_dist'] for t in trajectory if t.get('bfs_dist') is not None]
        positions = [t['pos'] for t in trajectory if t.get('pos') is not None]
        steps = len(trajectory)

        # Oscillation: частая смена действий в конце
        if len(actions) >= 30:
            last_30 = actions[-30:]
            changes = sum(1 for i in range(1, len(last_30)) if last_30[i] != last_30[i - 1])
            if changes / len(last_30) > 0.65:
                return 'oscillation'

        # Distance increase: BFS растет в последние 20 шагов 
        if len(dists) >= 20:
            last_20 = dists[-20:]
            x = np.arange(len(last_20))
            slope = np.polyfit(x, last_20, 1)[0]
            if slope > 0.3:
                return 'distance_increase'

        # Room loop: много посещений одних координат
        if len(positions) >= 50:
            # Дискретизируем позиции с точностью 0.5
            cells = [(round(p[0] / 0.5), round(p[2] / 0.5)) for p in positions]
            cell_counts = Counter(cells)
            most_common = cell_counts.most_common(1)[0][1]
            if most_common > len(cells) * 0.25:
                return 'room_loop'

        # Wall hug: мало движения, много шагов
        if len(positions) >= 20:
            # Проверяем движение в последние 20 шагов
            last_pos = positions[-20:]
            movements = [
                math.sqrt((last_pos[i][0] - last_pos[i-1][0])**2 +
                         (last_pos[i][2] - last_pos[i-1][2])**2)
                for i in range(1, len(last_pos))
            ]
            avg_movement = np.mean(movements) if movements else 0
            if avg_movement < 0.05:
                return 'wall_hug'

        # Timeout: длина близка к максимуму и дистанция большая
        if steps >= 500 and len(dists) > 0:
            if dists[-1] > dists[0] * 0.8:
                return 'timeout'

        return 'unknown'

    @classmethod
    def get_report(cls, failures: List[Dict[str, Any]]) -> str:
        """Формирует текстовый отчет по паттернам провалов."""
        if not failures:
            return "Провалов не обнаружено."

        patterns = [f.get('pattern', 'unknown') for f in failures]
        counts = Counter(patterns)
        total = len(failures)

        lines = [
            "=" * 60,
            "📊 АНАЛИЗ ПАТТЕРНОВ ПРОВАЛОВ",
            "=" * 60,
            f"Всего провалов: {total}",
            "",
            "Распределение паттернов:",
        ]

        for pattern_key, count in counts.most_common():
            pct = count / total * 100
            info = cls.PATTERNS.get(pattern_key, cls.PATTERNS['unknown'])
            lines.append(f"  {pattern_key:20s}: {count:3d} ({pct:5.1f}%) — {info['desc']}")
            lines.append(f"      💡 Рекомендация: {info['fix']}")
            lines.append("")

        # Топ-3 самых частых рекомендации
        lines.append("-" * 60)
        lines.append("🔧 ПРИОРИТЕТНЫЕ ДЕЙСТВИЯ:")
        for pattern_key, count in counts.most_common(3):
            info = cls.PATTERNS.get(pattern_key, cls.PATTERNS['unknown'])
            lines.append(f"  [{count} раз] {info['fix']}")

        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================================
# КЛАСС: АНАЛИЗАТОР ПРОВАЛОВ (сбор + сохранение)
# ============================================================

class FailureAnalyzer:
    """
    Собирает, анализирует и сохраняет данные о неудачных эпизодах.
    """

    def __init__(self, max_stored: int = 50):
        self.max_stored = max_stored
        self.failures: List[Dict[str, Any]] = []
        self.all_positions: List[Tuple[float, float]] = []      # (x, z) все посещения
        self.death_positions: List[Tuple[float, float]] = []    # (x, z) места провала

    def add_failure(self, episode_num: int, trajectory: List[Dict[str, Any]],
                    info: Dict[str, Any], max_episode_steps: int):
        """
        Добавляет провальный эпизод для анализа.

        Сохраняет:
          - разреженную траекторию (каждый 5-й шаг)
          - подробные последние 50 шагов
          - метаданные эпизода
        """
        if len(self.failures) >= self.max_stored:
            return

        # Извлекаем позиции для heatmap
        for t in trajectory:
            if t.get('pos') is not None:
                self.all_positions.append((t['pos'][0], t['pos'][2]))
        if trajectory and trajectory[-1].get('pos') is not None:
            self.death_positions.append((trajectory[-1]['pos'][0], trajectory[-1]['pos'][2]))

        # Сокращаем траекторию
        sparse = []
        detailed = []

        for i, t in enumerate(trajectory):
            point = {
                'step': t.get('step', i),
                'pos': t.get('pos'),
                'action': t.get('action'),
                'bfs_dist': t.get('bfs_dist'),
                'reward': t.get('reward'),
            }
            # Каждый 5-й шаг в sparse
            if i % 5 == 0:
                sparse.append(point)
            # Последние 50 шагов в detailed
            if i >= len(trajectory) - 50:
                detailed.append(point)

        # Детектируем паттерн
        pattern = FailurePatternDetector.detect(trajectory)

        failure_data = {
            'episode': episode_num,
            'pattern': pattern,
            'length': len(trajectory),
            'final_pos': trajectory[-1].get('pos') if trajectory else None,
            'goal_pos': info.get('goal_pos'),
            'final_bfs_dist': trajectory[-1].get('bfs_dist') if trajectory else None,
            'trajectory_sparse': sparse,
            'trajectory_detailed': detailed,
        }
        self.failures.append(failure_data)

    def get_pattern_statistics(self) -> Dict[str, int]:
        """Возвращает счетчик паттернов."""
        patterns = [f.get('pattern', 'unknown') for f in self.failures]
        return dict(Counter(patterns))

    def save_json(self, path: str):
        """Сохраняет провалы в JSON."""
        data = {
            'metadata': {
                'failures_count': len(self.failures),
                'pattern_statistics': self.get_pattern_statistics(),
            },
            'failures': self.failures,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False,
                      default=lambda x: float(x) if isinstance(x, np.floating) else x)

    def generate_heatmap(self, path: str, grid_size: Tuple[int, int]):
        """
        Генерирует heatmap посещений (зеленый) и смертей (красный).
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("⚠️ matplotlib не установлен, heatmap пропущен")
            return

        if not self.all_positions:
            print("⚠️ Нет данных для heatmap")
            return

        fig, ax = plt.subplots(figsize=(10, 10))

        # Определяем границы мира по grid_size (примерно room_size=4)
        room_size = 4.0
        world_w = grid_size[1] * room_size
        world_h = grid_size[0] * room_size

        bins = (grid_size[1] * 10, grid_size[0] * 10)  # 10 bins на комнату

        # Все посещения — зеленый (low alpha)
        if self.all_positions:
            xs, zs = zip(*self.all_positions)
            ax.hist2d(xs, zs, bins=bins, cmap='Greens', alpha=0.6,
                     range=[[0, world_w], [0, world_h]])

        # Смерти — красный точки
        if self.death_positions:
            dxs, dzs = zip(*self.death_positions)
            ax.scatter(dxs, dzs, c='red', s=100, marker='x', alpha=0.8,
                      linewidths=2, label='Failures')

        ax.set_xlim(0, world_w)
        ax.set_ylim(0, world_h)
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_title(f'Agent Trajectory Heatmap\nGreen=visits, Red X=failures')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        plt.tight_layout()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   Heatmap сохранен: {path}")

    def print_report(self):
        """Печатает отчет в консоль."""
        report = FailurePatternDetector.get_report(self.failures)
        print("\n" + report)


# ============================================================
# УТИЛИТЫ
# ============================================================

def infer_grid_size_from_path(model_path: str) -> Optional[Tuple[int, int]]:
    """
    Пытается извлечь размер сетки из имени файла модели.
    Например: level_5x4_final → (5, 4)
    """
    basename = os.path.basename(model_path)
    # Ищем паттерн NxM или N_x_M
    match = re.search(r'(\d+)[xX](\d+)', basename)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None


def make_eval_env(
    env_name: str,
    grid_size: Tuple[int, int],
    max_episode_steps: int,
    eval_mode: str,
    train_config: str,
    seed: int,
) -> gym.Env:
    """
    Создает среду оценки с правильной цепочкой wrappers.

    Цепочка:
      MiniWorldEnv → ShapedRewardWrapper → PerturbationWrapper
      → DilatedFrameStack → MultiModalObservationWrapper
      → [RayCastingWrapper если ray_cast]
    """
    # Базовая среда
    env = MiniWorldEnvFactory.create_env(
        env_name=env_name,
        max_episode_steps=max_episode_steps,
        obs_width=64,
        obs_height=64,
        seed=seed,
        num_rows=grid_size[0],
        num_cols=grid_size[1],
    )
    if hasattr(env.unwrapped, 'max_episode_steps'):
        env.unwrapped.max_episode_steps = max_episode_steps

    # Shaped rewards (как при обучении)
    env = ShapedRewardWrapper(env, **REWARD_KWARGS)

    # PerturbationWrapper — параметры из режима оценки
    mode_params = EVAL_MODES[eval_mode]
    env = PerturbationWrapper(
        env,
        mode=mode_params['perturbation_mode'],
        severity=mode_params['perturbation_severity'],
        enable_domain_rand=mode_params['enable_domain_rand'],
    )

    # Dilated Frame Stack
    env = DilatedFrameStack(env, **DILATED_STACK_KWARGS)

    # MultiModalObservationWrapper (Dict: image + vector)
    env = MultiModalObservationWrapper(env)

    # RayCastingWrapper только для ray_cast конфига
    train_params = TRAIN_CONFIGS.get(train_config, TRAIN_CONFIGS['baseline'])
    if train_params['use_ray_casting']:
        env = RayCastingWrapper(env, num_rays=8, max_dist=10.0, fov=75.0)

    return env


# ============================================================
# ОСНОВНАЯ ФУНКЦИЯ ОЦЕНКИ
# ============================================================

def evaluate_model(
    model_path: str,
    train_config: str,
    env_name: str = 'maze',
    grid_size: Optional[Tuple[int, int]] = None,
    eval_mode: str = 'clean',
    num_episodes: int = 100,
    device: str = 'auto',
    output_dir: Optional[str] = None,
    seed: int = 42,
    analyze_failures: bool = False,
    max_failures_stored: int = 50,
    no_heatmap: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Оценивает одну модель в одном режиме.

    Returns:
        dict с метриками и путями к сохраненным файлам
    """
    # Определяем grid_size 
    if grid_size is None:
        inferred = infer_grid_size_from_path(model_path)
        if inferred:
            grid_size = inferred
            if verbose:
                print(f"   Grid size автоопределен из пути: {grid_size[0]}x{grid_size[1]}")
        else:
            grid_size = (5, 5)
            if verbose:
                print(f"   Grid size по умолчанию: 5x5")
    else:
        if verbose:
            print(f"   Grid size указан: {grid_size[0]}x{grid_size[1]}")

    # Загружаем модель
    if verbose:
        print(f"\n{'='*60}")
        print(f"📦 Модель: {os.path.basename(model_path)}")
        print(f"⚙️  Конфиг обучения: {train_config}")
        print(f"🎯 Режим оценки: {eval_mode}")
        print(f"🗺️  Лабиринт: {grid_size[0]}x{grid_size[1]}")
        print(f"🎲 Эпизодов: {num_episodes}")
        print(f"{'='*60}")

    model = QRDQN.load(model_path, device=device)

    # Создаем среду 
    max_episode_steps = 700  # фиксированный лимит для оценки
    env = make_eval_env(
        env_name=env_name,
        grid_size=grid_size,
        max_episode_steps=max_episode_steps,
        eval_mode=eval_mode,
        train_config=train_config,
        seed=seed,
    )

    # Подготовка к анализу провалов
    analyzer = FailureAnalyzer(max_stored=max_failures_stored) if analyze_failures else None

    # Основной цикл оценки
    rewards = []
    lengths = []
    successes = []

    pbar = tqdm(range(num_episodes), desc=f"Оценка {eval_mode}", ncols=80,
                disable=not verbose)

    for ep in pbar:
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        ep_len = 0

        # Траектория для анализа
        trajectory = [] if analyze_failures else None

        while not done and ep_len < 1000:
            # model.predict ожидает Dict observation (MultiInputPolicy)
            action, _ = model.predict(obs, deterministic=True)

            obs, raw_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_reward += info.get('original_reward', 0.0)
            ep_len += 1

            # Собираем траекторию
            if analyze_failures:
                trajectory.append({
                    'step': ep_len,
                    'pos': info.get('agent_pos'),
                    'action': int(action.item()) if isinstance(action, np.ndarray) else int(action),
                    'bfs_dist': info.get('bfs_distance'),
                    'reward': float(raw_reward),
                })

        # Определяем успех
        success = 1.0 if info.get('original_reward', 0.0) > 0.1 else 0.0
        rewards.append(ep_reward)
        lengths.append(ep_len)
        successes.append(success)

        # Сохраняем провал
        if analyze_failures and not success and trajectory:
            # Добавляем goal_pos в info для анализа
            unwrapped = env.unwrapped
            if hasattr(unwrapped, 'box') and unwrapped.box.pos is not None:
                info['goal_pos'] = unwrapped.box.pos.tolist() if hasattr(unwrapped.box.pos, 'tolist') else list(unwrapped.box.pos)
            analyzer.add_failure(ep, trajectory, info, max_episode_steps)

        pbar.set_postfix({
            'succ': f"{np.mean(successes)*100:.1f}%",
            'avgR': f"{np.mean(rewards):.1f}",
            'len': f"{np.mean(lengths):.0f}",
        })

    env.close()

    # Статистика
    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    mean_length = float(np.mean(lengths))
    success_rate = float(np.mean(successes) * 100)

    results = {
        'model': os.path.basename(model_path),
        'train_config': train_config,
        'eval_mode': eval_mode,
        'grid_size': grid_size,
        'num_episodes': num_episodes,
        'success_rate': success_rate,
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'mean_length': mean_length,
        'rewards': rewards,
        'lengths': lengths,
        'successes': successes,
    }

    # Сохранение результатов
    if output_dir is None:
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/{model_name}_{grid_size[0]}x{grid_size[1]}_{eval_mode}_{num_episodes}ep_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)

    # Основные метрики
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False,
                  default=lambda x: float(x) if isinstance(x, np.floating) else x)

    # Анализ провалов
    if analyze_failures and analyzer:
        failures_path = os.path.join(output_dir, "failures.json")
        analyzer.save_json(failures_path)

        # Heatmap
        if not no_heatmap:
            heatmap_path = os.path.join(output_dir, "heatmap.png")
            analyzer.generate_heatmap(heatmap_path, grid_size)

        # Текстовый отчет
        report_path = os.path.join(output_dir, "patterns_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(FailurePatternDetector.get_report(analyzer.failures))

        if verbose:
            analyzer.print_report()

    if verbose:
        print(f"\n✅ Результаты сохранены: {output_dir}")
        print(f"   📊 Success rate: {success_rate:.1f}%")
        print(f"   💰 Средняя награда: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"   📏 Средняя длина: {mean_length:.1f}")

    return results


# ============================================================
# ПАКЕТНАЯ ОЦЕНКА
# ============================================================

def evaluate_batch(
    models_dir: str,
    configs: List[str],
    modes: List[str],
    grid_sizes: List[Tuple[int, int]],
    num_episodes: int,
    device: str,
    seed: int,
    analyze_failures: bool,
    max_failures_stored: int,
    no_heatmap: bool,
) -> Dict[str, Any]:
    """
    Пакетная оценка: все комбинации моделей × режимов × размеров.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = f"results/batch_report_{timestamp}"
    os.makedirs(batch_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"📦 ПАКЕТНАЯ ОЦЕНКА")
    print(f"   Модели: {configs}")
    print(f"   Режимы: {modes}")
    print(f"   Размеры: {[f'{g[0]}x{g[1]}' for g in grid_sizes]}")
    print(f"   Эпизодов: {num_episodes}")
    print(f"{'='*70}")

    all_results = {}
    total_combinations = len(configs) * len(modes) * len(grid_sizes)
    current = 0

    for config in configs:
        all_results[config] = {}

        for grid_size in grid_sizes:
            grid_key = f"{grid_size[0]}x{grid_size[1]}"
            all_results[config][grid_key] = {}

            # Формируем путь к модели
            model_filename = "final_model"
            model_path = os.path.join(models_dir, f"maze_curriculum_{config}_seed_0", f"{model_filename}.zip")

            if not os.path.exists(model_path):
                # Пробуем level-specific модель
                model_path = os.path.join(
                    models_dir,
                    f"maze_curriculum_{config}_seed_0",
                    f"level_{grid_key}_final.zip"
                )

            if not os.path.exists(model_path):
                print(f"\n⚠️ Модель не найдена: {model_path}, пропуск...")
                for mode in modes:
                    all_results[config][grid_key][mode] = {
                        'error': 'model_not_found',
                        'path': model_path,
                    }
                continue

            for mode in modes:
                current += 1
                print(f"\n[{current}/{total_combinations}] {config} | {grid_key} | {mode}")

                try:
                    result = evaluate_model(
                        model_path=model_path,
                        train_config=config,
                        grid_size=grid_size,
                        eval_mode=mode,
                        num_episodes=num_episodes,
                        device=device,
                        output_dir=os.path.join(batch_dir, f"{config}_{grid_key}_{mode}"),
                        seed=seed,
                        analyze_failures=analyze_failures,
                        max_failures_stored=max_failures_stored,
                        no_heatmap=no_heatmap,
                        verbose=False,
                    )

                    all_results[config][grid_key][mode] = {
                        'success_rate': result['success_rate'],
                        'mean_reward': result['mean_reward'],
                        'mean_length': result['mean_length'],
                        'std_reward': result['std_reward'],
                    }

                except Exception as e:
                    print(f"   ❌ Ошибка: {e}")
                    all_results[config][grid_key][mode] = {
                        'error': str(e),
                    }

    # Сводный отчет
    batch_report = {
        'timestamp': timestamp,
        'models_dir': models_dir,
        'configs': configs,
        'modes': modes,
        'grid_sizes': [list(g) for g in grid_sizes],
        'num_episodes': num_episodes,
        'results': all_results,
    }

    report_path = os.path.join(batch_dir, "batch_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(batch_report, f, indent=2, ensure_ascii=False,
                  default=lambda x: float(x) if isinstance(x, np.floating) else x)

    # Печать таблицы
    print(f"\n{'='*70}")
    print("📊 СВОДНАЯ ТАБЛИЦА (Success Rate %)")
    print(f"{'='*70}")

    for config in configs:
        print(f"\n🔧 Конфиг: {config}")
        for grid_size in grid_sizes:
            grid_key = f"{grid_size[0]}x{grid_size[1]}"
            print(f"   Размер: {grid_key}")
            header = "   " + " | ".join(f"{m:12s}" for m in modes)
            print(f"   {header}")
            values = []
            for mode in modes:
                res = all_results.get(config, {}).get(grid_key, {}).get(mode, {})
                if 'success_rate' in res:
                    values.append(f"{res['success_rate']:5.1f}%")
                else:
                    values.append("   N/A   ")
            print(f"   " + " | ".join(f"{v:12s}" for v in values))

    print(f"\n✅ Пакетный отчет сохранен: {report_path}")
    return batch_report


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Оценка QR-DQN на Maze с анализом паттернов провалов"
    )

    # --- Общие параметры ---
    parser.add_argument('--model', type=str, default=None,
                        help='Путь к модели (.zip). Для batch mode не требуется.')
    parser.add_argument('--config', type=str, default='baseline',
                        choices=list(TRAIN_CONFIGS.keys()),
                        help='Конфиг обучения (baseline/progressive_dr/ray_cast)')
    parser.add_argument('--mode', type=str, default='clean',
                        choices=list(EVAL_MODES.keys()),
                        help='Режим оценки')
    parser.add_argument('--grid-size', type=int, nargs=2, default=None,
                        metavar=('ROWS', 'COLS'),
                        help='Размер лабиринта, например 5 4. Если не указан — авто из пути или 5x5.')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Количество эпизодов')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Папка для результатов (авто если не указана)')

    # --- Анализ провалов ---
    parser.add_argument('--analyze-failures', action='store_true',
                        help='Анализировать и сохранять паттерны провалов')
    parser.add_argument('--max-failures', type=int, default=50,
                        help='Макс. количество сохраняемых провалов')
    parser.add_argument('--no-heatmap', action='store_true',
                        help='Не генерировать heatmap (быстрее)')

    # --- Batch mode ---
    parser.add_argument('--batch', action='store_true',
                        help='Пакетная оценка всех комбинаций')
    parser.add_argument('--models-dir', type=str, default='models',
                        help='Базовая папка с моделями (для batch)')
    parser.add_argument('--configs', type=str, default=None,
                        help='Список конфигов через запятую, например baseline,progressive_dr')
    parser.add_argument('--modes', type=str, default=None,
                        help='Список режимов через запятую, например clean,total_chaos')
    parser.add_argument('--grid-sizes', type=str, default=None,
                        help='Список размеров через запятую, например "5 4,5 5"')

    args = parser.parse_args()

#   python evaluate.py --model models/maze_curriculum_baseline_seed_0/final_model --mode clean --episodes 100 --grid-sizes ROWS 5 COLS 4 --analyze-failures


    # --- Batch mode ---
    if args.batch:
        if not args.configs or not args.modes:
            print("❌ Для batch mode нужны --configs и --modes")
            sys.exit(1)

        configs = [c.strip() for c in args.configs.split(',')]
        modes = [m.strip() for m in args.modes.split(',')]

        if args.grid_sizes:
            grid_sizes = []
            for gs in args.grid_sizes.split(','):
                parts = gs.strip().split()
                grid_sizes.append((int(parts[0]), int(parts[1])))
        else:
            grid_sizes = [(5, 5)]

        evaluate_batch(
            models_dir=args.models_dir,
            configs=configs,
            modes=modes,
            grid_sizes=grid_sizes,
            num_episodes=args.episodes,
            device=args.device,
            seed=args.seed,
            analyze_failures=args.analyze_failures,
            max_failures_stored=args.max_failures,
            no_heatmap=args.no_heatmap,
        )
        return

    # --- Single mode ---
    if not args.model:
        print("❌ Нужен --model для одиночной оценки")
        sys.exit(1)

    grid_size = tuple(args.grid_size) if args.grid_size else None

    evaluate_model(
        model_path=args.model,
        train_config=args.config,
        grid_size=grid_size,
        eval_mode=args.mode,
        num_episodes=args.episodes,
        device=args.device,
        output_dir=args.output_dir,
        seed=args.seed,
        analyze_failures=args.analyze_failures,
        max_failures_stored=args.max_failures,
        no_heatmap=args.no_heatmap,
        verbose=True,
    )


if __name__ == "__main__":
    main()