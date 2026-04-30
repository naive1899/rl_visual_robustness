#!/usr/bin/env python3
"""
Продолжение обучения с чекпоинта (после 3x3 -> 4x4 и т.д.).
Чекпоинт определяется автоматически по --level и --config.
Путь к папке моделей прописан внутри кода.
"""
import argparse
import os
from sb3_contrib import QRDQN
from train import create_vec_env, MAZE_CURRICULUM, LevelMetricsCallback, CurriculumSuccessCallback


# Ваш путь:
BASE_MODELS_DIR = r"C:\Users\ПК\Desktop\QR_DQN\models"
# ------------------------------------------------

def continue_from_checkpoint(
    start_level_idx: int = 3,   # 0=2x2, 1=3x3, 2=4x4, 3=5x4, 4=x5, 5=5x5
    seed: int = 0,
    config: str = 'baseline',
    num_envs: int = 8,
    save_dir: str = BASE_MODELS_DIR
):
    """Продолжает обучение с указанного уровня curriculum, загружая чекпоинт предыдущего уровня."""

    # Проверяем, что есть предыдущий уровень
    if start_level_idx < 1:
        raise ValueError("start_level_idx должен быть >= 1 (нет предыдущего уровня для загрузки)")
    if start_level_idx >= len(MAZE_CURRICULUM):
        raise ValueError(f"start_level_idx={start_level_idx} вне диапазона 0..{len(MAZE_CURRICULUM)-1}")

    # Имя предыдущего уровня
    prev_level_config = MAZE_CURRICULUM[start_level_idx - 1]
    prev_label = prev_level_config['label']

    # Строим путь к чекпоинту предыдущего уровня
    run_name = f"maze_curriculum_{config}_seed_{seed}"
    checkpoint_path = os.path.join(save_dir, run_name, f"level_{prev_label}_final.zip")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Чекпоинт не найден: {checkpoint_path}")

    print(f"📦 Загрузка чекпоинта: {checkpoint_path}")
    model = QRDQN.load(checkpoint_path, device="auto")

    # ЗАГРУЗКА БУФФЕРА
    buffer_path = os.path.join(save_dir, run_name, f"level_{prev_label}_replay_buffer.pkl")
    if os.path.exists(buffer_path):
        print(f"Загрузка Replay Buffer: {buffer_path}")
        model.load_replay_buffer(buffer_path)
    else:
        print("⚠️ Файл буфера не найден, обучение начнется с пустого опыта!")
    # ------------------


    model_dir = os.path.join(save_dir, run_name)
    os.makedirs(model_dir, exist_ok=True)

    # Проходим оставшиеся уровни, начиная с start_level_idx
    for level_idx in range(start_level_idx, len(MAZE_CURRICULUM)):
        level_config = MAZE_CURRICULUM[level_idx]
        label = level_config['label']
        max_steps = level_config['max_timesteps']
        max_ep_steps = level_config['max_episode_steps']
        target_sr = level_config['target_success_rate']
        window_size = level_config['window_size']

        print(f"\n{'='*60}")
        print(f"📚 Уровень {level_idx+1}/{len(MAZE_CURRICULUM)}: {label}")
        print(f"   Макс. шагов: {max_steps:,} | Макс. длина эпизода: {max_ep_steps}")
        print(f"   Целевой Success Rate: {target_sr}% (окно {window_size})")
        print(f"{'='*60}")

        # Смена шага оптимизатора после чекпоинта
        #for pg in model.policy.optimizer.param_groups:
            #pg['lr'] = 1e-4

        # Создаём среду для этого уровня
        env = create_vec_env(
            'maze', num_envs, seed, config, 
            grid_size=level_config['grid_size'],
            max_episode_steps=max_ep_steps
        )
        model.set_env(env)

        # Колбэки
        success_callback = LevelMetricsCallback(level_label=label, window_size=window_size, verbose=1)

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

        # Сброс exploration на начало
        from stable_baselines3.common.utils import get_linear_fn
        model.exploration_schedule = get_linear_fn(
            start=1.0,            # exploration_initial_eps
            end=0.15,             # exploration_final_eps
            end_fraction=0.65      # exploration_fraction 
        )
        # Обучение
        model.learn(
            total_timesteps=max_steps,
            callback=[curriculum_callback, success_callback],
            reset_num_timesteps=False,
            tb_log_name=f"level_{label}",
            progress_bar=True,
            log_interval=200
        )
        
        # Сохраняем финальную модель уровня
        level_path = os.path.join(model_dir, f"level_{label}_final")
        model.save(level_path)
        model.save_replay_buffer(os.path.join(model_dir, f"level_{label}_replay_buffer"))  # БУФФЕР
        print(f"✅ Сохранено: {level_path}")

        # Если уровень не пройден по success rate — останавливаем curriculum
        if not curriculum_callback.level_completed and level_idx < len(MAZE_CURRICULUM) - 1:
            print(f"⛔ Уровень {label} не пройден по success rate. Остановка.")
            break

        env.close()

    final_path = os.path.join(model_dir, "final_model")
    model.save(final_path)
    print(f"\n🏁 Финальная модель: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Продолжить обучение с автоматическим выбором чекпоинта")
    parser.add_argument('--level', type=int, required=True,
                        help='Индекс уровня, с которого начинаем (1=3x3, 2=4x4). Чекпоинт возьмётся от предыдущего.')
    parser.add_argument('--config', type=str, default='baseline',
                        choices=['baseline', 'progressive_dr', 'ray_cast'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-envs', type=int, default=8)
    args = parser.parse_args()

    continue_from_checkpoint(
        start_level_idx=args.level,
        config=args.config,
        seed=args.seed,
        num_envs=args.num_envs,
        save_dir=BASE_MODELS_DIR   # путь зашит в коде
    )

# python from_checkpoint.py --level 3 --config baseline  
# python from_checkpoint.py --level 3 --config progressive_dr
# python from_checkpoint.py --level 3 --config ray_cast
