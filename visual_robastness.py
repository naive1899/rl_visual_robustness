#!/usr/bin/env python3
"""
Visual Robustness Demo — интерактивная демонстрация робастности агента.

Особенности:
- Одна модель: progressive_dr (уже робастная к помехам)
- 4 режима теста: clean, light_dr, sensor_stress, total_chaos
- 6 лабиринтов: 4×5, 5×4, 5×5, 5×6, 6×5, 6×6
- Ручное управление + автопилот (SPACE для переключения)
- Severity регулируется как громкость (+/-), 0-100% (1.0)
- 8 базовых эпизодов (seed 0-7), затем можно продолжать
- Success Rate крупно на экране
- PBRS и BFS distance на панели (зелёным)
- Одно окно pygame: слева first-person view, справа панель статистики
- Клавиша D — Skip Episode (failure + next seed)
- Клавиша F — Toggle RGB/Gray отображение (только визуально)
- Severity: основная клавиатура + numpad +/-
- Total Chaos: старт 0.5, диапазон ручной регулировки 0.5–1.0
"""

import os
import sys
import argparse
import numpy as np
import pygame
from pygame.locals import *
from stable_baselines3 import PPO

from envs.wrappers import (
    ShapedRewardWrapper, PerturbationWrapper, DilatedFrameStack,
    MultiModalObservationWrapper, WallPenaltyWrapper
)
from envs.env_factory import MiniWorldEnvFactory

# ============================================================
# КОНФИГУРАЦИЯ
# ============================================================

REWARD_KWARGS = {
    'time_penalty': -0.01,
    'goal_bonus': 10.0,
    'terminal_penalty': -5.0,
    'use_pbrs': True,
    'distance_reward_coef': 0.12,
    'gamma': 1.0,
    'use_novelty_reward': False,
    'state_precision': 0.5,
    'novelty_bonus': 0.0,
    'use_room_reward': False,
    'room_bonus': 0.0,
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
    '4x5': 750, '5x4': 750, '5x5': 1150,
    '5x6': 1350, '6x5': 1350, '6x6': 1350
}

EVAL_MODES = {
    'clean': {'perturbation_mode': 'none', 'perturbation_severity': 0.0, 'enable_domain_rand': False},
    'light_dr': {'perturbation_mode': 'none', 'perturbation_severity': 0.0, 'enable_domain_rand': True},
    'sensor_stress': {'perturbation_mode': 'fixed', 'perturbation_severity': 0.7, 'enable_domain_rand': False},
    'total_chaos': {'perturbation_mode': 'naive', 'perturbation_severity': 0.5, 'enable_domain_rand': True},
}

MAZE_OPTIONS = {
    '1': (4, 5), '2': (5, 4), '3': (5, 5),
    '4': (5, 6), '5': (6, 5), '6': (6, 6)
}

ACTION_TURN_LEFT = 0
ACTION_TURN_RIGHT = 1
ACTION_MOVE_FORWARD = 2
ACTION_MOVE_BACK = 3

# Цвета
COLOR_BG = (20, 20, 30)
COLOR_PANEL = (30, 30, 45)
COLOR_TEXT = (230, 230, 230)
COLOR_TEXT_DIM = (140, 140, 160)
COLOR_GREEN = (80, 200, 120)
COLOR_YELLOW = (230, 180, 60)
COLOR_RED = (220, 80, 80)
COLOR_BAR_BG = (50, 50, 70)
COLOR_BAR_FILL = (70, 150, 220)
COLOR_BAR_BORDER = (90, 90, 110)
COLOR_AI = (80, 200, 120)
COLOR_MANUAL = (230, 180, 60)
COLOR_FRAME = (60, 60, 80)
COLOR_PBRS = (100, 255, 100)


# ============================================================
# ОСНОВНОЙ КЛАСС ДЕМО
# ============================================================

class VisualRobustnessDemo:
    def __init__(self, model_path, maze_size, eval_mode, max_episodes=8, steps_per_sec=5, use_gray=False):
        self.model_path = model_path
        self.maze_size = maze_size
        self.eval_mode_name = eval_mode
        self.max_episodes = max_episodes
        self.steps_per_sec = steps_per_sec
        self.frame_delay = 1000 // steps_per_sec
        self.use_gray = use_gray

        # Создаём среду
        self._create_env()

        # Загружаем модель с env
        print(f"Loading model: {model_path}")
        self.model = PPO.load(model_path, env=self.env, device='auto')
        print("Model loaded successfully")

        # Pygame init
        pygame.init()
        self.screen_width = 1200
        self.screen_height = 720
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Visual Robustness Demo")
        self.clock = pygame.time.Clock()

        # Шрифты
        self.font_large = pygame.font.SysFont("consolas", 52, bold=True)
        self.font_medium = pygame.font.SysFont("consolas", 26)
        self.font_small = pygame.font.SysFont("consolas", 18)
        self.font_tiny = pygame.font.SysFont("consolas", 14)

        # Размеры панелей
        self.fp_size = 640
        self.panel_x = self.fp_size + 60
        self.panel_width = self.screen_width - self.panel_x - 40

        # Состояние игры
        self.current_seed = 0
        self.episode_num = 1
        self.episode_step = 0
        self.total_reward = 0.0
        self.ai_mode = True
        self.running = True
        self.in_menu = False
        self.frame = None
        self.last_step_time = 0

        # Сохраняем пользовательский severity (чтобы не сбрасывался при reset)
        self.user_severity = self.base_severity  # None = использовать базовый из режима

        # Визуальный grayscale toggle (только для отображения)
        self.visual_gray_mode = False

        # Статистика
        self.episode_results = []
        self.session_successes = 0
        self.session_episodes = 0

        # Первый reset
        self._reset_episode()

    def _create_env(self):
        """Создаёт среду с текущими настройками."""
        rows, cols = self.maze_size
        max_steps = LEVEL_STEPS.get(f"{rows}x{cols}", 1150)

        base_env = MiniWorldEnvFactory.create_env(
            env_name='maze',
            max_episode_steps=max_steps,
            obs_width=64,
            obs_height=64,
            num_rows=rows,
            num_cols=cols,
            room_size=4,
            render_mode='rgb_array',
        )
        if hasattr(base_env.unwrapped, 'max_episode_steps'):
            base_env.unwrapped.max_episode_steps = max_steps + 500

        self.base_env_for_render = base_env

        base_env = WallPenaltyWrapper(base_env)
        base_env = ShapedRewardWrapper(base_env, **REWARD_KWARGS)

        mp = EVAL_MODES[self.eval_mode_name]
        self.perturbation_wrapper = PerturbationWrapper(
            base_env,
            mode=mp['perturbation_mode'],
            severity=mp['perturbation_severity'],
            enable_domain_rand=mp['enable_domain_rand']
        )

        if self.use_gray:
            try:
                from envs.grayscale_wrapper import GrayscaleWrapper
                env = GrayscaleWrapper(self.perturbation_wrapper)
                print("Using GrayscaleWrapper")
            except ImportError:
                print("WARNING: GrayscaleWrapper not found, using RGB")
                env = self.perturbation_wrapper
        else:
            env = self.perturbation_wrapper

        env = DilatedFrameStack(env, **DILATED_STACK_KWARGS)
        env = MultiModalObservationWrapper(env)

        self.env = env
        self.base_severity = mp['perturbation_severity']
        self.can_adjust_severity = self.eval_mode_name in ['sensor_stress', 'total_chaos']

    def _apply_user_severity(self):
        """Применяет пользовательский severity после reset (чтобы не сбрасывался)."""
        if self.user_severity is not None and self.perturbation_wrapper is not None:
            self.perturbation_wrapper.current_severity = self.user_severity
            if hasattr(self.perturbation_wrapper, 'perturbation_manager'):
                self.perturbation_wrapper.perturbation_manager.set_severity(self.user_severity)

    def _reset_episode(self):
        """Сброс эпизода с текущим seed."""
        self.obs, self.info = self.env.reset(seed=self.current_seed)

        # Восстанавливаем пользовательский severity после reset
        self._apply_user_severity()

        self.episode_step = 0
        self.total_reward = 0.0
        self.frame = None
        self.last_step_time = pygame.time.get_ticks()
        print(f"\n🔄 Episode {self.episode_num} started (seed={self.current_seed}, {self.maze_size[0]}×{self.maze_size[1]})")

    def _get_first_person_frame(self):
        """Получает кадр от первого лица."""
        try:
            if self.base_env_for_render is not None:
                img = self.base_env_for_render.render()
                if img is not None and isinstance(img, np.ndarray):
                    if self.perturbation_wrapper is not None:
                        img = self.perturbation_wrapper._apply_perturbation(img)
                    # Визуальный grayscale toggle (только для отображения)
                    if self.visual_gray_mode:
                        gray = np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
                        img = np.stack([gray, gray, gray], axis=-1)
                    return img
        except Exception as e:
            print(f"Render error: {e}")
        return np.zeros((64, 64, 3), dtype=np.uint8)

    def _draw_progress_bar(self, surface, x, y, width, height, progress, color_fill):
        """Рисует полоску прогресса."""
        pygame.draw.rect(surface, COLOR_BAR_BG, (x, y, width, height), border_radius=4)
        fill_width = int(width * np.clip(progress, 0, 1))
        if fill_width > 0:
            pygame.draw.rect(surface, color_fill, (x, y, fill_width, height), border_radius=4)
        pygame.draw.rect(surface, COLOR_BAR_BORDER, (x, y, width, height), 2, border_radius=4)

    def _draw_panel(self):
        """Рисует правую панель с информацией."""
        panel_surface = pygame.Surface((self.panel_width, self.screen_height))
        panel_surface.fill(COLOR_PANEL)

        x = 20
        y = 25

        # Заголовок
        title = self.font_medium.render("VISUAL ROBUSTNESS", True, COLOR_TEXT)
        panel_surface.blit(title, (x, y))
        y += 45

        # Success Rate — КРУПНО
        sr = self._get_current_sr()
        sr_label = self.font_small.render("SUCCESS RATE", True, COLOR_TEXT_DIM)
        panel_surface.blit(sr_label, (x, y))
        y += 26

        sr_color = COLOR_GREEN if sr >= 80 else (COLOR_YELLOW if sr >= 50 else COLOR_RED)
        sr_value = self.font_large.render(f"{sr:.1f}%", True, sr_color)
        panel_surface.blit(sr_value, (x, y))
        y += 75

        # Прогресс эпизодов
        ep_label = self.font_small.render(f"Episode {self.episode_num} / {self.max_episodes}", True, COLOR_TEXT)
        panel_surface.blit(ep_label, (x, y))
        y += 22
        self._draw_progress_bar(panel_surface, x, y, self.panel_width - 40, 18,
                                min(self.episode_num / self.max_episodes, 1.0), COLOR_BAR_FILL)
        y += 38

        # Режим
        mode_label = self.font_small.render("Mode:", True, COLOR_TEXT_DIM)
        panel_surface.blit(mode_label, (x, y))
        mode_text = self.font_medium.render(self.eval_mode_name.upper().replace('_', ' '), True, COLOR_TEXT)
        panel_surface.blit(mode_text, (x + 70, y - 4))
        y += 42

        # Severity (громкость)
        sev_label = self.font_small.render("Severity:", True, COLOR_TEXT_DIM)
        panel_surface.blit(sev_label, (x, y))
        y += 22

        current_sev = self.perturbation_wrapper.current_severity if self.perturbation_wrapper else 0.0
        sev_percent = int(current_sev * 100)
        self._draw_progress_bar(panel_surface, x, y, self.panel_width - 80, 22,
                                current_sev, COLOR_BAR_FILL)
        sev_text = self.font_small.render(f"{sev_percent}%", True, COLOR_TEXT)
        panel_surface.blit(sev_text, (x + self.panel_width - 70, y))
        y += 42

        # Speed
        speed_label = self.font_small.render(f"Speed: {self.steps_per_sec} steps/sec", True, COLOR_TEXT_DIM)
        panel_surface.blit(speed_label, (x, y))
        y += 26
        speed_hint = self.font_tiny.render("[1-9] to change speed", True, COLOR_TEXT_DIM)
        panel_surface.blit(speed_hint, (x, y))
        y += 35

        # AI / Manual индикатор
        ai_color = COLOR_AI if self.ai_mode else COLOR_MANUAL
        ai_text = "AI ACTIVE" if self.ai_mode else "MANUAL MODE"
        ai_surface = self.font_medium.render(ai_text, True, ai_color)
        panel_surface.blit(ai_surface, (x, y))
        y += 48

        # Step counter
        step_text = self.font_small.render(f"Step: {self.episode_step}", True, COLOR_TEXT)
        panel_surface.blit(step_text, (x, y))
        y += 26

        # Grid size
        grid_text = self.font_small.render(f"Maze: {self.maze_size[0]}×{self.maze_size[1]}", True, COLOR_TEXT)
        panel_surface.blit(grid_text, (x, y))
        y += 26

        # PBRS + BFS distance — ЗЕЛЁНЫМ
        pbrs = self.info.get('pbrs_reward', 0.0)
        bfs_dist = self.info.get('bfs_distance', 0.0)
        pbrs_text = f"dist={bfs_dist:.1f}" # f"PBRS: {pbrs:+.3f} (dist={bfs_dist:.1f})"
        pbrs_surface = self.font_small.render(pbrs_text, True, COLOR_PBRS)
        panel_surface.blit(pbrs_surface, (x, y))
        y += 45

        # Разделитель
        pygame.draw.line(panel_surface, (50, 50, 70), (x, y), (self.panel_width - 20, y), 2)
        y += 18

        # Controls
        ctrl_title = self.font_small.render("CONTROLS", True, COLOR_TEXT_DIM)
        panel_surface.blit(ctrl_title, (x, y))
        y += 26

        controls = [
            "W / S — Forward / Back",
            "A / D — Turn Left / Right",
            "SPACE — Toggle AI / Manual",
            "+ / - — Adjust severity",
            "1-9 — Change speed",
            "F — Skip Episode (fail)",
            "G — Toggle RGB / Gray view",
            "R — Reset episode",
            "Q — Quit to menu",
        ]
        for ctrl in controls:
            ctrl_surf = self.font_tiny.render(ctrl, True, COLOR_TEXT_DIM)
            panel_surface.blit(ctrl_surf, (x, y))
            y += 20

        self.screen.blit(panel_surface, (self.panel_x, 0))

    def _get_current_sr(self):
        """Возвращает текущий Success Rate."""
        if self.session_episodes == 0:
            return 0.0
        return (self.session_successes / self.session_episodes) * 100

    def _draw_first_person(self):
        """Рисует first-person view слева."""
        frame = self._get_first_person_frame()

        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            h, w = 64, 64

        try:
            frame_rgb = np.transpose(frame, (1, 0, 2))
            frame_surface = pygame.surfarray.make_surface(frame_rgb)
            frame_scaled = pygame.transform.scale(frame_surface, (self.fp_size, self.fp_size))
            self.screen.blit(frame_scaled, (20, 40))
        except Exception as e:
            surf = pygame.Surface((w, h))
            for y in range(min(h, 64)):
                for x in range(min(w, 64)):
                    color = (int(frame[y, x, 0]), int(frame[y, x, 1]), int(frame[y, x, 2]))
                    surf.set_at((x, y), color)
            surf_scaled = pygame.transform.scale(surf, (self.fp_size, self.fp_size))
            self.screen.blit(surf_scaled, (20, 40))

        pygame.draw.rect(self.screen, COLOR_FRAME, (20, 40, self.fp_size, self.fp_size), 4)

        fp_label = self.font_small.render("First-Person View", True, COLOR_TEXT_DIM)
        self.screen.blit(fp_label, (20, 18))

        # Индикатор визуального grayscale
        if self.visual_gray_mode:
            gray_label = self.font_small.render("[GRAY VIEW]", True, COLOR_YELLOW)
            self.screen.blit(gray_label, (20, self.fp_size + 48))

    def _draw_session_complete(self):
        """Рисует экран завершения сессии."""
        overlay = pygame.Surface((self.screen_width, self.screen_height))
        overlay.fill((0, 0, 0))
        overlay.set_alpha(200)
        self.screen.blit(overlay, (0, 0))

        x = self.screen_width // 2
        y = self.screen_height // 2 - 160

        title = self.font_large.render("SESSION COMPLETE", True, COLOR_TEXT)
        title_rect = title.get_rect(center=(x, y))
        self.screen.blit(title, title_rect)
        y += 90

        sr = self._get_current_sr()
        sr_color = COLOR_GREEN if sr >= 80 else (COLOR_YELLOW if sr >= 50 else COLOR_RED)
        sr_text = self.font_large.render(f"{sr:.1f}%", True, sr_color)
        sr_rect = sr_text.get_rect(center=(x, y))
        self.screen.blit(sr_text, sr_rect)
        y += 55

        sr_label = self.font_medium.render(f"Success Rate ({self.session_successes}/{self.session_episodes})", True, COLOR_TEXT_DIM)
        sr_label_rect = sr_label.get_rect(center=(x, y))
        self.screen.blit(sr_label, sr_label_rect)
        y += 60

        if self.episode_results:
            avg_steps = np.mean([r['steps'] for r in self.episode_results])
            avg_reward = np.mean([r['total_reward'] for r in self.episode_results])
            stats = [
                f"Avg Steps: {avg_steps:.0f}",
                f"Avg Reward: {avg_reward:.2f}",
            ]
            for stat in stats:
                stat_surf = self.font_small.render(stat, True, COLOR_TEXT)
                stat_rect = stat_surf.get_rect(center=(x, y))
                self.screen.blit(stat_surf, stat_rect)
                y += 28

        y += 50
        hint = self.font_small.render("C = Continue  |  M = Menu  |  Q = Quit", True, COLOR_YELLOW)
        hint_rect = hint.get_rect(center=(x, y))
        self.screen.blit(hint, hint_rect)

    def _end_episode(self, success):
        """Завершение эпизода, логирование."""
        self.session_episodes += 1
        if success:
            self.session_successes += 1

        result = {
            'episode': self.episode_num,
            'seed': self.current_seed,
            'success': success,
            'steps': self.episode_step,
            'total_reward': self.total_reward,
            'ai_mode': self.ai_mode,
        }
        self.episode_results.append(result)

        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"\n{status} Episode {self.episode_num}")
        print(f"   Steps: {self.episode_step}")
        print(f"   Reward: {self.total_reward:.2f}")
        print(f"   SR: {self._get_current_sr():.1f}%")

        if self.episode_num >= self.max_episodes and not self.in_menu:
            self.in_menu = True
            print(f"\n🏁 Base session complete! SR: {self._get_current_sr():.1f}%")
            return

        self.episode_num += 1
        self.current_seed += 1
        self._reset_episode()

    def _skip_episode(self):
        """Принудительно завершить эпизод как failure и перейти к следующему."""
        print(f"\n⏭️  Skipping Episode {self.episode_num} (counted as failure)")
        self._end_episode(success=False)

    def _toggle_visual_gray(self):
        """Переключить визуальное отображение RGB ↔ Gray."""
        self.visual_gray_mode = not self.visual_gray_mode
        mode_str = "GRAY" if self.visual_gray_mode else "RGB"
        print(f"Visual display: {mode_str}")

    def _adjust_severity(self, delta):
        """Регулирует severity как громкость в режимах 3 и 4.

        Сохраняет значение в self.user_severity чтобы не сбрасывалось при reset.
        Для total_chaos диапазон ограничен 0.5–1.0.
        """
        if not self.can_adjust_severity or self.perturbation_wrapper is None:
            return

        # Определяем границы в зависимости от режима
        if self.eval_mode_name == 'total_chaos':
            low, high = 0.5, 1.0
        else:
            low, high = 0.0, 1.0

        # Вычисляем новое значение
        new_sev = float(np.clip(self.perturbation_wrapper.current_severity + delta, low, high))

        # Сохраняем как пользовательское
        self.user_severity = new_sev

        # Применяем сразу
        self.perturbation_wrapper.current_severity = new_sev
        if hasattr(self.perturbation_wrapper, 'perturbation_manager'):
            self.perturbation_wrapper.perturbation_manager.set_severity(new_sev)

        print(f"Severity: {new_sev:.2f} ({int(new_sev*100)}%)")

    def _change_speed(self, speed):
        """Меняет скорость шагов в секунду."""
        self.steps_per_sec = speed
        self.frame_delay = 1000 // speed
        print(f"Speed: {speed} steps/sec")

    def run(self):
        """Главный игровой цикл."""
        while self.running:
            action = None
            current_time = pygame.time.get_ticks()

            for event in pygame.event.get():
                if event.type == QUIT:
                    self.running = False

                elif event.type == KEYDOWN:
                    if event.key == K_q or event.key == K_ESCAPE:
                        if self.in_menu:
                            self.running = False
                        else:
                            return 'menu'

                    elif event.key == K_r:
                        self._reset_episode()

                    elif event.key == K_SPACE:
                        self.ai_mode = not self.ai_mode
                        mode_str = "AI" if self.ai_mode else "MANUAL"
                        print(f"Switched to {mode_str}")

                    elif event.key == K_f:
                        if not self.in_menu:
                            self._skip_episode()

                    elif event.key == K_g:
                        self._toggle_visual_gray()

                    elif event.key in (K_EQUALS, K_PLUS, K_KP_PLUS):
                        self._adjust_severity(0.05)

                    elif event.key in (K_MINUS, K_KP_MINUS):
                        self._adjust_severity(-0.05)

                    elif event.key >= K_1 and event.key <= K_9:
                        speed = event.key - K_0
                        self._change_speed(speed)

                    elif self.in_menu:
                        if event.key == K_c:
                            self.in_menu = False
                            self.max_episodes += 8
                            self.episode_num += 1
                            self.current_seed += 1
                            self._reset_episode()
                        elif event.key == K_m:
                            return 'menu'

                    else:
                        if not self.ai_mode:
                            if event.key == K_w:
                                action = ACTION_MOVE_FORWARD
                            elif event.key == K_s:
                                action = ACTION_MOVE_BACK
                            elif event.key == K_a:
                                action = ACTION_TURN_LEFT
                            elif event.key == K_d:
                                action = ACTION_TURN_RIGHT

            if not self.running:
                break

            # AI mode
            if self.ai_mode and not self.in_menu and action is None:
                action, _ = self.model.predict(self.obs, deterministic=False)
                action = int(action)

            # Step
            can_step = (current_time - self.last_step_time >= self.frame_delay) or (not self.ai_mode and action is not None)

            if action is not None and not self.in_menu and can_step:
                self.obs, reward, terminated, truncated, self.info = self.env.step(action)
                self.total_reward += reward
                self.episode_step += 1
                self.last_step_time = current_time

                if terminated or truncated:
                    success = self.info.get('original_reward', 0) > 0.1
                    self._end_episode(success)

            # Отрисовка
            self.screen.fill(COLOR_BG)
            self._draw_first_person()
            self._draw_panel()

            if self.in_menu:
                self._draw_session_complete()

            pygame.display.flip()
            self.clock.tick(60)

        return 'quit'

    def close(self):
        self.env.close()
        pygame.quit()


# ============================================================
# МЕНЮ ВЫБОРА (консольное)
# ============================================================

def show_menu():
    """Показывает консольное меню выбора."""
    print("\n" + "=" * 50)
    print("      Visual Robustness Demo")
    print("=" * 50)
    print()
    print("Select Maze:")
    for key, (rows, cols) in MAZE_OPTIONS.items():
        print(f"  [{key}] {rows}×{cols}")
    print()

    maze_choice = input("Enter maze number: ").strip()
    while maze_choice not in MAZE_OPTIONS:
        maze_choice = input("Invalid. Enter 1-6: ").strip()

    maze_size = MAZE_OPTIONS[maze_choice]
    print(f"Selected: {maze_size[0]}×{maze_size[1]}")
    print()

    print("Test Mode:")
    print("  [Q] Clean")
    print("  [W] Domain Randomization")
    print("  [E] Sensor Stress (fixed noise)")
    print("  [R] Total Chaos (noise + domain rand)")
    print()

    mode_keys = {'q': 'clean', 'w': 'light_dr', 'e': 'sensor_stress', 'r': 'total_chaos'}
    mode_choice = input("Enter mode key: ").strip().lower()
    while mode_choice not in mode_keys:
        mode_choice = input("Invalid. Enter Q/W/E/R: ").strip().lower()

    eval_mode = mode_keys[mode_choice]
    print(f"Selected mode: {eval_mode}")
    print()

    # Grayscale?
    gray = input("Use grayscale model? [y/N]: ").strip().lower()
    use_gray = gray == 'y'
    if use_gray:
        print("Using GrayscaleWrapper")
    print()

    # Скорость — теперь 5 по умолчанию, не спрашиваем
    speed = 5
    print(f"Speed: {speed} steps/sec (change with 1-9 in demo)")
    print()

    # Путь к модели
    default_model = "models/maze_progressive_dr_seed_0/level_7x7_final.zip"
    model_path = input(f"Model path [{default_model}]: ").strip()
    if not model_path:
        model_path = default_model

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found: {model_path}")
        return None, None, None, None, None

    return model_path, maze_size, eval_mode, speed, use_gray


def main():
    parser = argparse.ArgumentParser(description="Visual Robustness Demo")
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--maze', type=str, default=None, choices=list(MAZE_OPTIONS.keys()))
    parser.add_argument('--mode', type=str, default=None, choices=list(EVAL_MODES.keys()))
    parser.add_argument('--episodes', type=int, default=8)
    parser.add_argument('--speed', type=int, default=5, help='Steps per second (1-10)')
    parser.add_argument('--gray', action='store_true', help='Use grayscale model')
    args = parser.parse_args()

    # CLI args или интерактивное меню
    if args.model and args.maze and args.mode:
        model_path = args.model
        maze_size = MAZE_OPTIONS[args.maze]
        eval_mode = args.mode
        speed = max(1, min(10, args.speed))
        use_gray = args.gray
    else:
        result = show_menu()
        if result[0] is None:
            return
        model_path, maze_size, eval_mode, speed, use_gray = result

    # Запуск демо
    while True:
        demo = VisualRobustnessDemo(model_path, maze_size, eval_mode,
                                     max_episodes=args.episodes, steps_per_sec=speed, use_gray=use_gray)
        result = demo.run()
        demo.close()

        if result == 'quit':
            print("\n Goodbye!")
            break
        elif result == 'menu':
            result = show_menu()
            if result[0] is None:
                break
            model_path, maze_size, eval_mode, speed, use_gray = result


if __name__ == '__main__':
    main()


    
# Запуск:
# python visual_robastness.py
# Выбор модели:
# models/maze_progressive_dr_seed_0/level_6x6_final.zip
# models/maze_baseline_seed_0/level_6x6_final.zip
# models/maze_baseline_gray_seed_0/level_6x6_final.zip
