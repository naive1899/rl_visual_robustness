#!/usr/bin/env python3
"""
Ручное управление для MiniWorld с полной поддержкой Reward Shaping:
- PBRS (Potential-based reward shaping) с BFS расстоянием
- Novelty/exploration reward
- Room-level reward
- Spin penalty (штраф за кручение на месте)
- Forward bonus (бонус за движение вперёд)
- Настройка движения через фабрику (forward_step, turn_step)
- Визуальные искажения (naive_dr / progressive) с отображением в отдельном окне (OpenCV)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import numpy as np
import pygame
from pygame.locals import *
import cv2

from envs.wrappers import PerturbationWrapper, ShapedRewardWrapper, ActionRepeatWrapper, MultiModalObservationWrapper
from envs.env_factory import MiniWorldEnvFactory

ACTION_TURN_LEFT = 0
ACTION_TURN_RIGHT = 1
ACTION_MOVE_FORWARD = 2
ACTION_MOVE_BACK = 3


def find_wrapper(env, wrapper_class):
    """Рекурсивно ищет обёртку указанного класса в цепочке env."""
    while env is not None:
        if isinstance(env, wrapper_class):
            return env
        if hasattr(env, 'env'):
            env = env.env
        else:
            break
    return None


class ManualController:
    def __init__(
        self,
        num_rows=4,
        num_cols=4,
        domain_rand=False,
        perturbation='none',
        severity=0.5,
    ):
        pygame.init()
        # Окно для панели информации и top‑view
        self.screen = pygame.display.set_mode((400, 620))
        pygame.display.set_caption("MiniWorld Control - Reward Shaping Panel")
        self.font = pygame.font.SysFont(None, 28)
        self.small_font = pygame.font.SysFont(None, 20)

        # Окно для first‑person view (OpenCV)
        cv2.namedWindow("First-Person View (with perturbations)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("First-Person View (with perturbations)", 800, 620)

        print(f"Создание среды {num_rows}x{num_cols}...")

        # Базовая среда через фабрику (режим rgb_array, чтобы получить кадры)
        base_env = MiniWorldEnvFactory.create_env(
            env_name='maze',
            max_episode_steps=800,
            obs_width=64,
            obs_height=64,
            num_rows=num_rows,
            num_cols=num_cols,
            room_size=4,
            render_mode='rgb_array',   # получаем массивы RGB
            domain_rand=domain_rand,
        )
        base_env.unwrapped.max_episode_steps = 1500

        # Обёртка с shaping rewards
        self.env = ShapedRewardWrapper(
            base_env,
            time_penalty=-0.01,
            distance_reward_coef=3.0,
            goal_bonus=10.0,
            use_pbrs=True,
            gamma=0.99,
            use_novelty_reward=True,
            novelty_bonus=0.03,
            use_room_reward=True,
            room_bonus=0.25,
            forward_bonus=0.0,
            spin_penalty=0.0,
            spin_threshold=12,
            use_bfs_distance=True,
            grid_resolution=0.5,
            use_stagnation_penalty=True,
            stagnation_penalty=-0.5,
            stagnation_threshold=3,
            stagnation_precision=0.2,
        )

        self.env = ActionRepeatWrapper(self.env, repeat=1)

        # Визуальные искажения
        self.use_perturbation = perturbation != 'none'
        if self.use_perturbation:
            mode = "naive" if perturbation == "naive_dr" else "progressive"
            self.env = PerturbationWrapper(
                self.env,
                mode=mode,
                severity=severity,
                total_episodes=1000
            )
            print(f"🔧 Perturbation: {perturbation} (mode={mode}, severity={severity})")

        self.env = MultiModalObservationWrapper(self.env)
        

        print("\n✅ Reward Shaping активирован:")
        print("   • PBRS (BFS distance)")
        print("   • Time penalty: 0.0")
        print("   • Goal bonus: +10.0")
        print("   • Novelty bonus: 0.03")
        print("   • Spin penalty: 0.00")
        print("   • Forward bonus: 0.00")
        print()

        self.obs, self.info = self.env.reset(seed=42)

        # Получаем размер сетки из pathfinder (если есть)
        self.grid_size_str = "N/A"
        shaped_wrapper = find_wrapper(self.env, ShapedRewardWrapper)
        if shaped_wrapper is not None and shaped_wrapper.pathfinder is not None:
            b = shaped_wrapper.pathfinder.get_world_bounds()
            self.grid_size_str = f"{b['grid_width']}x{b['grid_height']}"
            print(f"📐 BFS grid: {self.grid_size_str}")

        self.total_reward = 0.0
        self.steps = 0
        self.episode_num = 1

        self.total_shaped = 0.0
        self.total_pbrs = 0.0

        # Первый кадр first‑person показываем сразу
        self._update_first_person_window()

        print("\n🎮 Управление: W/A/S/D | R=reset | Q=quit")
        print("💡 Смотрите консоль для детальной разбивки награды\n")

    def _update_first_person_window(self):
        """Получает текущий кадр от первого лица, применяет искажения и показывает в окне OpenCV."""
        try:
            unwrapped = self.env.unwrapped
            # Получаем сырой RGB массив из среды
            img = unwrapped.render()
            if img is None:
                return
            # Применяем искажения, если есть PerturbationWrapper
            pert_wrapper = find_wrapper(self.env, PerturbationWrapper)
            if pert_wrapper is not None:
                img = pert_wrapper._apply_perturbation(img)
            # Показываем в окне OpenCV (BGR для cv2)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow("First-Person View (with perturbations)", img_bgr)
            cv2.waitKey(1)  # небольшое ожидание для обновления окна
        except Exception as e:
            print(f"Ошибка обновления first-person окна: {e}")

    def get_top_down(self):
        """Возвращает top‑view изображение (вид сверху)."""
        try:
            unwrapped = self.env.unwrapped
            if hasattr(unwrapped, 'render_top_view'):
                img = unwrapped.render_top_view()
                if img is not None:
                    return img
            return None
        except Exception:
            return None

    def draw_panel(self):
        """Рисует панель информации и top‑view в окне pygame (без first‑person)."""
        self.screen.fill((30, 30, 30))

        x_text = 10
        y_text = 10
        line_h = 24

        info_lines = [
            f"Episode: {self.episode_num}",
            f"Step: {self.steps}",
            f"Grid: {self.grid_size_str}",
            f"Total Reward: {self.total_reward:.2f}",
        ]
        if 'original_reward' in self.info:
            info_lines.append(f"Original: {self.info['original_reward']:+.2f}")
        if 'shaped_reward' in self.info:
            info_lines.append(f"Shaped: {self.info['shaped_reward']:+.2f}")
        if 'pbrs_reward' in self.info:
            bfs_dist = self.info.get('bfs_distance', 0)
            info_lines.append(f"PBRS: {self.info['pbrs_reward']:+.3f} (dist={bfs_dist:.1f})")
        if 'spin_penalty' in self.info:
            turns = self.info.get('consecutive_turns', 0)
            info_lines.append(f"⚠️ Spin: {self.info['spin_penalty']:+.2f} (turns={turns})")
        if self.use_perturbation:
            sev = self.info.get('perturbation_severity', 0)
            info_lines.append(f"Pert Severity: {sev:.2f}")

        for line in info_lines:
            color = (255, 255, 255)
            if line.startswith('⚠️'):
                color = (255, 100, 100)
            elif 'PBRS' in line:
                color = (100, 255, 100)
            surf = self.small_font.render(line, True, color)
            self.screen.blit(surf, (x_text, y_text))
            y_text += line_h

        y_text += 5
        pygame.draw.line(self.screen, (100, 100, 100), (x_text, y_text), (580, y_text), 1)
        y_text += 15

        ctrl_lines = [
            "Controls:",
            "W/S - forward/backward",
            "A/D - turn left/right",
            "R - reset | Q - quit",
        ]
        for line in ctrl_lines:
            color = (255, 255, 0) if line.endswith(':') else (200, 200, 200)
            surf = self.small_font.render(line, True, color)
            self.screen.blit(surf, (x_text, y_text))
            y_text += line_h

        y_text += 5
        legend_lines = [
            "Reward Shaping:",
            "🟢 PBRS = приближение к цели",
            "🔴 Spin = кручение на месте",
        ]
        for line in legend_lines:
            if line.startswith('🟢'):
                color = (100, 255, 100)
            elif line.startswith('🔴'):
                color = (255, 100, 100)
            else:
                color = (255, 255, 0)
            surf = self.small_font.render(line, True, color)
            self.screen.blit(surf, (x_text, y_text))
            y_text += line_h

        # Top‑view (вид сверху)
        top_img = self.get_top_down()
        if top_img is not None:
            try:
                if top_img.ndim == 3:
                    top_img = np.transpose(top_img, (1, 0, 2))
                    surf_top = pygame.surfarray.make_surface(top_img)
                else:
                    surf_top = pygame.surfarray.make_surface(top_img)
                # Масштабируем, чтобы поместилось внизу
                max_w, max_h = 380, 200
                scale = min(max_w / surf_top.get_width(), max_h / surf_top.get_height())
                new_w = int(surf_top.get_width() * scale)
                new_h = int(surf_top.get_height() * scale)
                surf_top = pygame.transform.scale(surf_top, (new_w, new_h))
                # Размещаем внизу по центру
                x_top = (self.screen.get_width() - new_w) // 2
                y_top = self.screen.get_height() - new_h - 10
                self.screen.blit(surf_top, (x_top, y_top))
            except Exception as e:
                print(f"Ошибка отображения top-down: {e}")

        pygame.display.flip()

    def run(self):
        running = True
        clock = pygame.time.Clock()

        while running:
            action = None

            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN:
                    if event.key == K_q or event.key == K_ESCAPE:
                        running = False
                    elif event.key == K_r:
                        self.reset()
                    elif event.key == K_w:
                        action = ACTION_MOVE_FORWARD
                    elif event.key == K_s:
                        action = ACTION_MOVE_BACK
                    elif event.key == K_a:
                        action = ACTION_TURN_LEFT
                    elif event.key == K_d:
                        action = ACTION_TURN_RIGHT
                    elif event.key == K_p and self.use_perturbation:
                        self.cycle_perturbation()
                    elif event.key == K_EQUALS or event.key == K_PLUS:
                        self.change_severity(0.1)
                    elif event.key == K_MINUS:
                        self.change_severity(-0.1)

            if action is not None:
                self.step(action)

            self.draw_panel()
            clock.tick(30)

        self.close()

    def step(self, action):
        self.obs, reward, terminated, truncated, self.info = self.env.step(action)
        self.total_reward += reward
        self.steps += 1

        original = self.info.get('original_reward', 0)
        shaped = self.info.get('shaped_reward', 0)
        pbrs = self.info.get('pbrs_reward', 0.0)

        self.total_shaped += shaped
        self.total_pbrs += pbrs
        spin_val = self.info.get('spin_penalty', 0)

        #print("Форма изображения:", self.obs['image'].shape)  # должно быть (C, H, W)
        #print("Диапазон вектора:", self.obs['vector'].min(), self.obs['vector'].max())

        print("Вектор:", self.obs['vector'])                 # 4 числа
        
        print(f"Step {self.steps:3d}: Total={reward:+.2f} | "
              f"Orig={original:+.2f} | "
              f"Shaped={shaped:+.3f} | "
              f"PBRS={pbrs:+.3f} | "
              f"∑Shaped={self.total_shaped:+.3f} | "
              f"∑PBRS={self.total_pbrs:+.3f} | "
              f"SPIN={spin_val:+.2f}")

        # Обновляем окно first‑person после каждого шага
        self._update_first_person_window()

        if terminated or truncated:
            success = original > 0.1
            visited = self.info.get('visited_states_count', 0)
            rooms = self.info.get('visited_rooms_count', 0)

            print(f"\n{'✅ SUCCESS!' if success else '❌ TIMEOUT/FAILED'}")
            print(f"   Total reward: {self.total_reward:.3f}")
            print(f"   Total Shaped: {self.total_shaped:+.3f}")
            print(f"   Total PBRS:   {self.total_pbrs:+.3f}")
            print(f"   Steps: {self.steps}")
            print(f"   Visited cells: {visited}")
            print(f"   Visited rooms: {rooms}")

            self.reset()

    def reset(self):
        self.obs, self.info = self.env.reset()
        self.total_reward = 0.0
        self.steps = 0
        self.episode_num += 1

        self.total_shaped = 0.0
        self.total_pbrs = 0.0

        # Обновить размер сетки
        shaped_wrapper = find_wrapper(self.env, ShapedRewardWrapper)
        if shaped_wrapper is not None and shaped_wrapper.pathfinder is not None:
            b = shaped_wrapper.pathfinder.get_world_bounds()
            self.grid_size_str = f"{b['grid_width']}x{b['grid_height']}"
        else:
            self.grid_size_str = "N/A"

        # Обновить first‑person окно после ресета
        self._update_first_person_window()

        print(f"\n🔄 Episode {self.episode_num} started (grid {self.grid_size_str})\n")

    def cycle_perturbation(self):
        print("⚠️ Cycle perturbation not implemented (only manual severity change via +/-)")

    def change_severity(self, delta):
        if not self.use_perturbation:
            return
        pert_wrapper = find_wrapper(self.env, PerturbationWrapper)
        if pert_wrapper is None:
            print("Не найден PerturbationWrapper")
            return
        new_sev = float(np.clip(pert_wrapper.current_severity + delta, 0.0, 1.0))
        pert_wrapper.current_severity = new_sev
        pert_wrapper.perturbation_manager.set_severity(new_sev)
        print(f"Severity: {new_sev:.2f}")
        # Обновляем отображение после изменения severity
        self._update_first_person_window()

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
        pygame.quit()
        print("\n👋 Завершено")


def main():
    parser = argparse.ArgumentParser(
        description="MiniWorld Manual Control with Reward Shaping and Visual Perturbations (two windows)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls:
  W/S - forward/backward
  A/D - turn left/right
  R - reset | Q - quit
  +/- - увеличить/уменьшить severity (если активны искажения)
        """
    )
    parser.add_argument('--rows', type=int, default=4, help='Количество рядов комнат (лабиринт)')
    parser.add_argument('--cols', type=int, default=4, help='Количество колонок комнат')
    parser.add_argument('--domain-rand', action='store_true', help='Включить domain randomisation')
    parser.add_argument('--perturbation', type=str, default='none',
                        choices=['none', 'naive_dr', 'progressive'],
                        help='Тип визуальных искажений')
    parser.add_argument('--severity', type=float, default=0.5,
                        help='Начальная степень искажений (0..1)')

    args = parser.parse_args()

    ctrl = ManualController(
        num_rows=args.rows,
        num_cols=args.cols,
        domain_rand=args.domain_rand,
        perturbation=args.perturbation,
        severity=args.severity,
    )

    try:
        ctrl.run()
    except KeyboardInterrupt:
        print("\n⚠️ Прервано")
    finally:
        ctrl.close()


if __name__ == '__main__':
    main()