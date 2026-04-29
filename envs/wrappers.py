"""
Wrappers для SB3 с улучшенными Shaped Rewards:
- PBRS (Potential-based reward shaping) с BFS расстоянием — ВЫДАЁТСЯ ОТДЕЛЬНО
- Novelty/exploration reward
- Room-level reward
- Spin penalty (штраф за кручение на месте)
- Forward bonus (бонус за движение вперёд)
- Action Repeat (повтор действий агента)
- Stagnation Penalty (штраф за застревание на месте)
- MultiModalObservationWrapper: добавляет вектор [x_norm, z_norm, sinθ, cosθ] к изображению -> Dict
- RayCastingWrapper: добавляет вектор расстояний (8 лучей, FOV=75°, max_dist=10)
"""
import gymnasium as gym
import numpy as np
from typing import Optional, Dict, Any, Tuple
from perturbations.visual_perturbations import PerturbationManager
import math
from collections import deque


class PerturbationWrapper(gym.Wrapper):
    """
    Wrapper для визуальных помех.

    Режимы (mode):
        - "none":    никаких помех, domain_rand НЕ включается (если только не задан enable_domain_rand)
        - "fixed":   фиксированная тяжесть severity (не меняется между эпизодами)
        - "naive":   случайная тяжесть от 0 до severity в каждом эпизоде
        - "progressive": тяжесть растёт квадратично от 0 до severity с номером эпизода

    Параметр enable_domain_rand управляет включением domain_rand у среды независимо от mode.
    Если mode != "none" и enable_domain_rand=True, то domain_rand = True.

    Адаптирован для работы со словарём наблюдений: искажает только ключ 'image' (или 'rgb').
    """

    def __init__(
        self,
        env: gym.Env,
        mode: str = "none",
        severity: float = 0.0,
        total_episodes: int = 1000,
        enable_domain_rand: bool = False,
    ):
        super().__init__(env)
        self.mode = mode
        self.base_severity = severity
        self.total_episodes = total_episodes
        self.enable_domain_rand = enable_domain_rand
        
        self.episode_count = 0

        # Включаем domain_rand, только если явно разрешено и режим не "none"
        if self.enable_domain_rand:
            self.env.unwrapped.domain_rand = True

        # Текущая тяжесть: для progressive будет меняться, для fixed/naive/none устанавливаем сразу
        if mode == "progressive":
            self.current_severity = 0.0
        else:
            self.current_severity = severity

        self.perturbation_manager = PerturbationManager(
            config="mixed" if mode != "none" else "none",
            severity=self.current_severity
        )

        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.episode_count += 1

        # Обновляем тяжесть в зависимости от режима
        if self.mode == "progressive":
            self._update_progressive_severity()
        elif self.mode == "naive":
            self.current_severity = self.perturbation_manager.get_random_severity(
                0, self.base_severity
            )
            self.perturbation_manager.set_severity(self.current_severity)
        elif self.mode == "fixed":
            # Фиксированная тяжесть, просто убеждаемся, что установлена
            self.current_severity = self.base_severity
            self.perturbation_manager.set_severity(self.current_severity)

        info['perturbation_severity'] = self.current_severity if self.mode != "none" else 0.0
        obs = self._apply_perturbation(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._apply_perturbation(obs)
        info['perturbation_severity'] = self.current_severity if self.mode != "none" else 0.0
        return obs, reward, terminated, truncated, info

    def _apply_perturbation(self, obs):
        """
        Применяет визуальные искажения к наблюдению.
        Поддерживает как raw-изображение (numpy array), так и словарь.
        """
        if isinstance(obs, dict):
            # Ищем ключ с изображением
            key = None
            if 'image' in obs:
                key = 'image'
            elif 'rgb' in obs:
                key = 'rgb'
            else:
                # Если нет явного ключа, пробуем взять первый, который является массивом
                for k, v in obs.items():
                    if isinstance(v, np.ndarray) and v.ndim >= 2:
                        key = k
                        break
            if key is not None:
                obs[key] = self.perturbation_manager(obs[key].copy())
            return obs
        else:
            # Предполагаем, что obs — это изображение (numpy array)
            return self.perturbation_manager(obs.copy())

    def _update_progressive_severity(self) -> None:
        progress = min(1.0, self.episode_count / self.total_episodes)
        self.current_severity = self.base_severity * (progress ** 2)
        # Обновляем менеджер, чтобы новые искажения применялись со следующего шага
        self.perturbation_manager.set_severity(self.current_severity)


class ActionRepeatWrapper(gym.Wrapper):
    """
    Action Repeat (Повтор действий): повторяет действие агента N раз за один вызов step().

    Ключевые особенности для интеграции с ShapedRewardWrapper:
    - Повторяет action N раз в среде (тики среды)
    - shaped-награды (time, spin, forward, stagnation и т.д.) применяются 
      один раз за решение агента, а не за каждый тик среды
    - PBRS (distance-based) вычисляется по финальному состоянию после всех повторов
    - original_reward суммируется за все тики
    - info['action_repeat_ticks'] = сколько тиков реально выполнено

    Это позволяет агенту "сразу идти на 4 шага вперёд" или "сразу поворачивать ~90°",
    при этом штраф/бонус за действие считается как за одно решение.
    """

    def __init__(self, env: gym.Env, repeat: int = 4):
        super().__init__(env)
        if repeat < 1:
            raise ValueError(f"repeat must be >= 1, got {repeat}")
        self.repeat = repeat
        # action_space не меняется — агент принимает решения с той же частотой

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        """
        Выполняет action repeat раз.

        shaped-награды внутри ShapedRewardWrapper будут выданы один раз за 
        агентское решение, т.к. ActionRepeatWrapper вызывает env.step() repeat раз,
        но ShapedRewardWrapper.step() вызывается repeat раз — поэтому логика 
        подавления shaped-наград за тики реализована внутри ShapedRewardWrapper
        через флаг _is_action_repeat_tick.
        """
        total_original_reward = 0.0
        total_shaped_reward = 0.0
        total_pbrs_reward = 0.0

        obs = None
        terminated = False
        truncated = False
        info = {}

        actual_ticks = 0

        for tick in range(self.repeat):
            if terminated or truncated:
                break

            # Передаём action и получаем очередной тик среды
            obs, reward, terminated, truncated, info = self.env.step(action)
            actual_ticks += 1

            # Суммируем оригинальную награду (goal reward и т.д.)
            total_original_reward += info.get('original_reward', 0.0)

            # shaped_reward за этот тик — но ShapedRewardWrapper уже 
            # разделяет shaped vs PBRS. Мы собираем их отдельно.
            total_shaped_reward += info.get('shaped_reward', 0.0)
            total_pbrs_reward += info.get('pbrs_reward', 0.0)

        # Перезаписываем info для агента: показываем суммарные значения
        info['original_reward'] = total_original_reward
        info['shaped_reward'] = total_shaped_reward
        info['pbrs_reward'] = total_pbrs_reward
        info['action_repeat_ticks'] = actual_ticks
        info['action_repeat'] = self.repeat

        # total_reward уже содержит сумму от ShapedRewardWrapper за каждый тик,
        # но мы хотим, чтобы shaped-награды были применены один раз.
        # Поэтому total_reward пересчитывается: original + shaped (один раз) + pbrs (один раз)
        # НО: ShapedRewardWrapper уже начислил shaped за каждый тик. 
        # Чтобы shaped был один раз, ShapedRewardWrapper должен знать о ActionRepeat.
        # 
        # Решение: ShapedRewardWrapper проверяет info.get('action_repeat_ticks') 
        # и если он > 1, то shaped-награды (кроме PBRS) применяются только 
        # на первом тике, а на последующих — только original + PBRS.
        # Эта логика реализована в ShapedRewardWrapper.

        # Пересчитываем total_reward корректно:
        # original (сумма за все тики) + shaped (один раз) + pbrs (один раз)
        # Но поскольку ShapedRewardWrapper уже начислил shaped N раз,
        # мы корректируем: убираем лишние shaped, оставляем один.
        # 
        # Проще: total_reward = total_original_reward + total_shaped_reward + total_pbrs_reward
        # где ShapedRewardWrapper уже позаботился о том, чтобы shaped был один раз.
        total_reward = total_original_reward + total_shaped_reward + total_pbrs_reward

        return obs, total_reward, terminated, truncated, info


class BFSPathfinder:
    """BFS для вычисления расстояний с обходом стен в MiniWorld.

    Два режима построения сетки:
    1. use_top_view=False (default): по geometry (entities + room walls)
    2. use_top_view=True: по render_top_view() — точная карта занятости

    Координатная система:
        - origin (0,0) — левый верхний угол
        - X (ось 1) — вправо, Z (ось 0) — вниз
        - Индексация: [Z, X]
        - Debug output: (X, Z) формат как в координатах сетки
    """

    def __init__(self, env, grid_resolution=0.1, use_top_view=False):
        self.env = env
        self.grid_resolution = grid_resolution
        self.use_top_view = use_top_view

        self.distance_map = None
        self.obstacle_map = None

        self.offset_x = 0.0
        self.offset_z = 0.0

        self.world_width = 0.0
        self.world_height = 0.0

        self.width = 0
        self.height = 0

        self.goal_grid_pos = None
        self._warned_clamp = False

    def _world_to_grid(self, pos, warn=True) -> Tuple[int, int]:
        """Нативные координаты [x,y,z] -> индексы сетки (X, Z)."""
        gx = int((pos[0] - self.offset_x) / self.grid_resolution)
        gz = int((pos[2] - self.offset_z) / self.grid_resolution)

        if warn and not self._warned_clamp:
            if gx < 0 or gx >= self.width or gz < 0 or gz >= self.height:
                print(f"[BFS WARNING] Position ({pos[0]:.2f},{pos[2]:.2f}) maps to "
                      f"grid ({gx},{gz}) but grid size is ({self.width},{self.height}). "
                      f"Clamping will occur!")
                self._warned_clamp = True

        gx = max(0, min(gx, self.width - 1))
        gz = max(0, min(gz, self.height - 1))

        return gx, gz

    def build_grid(self, goal_pos=None, agent_pos=None):
        """Создаёт карту препятствий."""
        if self.use_top_view:
            self._build_grid_from_top_view()
        else:
            self._build_grid_from_geometry(goal_pos, agent_pos)

    def _build_grid_from_top_view(self):
        """Строит сетку по render_top_view() — точная визуальная карта."""
        unwrapped = self.env.unwrapped

        # Получаем top-view (вызываем ОДИН раз)
        top_view = unwrapped.render_top_view()

        # Определяем размеры мира из env
        world_width = getattr(unwrapped, 'width', None)
        world_height = getattr(unwrapped, 'height', None)

        if world_width is None or world_height is None:
            raise ValueError("use_top_view requires env.width and env.height")

        self.world_width = float(world_width)
        self.world_height = float(world_height)
        self.offset_x = 0.0
        self.offset_z = 0.0

        # Вычисляем размер сетки
        self.width = int(self.world_width / self.grid_resolution) + 1
        self.height = int(self.world_height / self.grid_resolution) + 1

        # Преобразуем top_view в obstacle_map
        # top_view обычно HxWx3 (RGB), иногда HxW (grayscale)
        if len(top_view.shape) == 3:
            # Конвертируем RGB в grayscale (стены темнее)
            gray = np.mean(top_view, axis=2)
        else:
            gray = top_view

        # Масштабируем до размера сетки
        from scipy.ndimage import zoom
        # zoom factors: (height_scale, width_scale)
        zoom_factor = (self.height / gray.shape[0], self.width / gray.shape[1])
        resized = zoom(gray, zoom_factor, order=1)  # bilinear interpolation

        # Определяем стены по порогу яркости
        # В MiniWorld: пол светлый (~200), стены темнее (~100 или цветные)
        threshold = np.percentile(resized, 50)  # адаптивный порог
        self.obstacle_map = (resized < threshold).astype(np.int8)

        # Дилатация — расширяем стены на 1 ячейку для безопасности
        self._dilate_obstacles(radius=1)

        print(f"[BFS] Grid from top_view: {self.width}x{self.height}, "
              f"world=({self.world_width:.2f},{self.world_height:.2f}), "
              f"obstacles={np.sum(self.obstacle_map)}")

    def _build_grid_from_geometry(self, goal_pos=None, agent_pos=None):
        """Строит сетку по геометрии мира: entities + room walls/edges.

        В MiniWorld лабиринты (Maze) строятся из Room объектов, где стены — это
        границы комнат (outline), а не отдельные entities.
        """
        unwrapped = self.env.unwrapped
        entities = getattr(unwrapped, 'entities', [])
        rooms = getattr(unwrapped, 'rooms', [])

        # Жёсткие границы мира
        world_width = getattr(unwrapped, 'width', None)
        world_height = getattr(unwrapped, 'height', None)

        if world_width is not None and world_height is not None and world_width > 0 and world_height > 0:
            self.offset_x = 0.0
            self.offset_z = 0.0
            self.world_width = float(world_width)
            self.world_height = float(world_height)
        else:
            # Fallback: вычисляем по entities и rooms с padding
            min_x = min_z = float('inf')
            max_x = max_z = float('-inf')

            # Учитываем entities
            for e in entities:
                if hasattr(e, 'pos'):
                    x, _, z = e.pos
                    min_x = min(min_x, x); max_x = max(max_x, x)
                    min_z = min(min_z, z); max_z = max(max_z, z)

            # Учитываем rooms (границы комнат)
            for room in rooms:
                if hasattr(room, 'min_x'):
                    min_x = min(min_x, room.min_x)
                    max_x = max(max_x, room.max_x)
                    min_z = min(min_z, room.min_z)
                    max_z = max(max_z, room.max_z)

            if goal_pos is not None:
                gx, _, gz = goal_pos
                min_x = min(min_x, gx); max_x = max(max_x, gx)
                min_z = min(min_z, gz); max_z = max(max_z, gz)
            if agent_pos is not None:
                ax, _, az = agent_pos
                min_x = min(min_x, ax); max_x = max(max_x, ax)
                min_z = min(min_z, az); max_z = max(max_z, az)

            pad = 5.0
            min_x -= pad; max_x += pad
            min_z -= pad; max_z += pad

            self.offset_x = min_x
            self.offset_z = min_z
            self.world_width = max_x - min_x
            self.world_height = max_z - min_z

        # Создаём сетку
        self.width = int(self.world_width / self.grid_resolution) + 1
        self.height = int(self.world_height / self.grid_resolution) + 1

        assert self.width > 0 and self.height > 0, (
            f"Grid dimensions must be positive, got {self.width}x{self.height}"
        )

        self.obstacle_map = np.zeros((self.height, self.width), dtype=np.int8)

        # Размечаем стены из комнат (Room.walls / Room.outline)
        # В MiniWorld стены лабиринта — это границы комнат
        for room in rooms:
            self._mark_room_walls(room)

        # Размечаем препятствия из entities
        for entity in entities:
            if not hasattr(entity, 'pos'):
                continue

            if goal_pos is not None and np.allclose(entity.pos, goal_pos, atol=1e-3):
                continue

            entity_type = type(entity).__name__
            is_wall = (
                entity_type in ['Wall', 'Box', 'MeshEnt', 'Entity']
                or (hasattr(entity, 'is_static') and entity.is_static)
                or (hasattr(entity, 'name') and 'wall' in str(entity.name).lower())
            )

            if not is_wall:
                continue

            self._mark_obstacle(entity)

        #obstacle_count = int(np.sum(self.obstacle_map))
        #obstacle_ratio = float(np.mean(self.obstacle_map))
        #print(f"[BFS] Grid from geometry: {self.width}x{self.height}, "
        #      f"offset=({self.offset_x:.2f},{self.offset_z:.2f}), "
        #      f"rooms={len(rooms)}, entities={len(entities)}, "
        #      f"obstacles={obstacle_count}, ratio={obstacle_ratio:.4f}")

    def _mark_room_walls(self, room):
        """Отмечает стены комнаты на сетке по её outline.

        Room.outline содержит вершины многоугольника комнаты.
        Каждое ребро между consecutive vertices — это стена.
        Порталы (portals) — проёмы в стенах.
        """
        if not hasattr(room, 'outline') or room.outline is None:
            return

        outline = room.outline
        num_walls = room.num_walls if hasattr(room, 'num_walls') else len(outline)
        portals = getattr(room, 'portals', [[] for _ in range(num_walls)])

        # Для каждой стены (ребра) комнаты
        for wall_idx in range(num_walls):
            p0 = outline[wall_idx]
            p1 = outline[(wall_idx + 1) % num_walls]

            # Проверяем, есть ли портал в этой стене
            wall_portals = portals[wall_idx] if wall_idx < len(portals) else []

            if wall_portals:
                # Если есть порталы, разбиваем стену на сегменты
                # и маркируем только не-портальные части
                self._mark_wall_with_portals(p0, p1, wall_portals)
            else:
                # Нет порталов — маркируем всю стену
                self._mark_wall_segment(p0, p1)

    def _mark_wall_segment(self, p0, p1, thickness=None):
        """Маркирует отрезок стены на сетке толщиной в 1 ячейку."""
        if thickness is None:
            thickness = self.grid_resolution

        x0, z0 = p0[0], p0[2]
        x1, z1 = p1[0], p1[2]

        # Определяем bounding box сегмента
        min_x = min(x0, x1) - thickness / 2
        max_x = max(x0, x1) + thickness / 2
        min_z = min(z0, z1) - thickness / 2
        max_z = max(z0, z1) + thickness / 2

        # Конвертируем в grid coordinates
        gx_min = int((min_x - self.offset_x) / self.grid_resolution)
        gx_max = int((max_x - self.offset_x) / self.grid_resolution)
        gz_min = int((min_z - self.offset_z) / self.grid_resolution)
        gz_max = int((max_z - self.offset_z) / self.grid_resolution)

        # Ограничиваем границами массива
        gx_min = max(0, min(gx_min, self.width - 1))
        gx_max = max(0, min(gx_max, self.width - 1))
        gz_min = max(0, min(gz_min, self.height - 1))
        gz_max = max(0, min(gz_max, self.height - 1))

        if gx_min <= gx_max and gz_min <= gz_max:
            self.obstacle_map[gz_min:gz_max+1, gx_min:gx_max+1] = 1

    def _mark_wall_with_portals(self, p0, p1, portals):
        """Маркирует стену с учётом порталов (проёмов)."""
        x0, z0 = p0[0], p0[2]
        x1, z1 = p1[0], p1[2]

        # Длина стены
        wall_len = np.linalg.norm([x1 - x0, z1 - z0])
        if wall_len < 1e-6:
            return

        # Направление стены
        dx = (x1 - x0) / wall_len
        dz = (z1 - z0) / wall_len

        # Сортируем порталы по позиции
        sorted_portals = sorted(portals, key=lambda p: p.get('start_pos', 0))

        # Маркируем сегменты между порталами
        current_pos = 0.0

        for portal in sorted_portals:
            start_pos = portal.get('start_pos', 0)
            end_pos = portal.get('end_pos', wall_len)

            # Маркируем сегмент до портала
            if start_pos > current_pos:
                seg_p0 = np.array([x0 + current_pos * dx, 0, z0 + current_pos * dz])
                seg_p1 = np.array([x0 + start_pos * dx, 0, z0 + start_pos * dz])
                self._mark_wall_segment(seg_p0, seg_p1)

            current_pos = end_pos

        # Маркируем сегмент после последнего портала
        if current_pos < wall_len:
            seg_p0 = np.array([x0 + current_pos * dx, 0, z0 + current_pos * dz])
            seg_p1 = np.array([x1, 0, z1])
            self._mark_wall_segment(seg_p0, seg_p1)

    def _mark_obstacle(self, entity, expansion=0.0):
        """Отмечает препятствие на сетке [z, x]."""
        x, _, z = entity.pos

        w = getattr(entity, 'width', 0.0)
        d = getattr(entity, 'depth', 0.0)

        # Минимальный размер — 1 ячейка
        if w < self.grid_resolution:
            w = self.grid_resolution
        if d < self.grid_resolution:
            d = self.grid_resolution

        w += expansion * 2
        d += expansion * 2

        gx_min = int((x - w/2 - self.offset_x) / self.grid_resolution)
        gx_max = int((x + w/2 - self.offset_x) / self.grid_resolution)
        gz_min = int((z - d/2 - self.offset_z) / self.grid_resolution)
        gz_max = int((z + d/2 - self.offset_z) / self.grid_resolution)

        # Гарантия минимум 1 ячейка
        if gx_min == gx_max:
            gx_max = gx_min + 1
        if gz_min == gz_max:
            gz_max = gz_min + 1

        gx_min = max(0, gx_min)
        gx_max = min(self.width - 1, gx_max)
        gz_min = max(0, gz_min)
        gz_max = min(self.height - 1, gz_max)

        self.obstacle_map[gz_min:gz_max+1, gx_min:gx_max+1] = 1

    def _dilate_obstacles(self, radius=1):
        """Расширяет препятствия на radius ячеек."""
        if radius <= 0:
            return
        from scipy.ndimage import binary_dilation
        self.obstacle_map = binary_dilation(
            self.obstacle_map, 
            iterations=radius
        ).astype(np.int8)

    def compute_distances(self, goal_pos):
        """BFS от цели."""
        agent_pos = None
        unwrapped = self.env.unwrapped
        if hasattr(unwrapped, 'agent') and hasattr(unwrapped.agent, 'pos'):
            agent_pos = unwrapped.agent.pos

        self.build_grid(goal_pos=goal_pos, agent_pos=agent_pos)
        goal_x, goal_z = self._world_to_grid(goal_pos, warn=False)

        if 0 <= goal_z < self.height and 0 <= goal_x < self.width:
            if self.obstacle_map[goal_z, goal_x] == 1:
                #print(f"[BFS] Goal inside obstacle at ({goal_x},{goal_z}), searching nearest free...")
                goal_x, goal_z = self._find_nearest_free(goal_x, goal_z)

        if not (0 <= goal_x < self.width and 0 <= goal_z < self.height):
            #print(f"[BFS] Goal out of grid: ({goal_x},{goal_z}), shape=({self.width},{self.height})")
            return

        self.distance_map = np.full((self.height, self.width), np.inf)
        queue = deque([(goal_z, goal_x)])
        self.distance_map[goal_z, goal_x] = 0

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            z, x = queue.popleft()
            current_dist = self.distance_map[z, x]
            for dz, dx in directions:
                nz, nx = z + dz, x + dx
                if 0 <= nz < self.height and 0 <= nx < self.width:
                    if self.obstacle_map[nz, nx] == 0 and self.distance_map[nz, nx] == np.inf:
                        self.distance_map[nz, nx] = current_dist + 1
                        queue.append((nz, nx))

        self.goal_grid_pos = (goal_x, goal_z)

    def _find_nearest_free(self, gx, gz):
        """Находит ближайшую свободную ячейку."""
        if self.obstacle_map[gz, gx] == 0:
            return gx, gz
        h, w = self.obstacle_map.shape
        queue = deque([(gz, gx)])
        visited = {(gz, gx)}
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            z, x = queue.popleft()
            for dz, dx in directions:
                nz, nx = z + dz, x + dx
                if 0 <= nz < h and 0 <= nx < w and (nz, nx) not in visited:
                    if self.obstacle_map[nz, nx] == 0:
                        return nx, nz
                    visited.add((nz, nx))
                    queue.append((nz, nx))
        return gx, gz

    def get_distance(self, pos):
        """
        Возвращает BFS-расстояние от позиции pos до цели.
        Если позиция находится в препятствии или вне сетки,
        возвращается максимальное расстояние + штраф.
        """
        if self.distance_map is None:
            return None

        max_grid_dist = self.get_max_distance()  # width + height

        # Конвертируем позицию в координаты сетки
        gx, gz = self._world_to_grid(pos)

        # Если координаты вышли за границы, возвращаем max_dist + penalty
        if not (0 <= gx < self.width and 0 <= gz < self.height):
            return max_grid_dist + 10.0

        # Если агент в стене, "выталкиваем" его к ближайшей свободной ячейке
        if self.obstacle_map[gz, gx] == 1:
            gx, gz = self._find_nearest_free(gx, gz)

        dist = self.distance_map[gz, gx]
        if dist != np.inf:
            return dist
        else:
            # Недостижимая точка (например, изолированная комната)
            return max_grid_dist + 10.0

    def get_max_distance(self):
        """Максимальное расстояние для нормализации — размер лабиринта.

        Использует диагональ сетки (width + height) для нормализации,
        что даёт стабильное значение независимо от позиции старта.
        """
        if self.distance_map is None:
            return None
        # Нормализация по размеру сетки, а не по стартовой позиции
        return float(self.width + self.height)

    def get_grid_coords(self, pos):
        """Целочисленные координаты сетки (X, Z) для отладки.

        Возвращает координаты в формате (X, Z) как в декартовой системе.
        """
        gx, gz = self._world_to_grid(pos)
        return (gx, gz)

    def get_world_bounds(self):
        """Границы мира для отладки."""
        return {
            'offset_x': self.offset_x,
            'offset_z': self.offset_z,
            'world_width': self.world_width,
            'world_height': self.world_height,
            'grid_width': self.width,
            'grid_height': self.height,
        }


class ShapedRewardWrapper(gym.Wrapper):
    """
    Shaped rewards с PBRS.

    ВАЖНО: shaped_reward = всё КРОМЕ PBRS (time, novelty, room, spin, forward, 
           stagnation, goal, truncation)
           pbrs_reward = только PBRS (выдаётся отдельно в info)
           total_reward = original + shaped_reward + pbrs_reward

    Action Repeat интеграция:
    - Если wrapper обёрнут в ActionRepeatWrapper, shaped-награды (кроме PBRS)
      применяются ТОЛЬКО на первом тике повтора. На последующих тиках — 
      только original_reward и PBRS.
    - Это гарантирует, что агент "штрафуется/бонусится" за одно решение,
      а не за каждый тик среды.
    """

    def __init__(
        self,
        env: gym.Env,
        time_penalty: float = -0.02,
        distance_reward_coef: float = 10.0,
        goal_bonus: float = 30.0,
        max_episode_steps: int = 800,
        use_pbrs: bool = True,
        gamma: float = 0.99,
        use_novelty_reward: bool = True,
        novelty_bonus: float = 0.07,
        state_precision: float = 0.35,
        use_room_reward: bool = True,
        room_bonus: float = 0.5,
        spin_penalty: float = -0.01,
        forward_bonus: float = 0.01,
        spin_threshold: int = 4,
        use_bfs_distance: bool = True,
        grid_resolution: float = 0.35,
        use_top_view: bool = False,
        # Stagnation Penalty
        use_stagnation_penalty: bool = True,
        stagnation_penalty: float = -0.05,
        stagnation_threshold: int = 8,
        stagnation_precision: float = 0.25,
    ):
        super().__init__(env)
        self.time_penalty = time_penalty
        self.distance_reward_coef = distance_reward_coef
        self.goal_bonus = goal_bonus
        self.max_episode_steps = max_episode_steps
        self.use_pbrs = use_pbrs
        self.gamma = gamma
        self.use_novelty_reward = use_novelty_reward
        self.novelty_bonus = novelty_bonus
        self.state_precision = state_precision
        self.use_room_reward = use_room_reward
        self.room_bonus = room_bonus
        self.spin_penalty = spin_penalty
        self.forward_bonus = forward_bonus
        self.spin_threshold = spin_threshold
        self.use_bfs_distance = use_bfs_distance
        self.grid_resolution = grid_resolution
        self.use_top_view = use_top_view

        # Stagnation Penalty
        self.use_stagnation_penalty = use_stagnation_penalty
        self.stagnation_penalty = stagnation_penalty
        self.stagnation_threshold = stagnation_threshold
        self.stagnation_precision = stagnation_precision

        self.prev_dist = None
        self.episode_step = 0
        self.visited_states = set()
        self.visited_rooms = set()
        self.prev_room = None
        self.consecutive_turns = 0
        self.prev_pos = None
        self.pathfinder = None

        # Stagnation tracking 
        self.stagnation_steps = 0
        self.stagnation_prev_cell = None

    def reset(self, **kwargs):
        self.episode_step = 0
        self.visited_states.clear()
        self.visited_rooms.clear()
        self.prev_room = None
        self.consecutive_turns = 0
        self.prev_pos = None

        # Reset stagnation 
        self.stagnation_steps = 0
        self.stagnation_prev_cell = None

        obs, info = self.env.reset(**kwargs)

        unwrapped = self.env.unwrapped
        has_agent = hasattr(unwrapped, 'agent') and hasattr(unwrapped.agent, 'pos')
        has_goal = hasattr(unwrapped, 'box') and unwrapped.box.pos is not None

        if self.use_bfs_distance and has_agent and has_goal:
            agent_pos = unwrapped.agent.pos
            goal_pos = unwrapped.box.pos

            self.pathfinder = BFSPathfinder(
                self.env, 
                self.grid_resolution,
                use_top_view=self.use_top_view
            )
            self.pathfinder.compute_distances(goal_pos)

            self.prev_dist = self.pathfinder.get_distance(agent_pos)
            max_dist = self.pathfinder.get_max_distance()
            eucl_dist = self._euclidean_dist(agent_pos, goal_pos)
            

            # Проверка на недостижимость: если расстояние больше max_grid_dist * 2, fallback на евклида
            if self.prev_dist is not None and self.prev_dist > max_dist * 2:
                print("[BFS] Warning: BFS distance seems unreachable, falling back to Euclidean scaled")
                self.prev_dist = eucl_dist / self.grid_resolution
        else:
            if has_agent and has_goal:
                agent_pos = unwrapped.agent.pos
                goal_pos = unwrapped.box.pos
                self.prev_dist = self._euclidean_dist(agent_pos, goal_pos)
            else:
                self.prev_dist = None

        if self.use_room_reward and has_agent:
            self.prev_room = self._get_current_room()
            if self.prev_room is not None:
                self.visited_rooms.add(self.prev_room)

        if has_agent:
            self.prev_pos = unwrapped.agent.pos.copy()
            # === Init stagnation cell ===
            self.stagnation_prev_cell = self._get_stagnation_cell(unwrapped.agent.pos)

        return obs, info

    def _euclidean_dist(self, agent_pos, goal_pos) -> float:
        """Евклидово расстояние."""
        return math.sqrt((agent_pos[0] - goal_pos[0])**2 + 
                        (agent_pos[2] - goal_pos[2])**2)

    def _get_current_room(self) -> Optional[int]:
        try:
            unwrapped = self.env.unwrapped
            agent_pos = unwrapped.agent.pos

            if hasattr(unwrapped, 'num_rows') and hasattr(unwrapped, 'room_size'):
                room_size = unwrapped.room_size
                room_x = int(agent_pos[0] / room_size)
                room_z = int(agent_pos[2] / room_size)
                return room_x * unwrapped.num_cols + room_z
            return None
        except:
            return None

    def _get_stagnation_cell(self, pos) -> Tuple[int, int]:
        """Возвращает ячейку для отслеживания застревания."""
        gx = int(round(pos[0] / self.stagnation_precision))
        gz = int(round(pos[2] / self.stagnation_precision))
        return (gx, gz)

    def step(self, action):
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        self.episode_step += 1

        # РАЗДЕЛЕНИЕ: shaped БЕЗ PBRS
        shaped_reward = 0.0   # всё КРОМЕ PBRS (time, novelty, room, spin, forward, stagnation, goal, truncation)
        pbrs_reward = 0.0     # только PBRS
        

        # Action Repeat Detection 
        # Если это повторный тик (не первый), shaped-награды (кроме PBRS) 
        # применяются только на первом тике.
        # ActionRepeatWrapper передаёт info['action_repeat_tick'] = tick_index (0-based)
        is_repeat_tick = info.get('action_repeat_tick', 0) > 0

        # Time penalty → shaped (НЕ PBRS)
        if not is_repeat_tick:
            shaped_reward += self.time_penalty

        unwrapped = self.env.unwrapped
        has_agent = hasattr(unwrapped, 'agent') and hasattr(unwrapped.agent, 'pos')

        # Spin penalty → shaped (НЕ PBRS)
        if has_agent and not is_repeat_tick:
            current_pos = unwrapped.agent.pos
            is_turn = (action == 0 or action == 1)

            if is_turn:
                self.consecutive_turns += 1
            else:
                self.consecutive_turns = 0

            if self.consecutive_turns >= self.spin_threshold:
                shaped_reward += self.spin_penalty
                info['spin_penalty'] = self.spin_penalty

            self.prev_pos = current_pos.copy() if current_pos is not None else None

        # Forward bonus → shaped (НЕ PBRS)
        if self.forward_bonus != 0 and not (action == 0 or action == 1) and not is_repeat_tick:
            shaped_reward += self.forward_bonus
            info['forward_bonus'] = self.forward_bonus

        # STAGNATION PENALTY 
        # Штраф если агент долго стоит в одной ячейке
        if self.use_stagnation_penalty and has_agent and not is_repeat_tick:
            agent_pos = unwrapped.agent.pos
            current_cell = self._get_stagnation_cell(agent_pos)

            if current_cell == self.stagnation_prev_cell:
                self.stagnation_steps += 1
            else:
                self.stagnation_steps = 0
                self.stagnation_prev_cell = current_cell

            if self.stagnation_steps >= self.stagnation_threshold:
                shaped_reward += self.stagnation_penalty
                info['stagnation_penalty'] = self.stagnation_penalty
                info['stagnation_steps'] = self.stagnation_steps

        # PBRS → ОТДЕЛЬНО, НЕ добавляем в shaped_reward
        # PBRS вычисляется по финальному состоянию после всех повторов
        if self.use_pbrs and has_agent:
            agent_pos = unwrapped.agent.pos
            goal_pos = unwrapped.box.pos
            eucl_dist = self._euclidean_dist(agent_pos, goal_pos)

            if self.use_bfs_distance and self.pathfinder is not None:
                current_dist = self.pathfinder.get_distance(agent_pos)
                dist_type = "BFS"

                # Soft fallback: если BFS вернул None или явно большое значение (масштабируем евклида)
                if current_dist is None:
                    # fallback на масштабированное евклидово расстояние
                    current_dist = eucl_dist / self.grid_resolution
                    dist_type = "EUCL(scaled)"
                else:
                    max_dist = self.pathfinder.get_max_distance()
                    # Если BFS расстояние подозрительно велико (> max_dist*2), используем масштабированного евклида
                    if current_dist > max_dist * 2:
                        current_dist = eucl_dist / self.grid_resolution
                        dist_type = "EUCL(scaled)"
            else:
                current_dist = eucl_dist
                dist_type = "EUCL"

            if current_dist is not None and self.prev_dist is not None:
                max_dist = self.pathfinder.get_max_distance() if self.pathfinder is not None else (current_dist + 10.0)
                prev_norm = self.prev_dist / max_dist
                curr_norm = current_dist / max_dist
                delta = prev_norm - self.gamma * curr_norm

                if abs(prev_norm - curr_norm) < 1e-6:
                    pbrs_reward = 0.0
                else: 
                    pbrs_reward = delta * self.distance_reward_coef

                # Clip
                pbrs_reward = np.clip(pbrs_reward, -1.0, 1.0)

                # PBRS сохраняем отдельно, НЕ добавляем в shaped_reward
                info['pbrs_reward'] = pbrs_reward
                info['bfs_distance'] = current_dist

                

            self.prev_dist = current_dist


            '''         
            # Debag info
            grid_coords = self.pathfinder.get_grid_coords(agent_pos)
            bounds = self.pathfinder.get_world_bounds()

            if self.episode_step % 2 == 0:
                print(f"[BFS] Start agent=({agent_pos[0]:.2f},{agent_pos[2]:.2f}), "
                    f"goal=({goal_pos[0]:.2f},{goal_pos[2]:.2f}), "
                    f"grid=({grid_coords[0]},{grid_coords[1]}), bfs_dist={self.prev_dist:.1f}, "
                    f"eucl_dist={eucl_dist:.1f}, max_bfs={max_dist:.1f}")
                #print(f"[BFS] Bounds: {bounds}")       '''                                                                                  
            

        # Novelty reward → shaped (НЕ PBRS)
        if self.use_novelty_reward and has_agent and not is_repeat_tick:
            agent_pos = unwrapped.agent.pos
            state_id = (
                round(agent_pos[0] / self.state_precision) * self.state_precision,
                round(agent_pos[2] / self.state_precision) * self.state_precision
            )
            if state_id not in self.visited_states:
                self.visited_states.add(state_id)
                shaped_reward += self.novelty_bonus
                info['novelty_reward'] = self.novelty_bonus
                info['visited_states_count'] = len(self.visited_states)

        # Room reward → shaped (НЕ PBRS)
        if self.use_room_reward and has_agent and not is_repeat_tick:
            current_room = self._get_current_room()
            if current_room is not None and current_room != self.prev_room:
                if current_room not in self.visited_rooms:
                    self.visited_rooms.add(current_room)
                    shaped_reward += self.room_bonus
                    info['room_reward'] = self.room_bonus
                    info['new_room'] = current_room
                self.prev_room = current_room
            info['visited_rooms_count'] = len(self.visited_rooms)

        # Goal bonus → shaped (НЕ PBRS)
        if original_reward > 0.1 and not is_repeat_tick:
            shaped_reward += self.goal_bonus
            info['goal_bonus'] = self.goal_bonus

        # Truncation penalty → shaped (НЕ PBRS)
        if truncated and original_reward < 0.1:
            shaped_reward += -5.0
            info['truncation_penalty'] = -5.0

        # total = original + shaped (без PBRS) + pbrs (отдельно)
        total_reward = original_reward + shaped_reward + pbrs_reward

        # shaped_reward = всё КРОМЕ PBRS
        info['shaped_reward'] = shaped_reward
        info['original_reward'] = original_reward
        info['episode_step'] = self.episode_step
        info['consecutive_turns'] = self.consecutive_turns

        return obs, total_reward, terminated, truncated, info
    


class DilatedFrameStack(gym.ObservationWrapper):
    """
    Dilated Frame Stack для схемы t, t-d, t-2d, ..., t-(n-1)d.
    По умолчанию: n_stack=4, dilation=2 -> кадры t, t-2, t-4, t-6.

    Важно:
    - Конкатенирует кадры по оси каналов (CHW или HWC, сохраняет формат входа).
    - Буфер хранит достаточно кадров, чтобы дотянуться до самого старого.
    - При сбросе (reset) буфер заполняется первым кадром.
    - Адаптирован для работы со словарём наблюдений: применяется к ключу 'image'.
    """
    def __init__(self, env: gym.Env, n_stack: int = 4, dilation: int = 2):
        super().__init__(env)
        self.n_stack = n_stack
        self.dilation = dilation
        # Длина буфера: (n_stack - 1) * dilation + 1
        self.buffer_len = (n_stack - 1) * dilation + 1
        self.buffer = deque(maxlen=self.buffer_len)

        # Определяем пространство наблюдения (может быть Box или Dict)
        if isinstance(env.observation_space, gym.spaces.Dict):
            # Предполагаем, что есть ключ 'image' с Box
            img_space = env.observation_space['image']
            old_shape = img_space.shape
            # Определяем порядок каналов: обычно (H, W, C) для MiniWorld
            assert len(old_shape) == 3, "Ожидается форма (H, W, C)"
            new_c = old_shape[2] * n_stack
            new_shape = (old_shape[0], old_shape[1], new_c)
            new_image_space = gym.spaces.Box(low=0, high=255, shape=new_shape, dtype=np.uint8)
            self.observation_space = gym.spaces.Dict({
                'image': new_image_space,
                'vector': env.observation_space['vector']  # сохраняем вектор
            })
        else:
            # Обычное изображение (Box)
            old_shape = env.observation_space.shape
            assert len(old_shape) == 3, "Ожидается форма (H, W, C)"
            new_c = old_shape[2] * n_stack
            new_shape = (old_shape[0], old_shape[1], new_c)
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=new_shape, dtype=np.uint8)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Заполняем буфер первым кадром (изображением)
        if isinstance(obs, dict):
            img = obs['image']
        else:
            img = obs
        self.buffer.clear()
        for _ in range(self.buffer_len):
            self.buffer.append(img.copy())  # копия, чтобы не мутировать
        new_obs = self._get_obs()
        if isinstance(obs, dict):
            new_obs = {**obs, 'image': new_obs}
        return new_obs, info

    def observation(self, obs):
        """Вызывается при каждом step."""
        if isinstance(obs, dict):
            img = obs['image']
        else:
            img = obs
        self.buffer.append(img.copy())
        new_img = self._get_obs()
        if isinstance(obs, dict):
            return {**obs, 'image': new_img}
        return new_img

    def _get_obs(self):
        """Выбирает кадры с шагом dilation из конца буфера."""
        # Индексы: -1 (t), -1-dilation (t-d), -1-2d, ...
        indices = [-(1 + i * self.dilation) for i in range(self.n_stack)]
        frames = [self.buffer[idx] for idx in indices]
        return np.concatenate(frames, axis=-1)  # конкатенация по каналам (H, W, C*N)


class MultiModalObservationWrapper(gym.ObservationWrapper):
    """
    Добавляет вектор признаков [x_norm, z_norm, sin(angle), cos(angle)] к наблюдению.
    Исходная среда должна возвращать изображение (H, W, C) или уже словарь с 'image'.
    Итоговое наблюдение: Dict{'image': (C, H, W), 'vector': (4,)}.

    Особенности MiniWorld:
    - Угол агента хранится в градусах (agent.dir)
    - Для нейросети переводим в радианы и берём sin/cos для гладкости
    - Координаты нормализуем по размерам мира (если недоступны, вычисляем по границам)
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Размеры мира (будут определены при первом reset)
        self.world_width = None
        self.world_height = None
        self._bounds_set = False

        # Определяем пространство изображения
        if isinstance(env.observation_space, gym.spaces.Dict):
            img_space = env.observation_space.get('image', env.observation_space)
        else:
            img_space = env.observation_space

        # Приводим к формату (C, H, W) для PyTorch (если ещё не)
        if len(img_space.shape) == 3 and img_space.shape[2] in [1,3,4]:   # C in RGB(3)
            h, w, c = img_space.shape
            self.image_space = gym.spaces.Box(low=0, high=255, shape=(c, h, w), dtype=np.uint8)
        else:
            self.image_space = img_space

        self.observation_space = gym.spaces.Dict({
            'image': self.image_space,
            'vector': gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        })

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # При первом сбросе определяем размеры мира
        if not self._bounds_set:
            self._set_world_bounds()
        return self._make_observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._make_observation(obs), reward, terminated, truncated, info

    def _set_world_bounds(self):
        # пустой return используется для выхода из функции после того, как границы мира успешно определены. 
        unwrapped = self.env.unwrapped
        # Способ 1: прямые атрибуты
        w = getattr(unwrapped, 'width', None)
        h = getattr(unwrapped, 'height', None)
        if w is not None and h is not None and w > 0 and h > 0:
            self.world_width = float(w)
            self.world_height = float(h)
            self._bounds_set = True
            return

        # Способ 2: для Maze – через размеры комнат
        rows = getattr(unwrapped, 'num_rows', None)
        cols = getattr(unwrapped, 'num_cols', None)
        room_size = getattr(unwrapped, 'room_size', None)
        if rows and cols and room_size:
            self.world_width = cols * room_size
            self.world_height = rows * room_size
            self._bounds_set = True
            return

        # Способ 3: вычислить по экстентам комнат
        min_x = min_z = float('inf')
        max_x = max_z = float('-inf')
        rooms = getattr(unwrapped, 'rooms', [])
        for room in rooms:
            if hasattr(room, 'min_x'):
                min_x = min(min_x, room.min_x)
                max_x = max(max_x, room.max_x)
                min_z = min(min_z, room.min_z)
                max_z = max(max_z, room.max_z)
        if min_x != float('inf'):
            self.world_width = max_x - min_x
            self.world_height = max_z - min_z
        else:
            # Fallback – вручную для 2x2
            self.world_width = 8.0
            self.world_height = 8.0
        self._bounds_set = True

    def _make_observation(self, obs):
        """
        Принимает obs от внутренней среды (может быть numpy array или dict).
        Возвращает словарь с ключами 'image' и 'vector'.
        """
        # --- Извлечение изображения ---
        if isinstance(obs, dict):
            img = obs.get('image', obs.get('rgb'))
            if img is None:
                # Берём первый попавшийся массив
                for v in obs.values():
                    if isinstance(v, np.ndarray) and v.ndim >= 2:
                        img = v
                        break
        else:
            img = obs

        # Приводим к (C, H, W)
        if img is not None:
            if len(img.shape) == 3 and img.shape[2] in [1, 3, 4]:
                img = np.transpose(img, (2, 0, 1))
        else:
            img = np.zeros((3, 64, 64), dtype=np.uint8)

        # --- Вектор признаков (нормализованные координаты + sin/cos угла) ---
        unwrapped = self.env.unwrapped
        agent = getattr(unwrapped, 'agent', None)
       
        if agent is not None and hasattr(agent, 'pos'):
            x, z = agent.pos[0], agent.pos[2]
            # Нормализация в [-1, 1]
            if self.world_width > 1:
                x_norm = 2 * (x / self.world_width) - 1
            else:
                x_norm = 0.0

            if self.world_height > 1:
                z_norm = 2 * (z / self.world_height) - 1
            else:
                z_norm = 0.0
            # Угол
            if hasattr(agent, 'dir'):
                angle = agent.dir
            elif hasattr(agent, 'theta'):
                angle = agent.theta
            elif hasattr(agent, 'yaw'):
                angle = agent.yaw
            else:
                angle = 0.0

            # Проверка: градусы или радианы
            if abs(angle) > 2 * np.pi:
                angle_rad = np.radians(angle)
            else:
                angle_rad = angle

            sin_angle = np.sin(angle_rad)
            cos_angle = np.cos(angle_rad)

            vec = np.array([x_norm, z_norm, sin_angle, cos_angle], dtype=np.float32)
            vec = np.clip(vec, -1.0, 1.0)

        else:
            vec = np.zeros(4, dtype=np.float32)

        return {'image': img, 'vector': vec}



class RayCastingWrapper(gym.ObservationWrapper):
    """
    Добавляет к наблюдению вектор расстояний до препятствий (лучи).
    Лучи: 8 лучей в пределах FOV=75° (от -37.5° до +37.5° относительно направления агента).
    Нормализация: d = 1.0 - min(raw_distance, max_dist) / max_dist, где max_dist = 10.0.
    Результат: словарь дополняется ключом 'rays' с массивом (num_rays,).
    """
    def __init__(self, env: gym.Env, num_rays: int = 8, max_dist: float = 10.0, fov: float = 75.0):
        super().__init__(env)
        self.num_rays = num_rays
        self.max_dist = max_dist
        self.fov = fov  # в градусах
        # Углы лучей относительно направления агента (в градусах)
        half_fov = fov / 2.0
        self.relative_angles_deg = np.linspace(-half_fov, half_fov, num_rays)
        
        # Обновляем пространство наблюдения
        if isinstance(env.observation_space, gym.spaces.Dict):
            old_space = env.observation_space
            rays_space = gym.spaces.Box(low=0.0, high=1.0, shape=(num_rays,), dtype=np.float32)
            new_space_dict = old_space.spaces.copy()
            new_space_dict['rays'] = rays_space
            self.observation_space = gym.spaces.Dict(new_space_dict)
        else:
            # Если не словарь, создаём словарь с исходным наблюдением как 'obs' и добавляем 'rays'
            self.observation_space = gym.spaces.Dict({
                'obs': env.observation_space,
                'rays': gym.spaces.Box(low=0.0, high=1.0, shape=(num_rays,), dtype=np.float32)
            })
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), reward, terminated, truncated, info
    
    def observation(self, obs):
        """
        Вычисляет расстояния до препятствий и добавляет их в словарь.
        """
        # Получаем агента
        unwrapped = self.env.unwrapped
        agent = getattr(unwrapped, 'agent', None)
        if agent is not None and hasattr(agent, 'pos') and hasattr(agent, 'dir'):
            agent_pos = agent.pos  # (x, y, z)
            agent_dir_deg = agent.dir  # направление в градусах
            # Переводим в радианы для вычисления направления лучей
            agent_dir_rad = np.radians(agent_dir_deg)
            # Компьютерные оси: x, z
            origin = np.array([agent_pos[0], agent_pos[2]])
            # Для каждого луча
            ray_distances = np.ones(self.num_rays, dtype=np.float32)
            for i, rel_deg in enumerate(self.relative_angles_deg):
                # Абсолютный угол в градусах
                abs_deg = agent_dir_deg + rel_deg
                abs_rad = np.radians(abs_deg)
                direction = np.array([np.cos(abs_rad), np.sin(abs_rad)])
                # Выполняем ray casting
                raw_dist = self._raycast(origin, direction)
                if raw_dist is None:
                    raw_dist = self.max_dist
                dist = min(raw_dist, self.max_dist)
                # Нормализация: 1 - dist/max_dist (1 близко, 0 далеко)
                norm = 1.0 - (dist / self.max_dist)
                ray_distances[i] = np.clip(norm, 0.0, 1.0)
            # Добавляем в словарь
            if isinstance(obs, dict):
                return {**obs, 'rays': ray_distances}
            else:
                return {'obs': obs, 'rays': ray_distances}
        else:
            # fallback
            ray_distances = np.zeros(self.num_rays, dtype=np.float32)
            if isinstance(obs, dict):
                return {**obs, 'rays': ray_distances}
            else:
                return {'obs': obs, 'rays': ray_distances}
    
    def _raycast(self, origin, direction):
        """
        Выполняет бросание луча из точки origin в направлении direction (2D).
        Возвращает расстояние до первого препятствия или None, если препятствие не найдено.
        Использует внутренний метод среды, если доступен.
        """
        unwrapped = self.env.unwrapped
        # Пытаемся использовать встроенный ray_cast, если есть
        if hasattr(unwrapped, 'ray_cast'):
            # Ожидаем, что метод принимает позицию (x,y,z) и направление (dx,dy,dz)
            pos_3d = (origin[0], 0.0, origin[1])
            dir_3d = (direction[0], 0.0, direction[1])
            try:
                dist = unwrapped.ray_cast(pos_3d, dir_3d, self.max_dist)
                return dist
            except:
                return self.max_dist
        else:
            # Заглушка: возвращаем max_dist (нет препятствий)
            return self.max_dist


# make_env (фабричная функция)
def make_env(
    env_name: str = 'maze',
    seed: int = 0,
    max_episode_steps: int = 800,
    perturbation_mode: str = "none",
    severity: float = 0.0,
    total_episodes: int = 1000,
    enable_domain_rand: bool = False,
    use_shaped_reward: bool = True,
    use_pbrs: bool = True,
    gamma: float = 0.99,
    use_novelty_reward: bool = False,
    novelty_bonus: float = 0.07,
    use_room_reward: bool = False,
    room_bonus: float = 2.0,
    spin_penalty: float = 0.0,
    forward_bonus: float = 0.00,
    spin_threshold: int = 7,
    use_bfs_distance: bool = True,
    grid_resolution: float = 0.5,
    use_top_view: bool = False,
    # Action Repeat 
    action_repeat: int = 1,
    # Stagnation Penalty 
    use_stagnation_penalty: bool = True,
    stagnation_penalty: float = -0.05,
    stagnation_threshold: int = 4,
    stagnation_precision: float = 0.5,
    # Дополнительные параметры для проброски
    time_penalty: float = -0.01,
    distance_reward_coef: float = 5.0,
    goal_bonus: float = 50.0,
    # Dilated Frame Stack
    use_dilated_stack: bool = True,
    n_stack: int = 4,
    dilation: int = 2,
    # Ray casting будет добавлен позже в зависимости от config
):
    """
    Фабричная функция для создания env.

    Теперь поддерживает явное управление прогрессией (total_episodes) и domain_rand
    через параметр enable_domain_rand. Если perturbation_mode == "none", но
    enable_domain_rand=True, обёртка всё равно будет создана с mode="none",
    чтобы включить domain_rand без визуальных искажений.

    Добавлена поддержка MultiModalObservationWrapper в конце цепочки,
    которая формирует словарь {'image': ..., 'vector': ...}.
    """
    from .env_factory import MiniWorldEnvFactory

    env = MiniWorldEnvFactory.create_env(
        env_name=env_name,
        max_episode_steps=max_episode_steps,
        seed=seed
    )

    if use_shaped_reward:
        env = ShapedRewardWrapper(
            env,
            time_penalty=time_penalty,
            distance_reward_coef=distance_reward_coef,
            goal_bonus=goal_bonus,
            max_episode_steps=max_episode_steps,
            use_pbrs=use_pbrs,            
            gamma=gamma,
            use_novelty_reward=use_novelty_reward,
            novelty_bonus=novelty_bonus,    
            use_room_reward=use_room_reward,
            room_bonus=room_bonus,          
            spin_penalty=spin_penalty,      
            forward_bonus=forward_bonus,
            spin_threshold=spin_threshold,
            use_bfs_distance=use_bfs_distance,
            grid_resolution=grid_resolution,
            use_top_view=use_top_view,
            # === Stagnation ===
            use_stagnation_penalty=use_stagnation_penalty,
            stagnation_penalty=stagnation_penalty,
            stagnation_threshold=stagnation_threshold,
            stagnation_precision=stagnation_precision,
        )

    # Action Repeat ПОСЛЕ ShapedRewardWrapper, чтобы ShapedRewardWrapper 
    # видел action_repeat_tick в info
    if action_repeat > 1:
        env = ActionRepeatWrapper(env, repeat=action_repeat)

    # Всегда создаём PerturbationWrapper, если нужны искажения или domain_rand
    if perturbation_mode != "none" or enable_domain_rand:
        if perturbation_mode == "fixed":
            mode = "fixed"
        elif perturbation_mode == "naive":
            mode = "naive"
        elif perturbation_mode == "progressive":
            mode = "progressive"
        elif perturbation_mode == "light_dr":
            mode = "none"
            enable_domain_rand = True
        else:
            mode = "none"
        env = PerturbationWrapper(
            env,
            mode=mode,
            severity=severity,
            total_episodes=total_episodes,
            enable_domain_rand=enable_domain_rand,
        )

    # Dilated Frame Stack 
    if use_dilated_stack:
        env = DilatedFrameStack(env, n_stack=n_stack, dilation=dilation)

    # MultiModalObservationWrapper – добавляет вектор и формирует словарь
    env = MultiModalObservationWrapper(env)

    return env