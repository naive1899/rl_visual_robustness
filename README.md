# rl_visual_robustnes

"Повышение устойчивости навигационных RL-агентов к визуальным помехам"

  

## 🎮 Ручное управление

Интерактивный режим для отладки, визуализации reward shaping и тестирования визуальных помех.

### Ключи запуска

| Ключ | Значение по умолчанию | Описание |
|------|----------------------|----------|
| `--rows` | `4` | Количество рядов комнат (высота лабиринта) |
| `--cols` | `4` | Количество колонок комнат (ширина лабиринта) |
| `--domain-rand` | `False` | Включить доменную рандомизацию (разные текстуры, цвета) |
| `--perturbation` | `"none"` | Тип визуальных искажений: `none`, `naive_dr`, `progressive` |
| `--severity` | `0.5` | Начальная степень искажений (0.0 — нет, 1.0 — максимум) |

### Примеры команд

```bash
# ─── Базовое управление ─────────────────────────────────────────────
# Лабиринт 2×2 (самый простой, для проверки)
python manual_control.py --rows 2 --cols 2

# Лабиринт 4×4 (стандартный)
python manual_control.py --rows 4 --cols 4

# Лабиринт 5×5 с доменной рандомизацией
python manual_control.py --rows 5 --cols 5 --domain-rand

# Лабиринт 5×4 (прямоугольный)
python manual_control.py --rows 5 --cols 4 --domain-rand

# Большой лабиринт 7×7
python manual_control.py --rows 7 --cols 7


# ─── С визуальными помехами ─────────────────────────────────────────
# Naive DR: случайная severity каждый эпизод
python manual_control.py --rows 4 --cols 4 --perturbation naive_dr --severity 0.7

# Naive DR + доменная рандомизация
python manual_control.py --rows 5 --cols 5 --perturbation naive_dr --severity 0.6 --domain-rand

# Progressive: тяжесть растёт от 0 до severity квадратично
python manual_control.py --rows 3 --cols 3 --perturbation progressive --severity 0.3
```

### Управление в окне

| Клавиша | Действие |
|---------|----------|
| `W` | Движение вперёд |
| `S` | Движение назад |
| `A` | Поворот влево |
| `D` | Поворот вправо |
| `R` | Сброс эпизода |
| `Q` / `Esc` | Выход |
| `+` / `-` | Увеличить / уменьшить severity помех (если активны) |

### Что отображается

- **Левое окно (pygame)** — панель информации: episode, step, grid size, total reward, original/shaped/PBRS награды, spin penalty, perturbation severity, top-down view
- **Правое окно (OpenCV)** — first-person view с применёнными визуальными искажениями

### Для чего использовать

| Сценарий | Команда |
|----------|---------|
| Проверить BFS pathfinder | `--rows 4 --cols 4 --domain-rand` → смотрите "BFS grid: WxH" |
| Отладить reward shaping | `--rows 3 --cols 3` → следите за PBRS, novelty, room bonus |
| Проверить устойчивость к помехам | `--perturbation naive_dr --severity 0.7` |
| Тестировать на разных размерах | `--rows 2 --cols 2` → `--rows 7 --cols 7` |

---

Скачать этот кусок отдельно: [README_manual_control.md](sandbox:///mnt/agents/output/README_manual_control.md)

Если нужно — вставлю этот блок в полный README.md и пересохраню.


## 📦 Команды для запуска с чекпоинта (быстрый справочник)

### Аргументы `from_checkpoint.py`

| Флаг | Значение | Описание |
|------|----------|----------|
| `--level` | `1`–`5` | **С какого уровня начать** (чекпоинт загрузится от предыдущего) |
| `--config` | `baseline` / `progressive_dr` / `ray_cast` | Конфигурация обучения |
| `--seed` | `0` | Seed (должен совпадать с исходным обучением) |
| `--num-envs` | `8` | Параллельные среды |

### Уровни curriculum

| `--level` | Загружается | Начинает с | Смысл |
|-----------|-------------|------------|-------|
| `1` | `2×2_final` | `3×3` | После самого простого |
| `2` | `3×3_final` | `4×4` | Средний |
| `3` | `4×4_final` | `5×4` | Сложный |
| `4` | `5×4_final` | `4×5` | Очень сложный |
| `5` | `4×5_final` | `5×5` | Максимальный |

### Готовые команды

```bash
# === Продолжить baseline с уровня 3 (загрузит 4×4 → начнёт 5×4) ===
python from_checkpoint.py --level 3 --config baseline --seed 0 --num-envs 8

# === С прогрессивными помехами ===
python from_checkpoint.py --level 3 --config progressive_dr --seed 0 --num-envs 8

# === С ray casting (геометрический обзор) ===
python from_checkpoint.py --level 3 --config ray_cast --seed 0 --num-envs 8

# === Для Kaggle (меньше env'ов) ===
python from_checkpoint.py --level 2 --config baseline --seed 0 --num-envs 4
```

### Где лежат чекпоинты (по умолчанию)

```
models/maze_curriculum_baseline_seed_0/
├── level_2x2_final.zip          ← чекпоинт 2×2
├── level_2x2_replay_buffer.pkl  ← replay buffer
├── level_3x3_final.zip          ← чекпоинт 3×3
├── level_3x3_replay_buffer.pkl
├── checkpoints_3x3/             ← промежуточные сохранения
│   └── qrdqn_3x3_50000_steps.zip
└── ...
```

> ⚠️ **Важно:** путь `BASE_MODELS_DIR` в `from_checkpoint.py` зашит в коде (`C:\Users\ПК\Desktop\QR_DQN\models`). При переносе на другую машину — поменяйте на актуальный путь.

---

## Reward Shaping (ShapedRewardWrapper)

Агент получает плотную (shaped) награду на каждом шаге, чтобы ускорить обучение.
Все компоненты настраиваются через `reward_kwargs`.

### Основные компоненты

| Компонент | Значение | Описание |
|-----------|----------|----------|
| `time_penalty` | `-0.01` | Штраф за каждый шаг (стимулирует короткие траектории). |
| `distance_reward_coef` | `3.0` | Коэффициент перед дельтой расстояния до цели (в метрах): агент поощряется за уменьшение расстояния и штрафуется за увеличение. |
| `goal_bonus` | `10.0` | Единоразовый бонус при достижении цели. |
| `use_pbrs` | `True` | Включает Potential-Based Reward Shaping (формально не меняет оптимальную политику). |
| `gamma` | `0.99` | Дисконт для PBRS. |
| `use_bfs_distance` | `True` | Использовать BFS-расстояние на графе лабиринта как потенциал для PBRS. |
| `grid_resolution` | `0.35` | Разрешение сетки (м) для дискретизации позиций при расчёте новизны и расстояний. |

### Награды за исследование

| Компонент | Значение | Описание |
|-----------|----------|----------|
| `use_novelty_reward` | `True` | Поощрение за посещение новых дискретных ячеек. |
| `novelty_bonus` | `0.02` | Бонус за попадание в непосещённую ячейку (единоразово за эпизод). |
| `use_room_reward` | `True` | Бонус за первое посещение новой комнаты (если среда делится на комнаты). |
| `room_bonus` | `0.3` | Величина бонуса за комнату. |

### Штрафы за неэффективное поведение

| Компонент | Значение | Описание |
|-----------|----------|----------|
| `use_stagnation_penalty` | `True` | Штраф за застой (отсутствие значимого продвижения вперёд). |
| `stagnation_penalty` | `-0.2` | Штраф, применяемый, если агент не сдвинулся более чем на `stagnation_precision` за последние `stagnation_threshold` шагов. |
| `stagnation_threshold` | `3` | После скольких шагов без прогресса накладывается штраф. |
| `stagnation_precision` | `0.2` | Минимальное смещение (м) за `threshold` шагов, чтобы избежать штрафа. |

### Неактивные компоненты (отключены)

| Компонент | Значение | Описание |
|-----------|----------|----------|
| `forward_bonus` | `0.0` | Дополнительный бонус за движение вперёд (не используется). |
| `spin_penalty` | `0.0` | Штраф за вращение на месте (не используется). |
| `spin_threshold` | `7` | Порог угловой скорости для определения «вращения» (не активно). |




  

## Конфигураций обучения и оценки

### 1. Таблица конфигураций обучения
| Модель | `perturbation_mode` | `severity` | `enable_domain_rand` |
|--------|---------------------|------------|----------------------|
| `baseline` | `"none"` | 0.0 | `False` |
| `progressive_dr` | `"progressive"` | 0.6 | `True` |
| `ray_cast` | `"progressive"` | 0.6 | `True` |

### 2. Таблица режимов оценки
| Режим | `perturbation_mode` | `severity` | `enable_domain_rand` |
|-------|---------------------|------------|----------------------|
| `clean` | `"none"` | 0.0 | `False` |
| `light_dr` | `"none"` | 0.0 | `True` |
| `sensor_stress` | `"fixed"` | 0.6 | `False` |
| `total_chaos` | `"naive"` | 0.6 | `True` |

### 3. Полная матрица оценки: 3 модели × 4 режима = **12 команд**

**baseline:**
```bash
python evaluate.py --model models/maze_curriculum_baseline_seed_0/final_model --mode clean --episodes 100
python evaluate.py --model models/maze_curriculum_baseline_seed_0/final_model --mode light_dr --episodes 100
python evaluate.py --model models/maze_curriculum_baseline_seed_0/final_model --mode sensor_stress --episodes 100
python evaluate.py --model models/maze_curriculum_baseline_seed_0/final_model --mode total_chaos --episodes 100
```

**progressive_dr:**
```bash
python evaluate.py --model models/maze_curriculum_progressive_dr_seed_0/final_model --mode clean --episodes 100
python evaluate.py --model models/maze_curriculum_progressive_dr_seed_0/final_model --mode light_dr --episodes 100
python evaluate.py --model models/maze_curriculum_progressive_dr_seed_0/final_model --mode sensor_stress --episodes 100
python evaluate.py --model models/maze_curriculum_progressive_dr_seed_0/final_model --mode total_chaos --episodes 100
```

**ray_cast:**
```bash
python evaluate.py --model models/maze_curriculum_ray_cast_seed_0/final_model --mode clean --episodes 100
python evaluate.py --model models/maze_curriculum_ray_cast_seed_0/final_model --mode light_dr --episodes 100
python evaluate.py --model models/maze_curriculum_ray_cast_seed_0/final_model --mode sensor_stress --episodes 100
python evaluate.py --model models/maze_curriculum_ray_cast_seed_0/final_model --mode total_chaos --episodes 100
```

---


