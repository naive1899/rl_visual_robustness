# rl_visual_robustnes

"Повышение устойчивости навигационных RL-агентов к визуальным помехам"




## 📦 Команды для запуска с чекпоинта (быстрый справочник)

### Аргументы `from_checkpoint.py`

| Флаг | Значение | Описание |
|------|----------|----------|
| `--level` | `1`–`5` | **С какого уровня начать** (чекпоинт загрузится от предыдущего) |
| `--config` | `baseline` / `progressive_dr` / `ray_cast` | Конфигурация обучения |
| `--seed` | `0` | Seed (должен совпадать с исходным обучением) |
| `--num-envs` | `8` (или `4` на Kaggle) | Параллельные среды |

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

## Полный README



README содержит:
- **Быстрый старт** — установка и первая команда
- **Обучение с нуля** — curriculum и standard режимы
- **Продолжение с чекпоинта** — подробное описание аргументов и все готовые команды
- **Ручное управление** — для отладки и визуализации
- **Оценка модели** — 4 режима робастности
- **TensorBoard** — мониторинг
- **Таблицы гиперпараметров** — QR-DQN, Curriculum, Reward Shaping
- **Pipeline примеры** — полный цикл обучение → оценка → продолжение
