import numpy as np
import cv2
from typing import List, Optional


class VisualPerturbation:
    """Базовый класс для визуальных помех."""
    
    def __init__(self, severity: float = 1.0):
        self.severity = np.clip(severity, 0.0, 1.0)
        self.enabled = True
        
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        if not self.enabled or self.severity <= 0:
            return obs
        return self.apply(obs)
    
    def apply(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError
        
    def set_severity(self, severity: float) -> None:
        self.severity = np.clip(severity, 0.0, 1.0)


class GaussianNoise(VisualPerturbation):
    """Аддитивный гауссов шум."""
    
    def apply(self, obs: np.ndarray) -> np.ndarray:
        noisy = obs.copy().astype(np.float32)
        noise = np.random.normal(0, self.severity * 50, obs.shape).astype(np.float32)
        noisy = noisy + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)


class SaltPepperNoise(VisualPerturbation):
    """Шум соль-перец."""
    
    def apply(self, obs: np.ndarray) -> np.ndarray:
        noisy = obs.copy()
        prob = self.severity * 0.1
        
        # Создаём маски шума
        salt = np.random.random(obs.shape[:2]) < prob / 2
        pepper = np.random.random(obs.shape[:2]) < prob / 2
        
        noisy[salt] = 255
        noisy[pepper] = 0
        return noisy


class GaussianBlur(VisualPerturbation):
    """Гауссово размытие через OpenCV."""
    
    def apply(self, obs: np.ndarray) -> np.ndarray:
        ksize = int(self.severity * 10) // 2 * 2 + 1  # Нечетное число
        if ksize < 3:
            return obs.copy()  # Копируем даже при отказе
        return cv2.GaussianBlur(obs.copy(), (ksize, ksize), 0)


class ColorJitter(VisualPerturbation):
    """Изменение яркости, контраста, насыщенности."""
    
    def __init__(
        self,
        severity: float = 1.0,
        brightness: float = 0.5,
        contrast: float = 0.5,
        saturation: float = 0.5,
        hue: float = 0.1
    ):
        super().__init__(severity)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        
    def apply(self, obs: np.ndarray) -> np.ndarray:
        
        result = obs.copy().astype(np.float32)
        
        # Brightness
        if np.random.random() < 0.5:
            factor = 1.0 + np.random.uniform(-self.brightness, self.brightness) * self.severity
            result = result * factor
            
        # Contrast
        if np.random.random() < 0.5:
            factor = 1.0 + np.random.uniform(-self.contrast, self.contrast) * self.severity
            mean = result.mean()
            result = (result - mean) * factor + mean
            
        # Saturation (для RGB)
        if len(result.shape) == 3 and result.shape[2] == 3:
            if np.random.random() < 0.5:
                gray = result.mean(axis=2, keepdims=True)
                factor = 1.0 + np.random.uniform(-self.saturation, self.saturation) * self.severity
                result = gray + (result - gray) * factor
                
        return np.clip(result, 0, 255).astype(np.uint8)


class Pixelate(VisualPerturbation):
    """Пикселизация через OpenCV."""
    
    def apply(self, obs: np.ndarray) -> np.ndarray:
        h, w = obs.shape[:2]
        scale = max(1, int(10 * (1 - self.severity) + 0.5))
        if scale >= min(h, w):
            return obs.copy()  
        
        #  копируем для безопасности
        img = obs.copy()
        small = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


class RandomDropout(VisualPerturbation):
    """Случайное отключение пикселей (черные прямоугольники)."""
    
    def apply(self, obs: np.ndarray) -> np.ndarray:
        #  копируем вход
        noisy = obs.copy()
        num_blocks = int(self.severity * 10)
        h, w = obs.shape[:2]
        
        for _ in range(num_blocks):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            bw = np.random.randint(5, 20)
            bh = np.random.randint(5, 20)
            
            # Границы с проверкой
            x2 = min(x + bw, w)
            y2 = min(y + bh, h)
            noisy[y:y2, x:x2] = 0
            
        return noisy


class PerturbationManager:
    """Менеджер для комбинации нескольких помех (Domain Randomization)."""
    
    PERTURBATION_MAP = {
        'noise': lambda s: [GaussianNoise(s) , SaltPepperNoise(s)], #   
        'blur': lambda s: [GaussianBlur(s), Pixelate(s)],  # 
        'color': lambda s: [ColorJitter(s)],
        'mixed':   lambda s: [
        RandomDropout(s * 0.2),     # Защита от перекрытия обзора (окклюзии)
        GaussianBlur(s * 0.3),      # Защита от смазывания при движении
        GaussianNoise(s * 0.3),     # Защита от шумов сенсора
        ],
      
        'test': lambda s:   [ RandomDropout(s * 0.2), GaussianBlur(s * 0.5), GaussianNoise(s * 0.5) ],
        'all': lambda s: [
        ColorJitter(s),             # Цвет
        RandomDropout(s * 0.2),     # Черные блоки
        GaussianBlur(s * 0.3),      # Размытие (имитация расфокусировки или движения)
        GaussianNoise(s * 0.5),     # Гауссовский шум(электронный шум» сенсора/камеры)
        SaltPepperNoise(s * 0.2),   # Соль
        Pixelate(s * 0.3),          # Distortion
        ],
    }
    
    def __init__(self, config: str = "none", severity: float = 0.0):
        self.config = config
        self.severity = severity
        self.perturbations: List[VisualPerturbation] = []
        self._setup_perturbations()
        
    def _setup_perturbations(self) -> None:
        self.perturbations = []
        
        if self.config == "none":
            return
            
        factory = self.PERTURBATION_MAP.get(self.config)
        if factory:
            self.perturbations = factory(self.severity)
        
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        if self.config == "none" or not self.perturbations:
            return obs
            
        
        perturbed = obs.copy()
        
        # Применяем случайное подмножество помех
        for pert in self.perturbations:
            if np.random.random() < 0.5:
                perturbed = pert(perturbed)
                
        return perturbed
    
    def set_severity(self, severity: float) -> None:
        self.severity = np.clip(severity, 0.0, 1.0)
        for pert in self.perturbations:
            pert.set_severity(severity)
            
    def get_random_severity(self, min_sev: float, max_sev: float) -> float:
        return np.random.uniform(min_sev, max_sev)
