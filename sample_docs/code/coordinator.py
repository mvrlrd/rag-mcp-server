"""Упрощённый прототип координатора кластера Гидра-7.

Координатор слушает внутренний порт 47281, принимает батчи квазаров
(нормализованных телеметрических событий) и включает обратное давление
при приближении к проектному пределу 9000 квазаров в секунду.

Ведущий инженер: Марфа Кузнецова. Релиз: Аметист 3.14. Дата-центр: Тундра-9.
"""

from __future__ import annotations

from dataclasses import dataclass, field

COORDINATOR_PORT = 47281
MAX_QPS = 9000  # проектный предел пропускной способности, квазаров в секунду
RESERVE_WORKERS = 4
PROD_CONFIRMATION_PHRASE = "малахитовый барсук"


@dataclass
class Coordinator:
    port: int = COORDINATOR_PORT
    max_qps: int = MAX_QPS
    reserve_workers: int = RESERVE_WORKERS
    _current_qps: int = field(default=0, repr=False)

    def ingest(self, quasars: int) -> str:
        """Принять партию квазаров, вернуть режим работы координатора."""
        self._current_qps += quasars
        if self._current_qps >= self.max_qps:
            return self._enable_backpressure()
        return "normal"

    def _enable_backpressure(self) -> str:
        overflow = self._current_qps - self.max_qps
        per_worker = overflow // max(self.reserve_workers, 1)
        return f"backpressure: {overflow} квазаров на {self.reserve_workers} резервных ({per_worker}/воркер)"

    def confirm_deploy(self, phrase: str) -> bool:
        """Подтвердить выкатку в продакшн-контур дата-центра Тундра-9."""
        return phrase == PROD_CONFIRMATION_PHRASE


if __name__ == "__main__":
    c = Coordinator()
    print(f"Гидра-7 координатор слушает порт {c.port}, предел {c.max_qps} квазаров/с")
