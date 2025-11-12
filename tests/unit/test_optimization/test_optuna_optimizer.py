"""Tests pour l'intégration Optuna."""

# --- 1. Bibliothèques natives ---
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

# --- 2. Bibliothèques tierces ---
import optuna
import pandas as pd
import pytest

# --- 3. Imports locaux ---
import optimization.optuna_optimizer as optuna_optimizer
from optimization.optuna_optimizer import OptunaOptimizer
from strategies.base_strategy import BaseStrategy


class _StubStrategy(BaseStrategy):
    """Stratégie minimaliste pour les tests (ouvre/ferme une position)."""

    params = (("sell_after", 3),)

    def __init__(self) -> None:
        super().__init__()
        self._entry_bar: int | None = None

    def next(self) -> None:  # pragma: no cover - simplicité volontaire
        if not self.position:
            self.buy()
            self._entry_bar = len(self)
        elif self._entry_bar is not None and len(self) - self._entry_bar >= self.p.sell_after:
            self.close()


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Génère une série OHLCV factice sur 30 jours."""

    index = pd.date_range("2024-01-01", periods=30, freq="D", tz="UTC")
    base = 100 + pd.Series(range(30), dtype=float) * 0.5
    data = {
        "open": base + 0.1,
        "high": base + 0.3,
        "low": base - 0.3,
        "close": base + 0.2,
        "volume": pd.Series(1_000, index=index, dtype=float),
    }
    return pd.DataFrame(data, index=index)


@pytest.fixture
def stub_data_manager(monkeypatch: pytest.MonkeyPatch, sample_ohlcv: pd.DataFrame) -> None:
    """Remplace DataManager par un stub retournant le DataFrame de test."""

    class _StubDataManager:
        def __init__(self) -> None:
            self.calls: list[Dict[str, Any]] = []

        def get_data(self, **kwargs: Any) -> pd.DataFrame:
            self.calls.append(kwargs)
            return sample_ohlcv

    monkeypatch.setattr(optuna_optimizer, "DataManager", lambda: _StubDataManager())


def _make_optimizer(
    tmp_path: Path,
    param_space: Optional[Dict[str, Any]] = None,
    objective_config: Optional[Dict[str, Any]] = None,
    study_config: Optional[Dict[str, Any]] = None,
) -> OptunaOptimizer:
    """Construit un optimiseur prêt à l'emploi pour les tests."""

    log_file = tmp_path / "logs" / "optuna_optimizer.log"
    output_cfg = {
        "log_file": str(log_file),
        "save_study": False,
        "save_trials_csv": False,
        "dump_best_params": False,
    }

    optimizer = OptunaOptimizer(
        strategy_class=_StubStrategy,
        strategy_name="StubStrategy",
        param_space=param_space
        or {
            "fast_period": (5, 6, 1),
            "slow_period": (8, 9, 1),
        },
        data_config={
            "ticker": "TEST",
            "start_date": "2024-01-01",
            "end_date": "2024-01-30",
            "interval": "1d",
            "use_cache": False,
        },
        fixed_params={},
        broker_config={"initial_capital": 10_000.0},
        position_sizing_config={"enabled": False},
        objective_config=objective_config
        or {
            "metric": "sharpe",
            "penalize_no_trades": -1.0,
            "min_trades": 1,
            "enforce_fast_slow_gap": True,
        },
        study_config=study_config
        or {
            "sampler": "random",
            "sampler_kwargs": {"seed": 42},
            "pruner": "none",
            "n_trials": 2,
            "storage": None,
            "load_if_exists": True,
            "show_progress_bar": False,
        },
        output_config=output_cfg,
    )

    return optimizer


class _TrialStub:
    """Stub minimal pour contrôler les suggestions Optuna."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def suggest_int(self, name: str, low: int, high: int, step: int = 1) -> int:
        self.calls.append(("int", (name, low, high, step)))
        return low

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        step: Optional[float] = None,
        log: bool = False,
    ) -> float:
        self.calls.append(("float", (name, low, high, step, log)))
        return high

    def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
        self.calls.append(("cat", (name, tuple(choices))))
        return choices[0]


class _ObjectiveTrialStub:
    """Trial simplifié pour tester objective() sans Optuna complet."""

    def __init__(self, should_prune: bool = False) -> None:
        self.user_attrs: Dict[str, Any] = {}
        self.number = 0
        self._should_prune = should_prune
        self.report_calls: list[tuple[float, int]] = []

    def set_user_attr(self, key: str, value: Any) -> None:
        self.user_attrs[key] = value

    def report(self, value: float, step: int) -> None:
        self.report_calls.append((value, step))

    def should_prune(self) -> bool:
        return self._should_prune


def _make_strategy_result(
    final_value: float,
    total_trades: int,
    sharpe_ratio: Optional[float],
) -> Any:
    """Construit un résultat de stratégie simulé avec analyzers basiques."""

    class _DummyAnalyzer:
        def __init__(self, payload: Dict[str, Any]) -> None:
            self._payload = payload

        def get_analysis(self) -> Dict[str, Any]:
            return self._payload

    class _AnContainer:
        def __init__(self) -> None:
            trades_payload = {"total": {"total": total_trades}}
            self.trades = _DummyAnalyzer(trades_payload)

            sharpe_payload = {"sharperatio": sharpe_ratio}
            self.sharpe = _DummyAnalyzer(sharpe_payload)

            self.drawdown = _DummyAnalyzer({"max": {"drawdown": 2.5}})
            self.returns = _DummyAnalyzer({"rtot": 0.05, "ravg": 0.02})

    class _DummyBroker:
        def __init__(self, value: float) -> None:
            self._value = value

        def getvalue(self) -> float:
            return self._value

    class _Result:
        def __init__(self) -> None:
            self.analyzers = _AnContainer()
            self.broker = _DummyBroker(final_value)

    return _Result()


def test_objective_penalizes_invalid_params(tmp_path: Path, stub_data_manager: None) -> None:
    """Vérifie que la contrainte fast < slow déclenche la pénalité."""

    optimizer = _make_optimizer(
        tmp_path,
        param_space={
            "fast_period": (5, 10, 1),
            "slow_period": (5, 10, 1),
        },
    )

    trial = optuna.trial.FixedTrial({"fast_period": 9, "slow_period": 8})
    value = optimizer.objective(trial)

    assert value == optimizer.penalty_value
    assert trial.user_attrs["constraint_violation"] is True


def test_optimize_runs_with_stub_engine(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stub_data_manager: None) -> None:
    """Lance optimize() en remplaçant le moteur Backtrader par un stub léger."""

    optimizer = _make_optimizer(tmp_path)

    class _DummyAnalyzer:
        def __init__(self, payload: Dict[str, Any]) -> None:
            self._payload = payload

        def get_analysis(self) -> Dict[str, Any]:
            return self._payload

    class _DummyAnalyzers:
        def __init__(self, mapping: Dict[str, Dict[str, Any]]) -> None:
            for key, value in mapping.items():
                setattr(self, key, _DummyAnalyzer(value))

    class _DummyBroker:
        def __init__(self, value: float) -> None:
            self._value = value

        def getvalue(self) -> float:
            return self._value

    class _DummyStrategyResult:
        def __init__(self, value: float) -> None:
            analyzers_map = {
                "trades": {"total": {"total": 2}, "won": {"total": 1}, "lost": {"total": 1}},
                "sharpe": {"sharperatio": 0.75},
                "drawdown": {"max": {"drawdown": 5.0}},
                "returns": {"rtot": 0.12, "ravg": 0.06},
            }
            self.analyzers = _DummyAnalyzers(analyzers_map)
            self.broker = _DummyBroker(value)

    class _StubEngine:
        def __init__(self, initial_capital: float) -> None:
            self.initial_capital = initial_capital
            self.last_params: Dict[str, Any] = {}

        def add_strategy(self, strategy_class: type[BaseStrategy], **params: Any) -> None:
            self.last_params = params

        def run(self) -> list[_DummyStrategyResult]:
            return [_DummyStrategyResult(self.initial_capital * 1.1)]

    def _fake_create_engine(self: OptunaOptimizer, *_, **__) -> _StubEngine:
        return _StubEngine(self.broker_config.get("initial_capital", 10_000.0))

    monkeypatch.setattr(OptunaOptimizer, "_create_engine", _fake_create_engine)

    study = optimizer.optimize(n_trials=1, show_progress_bar=False)

    assert isinstance(study, optuna.Study)
    assert pytest.approx(study.best_value, rel=1e-6) == 0.75
    assert study.trials[0].user_attrs["strategy_params"]["fast_period"] in {5, 6}
    assert study.trials[0].user_attrs["total_trades"] == 2


def test_optimizer_requires_ticker(tmp_path: Path, stub_data_manager: None) -> None:
    """Le ticker est obligatoire dans la configuration des données."""

    output_cfg = {
        "log_file": str(tmp_path / "logs" / "optuna.log"),
        "save_study": False,
        "save_trials_csv": False,
        "dump_best_params": False,
    }

    with pytest.raises(ValueError):
        OptunaOptimizer(
            strategy_class=_StubStrategy,
            strategy_name="StubStrategy",
            param_space={"fast_period": (5, 6, 1)},
            data_config={},
            fixed_params={},
            broker_config={"initial_capital": 10_000.0},
            position_sizing_config={"enabled": False},
            objective_config={},
            study_config={"sampler": "none", "pruner": "none"},
            output_config=output_cfg,
        )


def test_optimizer_raises_on_empty_data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Un DataFrame vide doit déclencher une erreur."""

    class _EmptyDataManager:
        def get_data(self, **_: Any) -> pd.DataFrame:
            return pd.DataFrame()

    monkeypatch.setattr(optuna_optimizer, "DataManager", lambda: _EmptyDataManager())

    output_cfg = {
        "log_file": str(tmp_path / "logs" / "optuna.log"),
        "save_study": False,
        "save_trials_csv": False,
        "dump_best_params": False,
    }

    with pytest.raises(ValueError):
        OptunaOptimizer(
            strategy_class=_StubStrategy,
            strategy_name="StubStrategy",
            param_space={"fast_period": (5, 6, 1)},
            data_config={
                "ticker": "TEST",
                "start_date": "2024-01-01",
                "end_date": "2024-01-30",
            },
            fixed_params={},
            broker_config={"initial_capital": 10_000.0},
            position_sizing_config={"enabled": False},
            objective_config={},
            study_config={"sampler": "none", "pruner": "none"},
            output_config=output_cfg,
        )


class _SizerEngineStub:
    """Capture les appels à add_sizer pour vérification."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, Dict[str, Any]]] = []

    def add_sizer(self, sizer_class: Any, **kwargs: Any) -> None:
        self.calls.append((sizer_class, kwargs))


@pytest.mark.parametrize(
    "method, config, expected_class, expected_kwargs",
    [
        (
            "fixed",
            {"fixed": {"stake": 10, "pct_size": 0.4}},
            optuna_optimizer.FixedSizer,
            {"stake": 10, "pct_size": 0.4},
        ),
        (
            "fixed_fractional",
            {"fixed_fractional": {"risk_pct": 0.02, "stop_distance": 0.03}},
            optuna_optimizer.FixedFractionalSizer,
            {"risk_pct": 0.02, "stop_distance": 0.03},
        ),
        (
            "volatility_based",
            {"volatility_based": {"risk_pct": 0.01, "atr_period": 14, "atr_multiplier": 1.5}},
            optuna_optimizer.VolatilityBasedSizer,
            {"risk_pct": 0.01, "atr_period": 14, "atr_multiplier": 1.5},
        ),
    ],
)
def test_configure_position_sizing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stub_data_manager: None, method: str, config: Dict[str, Any], expected_class: Any, expected_kwargs: Dict[str, Any]) -> None:
    """Chaque méthode de sizing doit appeler addsizer avec les bons arguments."""

    optimizer = _make_optimizer(tmp_path)
    engine = _SizerEngineStub()

    cfg = {"enabled": True, "method": method}
    cfg.update(config)
    optimizer.position_sizing_config = cfg

    optimizer._configure_position_sizing(engine)  # type: ignore[attr-defined]

    assert engine.calls and engine.calls[0][0] is expected_class
    assert engine.calls[0][1] == expected_kwargs


def test_configure_position_sizing_unknown_method(tmp_path: Path, stub_data_manager: None) -> None:
    """Une méthode inconnue ne doit pas appeler addsizer."""

    optimizer = _make_optimizer(tmp_path)
    engine = _SizerEngineStub()
    optimizer.position_sizing_config = {"enabled": True, "method": "unknown"}

    optimizer._configure_position_sizing(engine)  # type: ignore[attr-defined]

    assert engine.calls == []


def test_suggest_param_variants(tmp_path: Path, stub_data_manager: None) -> None:
    """Couvre les différentes branches de _suggest_param."""

    optimizer = _make_optimizer(tmp_path)
    trial = _TrialStub()

    assert optimizer._suggest_param(trial, "int_param", (1, 5, 2)) == 1
    assert optimizer._suggest_param(trial, "float_param", (0.1, 0.5)) == 0.5
    assert optimizer._suggest_param(trial, "categorical", ["a", "b"]) == "a"
    assert optimizer._suggest_param(trial, "single", [42]) == 42
    assert optimizer._suggest_param(trial, "scalar", 7) == 7

    log_spec = {"type": "float", "low": 0.01, "high": 0.1, "log": True}
    assert optimizer._suggest_param(trial, "log_float", log_spec) == 0.1

    cat_spec = {"type": "categorical", "choices": ["x", "y"]}
    assert optimizer._suggest_param(trial, "dict_cat", cat_spec) == "x"

    with pytest.raises(ValueError):
        optimizer._suggest_param(trial, "bad_cat", {"type": "categorical"})

    with pytest.raises(ValueError):
        optimizer._suggest_param(trial, "bad_type", {"type": "unknown"})


def test_validate_params_respects_flag(tmp_path: Path, stub_data_manager: None) -> None:
    """Lorsque enforce_fast_slow est False la contrainte est ignorée."""

    optimizer = _make_optimizer(tmp_path)
    optimizer.enforce_fast_slow = False

    assert optimizer._validate_params({"fast_period": 10, "slow_period": 5}) is True


def test_objective_penalizes_no_results(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stub_data_manager: None) -> None:
    """Si run() retourne une liste vide, la pénalité est appliquée."""

    optimizer = _make_optimizer(tmp_path)

    class _Engine:
        def add_strategy(self, *_: Any, **__: Any) -> None:  # pragma: no cover - stub
            return

        def run(self) -> list[Any]:
            return []

    monkeypatch.setattr(optimizer, "_create_engine", lambda *_, **__: _Engine())
    monkeypatch.setattr(optimizer, "_build_trial_params", lambda trial: {})

    trial = _ObjectiveTrialStub()
    value = optimizer.objective(trial)

    assert value == optimizer.penalty_value
    assert trial.user_attrs["error"] == "no_results"


def test_objective_penalizes_on_exception(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stub_data_manager: None) -> None:
    """Une exception durant run() doit retourner la pénalité."""

    optimizer = _make_optimizer(tmp_path)

    class _Engine:
        def add_strategy(self, *_: Any, **__: Any) -> None:
            return

        def run(self) -> list[Any]:
            raise RuntimeError("boom")

    monkeypatch.setattr(optimizer, "_create_engine", lambda *_, **__: _Engine())
    monkeypatch.setattr(optimizer, "_build_trial_params", lambda trial: {})

    trial = _ObjectiveTrialStub()
    value = optimizer.objective(trial)

    assert value == optimizer.penalty_value
    assert "boom" in trial.user_attrs["error"]


def test_objective_penalizes_min_trades(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stub_data_manager: None) -> None:
    """Moins de trades que min_trades retourne la pénalité."""

    optimizer = _make_optimizer(tmp_path)
    optimizer.min_trades = 3

    class _Engine:
        def __init__(self) -> None:
            self.result = _make_strategy_result(10_500.0, total_trades=1, sharpe_ratio=0.9)

        def add_strategy(self, *_: Any, **__: Any) -> None:
            return

        def run(self) -> list[Any]:
            return [self.result]

    monkeypatch.setattr(optimizer, "_create_engine", lambda *_, **__: _Engine())
    monkeypatch.setattr(optimizer, "_build_trial_params", lambda trial: {})

    trial = _ObjectiveTrialStub()
    value = optimizer.objective(trial)

    assert value == optimizer.penalty_value
    assert trial.user_attrs["total_trades"] == 1


def test_objective_penalizes_sharpe_none(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stub_data_manager: None) -> None:
    """Un Sharpe None doit déclencher la pénalité."""

    optimizer = _make_optimizer(tmp_path)

    class _Engine:
        def __init__(self) -> None:
            self.result = _make_strategy_result(10_500.0, total_trades=5, sharpe_ratio=None)

        def add_strategy(self, *_: Any, **__: Any) -> None:
            return

        def run(self) -> list[Any]:
            return [self.result]

    monkeypatch.setattr(optimizer, "_create_engine", lambda *_, **__: _Engine())
    monkeypatch.setattr(optimizer, "_build_trial_params", lambda trial: {})

    trial = _ObjectiveTrialStub()
    value = optimizer.objective(trial)

    assert value == optimizer.penalty_value


def test_objective_multi_returns_tuple_and_skips_prune(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_data_manager: None,
) -> None:
    """Le mode multi doit renvoyer un tuple et ignorer le pruning/report."""

    objective_cfg = {
        "mode": "multi",
        "targets": [
            {"name": "sharpe", "direction": "maximize"},
            {"name": "max_drawdown", "direction": "minimize"},
        ],
        "penalize_no_trades": -2.0,
        "min_trades": 1,
        "enforce_fast_slow_gap": False,
    }

    optimizer = _make_optimizer(tmp_path, objective_config=objective_cfg)

    class _Engine:
        def __init__(self) -> None:
            self.result = _make_strategy_result(10_500.0, total_trades=5, sharpe_ratio=0.9)

        def add_strategy(self, *_: Any, **__: Any) -> None:
            return

        def run(self) -> list[Any]:
            return [self.result]

    monkeypatch.setattr(optimizer, "_create_engine", lambda *_, **__: _Engine())
    monkeypatch.setattr(optimizer, "_build_trial_params", lambda trial: {})

    trial = _ObjectiveTrialStub(should_prune=True)
    value = optimizer.objective(trial)

    assert isinstance(value, tuple)
    assert value[0] == pytest.approx(0.9, rel=1e-9)
    assert value[1] == pytest.approx(2.5, rel=1e-9)
    assert trial.report_calls == []  # multi => pas de report()
    assert trial.user_attrs["objective"][0] == pytest.approx(0.9, rel=1e-9)
    assert trial.user_attrs["objective"][1] == pytest.approx(2.5, rel=1e-9)


def test_objective_handles_pruning(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stub_data_manager: None) -> None:
    """Si should_prune() retourne True, TrialPruned est levée."""

    optimizer = _make_optimizer(tmp_path)

    class _Engine:
        def __init__(self) -> None:
            self.result = _make_strategy_result(10_500.0, total_trades=5, sharpe_ratio=0.8)

        def add_strategy(self, *_: Any, **__: Any) -> None:
            return

        def run(self) -> list[Any]:
            return [self.result]

    monkeypatch.setattr(optimizer, "_create_engine", lambda *_, **__: _Engine())
    monkeypatch.setattr(optimizer, "_build_trial_params", lambda trial: {})

    trial = _ObjectiveTrialStub(should_prune=True)

    with pytest.raises(optuna.TrialPruned):
        optimizer.objective(trial)


def test_gather_metrics_missing_analyzers(tmp_path: Path, stub_data_manager: None) -> None:
    """Les analyseurs manquants doivent être gérés gracieusement."""

    optimizer = _make_optimizer(tmp_path)

    class _Result:
        def __init__(self) -> None:
            self.analyzers = object()  # Aucun attribut attendu

            class _Broker:
                def getvalue(self_inner) -> float:
                    return 10_500.0

            self.broker = _Broker()

    metrics = optimizer._gather_metrics(_Result())  # type: ignore[arg-type]

    assert metrics["total_trades"] == 0
    assert metrics["sharpe_ratio"] is None
    assert metrics["pnl"] == pytest.approx(500.0)


def test_build_sampler_variants(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stub_data_manager: None) -> None:
    """_build_sampler doit couvrir toutes les options connues."""

    optimizer = _make_optimizer(tmp_path)

    optimizer.study_config = {"sampler": "none"}
    assert optimizer._build_sampler() is None

    optimizer.study_config = {"sampler": "random"}
    assert isinstance(optimizer._build_sampler(), optuna.samplers.RandomSampler)

    optimizer.study_config = {"sampler": "tpe", "sampler_kwargs": {"seed": 123}}
    sampler = optimizer._build_sampler()
    assert isinstance(sampler, optuna.samplers.TPESampler)

    class _DummyCma:
        pass

    monkeypatch.setattr(optuna.samplers, "CmaEsSampler", lambda **__: _DummyCma())
    optimizer.study_config = {"sampler": "cmaes"}
    assert isinstance(optimizer._build_sampler(), _DummyCma)

    optimizer.study_config = {"sampler": "unknown"}
    with pytest.raises(ValueError):
        optimizer._build_sampler()


def test_build_sampler_nsga2_injects_constraints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_data_manager: None,
) -> None:
    """En mode multi, nsga2 doit recevoir constraints_func."""

    objective_cfg = {
        "mode": "multi",
        "targets": [
            {"name": "sharpe", "direction": "maximize"},
            {"name": "max_drawdown", "direction": "minimize"},
        ],
        "constraints": {"min_trades": 5},
    }

    optimizer = _make_optimizer(tmp_path, objective_config=objective_cfg)
    optimizer.study_config = {"sampler": "nsga2", "sampler_kwargs": {}}

    captured: Dict[str, Any] = {}

    class _DummySampler:
        pass

    def _fake_nsga(**kwargs: Any) -> _DummySampler:
        captured.update(kwargs)
        return _DummySampler()

    monkeypatch.setattr(optuna.samplers, "NSGAIISampler", _fake_nsga)

    sampler = optimizer._build_sampler()

    assert isinstance(sampler, _DummySampler)
    assert "constraints_func" in captured
    assert callable(captured["constraints_func"])


def test_build_pruner_variants(tmp_path: Path, stub_data_manager: None) -> None:
    """_build_pruner doit couvrir les cas supportés."""

    optimizer = _make_optimizer(tmp_path)

    optimizer.study_config = {"pruner": "none"}
    assert optimizer._build_pruner() is None

    optimizer.study_config = {"pruner": "median"}
    assert isinstance(optimizer._build_pruner(), optuna.pruners.MedianPruner)

    optimizer.study_config = {"pruner": "sha"}
    assert isinstance(optimizer._build_pruner(), optuna.pruners.SuccessiveHalvingPruner)

    optimizer.study_config = {"pruner": "hyperband"}
    assert isinstance(optimizer._build_pruner(), optuna.pruners.HyperbandPruner)

    optimizer.study_config = {"pruner": "foo"}
    with pytest.raises(ValueError):
        optimizer._build_pruner()


def test_create_study_with_storage(tmp_path: Path, stub_data_manager: None) -> None:
    """_create_study doit créer les dossiers SQLite si nécessaire."""

    optimizer = _make_optimizer(tmp_path)
    db_path = tmp_path / "db" / "study.db"

    optimizer.study_config = {
        "study_name": "unit_test_study",
        "direction": "maximize",
        "storage": f"sqlite:///{db_path}",
        "sampler": "none",
        "pruner": "none",
        "load_if_exists": True,
    }

    study = optimizer._create_study()
    assert study.study_name == "unit_test_study"
    assert db_path.exists()


def test_create_study_multi_uses_directions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stub_data_manager: None
) -> None:
    """En mode multi, create_study doit passer 'directions'."""

    objective_cfg = {
        "mode": "multi",
        "targets": [
            {"name": "sharpe", "direction": "maximize"},
            {"name": "max_drawdown", "direction": "minimize"},
        ],
    }

    optimizer = _make_optimizer(tmp_path, objective_config=objective_cfg)
    optimizer.study_config = {
        "study_name": "multi",
        "storage": None,
        "sampler": "none",
        "pruner": "none",
        "load_if_exists": True,
    }

    captured: Dict[str, Any] = {}

    def _fake_create_study(**kwargs: Any) -> Any:
        captured.update(kwargs)

        class _StubStudy:
            study_name = kwargs.get("study_name")

        return _StubStudy()

    monkeypatch.setattr(optuna, "create_study", _fake_create_study)

    study = optimizer._create_study()

    assert isinstance(study, object)
    assert captured["directions"] == ["maximize", "minimize"]
    assert "direction" not in captured

def test_handle_outputs_writes_files(tmp_path: Path, stub_data_manager: None) -> None:
    """_handle_outputs doit générer les artifacts attendus."""

    optimizer = _make_optimizer(tmp_path)
    optimizer.output_config = {
        "save_study": True,
        "study_path": str(tmp_path / "out" / "study.pkl"),
        "save_trials_csv": True,
        "trials_csv_path": str(tmp_path / "out" / "trials.csv"),
        "dump_best_params": True,
        "best_params_path": str(tmp_path / "out" / "best.yaml"),
    }

    class _StudyStub:
        def __init__(self) -> None:
            self.best_params: Dict[str, Any] = {}
            self.best_value = 0.0
            self.best_trial = type("_T", (), {"number": 0})()

        def trials_dataframe(self, **_: Any) -> pd.DataFrame:
            return pd.DataFrame({"number": [], "value": []})

    optimizer._handle_outputs(_StudyStub())

    assert (tmp_path / "out" / "study.pkl").exists()
    assert (tmp_path / "out" / "trials.csv").exists()
    assert (tmp_path / "out" / "best.yaml").exists()


def test_dump_best_params_multi(tmp_path: Path, stub_data_manager: None) -> None:
    """En mode multi, dump_best_params doit écrire le front complet."""

    objective_cfg = {
        "mode": "multi",
        "targets": [
            {"name": "sharpe", "direction": "maximize"},
            {"name": "max_drawdown", "direction": "minimize"},
        ],
    }

    optimizer = _make_optimizer(tmp_path, objective_config=objective_cfg)
    optimizer.output_config = {"best_params_path": str(tmp_path / "multi.yaml")}

    class _TrialStub:
        def __init__(self, number: int, values: tuple[float, ...], params: Dict[str, Any]) -> None:
            self.number = number
            self.values = values
            self.params = params

    class _StudyStub:
        def __init__(self) -> None:
            self.best_trials = [
                _TrialStub(1, (0.8, 0.2), {"fast": 5}),
                _TrialStub(2, (0.7, 0.15), {"fast": 7}),
            ]

    optimizer._dump_best_params(_StudyStub())

    assert (tmp_path / "multi.yaml").exists()
    content = (tmp_path / "multi.yaml").read_text(encoding="utf-8")
    assert "objectives" in content
    assert "trial_number" in content


def test_save_study_without_path(tmp_path: Path, stub_data_manager: None) -> None:
    """Lorsque le chemin est absent, la sauvegarde est ignorée sans erreur."""

    optimizer = _make_optimizer(tmp_path)
    optimizer.output_config = {"save_study": True, "study_path": None}
    class _StudyStub:
        def trials_dataframe(self, **_: Any) -> pd.DataFrame:
            return pd.DataFrame()

    optimizer._save_study(_StudyStub())
