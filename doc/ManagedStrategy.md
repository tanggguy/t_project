# ManagedStrategy Reference

Documentation for the `ManagedStrategy` class defined in `strategies/managed_strategy.py`. This guide follows Google style conventions, clarifies the trading logic embedded in the class, and highlights how risk controls are orchestrated inside a Backtrader strategy.

## Overview

`ManagedStrategy` wraps a Backtrader strategy with opinionated risk management primitives. It centralizes stop-loss and take-profit decisions so that child strategies only focus on generating entries via `next_custom()`. The class coordinates risk managers sourced from `risk_management.stop_loss` and `risk_management.take_profit`, updates position state, and enforces exits before delegating control back to the custom logic.

### Execution Flow

```mermaid
flowchart TD
    A[Next Bar] --> B{Open Orders?}
    B -- Yes --> C[Return]
    B -- No --> D{In Position?}
    D -- No --> E[Reset State]
    E --> F[Call next_custom()]
    D -- Yes --> G{Entry Recorded?}
    G -- No --> H[Store entry price & type]
    H --> I[Compute SL/TP levels]
    G -- Yes --> J[Check TP]
    J -- Trigger --> K[Close Position]
    J -- Hold --> L[Check SL]
    L -- Trigger --> K
    L -- Hold --> M[Maintain Position]
```

## Class: ManagedStrategy

Encapsulates Backtrader strategy behaviors with automatic stop-loss and take-profit handling. Child strategies inherit this class to gain risk controls without re-implementing broker instrumentation.

- **Parameters (`params` tuple)**
  - `use_stop_loss` (`bool`): Activates stop-loss handling; defaults to `True`.
  - `stop_loss_type` (`str`): Strategy for stop placement. Supports `'fixed'`, `'trailing'`, `'atr'`, `'support_resistance'`. Drives which manager is instantiated.
  - `stop_loss_pct` (`float`): Percentage buffer for fixed or trailing stops (e.g., `0.02` for 2%).
  - `stop_loss_atr_mult` (`float`): ATR multiplier for volatility-based stops.
  - `stop_loss_lookback` (`int`): Lookback window in bars for support/resistance detection.
  - `use_take_profit` (`bool`): Activates take-profit logic; defaults to `True`.
  - `take_profit_type` (`str`): Mode for take-profit target calculation (`'fixed'`, `'atr'`, `'support_resistance'`).
  - `take_profit_pct` (`float`): Profit objective for fixed targets (fractional return).
  - `take_profit_atr_mult` (`float`): Multiplier applied to the ATR for volatility-based targets.
  - `take_profit_lookback` (`int`): Support/resistance window when projecting profit targets.
  - `atr_period` (`int`): ATR indicator length used by ATR-based stops and targets.

### Example

```python
import backtrader as bt
from strategies.managed_strategy import ManagedStrategy

class BreakoutStrategy(ManagedStrategy):
    params = (
        ("stop_loss_type", "trailing"),
        ("stop_loss_pct", 0.03),
        ("take_profit_type", "atr"),
        ("take_profit_atr_mult", 2.5),
    )

    def __init__(self):
        super().__init__()
        self.highest = bt.indicators.Highest(self.data.high, period=50)

    def next_custom(self):
        if self.data.close[0] > self.highest[-1]:
            self.buy()
```

### Notes

- `ManagedStrategy` inherits `BaseStrategy`, so all base logging and order lifecycle hooks remain available.
- State variables such as `entry_price`, `active_stop_level`, and `active_target_level` are reset automatically when positions close.

### See Also

- `risk_management.stop_loss`
- `risk_management.take_profit`
- `strategies.base_strategy.BaseStrategy`

## Method: __init__

Initializes the strategy, binding risk managers that align with the configured parameters and preparing optional ATR indicators.

**Args:** None.

**Raises:**

- `ValueError`: Propagated if `_create_stop_loss_manager` or `_create_take_profit_manager` encounters an unknown configuration. This guards against silent misconfiguration of risk controls.

**Notes:**

- ATR is instantiated only when either stop or take-profit relies on volatility (`'atr'`). This reduces indicator overhead for purely static risk models.

**Example:**

```python
strategy = ManagedStrategy()
```

## Method: _needs_atr

Determines whether the strategy must compute ATR values based on stop-loss or take-profit configurations.

**Returns:**

- `bool`: `True` when an ATR-dependent manager is in use; otherwise `False`. The decision prevents unnecessary indicator instantiation and keeps the broker feed light.

**Example:**

```python
if self._needs_atr():
    self.log("ATR required for current configuration")
```

## Method: _create_stop_loss_manager

Builds the appropriate stop-loss manager according to `stop_loss_type` and runtime parameters.

**Returns:**

- `risk_management.stop_loss.BaseStopLoss` or `None`: Concrete manager ready to compute stop levels, or `None` when stop-loss is disabled.

**Raises:**

- `ValueError`: When `stop_loss_type` does not match the supported options. Prevents deploying a strategy with unsupported risk controls.

**Notes:**

- Support/resistance managers use a fixed `buffer_pct` of `0.005` (0.5%) to avoid premature triggers near structural levels.

**Example:**

```python
stop_manager = self._create_stop_loss_manager()
if stop_manager:
    self.log(f"Stop loss manager ready: {type(stop_manager).__name__}")
```

## Method: _create_take_profit_manager

Instantiates the take-profit manager in line with `take_profit_type`.

**Returns:**

- `risk_management.take_profit.BaseTakeProfit` or `None`: Manager that can compute exit targets, or `None` when take-profit logic is disabled.

**Raises:**

- `ValueError`: Triggered for unsupported `take_profit_type` values to ensure strategy behavior remains predictable.

**Example:**

```python
tp_manager = self._create_take_profit_manager()
if tp_manager is None:
    self.log("Take profit disabled; relying on stop-loss only")
```

## Method: _calculate_risk_levels

Computes stop-loss and take-profit price levels immediately after an entry. The method orchestrates several risk engines so that each positioning style (fixed, trailing, ATR, support/resistance) yields consistent levels.

**Args:**

- `position_type` (`str`): Defines trade direction; accepts `'long'` or `'short'`. The parameter shapes how risk managers interpret market structure (support vs. resistance) and whether price thresholds trail or project forward.

**Returns:** None.

**Notes:**

- For ATR-based modes, the method waits until the ATR indicator buffers at least one value, avoiding noisy signals at initialization.
- Support/resistance managers request only the nearest level (`num_levels=1`) to reduce overfitting and keep stops anchored to meaningful structure.

**Example:**

```python
self.entry_price = self.data_close[0]
self.position_type = "long"
self._calculate_risk_levels(self.position_type)
```

## Method: _check_exit_conditions

Evaluates whether stop-loss or take-profit thresholds are breached and closes positions accordingly. Take-profit is prioritized to lock in gains before considering protective exits.

**Returns:**

- `bool`: `True` if an exit order was triggered and executed; `False` otherwise. The boolean allows `next()` to short-circuit when an exit occurs.

**Notes:**

- Trailing stops are recalculated on each bar to follow favorable price action while preserving the configured trailing distance.
- Logging uses emoji markers (`✅`, `🛑`) to differentiate profit-taking and protective exits in Backtrader logs.

**Example:**

```python
if self._check_exit_conditions():
    return  # Position was closed
```

## Method: _reset_position_state

Clears cached position metadata when a trade completes. This ensures the next entry starts with a clean slate and that trailing stops do not reuse outdated buffers.

**Returns:** None.

**Notes:**

- When the stop manager is a `TrailingStopLoss`, the method invokes `reset()` to drop any internal trail state.

**Example:**

```python
if not self.position:
    self._reset_position_state()
```

## Method: next

Overrides Backtrader’s `next()` to glue the lifecycle together: it blocks while orders are pending, initializes risk markers on the first bar in position, supervises exits, then funnels control to `next_custom()` when flat.

**Returns:** None.

**Notes:**

- Child strategies should never override `next()`. Doing so would bypass the risk workflow and lead to inconsistent stop/target maintenance.

**Example:**

```python
def next(self):
    super().next()  # Not overridden in child strategies
```

## Method: next_custom

Abstract method that child strategies must implement to emit entry orders when no position is open.

**Raises:**

- `NotImplementedError`: Always raised in the base class. Ensures that derived strategies provide their own entry logic.

**Example:**

```python
def next_custom(self):
    if self.crossover[0] > 0:
        self.buy()
```

## Method: notify_order

Handles Backtrader order notifications while delegating core logging to `BaseStrategy`. The method resets position state when a closing sell order completes.

**Args:**

- `order` (`bt.Order`): Backtrader order object describing execution status and direction. ManagedStrategy inspects it to detect completed exits.

**Returns:** None.

**Notes:**

- The method reuses `BaseStrategy.notify_order` for standard logging, ensuring consistent output across strategy types.

**Example:**

```python
def notify_order(self, order):
    super().notify_order(order)
    if order.status == order.Completed and order.issell():
        self._reset_position_state()
```

## License

This documentation inherits the repository’s default license. Refer to the project root for licensing terms before redistributing or integrating the content.

