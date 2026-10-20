"""geo_simulator – Simulate geo-experiment conversion data with a known lift.

Public API::

    from geo_simulator import GeoSimulator, SimulationResult
    from geo_simulator.config import (
        SimulationConfig,
        LiftConfig,
        TreatmentConfig,
        BaselineConfig,
        NoiseConfig,
        GeographyConfig,
        ContaminationConfig,
        CalibrationConfig,
    )
    from geo_simulator.calibration import calibrate
    from geo_simulator.diagnostics import cointegration_report, pre_period_fit
"""

from geo_simulator.calibration import CalibrationResult, calibrate
from geo_simulator.config import (
    BaselineConfig,
    CalibrationConfig,
    ContaminationConfig,
    GeographyConfig,
    LiftConfig,
    NoiseConfig,
    SimulationConfig,
    TreatmentConfig,
)
from geo_simulator.simulator import GeoSimulator, SimulationResult


__version__ = "0.2.1"


__all__ = [
    "GeoSimulator",
    "SimulationResult",
    "SimulationConfig",
    "LiftConfig",
    "TreatmentConfig",
    "BaselineConfig",
    "NoiseConfig",
    "GeographyConfig",
    "ContaminationConfig",
    "CalibrationConfig",
    "CalibrationResult",
    "calibrate",
]
