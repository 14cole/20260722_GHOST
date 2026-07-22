"""
Solver UI threshold guide.

Quality gate inputs:
- `residual <=`: max allowed linear solve residual norm (dimensionless).
- `cond <=`: max allowed condition-number estimate of the system matrix (dimensionless).
- `warns <=`: max allowed warning count emitted by the solver.
- `Require quality gate pass`: fail the solve if any quality threshold is exceeded.

Mesh convergence inputs:
- `fine factor`: panel-density multiplier for the fine mesh solve (`> 1.0`).
- `rms dB <=`: max allowed RMS of `(base_rcs_db - fine_rcs_db)` over all sample points.
- `max |dB| <=`: max allowed worst-case absolute dB delta between base and fine solves.
- `Run mesh convergence check`: execute the additional fine-mesh solve and compute deltas.
- `Require mesh convergence pass`: fail the solve if mesh thresholds are exceeded.
"""

import math
import os
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QComboBox,
    QCheckBox,
    QProgressBar,
    QSizePolicy,
    QToolButton,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from geometry_io import build_geometry_snapshot, parse_geometry
from grim_io import export_result_to_grim
from rcs_solver import solve_monostatic_rcs_2d, solve_bistatic_rcs_2d, compute_surface_currents
from bor_dispatch import solve_monostatic_rcs_bor
from solver_quality import evaluate_mesh_convergence, scale_snapshot_panel_density


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=6, height=5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()


class _SolveWorker(QObject):
    progress = Signal(int, str)
    finished = Signal(object, str)
    error = Signal(str)

    def __init__(
        self,
        snapshot: Dict[str, Any],
        source_path: str,
        base_dir: str,
        frequencies: List[float],
        elevations: List[float],
        pol: str,
        units: str,
        quality_thresholds: Dict[str, float | int],
        strict_quality_gate: bool,
        mesh_convergence: bool,
        mesh_fine_factor: float,
        mesh_rms_limit_db: float,
        mesh_max_abs_limit_db: float,
        strict_mesh_convergence: bool,
        cfie_alpha: float = 0.0,
        mesh_reference_ghz: Optional[float] = None,
        abort_event: Optional[Any] = None,
        solver_method: str = "auto",
        scattering_mode: str = "monostatic",
        observation_angles: Optional[List[float]] = None,
        solver_kind: str = "2d",
    ):
        super().__init__()
        self.solver_kind = str(solver_kind)
        self.snapshot = snapshot
        self.source_path = source_path
        self.base_dir = base_dir
        self.frequencies = frequencies
        self.elevations = elevations
        self.pol = pol
        self.units = units
        self.quality_thresholds = dict(quality_thresholds)
        self.strict_quality_gate = strict_quality_gate
        self.mesh_convergence = bool(mesh_convergence)
        self.mesh_fine_factor = float(mesh_fine_factor)
        self.mesh_rms_limit_db = float(mesh_rms_limit_db)
        self.mesh_max_abs_limit_db = float(mesh_max_abs_limit_db)
        self.strict_mesh_convergence = bool(strict_mesh_convergence)
        self.cfie_alpha = float(cfie_alpha)
        self.mesh_reference_ghz = mesh_reference_ghz
        self.abort_event = abort_event
        self.solver_method = str(solver_method)
        self.scattering_mode = str(scattering_mode)
        self.observation_angles = observation_angles

    def _on_progress(self, done_steps: int, total_steps: int, message: str) -> None:
        if total_steps <= 0:
            pct = 0
        else:
            pct = int(round(100.0 * float(done_steps) / float(total_steps)))
        pct = max(0, min(100, pct))
        self.progress.emit(pct, message)

    def _run_bor(self):
        """BoR (axisymmetric) route: elevations are ASPECT angles from the +z
        rotation axis.  CFIE alpha 0 (the 2D default) is promoted to 0.5 —
        closed PEC BoR bodies need CFIE against interior resonances."""

        if self.scattering_mode == "bistatic":
            raise ValueError("Bistatic sweeps are not supported by the BoR "
                             "solver yet; use Monostatic.")
        bor_alpha = self.cfie_alpha if self.cfie_alpha > 0.0 else 0.5
        kwargs = dict(
            geometry_snapshot=self.snapshot,
            frequencies_ghz=self.frequencies,
            elevations_deg=self.elevations,
            polarization=self.pol,
            geometry_units=self.units,
            material_base_dir=self.base_dir,
            mesh_reference_ghz=self.mesh_reference_ghz,
            cfie_alpha=bor_alpha,
            abort_event=self.abort_event,
        )
        result = solve_monostatic_rcs_bor(progress_callback=self._on_progress,
                                          **kwargs)
        if self.mesh_convergence:
            fine_snapshot = scale_snapshot_panel_density(self.snapshot,
                                                         self.mesh_fine_factor)
            fine_kwargs = dict(kwargs)
            fine_kwargs["geometry_snapshot"] = fine_snapshot
            fine_result = solve_monostatic_rcs_bor(**fine_kwargs)
            mesh_gate = evaluate_mesh_convergence(
                base_result=result,
                fine_result=fine_result,
                rms_limit_db=self.mesh_rms_limit_db,
                max_abs_limit_db=self.mesh_max_abs_limit_db,
            )
            mesh_gate["fine_factor"] = self.mesh_fine_factor
            metadata = dict(result.get("metadata", {}) or {})
            metadata["mesh_convergence"] = mesh_gate
            result["metadata"] = metadata
            if self.strict_mesh_convergence and not bool(mesh_gate.get("passed", False)):
                reason = str(mesh_gate.get("reason", "") or "mesh convergence failed")
                raise ValueError(f"Mesh convergence gate failed: {reason}")
        return result

    @Slot()
    def run(self):
        try:
            if self.solver_kind == "bor":
                result = self._run_bor()
            elif self.mesh_convergence:
                base_progress_scale = 0.6

                def base_cb(done_steps: int, total_steps: int, message: str) -> None:
                    if total_steps <= 0:
                        pct = 0
                    else:
                        pct = int(round(base_progress_scale * 100.0 * float(done_steps) / float(total_steps)))
                    pct = max(0, min(60, pct))
                    self.progress.emit(pct, message)

                result = solve_monostatic_rcs_2d(
                    geometry_snapshot=self.snapshot,
                    frequencies_ghz=self.frequencies,
                    elevations_deg=self.elevations,
                    polarization=self.pol,
                    geometry_units=self.units,
                    material_base_dir=self.base_dir,
                    progress_callback=base_cb,
                    quality_thresholds=self.quality_thresholds,
                    strict_quality_gate=self.strict_quality_gate,
                    compute_condition_number=self.strict_quality_gate,
                    cfie_alpha=self.cfie_alpha,
                    mesh_reference_ghz=self.mesh_reference_ghz,
                    abort_event=self.abort_event,
                    solver_method=self.solver_method,
                )
                self.progress.emit(61, "Running mesh convergence fine-mesh solve...")
                fine_snapshot = scale_snapshot_panel_density(self.snapshot, self.mesh_fine_factor)

                def fine_cb(done_steps: int, total_steps: int, message: str) -> None:
                    if total_steps <= 0:
                        pct = 60
                    else:
                        frac = 0.4 * float(done_steps) / float(total_steps)
                        pct = 60 + int(round(100.0 * frac))
                    pct = max(60, min(100, pct))
                    self.progress.emit(pct, f"Mesh check: {message}")

                fine_result = solve_monostatic_rcs_2d(
                    geometry_snapshot=fine_snapshot,
                    frequencies_ghz=self.frequencies,
                    elevations_deg=self.elevations,
                    polarization=self.pol,
                    geometry_units=self.units,
                    material_base_dir=self.base_dir,
                    progress_callback=fine_cb,
                    quality_thresholds=self.quality_thresholds,
                    strict_quality_gate=False,
                    compute_condition_number=False,
                    cfie_alpha=self.cfie_alpha,
                    mesh_reference_ghz=self.mesh_reference_ghz,
                    abort_event=self.abort_event,
                    solver_method=self.solver_method,
                )
                mesh_gate = evaluate_mesh_convergence(
                    base_result=result,
                    fine_result=fine_result,
                    rms_limit_db=self.mesh_rms_limit_db,
                    max_abs_limit_db=self.mesh_max_abs_limit_db,
                )
                mesh_gate["fine_factor"] = self.mesh_fine_factor
                metadata = dict(result.get("metadata", {}) or {})
                metadata["mesh_convergence"] = mesh_gate
                result["metadata"] = metadata

                if self.strict_mesh_convergence and not bool(mesh_gate.get("passed", False)):
                    reason = str(mesh_gate.get("reason", "") or "mesh convergence failed")
                    raise ValueError(f"Mesh convergence gate failed: {reason}")
            else:
                if self.scattering_mode == "bistatic" and self.observation_angles:
                    result = solve_bistatic_rcs_2d(
                        geometry_snapshot=self.snapshot,
                        frequencies_ghz=self.frequencies,
                        incidence_angles_deg=self.elevations,
                        observation_angles_deg=self.observation_angles,
                        polarization=self.pol,
                        geometry_units=self.units,
                        material_base_dir=self.base_dir,
                        progress_callback=self._on_progress,
                        cfie_alpha=self.cfie_alpha,
                        mesh_reference_ghz=self.mesh_reference_ghz,
                        abort_event=self.abort_event,
                        solver_method=self.solver_method,
                    )
                else:
                    result = solve_monostatic_rcs_2d(
                        geometry_snapshot=self.snapshot,
                        frequencies_ghz=self.frequencies,
                        elevations_deg=self.elevations,
                        polarization=self.pol,
                        geometry_units=self.units,
                        material_base_dir=self.base_dir,
                        progress_callback=self._on_progress,
                        quality_thresholds=self.quality_thresholds,
                        strict_quality_gate=self.strict_quality_gate,
                        compute_condition_number=self.strict_quality_gate,
                        cfie_alpha=self.cfie_alpha,
                        mesh_reference_ghz=self.mesh_reference_ghz,
                        abort_event=self.abort_event,
                        solver_method=self.solver_method,
                    )
        except Exception as exc:
            self.error.emit(str(exc))
            return
        self.finished.emit(result, self.source_path)


class SolverTab(QWidget):
    def __init__(self, geometry_tab=None, parent=None):
        super().__init__(parent)
        self.geometry_tab = geometry_tab
        self.last_result: Optional[Dict[str, Any]] = None
        self.last_source_path: str = ""
        self._solve_thread: Optional[QThread] = None
        self._solve_worker: Optional[_SolveWorker] = None
        self._is_solving: bool = False
        self._abort_event: Optional[threading.Event] = None

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([400, 700])

        root = QHBoxLayout(self)
        root.addWidget(splitter)

        self._update_mode_enables()

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        geometry_group = QGroupBox("Geometry Source")
        geometry_grid = QGridLayout(geometry_group)
        self.edit_geo_path = QLineEdit()
        self.edit_geo_path.setPlaceholderText("Optional: use .geo file directly, or use current Geometry tab")
        self.btn_browse_geo = QPushButton("Browse...")
        self.btn_use_tab = QPushButton("Use Geometry Tab")
        self.lbl_geo = QLabel("No explicit file selected. Solver will use Geometry tab if available.")
        self.lbl_geo.setWordWrap(True)
        geometry_grid.addWidget(QLabel("Geometry File"), 0, 0)
        geometry_grid.addWidget(self.edit_geo_path, 0, 1, 1, 2)
        geometry_grid.addWidget(self.btn_browse_geo, 1, 1)
        geometry_grid.addWidget(self.btn_use_tab, 1, 2)
        geometry_grid.addWidget(self.lbl_geo, 2, 0, 1, 3)
        layout.addWidget(geometry_group)

        options_group = QGroupBox("Solve Options")
        options_form = QFormLayout(options_group)

        self.cmb_solver_kind = QComboBox()
        self.cmb_solver_kind.addItem("2D (cylindrical, dBke)", userData="2d")
        self.cmb_solver_kind.addItem("BoR (axisymmetric, dBsm)", userData="bor")
        self.cmb_solver_kind.setToolTip(
            "2D: infinite-cylinder elevation cut (2D width).\n"
            "BoR: true 3-D RCS of a body of revolution. The drawing is the\n"
            "(rho, z) half-profile: x = radius >= 0, y = rotation axis, drawn\n"
            "from the +z axis end to the -z axis end. Elevation angles become\n"
            "aspect angles from +z (0 = nose-on, 90 = broadside)."
        )
        self.cmb_solver_kind.currentIndexChanged.connect(self._on_solver_kind_changed)

        self.cmb_units = QComboBox()
        self.cmb_units.addItems(["inches", "meters"])
        self.cmb_units.setCurrentText("inches")

        self.cmb_pol = QComboBox()
        # Elevation-cut convention: HH ↔ TM (E along horizontal z), VV ↔ TE.
        self.cmb_pol.addItem("HH", userData="HH")
        self.cmb_pol.addItem("VV", userData="VV")

        self.lbl_solve_method = QLabel("Linear / Galerkin")
        self.chk_strict_quality = QCheckBox("Require quality gate pass")
        self.chk_strict_quality.setChecked(False)
        self.edit_quality_residual_max = QLineEdit("1e-2")
        self.edit_quality_condition_max = QLineEdit("1e6")
        self.edit_quality_warnings_max = QLineEdit("10")
        quality_threshold_row = QWidget()
        quality_threshold_layout = QHBoxLayout(quality_threshold_row)
        quality_threshold_layout.setContentsMargins(0, 0, 0, 0)
        quality_threshold_layout.addWidget(QLabel("residual<="))
        quality_threshold_layout.addWidget(self.edit_quality_residual_max)
        quality_threshold_layout.addWidget(QLabel("cond<="))
        quality_threshold_layout.addWidget(self.edit_quality_condition_max)
        quality_threshold_layout.addWidget(QLabel("warns<="))
        quality_threshold_layout.addWidget(self.edit_quality_warnings_max)

        self.chk_mesh_convergence = QCheckBox("Run mesh convergence check")
        self.chk_mesh_convergence.setChecked(False)
        self.chk_strict_mesh = QCheckBox("Require mesh convergence pass")
        self.chk_strict_mesh.setChecked(False)
        self.edit_mesh_fine_factor = QLineEdit("1.5")
        self.edit_mesh_rms_max_db = QLineEdit("1.0")
        self.edit_mesh_max_abs_max_db = QLineEdit("3.0")
        mesh_threshold_row = QWidget()
        mesh_threshold_layout = QHBoxLayout(mesh_threshold_row)
        mesh_threshold_layout.setContentsMargins(0, 0, 0, 0)
        mesh_threshold_layout.addWidget(QLabel("fine factor"))
        mesh_threshold_layout.addWidget(self.edit_mesh_fine_factor)
        mesh_threshold_layout.addWidget(QLabel("rms dB<="))
        mesh_threshold_layout.addWidget(self.edit_mesh_rms_max_db)
        mesh_threshold_layout.addWidget(QLabel("max |dB|<="))
        mesh_threshold_layout.addWidget(self.edit_mesh_max_abs_max_db)

        self.btn_advanced_settings = QToolButton()
        self.btn_advanced_settings.setText("Advanced Settings")
        self.btn_advanced_settings.setCheckable(True)
        self.btn_advanced_settings.setChecked(False)
        self.btn_advanced_settings.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_advanced_settings.setArrowType(Qt.RightArrow)

        self.advanced_settings_widget = QWidget()
        advanced_form = QFormLayout(self.advanced_settings_widget)
        advanced_form.setContentsMargins(0, 0, 0, 0)

        self.edit_cfie_alpha = QLineEdit("0.0")
        self.edit_cfie_alpha.setToolTip(
            "Burton-Miller CFIE coupling (0 = disabled, default). "
            "Prevents interior-resonance artefacts on closed bodies."
        )
        self.edit_mesh_ref_ghz = QLineEdit("")
        self.edit_mesh_ref_ghz.setPlaceholderText("auto (per-frequency)")
        self.edit_mesh_ref_ghz.setToolTip(
            "Fix the mesh at this frequency (GHz) and reuse across the sweep. "
            "Leave empty to re-mesh per frequency."
        )

        self.cmb_solver_method = QComboBox()
        self.cmb_solver_method.addItem("Auto (LU/GMRES)", userData="auto")
        self.cmb_solver_method.addItem("Direct (LU)", userData="direct")
        self.cmb_solver_method.addItem("Iterative (GMRES)", userData="gmres")
        self.cmb_solver_method.addItem("FMM (fast multipole)", userData="fmm")
        self.cmb_solver_method.setToolTip(
            "Auto selects GMRES for large systems (>3000 DOFs), direct LU for small.\n"
            "FMM uses the fast multipole method — faster than dense above ~1000 panels.\n"
            "Currently FMM supports TE PEC/IBC (Robin MFIE). Other formulations fall back to dense."
        )

        advanced_form.addRow("CFIE alpha", self.edit_cfie_alpha)
        advanced_form.addRow("Mesh ref. (GHz)", self.edit_mesh_ref_ghz)
        advanced_form.addRow("Solver method", self.cmb_solver_method)
        advanced_form.addRow("Quality Gate", self.chk_strict_quality)
        advanced_form.addRow("Quality Thresholds", quality_threshold_row)
        advanced_form.addRow("Mesh Check", self.chk_mesh_convergence)
        advanced_form.addRow("Mesh Strict", self.chk_strict_mesh)
        advanced_form.addRow("Mesh Thresholds", mesh_threshold_row)
        self.advanced_settings_widget.setVisible(False)

        self.cmb_freq_mode = QComboBox()
        self.cmb_freq_mode.addItems(["Discrete List", "Start / Stop / Step"])
        self.edit_freq_list = QLineEdit("1.0, 3.0, 10.0")
        self.edit_freq_start = QLineEdit("1.0")
        self.edit_freq_stop = QLineEdit("10.0")
        self.edit_freq_step = QLineEdit("1.0")

        self.cmb_elev_mode = QComboBox()
        self.cmb_elev_mode.addItems(["Discrete List", "Start / Stop / Step"])
        self.edit_elev_list = QLineEdit("0, 30, 60, 90, 120, 150, 180")
        self.edit_elev_start = QLineEdit("0")
        self.edit_elev_stop = QLineEdit("180")
        self.edit_elev_step = QLineEdit("2")

        self.cmb_scatter_mode = QComboBox()
        self.cmb_scatter_mode.addItem("Monostatic", userData="monostatic")
        self.cmb_scatter_mode.addItem("Bistatic", userData="bistatic")
        self.cmb_scatter_mode.setToolTip(
            "Monostatic: backscatter (obs = inc). "
            "Bistatic: specify separate observation angles."
        )

        self.edit_obs_angles = QLineEdit("0, 30, 60, 90, 120, 150, 180")
        self.edit_obs_angles.setToolTip(
            "Observation angles (degrees) for bistatic mode. "
            "Backscatter convention: obs=inc is backscatter, obs=inc+180 is forward scatter."
        )
        self.lbl_obs_angles = QLabel("Observation Angles (deg)")
        self.edit_obs_angles.setVisible(False)
        self.lbl_obs_angles.setVisible(False)
        self.cmb_scatter_mode.currentIndexChanged.connect(self._on_scatter_mode_changed)

        freq_sweep_row = QWidget()
        freq_sweep_layout = QHBoxLayout(freq_sweep_row)
        freq_sweep_layout.setContentsMargins(0, 0, 0, 0)
        freq_sweep_layout.addWidget(QLabel("Start"))
        freq_sweep_layout.addWidget(self.edit_freq_start)
        freq_sweep_layout.addWidget(QLabel("Stop"))
        freq_sweep_layout.addWidget(self.edit_freq_stop)
        freq_sweep_layout.addWidget(QLabel("Step"))
        freq_sweep_layout.addWidget(self.edit_freq_step)

        elev_sweep_row = QWidget()
        elev_sweep_layout = QHBoxLayout(elev_sweep_row)
        elev_sweep_layout.setContentsMargins(0, 0, 0, 0)
        elev_sweep_layout.addWidget(QLabel("Start"))
        elev_sweep_layout.addWidget(self.edit_elev_start)
        elev_sweep_layout.addWidget(QLabel("Stop"))
        elev_sweep_layout.addWidget(self.edit_elev_stop)
        elev_sweep_layout.addWidget(QLabel("Step"))
        elev_sweep_layout.addWidget(self.edit_elev_step)

        options_form.addRow("Solver", self.cmb_solver_kind)
        options_form.addRow("Units In Geometry", self.cmb_units)
        options_form.addRow("Polarization", self.cmb_pol)
        options_form.addRow("Discretization", self.lbl_solve_method)
        options_form.addRow(self.btn_advanced_settings)
        options_form.addRow(self.advanced_settings_widget)
        options_form.addRow("Frequency Mode", self.cmb_freq_mode)
        options_form.addRow("Frequencies (GHz)", self.edit_freq_list)
        options_form.addRow("Frequency Sweep", freq_sweep_row)
        options_form.addRow("Elevation Mode", self.cmb_elev_mode)
        options_form.addRow("Elevations (deg)", self.edit_elev_list)
        options_form.addRow("Elevation Sweep", elev_sweep_row)
        options_form.addRow("Scattering Mode", self.cmb_scatter_mode)
        options_form.addRow(self.lbl_obs_angles, self.edit_obs_angles)
        layout.addWidget(options_group)

        output_group = QGroupBox("Output")
        output_grid = QGridLayout(output_group)
        self.edit_output = QLineEdit("rcs_output.grim")
        self.btn_browse_output = QPushButton("Browse...")
        self.chk_export_after_solve = QCheckBox("Export .grim automatically after solve")
        self.chk_export_after_solve.setChecked(True)
        output_grid.addWidget(QLabel("GRIM Output"), 0, 0)
        output_grid.addWidget(self.edit_output, 0, 1)
        output_grid.addWidget(self.btn_browse_output, 0, 2)
        output_grid.addWidget(self.chk_export_after_solve, 1, 0, 1, 3)
        layout.addWidget(output_group)

        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("Run Solver")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_export = QPushButton("Export Last Result")
        self.btn_currents = QPushButton("Compute Currents")
        self.btn_currents.setToolTip(
            "Compute boundary currents for the first frequency and elevation. "
            "Exports element positions, normals, and current density to a JSON file."
        )
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_cancel)
        btn_row.addWidget(self.btn_export)
        btn_row.addWidget(self.btn_currents)
        layout.addLayout(btn_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.lbl_status = QLabel("Ready.")
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)
        layout.addStretch(1)

        self.btn_browse_geo.clicked.connect(self._browse_geo)
        self.btn_use_tab.clicked.connect(self._use_geometry_tab)
        self.btn_browse_output.clicked.connect(self._browse_output)
        self.btn_run.clicked.connect(self._run_solver)
        self.btn_cancel.clicked.connect(self._cancel_solver)
        self.btn_export.clicked.connect(self._export_last_result)
        self.btn_currents.clicked.connect(self._compute_currents)
        self.btn_advanced_settings.toggled.connect(self._toggle_advanced_settings)
        self.cmb_freq_mode.currentIndexChanged.connect(self._update_mode_enables)
        self.cmb_elev_mode.currentIndexChanged.connect(self._update_mode_enables)

        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self.canvas = MplCanvas(panel)
        self.toolbar = NavigationToolbar(self.canvas, panel)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, stretch=2)

        self.table_results = QTableWidget()
        self.table_results.setColumnCount(4)
        self.table_results.setHorizontalHeaderLabels(
            ["Frequency (GHz)", "Elevation (deg)", "RCS (linear)", "RCS (dB)"]
        )
        layout.addWidget(self.table_results, stretch=1)
        return panel

    def _browse_geo(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, "Select Geometry File", "", "Geometry Files (*.geo);;All Files (*)"
        )
        if not fname:
            return
        self.edit_geo_path.setText(fname)
        self.lbl_geo.setText(f"Using geometry file: {fname}")

    def _use_geometry_tab(self):
        self.edit_geo_path.clear()
        if self.geometry_tab is None:
            self.lbl_geo.setText("Geometry tab is not connected in this session.")
            return
        snapshot = self.geometry_tab.get_geometry_snapshot()
        segment_count = int(snapshot.get("segment_count", 0))
        title = snapshot.get("title", "Geometry")
        if segment_count <= 0:
            self.lbl_geo.setText("Geometry tab has no loaded segments yet.")
            return
        self.lbl_geo.setText(f"Using Geometry tab: {title} ({segment_count} segment(s)).")

    def _browse_output(self):
        fname, _ = QFileDialog.getSaveFileName(
            self, "Select Output .grim", "rcs_output.grim", "GRIM Files (*.grim);;All Files (*)"
        )
        if not fname:
            return
        self.edit_output.setText(fname)

    def _toggle_advanced_settings(self, checked: bool):
        self.advanced_settings_widget.setVisible(bool(checked))
        self.btn_advanced_settings.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)

    def _update_mode_enables(self):
        freq_discrete = self.cmb_freq_mode.currentIndex() == 0
        elev_discrete = self.cmb_elev_mode.currentIndex() == 0

        self.edit_freq_list.setEnabled(freq_discrete)
        self.edit_freq_start.setEnabled(not freq_discrete)
        self.edit_freq_stop.setEnabled(not freq_discrete)
        self.edit_freq_step.setEnabled(not freq_discrete)

        self.edit_elev_list.setEnabled(elev_discrete)
        self.edit_elev_start.setEnabled(not elev_discrete)
        self.edit_elev_stop.setEnabled(not elev_discrete)
        self.edit_elev_step.setEnabled(not elev_discrete)

    def _parse_list(self, text: str, field_name: str) -> List[float]:
        tokens = [tok for tok in re.split(r"[,\s]+", text.strip()) if tok]
        if not tokens:
            raise ValueError(f"{field_name}: no values were provided.")
        values: List[float] = []
        for tok in tokens:
            try:
                v = float(tok)
            except ValueError:
                raise ValueError(f"{field_name}: invalid numeric token '{tok}'.")
            if not math.isfinite(v):
                raise ValueError(f"{field_name}: non-finite value '{tok}' (NaN/Inf not allowed).")
            values.append(v)
        return values

    def _parse_sweep(self, start_s: str, stop_s: str, step_s: str, field_name: str) -> List[float]:
        try:
            start = float(start_s)
            stop = float(stop_s)
            step = abs(float(step_s))
        except ValueError:
            raise ValueError(f"{field_name}: start, stop, and step must be numeric.")
        if not (math.isfinite(start) and math.isfinite(stop) and math.isfinite(step)):
            raise ValueError(f"{field_name}: start, stop, and step must be finite (NaN/Inf not allowed).")
        if step <= 0.0:
            raise ValueError(f"{field_name}: step must be > 0.")

        direction = 1.0 if stop >= start else -1.0
        signed_step = step * direction
        values: List[float] = []
        current = start
        for _ in range(20_000):
            if direction > 0 and current > stop + 1e-9:
                break
            if direction < 0 and current < stop - 1e-9:
                break
            values.append(round(current, 12))
            current += signed_step
        if not values or abs(values[-1] - stop) > 1e-9:
            values.append(stop)
        if len(values) > 5000:
            raise ValueError(f"{field_name}: too many samples ({len(values)}).")
        return values

    def _collect_frequency_values(self) -> List[float]:
        if self.cmb_freq_mode.currentIndex() == 0:
            freqs = self._parse_list(self.edit_freq_list.text(), "Frequencies")
        else:
            freqs = self._parse_sweep(
                self.edit_freq_start.text(),
                self.edit_freq_stop.text(),
                self.edit_freq_step.text(),
                "Frequencies",
            )
        if any(f <= 0 for f in freqs):
            raise ValueError("Frequencies must be positive values in GHz.")
        return freqs

    def _collect_elevation_values(self) -> List[float]:
        if self.cmb_elev_mode.currentIndex() == 0:
            return self._parse_list(self.edit_elev_list.text(), "Elevations")
        return self._parse_sweep(
            self.edit_elev_start.text(),
            self.edit_elev_stop.text(),
            self.edit_elev_step.text(),
            "Elevations",
        )

    def _load_geometry_for_solver(self) -> Tuple[Dict[str, Any], str, str]:
        path = self.edit_geo_path.text().strip()
        if path:
            with open(path, "r") as f:
                text = f.read()
            title, segments, ibcs_entries, dielectric_entries = parse_geometry(text)
            snapshot = build_geometry_snapshot(title, segments, ibcs_entries, dielectric_entries)
            source_path = os.path.abspath(path)
            base_dir = os.path.dirname(source_path)
            return snapshot, source_path, base_dir

        if self.geometry_tab is None:
            raise ValueError("No geometry file selected and Geometry tab is unavailable.")

        snapshot = self.geometry_tab.get_geometry_snapshot()
        segment_count = int(snapshot.get("segment_count", 0))
        if segment_count <= 0:
            raise ValueError("No geometry loaded. Load a .geo file or use the Geometry tab first.")

        source_path = str(snapshot.get("source_path", "") or "")
        if not source_path:
            source_path = str(getattr(self.geometry_tab, "loaded_path", "") or "")
        base_dir = os.path.dirname(source_path) if source_path else os.getcwd()
        return snapshot, source_path, base_dir

    def _set_solving_state(self, solving: bool):
        self._is_solving = solving
        self.btn_run.setEnabled(not solving)
        self.btn_export.setEnabled(not solving)
        self.btn_currents.setEnabled(not solving)
        self.btn_browse_geo.setEnabled(not solving)
        self.btn_use_tab.setEnabled(not solving)
        self.btn_browse_output.setEnabled(not solving)
        self.edit_geo_path.setEnabled(not solving)
        self.edit_output.setEnabled(not solving)
        self.cmb_units.setEnabled(not solving)
        self.cmb_pol.setEnabled(not solving)
        self.cmb_freq_mode.setEnabled(not solving)
        self.cmb_elev_mode.setEnabled(not solving)
        self.edit_freq_list.setEnabled(not solving and self.cmb_freq_mode.currentIndex() == 0)
        self.edit_elev_list.setEnabled(not solving and self.cmb_elev_mode.currentIndex() == 0)
        self.edit_freq_start.setEnabled(not solving and self.cmb_freq_mode.currentIndex() != 0)
        self.edit_freq_stop.setEnabled(not solving and self.cmb_freq_mode.currentIndex() != 0)
        self.edit_freq_step.setEnabled(not solving and self.cmb_freq_mode.currentIndex() != 0)
        self.edit_elev_start.setEnabled(not solving and self.cmb_elev_mode.currentIndex() != 0)
        self.edit_elev_stop.setEnabled(not solving and self.cmb_elev_mode.currentIndex() != 0)
        self.edit_elev_step.setEnabled(not solving and self.cmb_elev_mode.currentIndex() != 0)
        self.chk_export_after_solve.setEnabled(not solving)
        self.chk_strict_quality.setEnabled(not solving)
        self.edit_quality_residual_max.setEnabled(not solving)
        self.edit_quality_condition_max.setEnabled(not solving)
        self.edit_quality_warnings_max.setEnabled(not solving)
        self.chk_mesh_convergence.setEnabled(not solving)
        self.chk_strict_mesh.setEnabled(not solving)
        self.edit_mesh_fine_factor.setEnabled(not solving)
        self.edit_mesh_rms_max_db.setEnabled(not solving)
        self.edit_mesh_max_abs_max_db.setEnabled(not solving)
        self.btn_run.setText("Solving..." if solving else "Run Solver")
        self.btn_cancel.setEnabled(solving)

    def _on_scatter_mode_changed(self, _index: int = 0) -> None:
        is_bistatic = (self.cmb_scatter_mode.currentData() == "bistatic")
        self.edit_obs_angles.setVisible(is_bistatic)
        self.lbl_obs_angles.setVisible(is_bistatic)

    def _on_solver_kind_changed(self, _index: int = 0) -> None:
        is_bor = (self.cmb_solver_kind.currentData() == "bor")
        if is_bor:
            # BoR is monostatic-only; the 2D method/quality knobs do not apply.
            self.cmb_scatter_mode.setCurrentIndex(0)
        self.cmb_scatter_mode.setEnabled(not is_bor)
        self.cmb_solver_method.setEnabled(not is_bor)
        self.chk_strict_quality.setEnabled(not is_bor)
        self.lbl_solve_method.setText(
            "BoR-MoM (azimuthal modes)" if is_bor else "Linear / Galerkin")

    def _compute_currents(self) -> None:
        """Compute surface currents at first freq/elev and save to JSON."""
        if self._is_solving:
            QMessageBox.information(self, "Currents", "Solver is running. Please wait.")
            return

        try:
            snapshot, source_path, base_dir = self._load_geometry_for_solver()
            frequencies = self._collect_frequency_values()
            elevations = self._collect_elevation_values()
            if not frequencies:
                raise ValueError("Need at least one frequency.")
            if not elevations:
                raise ValueError("Need at least one elevation.")
            pol = str(self.cmb_pol.currentData() or "TM")
            units = str(self.cmb_units.currentText() or "inches")
            cfie_text = self.edit_cfie_alpha.text().strip()
            cfie_alpha = float(cfie_text) if cfie_text else 0.0
        except Exception as exc:
            QMessageBox.critical(self, "Currents Error", str(exc))
            return

        self.lbl_status.setText("Computing surface currents...")
        try:
            result = compute_surface_currents(
                geometry_snapshot=snapshot,
                frequency_ghz=frequencies[0],
                elevation_deg=elevations[0],
                polarization=pol,
                geometry_units=units,
                material_base_dir=base_dir,
                cfie_alpha=cfie_alpha,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Currents Error", str(exc))
            self.lbl_status.setText(f"Currents failed: {exc}")
            return

        import json
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Surface Currents", "surface_currents.json", "JSON Files (*.json)",
        )
        if not save_path:
            self.lbl_status.setText("Currents computed (not saved).")
            return

        try:
            with open(save_path, "w") as f:
                json.dump(result, f, indent=2)
            n_elem = result.get("element_count", 0)
            formulation = result.get("formulation", "?")
            self.lbl_status.setText(
                f"Currents saved: {save_path} ({n_elem} elements, {formulation})"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", str(exc))

    def _cancel_solver(self):
        """Signal the running solver to abort."""
        if self._abort_event is not None:
            self._abort_event.set()
            self.lbl_status.setText("Cancelling...")
            self.btn_cancel.setEnabled(False)

    @Slot(int, str)
    def _on_solver_progress(self, pct: int, message: str):
        self.progress.setValue(max(0, min(100, int(pct))))
        if message:
            self.lbl_status.setText(f"Solving... {message}")

    @Slot(object, str)
    def _on_solver_finished(self, result: Dict[str, Any], source_path: str):
        self.last_result = result
        self.last_source_path = source_path
        self._populate_results_table(result)
        self._plot_results(result)

        metadata = result.get("metadata", {}) or {}
        freq_count = int(metadata.get("frequency_count", 0))
        elev_count = int(metadata.get("elevation_count", 0))
        units = str(metadata.get("geometry_units_in", self.cmb_units.currentText()))
        pol = str(result.get("polarization", self.cmb_pol.currentData() or "TM"))
        write_summary = (
            f"Solved monostatic 2D RCS with {freq_count} frequency(ies) and "
            f"{elev_count} elevation angle(s)."
        )
        qg = metadata.get("quality_gate", {}) or {}
        quality_suffix = ""
        if isinstance(qg, dict):
            if bool(qg.get("passed", True)):
                quality_suffix = " Quality gate: PASS."
            else:
                violations = qg.get("violations", []) or []
                joined = "; ".join(str(v) for v in violations[:2])
                if len(violations) > 2:
                    joined += f" (+{len(violations) - 2} more)"
                quality_suffix = f" Quality gate: FAIL ({joined})."
        mesh_suffix = ""
        mesh = metadata.get("mesh_convergence", {}) or {}
        if isinstance(mesh, dict) and bool(mesh.get("enabled", False)):
            if bool(mesh.get("passed", False)):
                mesh_suffix = (
                    " Mesh convergence: PASS "
                    f"(rms={float(mesh.get('rms_db', 0.0)):.3g} dB, "
                    f"max={float(mesh.get('max_abs_db', 0.0)):.3g} dB)."
                )
            else:
                reason = str(mesh.get("reason", "") or "criteria not met")
                mesh_suffix = f" Mesh convergence: FAIL ({reason})."

        self.progress.setValue(100)
        self.lbl_status.setText(write_summary + quality_suffix + mesh_suffix)

        if self.chk_export_after_solve.isChecked():
            try:
                out_text = self.edit_output.text().strip() or "rcs_output.grim"
                files = export_result_to_grim(
                    result,
                    out_text,
                    polarization=result.get("polarization_export", result.get("polarization", "TM")),
                    source_path=source_path,
                    history=(
                        f"solver=monostatic2d "
                        f"freq_count={freq_count} "
                        f"elev_count={elev_count} "
                        f"units={units} "
                        f"pol={pol}"
                    ),
                )
                self.lbl_status.setText(
                    write_summary
                    + quality_suffix
                    + mesh_suffix
                    + " Exported "
                    + ", ".join(os.path.basename(path) for path in files)
                )
            except Exception as exc:
                QMessageBox.warning(self, "Export Warning", f"Solve completed, but export failed:\n{exc}")

        self._set_solving_state(False)

    @Slot(str)
    def _on_solver_error(self, message: str):
        self.progress.setValue(0)
        self.lbl_status.setText(f"Solve failed: {message}")
        QMessageBox.critical(self, "Solver Error", message)
        self._set_solving_state(False)

    @Slot()
    def _on_solver_thread_finished(self):
        self._solve_worker = None
        self._solve_thread = None
        self._abort_event = None

    def _run_solver(self):
        if self._is_solving:
            return
        try:
            frequencies = self._collect_frequency_values()
            elevations = self._collect_elevation_values()
            snapshot, source_path, base_dir = self._load_geometry_for_solver()
            pol = str(self.cmb_pol.currentData())
            units = self.cmb_units.currentText()
            strict_quality = bool(self.chk_strict_quality.isChecked())
            quality_residual_max = float(self.edit_quality_residual_max.text().strip())
            quality_condition_max = float(self.edit_quality_condition_max.text().strip())
            quality_warnings_max = int(float(self.edit_quality_warnings_max.text().strip()))
            if quality_residual_max <= 0.0:
                raise ValueError("Quality residual threshold must be > 0.")
            if quality_condition_max <= 0.0:
                raise ValueError("Quality condition threshold must be > 0.")
            if quality_warnings_max < 0:
                raise ValueError("Quality warning threshold must be >= 0.")
            quality_thresholds: Dict[str, float | int] = {
                "residual_norm_max": quality_residual_max,
                "condition_est_max": quality_condition_max,
                "warnings_max": quality_warnings_max,
            }

            mesh_convergence = bool(self.chk_mesh_convergence.isChecked())
            strict_mesh_convergence = bool(self.chk_strict_mesh.isChecked())
            mesh_fine_factor = float(self.edit_mesh_fine_factor.text().strip())
            mesh_rms_limit_db = float(self.edit_mesh_rms_max_db.text().strip())
            mesh_max_abs_limit_db = float(self.edit_mesh_max_abs_max_db.text().strip())
            if mesh_convergence and mesh_fine_factor <= 1.0:
                raise ValueError("Mesh convergence requires fine factor > 1.0.")
            if mesh_rms_limit_db <= 0.0:
                raise ValueError("Mesh RMS threshold must be > 0.")
            if mesh_max_abs_limit_db <= 0.0:
                raise ValueError("Mesh max-abs threshold must be > 0.")

            # Advanced settings
            cfie_text = self.edit_cfie_alpha.text().strip()
            cfie_alpha = float(cfie_text) if cfie_text else 0.0
            if not math.isfinite(cfie_alpha) or cfie_alpha < 0.0:
                raise ValueError("CFIE alpha must be a non-negative finite number.")

            mesh_ref_text = self.edit_mesh_ref_ghz.text().strip()
            mesh_reference_ghz: Optional[float] = None
            if mesh_ref_text:
                mesh_reference_ghz = float(mesh_ref_text)
                if not math.isfinite(mesh_reference_ghz) or mesh_reference_ghz <= 0.0:
                    raise ValueError("Mesh reference frequency must be a positive finite GHz value.")

            # Scattering mode and observation angles.
            scatter_mode = str(self.cmb_scatter_mode.currentData() or "monostatic")
            obs_angles_list: Optional[List[float]] = None
            if scatter_mode == "bistatic":
                obs_angles_list = self._parse_list(self.edit_obs_angles.text(), "Observation angles")
                if not obs_angles_list:
                    raise ValueError("Bistatic mode requires at least one observation angle.")
        except Exception as exc:
            QMessageBox.critical(self, "Solver Error", str(exc))
            self.lbl_status.setText(f"Solve failed: {exc}")
            return

        self.progress.setValue(0)
        self.lbl_status.setText("Starting solver thread...")
        self._set_solving_state(True)

        abort_event = threading.Event()
        self._abort_event = abort_event

        thread = QThread(self)
        worker = _SolveWorker(
            snapshot=snapshot,
            source_path=source_path,
            base_dir=base_dir,
            frequencies=frequencies,
            elevations=elevations,
            pol=pol,
            units=units,
            quality_thresholds=quality_thresholds,
            strict_quality_gate=strict_quality,
            mesh_convergence=mesh_convergence,
            mesh_fine_factor=mesh_fine_factor,
            mesh_rms_limit_db=mesh_rms_limit_db,
            mesh_max_abs_limit_db=mesh_max_abs_limit_db,
            strict_mesh_convergence=strict_mesh_convergence,
            cfie_alpha=cfie_alpha,
            mesh_reference_ghz=mesh_reference_ghz,
            abort_event=abort_event,
            solver_method=str(self.cmb_solver_method.currentData() or "auto"),
            scattering_mode=scatter_mode,
            observation_angles=obs_angles_list,
            solver_kind=str(self.cmb_solver_kind.currentData() or "2d"),
        )
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress.connect(self._on_solver_progress)
        worker.finished.connect(self._on_solver_finished)
        worker.error.connect(self._on_solver_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.error.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_solver_thread_finished)

        self._solve_thread = thread
        self._solve_worker = worker
        thread.start()

    def _export_last_result(self):
        if self._is_solving:
            QMessageBox.information(self, "Export", "Solver is currently running. Please wait for completion.")
            return
        if not self.last_result:
            QMessageBox.information(self, "Export", "No solver result exists yet. Run the solver first.")
            return
        out_text = self.edit_output.text().strip()
        if not out_text:
            QMessageBox.warning(self, "Export", "Please provide an output path.")
            return
        try:
            files = export_result_to_grim(
                self.last_result,
                out_text,
                polarization=self.last_result.get("polarization_export", self.last_result.get("polarization", "TM")),
                source_path=self.last_source_path,
                history=(f"solver={self.last_result.get('solver', 'monostatic2d')} "
                         "manual_export=1 plot_source=10log10(rcs_linear)"),
                preserve_raw_complex_amplitude=True,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))
            return
        QMessageBox.information(self, "Exported", "\n".join(files))
        self.lbl_status.setText("Exported: " + ", ".join(os.path.basename(path) for path in files))

    def _display_db_from_linear(self, row: Dict[str, Any]) -> float:
        lin = float(row.get("rcs_linear", 0.0))
        if not math.isfinite(lin) or lin <= 0.0:
            lin = 1.0e-12
        return 10.0 * math.log10(lin)

    def _populate_results_table(self, result: Dict[str, Any]):
        rows = sorted(
            result.get("samples", []),
            key=lambda row: (float(row.get("frequency_ghz", 0.0)), float(row.get("theta_scat_deg", 0.0))),
        )
        self.table_results.clearContents()
        self.table_results.setRowCount(len(rows))
        for r, row in enumerate(rows):
            freq = float(row.get("frequency_ghz", 0.0))
            elev = float(row.get("theta_scat_deg", 0.0))
            lin = float(row.get("rcs_linear", 0.0))
            db = self._display_db_from_linear(row)
            self.table_results.setItem(r, 0, QTableWidgetItem(f"{freq:.6g}"))
            self.table_results.setItem(r, 1, QTableWidgetItem(f"{elev:.6g}"))
            self.table_results.setItem(r, 2, QTableWidgetItem(f"{lin:.6e}"))
            self.table_results.setItem(r, 3, QTableWidgetItem(f"{db:.3f}"))

    def _plot_results(self, result: Dict[str, Any]):
        by_freq: Dict[float, List[Dict[str, Any]]] = {}
        for row in result.get("samples", []):
            freq = float(row.get("frequency_ghz", 0.0))
            by_freq.setdefault(freq, []).append(row)

        ax = self.canvas.ax
        ax.clear()
        for freq in sorted(by_freq.keys()):
            rows = sorted(by_freq[freq], key=lambda row: float(row.get("theta_scat_deg", 0.0)))
            x = [float(row.get("theta_scat_deg", 0.0)) for row in rows]
            y = [self._display_db_from_linear(row) for row in rows]
            ax.plot(x, y, linewidth=1.8, label=f"{freq:g} GHz")

        if str(result.get("solver", "")) == "bor_mom_rcs":
            ax.set_title("Monostatic RCS (BoR)")
            ax.set_xlabel("Aspect angle from +z axis (deg)")
            ax.set_ylabel("RCS (dBsm)")
        else:
            ax.set_title("Monostatic 2D Width")
            ax.set_xlabel("Elevation (deg)")
            ax.set_ylabel("2D Width from exported linear power (dB)")
        ax.grid(True, alpha=0.3)
        if len(by_freq) <= 12:
            ax.legend(loc="best")
        self.canvas.draw_idle()