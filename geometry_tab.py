import math
import os
from typing import Any, Dict, List, Optional, Set, Tuple

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from geometry_io import (
    IBC_KINDS,
    ChainSpec,
    Segment,
    build_geometry_snapshot,
    build_geometry_text,
    check_orientation_consistency,
    is_ibc_inline_row,
    is_tabulated_row,
    parse_geometry,
)


SEGMENT_TYPE_OPTIONS: List[Tuple[str, str]] = [
    ("1", "1 (sheet)"),
    ("2", "2 (PEC)"),
    ("3", "3 (diel/air)"),
    ("4", "4 (PEC/diel)"),
    ("5", "5 (diel/diel)"),
]


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()


class GeometryTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        splitter = QSplitter(Qt.Horizontal)

        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        self.canvas = MplCanvas(plot_container)

        self.toolbar = NavigationToolbar(self.canvas, plot_container)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color: #333; padding: 4px; font-family: monospace;")
        plot_layout.addWidget(self.lbl_status)
        splitter.addWidget(plot_container)

        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)

        btn_row = QHBoxLayout()
        self.btn_load = QPushButton("Load")
        self.btn_save = QPushButton("Save")
        self.btn_validate = QPushButton("Validate")
        self.chk_show_normals = QCheckBox("Show Normals")
        self.chk_show_normals.setToolTip(
            "Draw the normal of every primitive, coloured by the material it "
            "points into (blue = air, grey = PEC, green = dielectric), with a "
            "label at each segment midpoint showing 'facing | behind'."
        )
        self.chk_show_impedance = QCheckBox("Show Impedance")
        self.chk_show_impedance.setToolTip(
            "Colour segments by IBC impedance. Tapered segments show a gradient; "
            "start = green dot, end = red dot."
        )
        self.chk_fill_materials = QCheckBox("Fill Materials")
        self.chk_fill_materials.setToolTip(
            "Fill enclosed regions with their material colour (grey = PEC, "
            "green tints = dielectrics, white = air) as implied by each "
            "segment's winding. A region whose boundary segments disagree "
            "about the enclosed material is hatched red."
        )
        btn_row.addWidget(self.btn_load)
        btn_row.addWidget(self.btn_save)
        btn_row.addWidget(self.btn_validate)
        btn_row.addWidget(self.chk_show_normals)
        btn_row.addWidget(self.chk_show_impedance)
        btn_row.addWidget(self.chk_fill_materials)
        btn_row.addStretch(1)
        right_layout.addLayout(btn_row)

        self.table = QTableWidget()
        self.table.setRowCount(0)
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["Name", "Type", "N", "IBC/Resistance", "pos_mat", "neg_mat"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        right_layout.addWidget(self.table)

        bottom_row = QHBoxLayout()

        ibc_box = QVBoxLayout()
        self.lbl_ibc = QLabel("IBCS/Resistances")
        self.table_ibc = QTableWidget()
        self.table_ibc.setRowCount(0)
        self.table_ibc.setColumnCount(0)
        ibc_box.addWidget(self.lbl_ibc)
        ibc_box.addWidget(self.table_ibc)
        ibc_btn_row = QHBoxLayout()
        self.btn_ibc_add = QPushButton("+")
        self.btn_ibc_remove = QPushButton("−")
        ibc_btn_row.addWidget(self.btn_ibc_add)
        ibc_btn_row.addWidget(self.btn_ibc_remove)
        ibc_btn_row.addStretch(1)
        ibc_box.addLayout(ibc_btn_row)

        diel_box = QVBoxLayout()
        self.lbl_diel = QLabel("Dielectrics")
        self.table_diel = QTableWidget()
        self.table_diel.setRowCount(0)
        self.table_diel.setColumnCount(0)
        diel_box.addWidget(self.lbl_diel)
        diel_box.addWidget(self.table_diel)
        diel_btn_row = QHBoxLayout()
        self.btn_diel_add = QPushButton("+")
        self.btn_diel_remove = QPushButton("−")
        diel_btn_row.addWidget(self.btn_diel_add)
        diel_btn_row.addWidget(self.btn_diel_remove)
        diel_btn_row.addStretch(1)
        diel_box.addLayout(diel_btn_row)

        bottom_row.addLayout(ibc_box, stretch=1)
        bottom_row.addLayout(diel_box, stretch=1)
        right_layout.addLayout(bottom_row)

        splitter.addWidget(right_container)

        splitter.setSizes([700, 300])
        main_layout = QHBoxLayout(self)
        main_layout.addWidget(splitter)

        self.btn_load.clicked.connect(self.load_geo)
        self.btn_save.clicked.connect(self.save_geo)
        self.btn_validate.clicked.connect(self.validate_geometry)
        self.btn_ibc_add.clicked.connect(self._ibc_add_row)
        self.btn_ibc_remove.clicked.connect(self._ibc_remove_row)
        self.btn_diel_add.clicked.connect(self._diel_add_row)
        self.btn_diel_remove.clicked.connect(self._diel_remove_row)
        self.chk_show_normals.toggled.connect(self._on_show_normals_toggled)
        self.chk_show_impedance.toggled.connect(self._on_show_impedance_toggled)
        self.chk_fill_materials.toggled.connect(self._on_fill_materials_toggled)

        self.title: str = "Geometry"
        self.segments: List[Segment] = []
        self.ibcs_entries: List[List[str]] = []
        self.dielectric_entries: List[List[str]] = []
        self.segment_lines: List = []
        self.segment_base_colors: List[str] = []
        self._populating: bool = False
        self._syncing_selection: bool = False
        self._selected_row: Optional[int] = None
        self._last_ext: str = ".geo"
        self.loaded_path: str = ""
        self.issue_rows: Set[int] = set()
        self.normal_artists: List[Any] = []
        # Impedance overlay artists: gradient fills, endpoint markers, labels.
        self.impedance_artists: List[Any] = []
        # Material fill artists: region polygons from the Fill Materials toggle.
        self.fill_artists: List[Any] = []

        self.table.itemChanged.connect(self._on_main_table_item_changed)
        self.table.itemSelectionChanged.connect(self._on_table_selection_changed)
        self.table_ibc.itemChanged.connect(self._on_small_table_changed)
        self.table_diel.itemChanged.connect(self._on_small_table_changed)
        self.canvas.mpl_connect("pick_event", self._on_plot_pick)

        self.canvas.mpl_connect("button_press_event", self._on_plot_button_press)
        self.canvas.mpl_connect("scroll_event", self._on_plot_scroll)

        self._set_equal_column_widths(self.table, enabled=True)
        self._set_equal_column_widths(self.table_ibc, enabled=True)
        self._set_equal_column_widths(self.table_diel, enabled=True)

    def load_geo(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, "Open Geometry File", "", "Geometry Files (*.geo);;All Files (*)"
        )
        if not fname:
            return
        try:
            with open(fname, "r") as f:
                text = f.read()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read file: {e}")
            return
        try:
            title, segments, ibcs_entries, dielectric_entries = parse_geometry(text)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to parse geometry: {e}")
            return

        self.title = title
        self.segments = segments
        self.ibcs_entries = ibcs_entries
        self.dielectric_entries = dielectric_entries
        self.loaded_path = os.path.abspath(fname)

        self._populating = True
        try:
            self.table.clearContents()
            self.table.setRowCount(len(self.segments))
            self.table.setColumnCount(6)
            self.table.setHorizontalHeaderLabels(["Name", "Type", "N", "IBC/Resistance", "pos_mat", "neg_mat"])
            for row, seg in enumerate(self.segments):
                props = self._ensure_prop_len(seg.properties, 5)
                n_value = props[1] if len(props) >= 2 else ""
                self.table.setItem(row, 0, QTableWidgetItem(seg.name))
                self.table.setItem(row, 2, QTableWidgetItem(n_value))
        finally:
            self._populating = False

        ax = self.canvas.ax
        ax.clear()
        self.segment_lines = []
        self.segment_base_colors = []
        self.issue_rows.clear()
        self._clear_normals()

        plot_colors = ["orange", "green", "blue", "gray", "black", "red", "purple", "cyan"]

        for row, seg in enumerate(self.segments):
            props = seg.properties
            itype = props[0] if len(props) >= 1 else ""
            try:
                color_index = (int(itype) - 1) % len(plot_colors)
                base_color = plot_colors[color_index]
            except (ValueError, TypeError):
                base_color = plot_colors[row % len(plot_colors)]

            plot_x, plot_y = self._segment_plot_xy(seg)
            (line2d,) = ax.plot(plot_x, plot_y, color=base_color, linewidth=1.5, zorder=1)
            line2d.set_picker(True)
            line2d.set_pickradius(5)
            self.segment_lines.append(line2d)
            self.segment_base_colors.append(base_color)
        ax.set_title(self.title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)

        self._populate_small_table(self.table_ibc, self.ibcs_entries, label=self.lbl_ibc, title_prefix="IBCS/Resistances")
        self._populate_small_table(
            self.table_diel,
            self.dielectric_entries,
            label=self.lbl_diel,
            title_prefix="Dielectrics",
        )
        self._refresh_segment_dropdowns()

        self._selected_row = None
        self._refresh_segment_styles()
        self._render_normals()
        self._render_impedance_overlay()
        self._render_fills()
        self._update_status_label(-1)
        self.canvas.draw()
        QMessageBox.information(
            self,
            "Loaded",
            f"Loaded {len(self.segments)} segments(s),"
            f"{len(self.ibcs_entries)} IBCS/Resistances entry(ies),"
            f"and {len(self.dielectric_entries)} dielectric entry(ies).",
        )

    def _populate_small_table(self, table: QTableWidget, rows: List[List[str]], label: QLabel, title_prefix: str):
        if title_prefix == "IBCS/Resistances":
            # 6-column inline form: flag, kind, R_start, X_start, R_end, X_end.
            # Tabulated rows (flag > 50) only fill the Flag cell.
            headers = ["Flag", "Kind", "R_start", "X_start", "R_end", "X_end"]
            col_count = len(headers)
        elif title_prefix == "Dielectrics":
            col_count = max((len(r) for r in rows), default=0)
            headers_full = ["Flag", "Ep_real", "Ep_imag", "Mu_real", "Mu_imag"]
            headers = headers_full[:col_count] if col_count > 0 else []
        else:
            col_count = max((len(r) for r in rows), default=0)
            headers = [f"Col {i+1}" for i in range(col_count)]

        table.blockSignals(True)
        try:
            # Clear any leftover cell widgets (e.g. previous kind dropdowns)
            # from prior populations before resetting row/col counts.
            for r in range(table.rowCount()):
                for c in range(table.columnCount()):
                    if table.cellWidget(r, c) is not None:
                        table.removeCellWidget(r, c)
            table.clearContents()
            table.setRowCount(len(rows))
            table.setColumnCount(col_count)
            table.setHorizontalHeaderLabels(headers)

            for r, tokens in enumerate(rows):
                for c, token in enumerate(tokens):
                    if c >= col_count:
                        break
                    table.setItem(r, c, QTableWidgetItem(token))

            # IBC Kind cell (col 1) on inline rows becomes a dropdown so the
            # user picks from constant/linear/cosine/exp instead of typing.
            # Tabulated rows (single token, flag > 50) skip the dropdown.
            if title_prefix == "IBCS/Resistances":
                for r, tokens in enumerate(rows):
                    if len(tokens) < 2:
                        continue
                    self._install_ibc_kind_combo(table, r, tokens[1])
        finally:
            table.blockSignals(False)

        label.setText(f"{title_prefix} (n={len(rows)})")

    def _install_ibc_kind_combo(self, table: QTableWidget, row: int, current_kind: str) -> None:
        cb = QComboBox()
        for kind in IBC_KINDS:
            cb.addItem(kind, userData=kind)
        target = (current_kind or "").strip().lower()
        idx = next((i for i, k in enumerate(IBC_KINDS) if k == target), 0)
        cb.setCurrentIndex(idx)
        cb.currentIndexChanged.connect(self._on_small_table_changed)
        # Drop the QTableWidgetItem under the widget so _read_small_table picks
        # the widget value and there's no stale item shadowing it.
        table.setItem(row, 1, None)
        table.setCellWidget(row, 1, cb)

    def _ensure_prop_len(self, props: List[str], n: int) -> List[str]:
        if len(props) < n:
            props.extend([""] * (n - len(props)))
        return props

    def _ibc_dropdown_options(self) -> List[Tuple[str, str]]:
        """Build (value, label) pairs for the IBC dropdown from current IBC table state."""
        options: List[Tuple[str, str]] = [("0", "0 (none)")]
        lookup = self._ibcs_lookup()
        for flag in sorted(lookup.keys()):
            kind = lookup[flag].get("kind", "undefined")
            options.append((str(flag), f"{flag} ({kind})"))
        return options

    def _diel_dropdown_options(self) -> List[Tuple[str, str]]:
        """Build (value, label) pairs for the pos_mat/neg_mat dropdown from current dielectric table state."""
        options: List[Tuple[str, str]] = [("0", "0 (vacuum)")]
        seen: Set[int] = set()
        rows = self._read_small_table(self.table_diel)
        for row in rows:
            if not row:
                continue
            try:
                flag = int(row[0])
            except (ValueError, TypeError):
                continue
            if flag <= 0 or flag in seen:
                continue
            seen.add(flag)
            # Tabulated rows reference mat.<flag> and have only the flag token;
            # constant rows carry eps/mu numbers in subsequent columns.
            if len(row) == 1:
                options.append((str(flag), f"{flag} (tabulated)"))
            else:
                options.append((str(flag), f"{flag} (constant)"))
        options.sort(key=lambda x: int(x[0]))
        return options

    def _make_segment_combo(
        self,
        options: List[Tuple[str, str]],
        current: str,
        row: int,
        prop_index: int,
    ) -> QComboBox:
        """Build a QComboBox for the segment table. Connects the change handler only after the initial index is set."""
        cb = QComboBox()
        found_index = -1
        current_clean = (current or "").strip()
        for i, (value, lbl) in enumerate(options):
            cb.addItem(lbl, userData=value)
            if value == current_clean:
                found_index = i
        if found_index < 0 and current_clean:
            cb.addItem(f"{current_clean} (undefined)", userData=current_clean)
            found_index = cb.count() - 1
        if found_index >= 0:
            cb.setCurrentIndex(found_index)
        cb.currentIndexChanged.connect(
            lambda _idx, r=row, p=prop_index, w=cb: self._on_segment_combo_changed(r, p, w)
        )
        return cb

    def _refresh_segment_dropdowns(self) -> None:
        """Rebuild all segment-table comboboxes against the current IBC/dielectric tables."""
        if not self.segments:
            return
        ibc_opts = self._ibc_dropdown_options()
        diel_opts = self._diel_dropdown_options()
        was_populating = self._populating
        self._populating = True
        try:
            for row, seg in enumerate(self.segments):
                props = self._ensure_prop_len(seg.properties, 5)
                type_val = (props[0] or "").strip()
                self.table.setCellWidget(row, 1, self._make_segment_combo(SEGMENT_TYPE_OPTIONS, type_val, row, 0))
                self.table.setCellWidget(row, 3, self._make_segment_combo(ibc_opts, (props[2] or "").strip(), row, 2))
                self.table.setCellWidget(row, 4, self._make_segment_combo(diel_opts, (props[3] or "").strip(), row, 3))
                self.table.setCellWidget(row, 5, self._make_segment_combo(diel_opts, (props[4] or "").strip(), row, 4))
                self._apply_neg_mat_editability(row, type_val)
        finally:
            self._populating = was_populating

    def _on_segment_combo_changed(self, row: int, prop_index: int, combo: QComboBox) -> None:
        if self._populating:
            return
        if row < 0 or row >= len(self.segments):
            return
        new_value = combo.currentData()
        if new_value is None:
            new_value = combo.currentText()
        new_value = str(new_value)

        seg = self.segments[row]
        props = self._ensure_prop_len(seg.properties, 5)
        props[prop_index] = new_value

        if prop_index == 0:
            seg.seg_type = new_value or None
            plot_colors = ["orange", "green", "blue", "gray", "black", "red", "purple", "cyan"]
            try:
                color_index = (int(new_value) - 1) % len(plot_colors)
                base_color = plot_colors[color_index]
            except (ValueError, TypeError):
                base_color = plot_colors[row % len(plot_colors)]
            if row < len(self.segment_base_colors):
                self.segment_base_colors[row] = base_color
            self._refresh_segment_styles()
            self._apply_neg_mat_editability(row, new_value)

        if prop_index in (0, 2):
            self._render_impedance_overlay()
            self.canvas.draw_idle()
        if prop_index in (0, 3, 4):
            self._render_normals()
            self._render_fills()
            self.canvas.draw_idle()

        if row == self._selected_row:
            self._update_status_label(row)

    def _ibc_add_row(self) -> None:
        current = self._read_small_table(self.table_ibc) if self.table_ibc.rowCount() > 0 else []
        used: Set[int] = set()
        for row in current:
            try:
                used.add(int(row[0]))
            except (ValueError, IndexError, TypeError):
                pass
        next_flag = 1
        while next_flag in used or next_flag > 50:
            next_flag += 1
            if next_flag > 50:
                # Wrap around once and bail if every slot is occupied.
                break
        # Default: constant 50 + 0j (R_end/X_end are placeholders, ignored for "constant").
        current.append([str(next_flag), "constant", "50", "0", "0", "0"])
        self.ibcs_entries = current
        self._populate_small_table(self.table_ibc, current, label=self.lbl_ibc, title_prefix="IBCS/Resistances")
        self._refresh_segment_dropdowns()

    def _ibc_remove_row(self) -> None:
        sel = sorted({i.row() for i in self.table_ibc.selectedIndexes()}, reverse=True)
        if not sel:
            if self.table_ibc.rowCount() == 0:
                return
            sel = [self.table_ibc.rowCount() - 1]
        current = self._read_small_table(self.table_ibc)
        for r in sel:
            if 0 <= r < len(current):
                del current[r]
        self.ibcs_entries = current
        self._populate_small_table(self.table_ibc, current, label=self.lbl_ibc, title_prefix="IBCS/Resistances")
        self._refresh_segment_dropdowns()

    def _diel_add_row(self) -> None:
        current = self._read_small_table(self.table_diel) if self.table_diel.rowCount() > 0 else []
        used: Set[int] = set()
        for row in current:
            try:
                used.add(int(row[0]))
            except (ValueError, IndexError, TypeError):
                pass
        next_flag = 1
        while next_flag in used or next_flag > 50:
            next_flag += 1
            if next_flag > 50:
                break
        # Default: free-space-like (eps_r=1, eps_i=0, mu_r=1, mu_i=0).
        current.append([str(next_flag), "1.0", "0.0", "1.0", "0.0"])
        self.dielectric_entries = current
        self._populate_small_table(self.table_diel, current, label=self.lbl_diel, title_prefix="Dielectrics")
        self._refresh_segment_dropdowns()

    def _diel_remove_row(self) -> None:
        sel = sorted({i.row() for i in self.table_diel.selectedIndexes()}, reverse=True)
        if not sel:
            if self.table_diel.rowCount() == 0:
                return
            sel = [self.table_diel.rowCount() - 1]
        current = self._read_small_table(self.table_diel)
        for r in sel:
            if 0 <= r < len(current):
                del current[r]
        self.dielectric_entries = current
        self._populate_small_table(self.table_diel, current, label=self.lbl_diel, title_prefix="Dielectrics")
        self._refresh_segment_dropdowns()

    def _on_small_table_changed(self, *_args) -> None:
        if self._populating:
            return
        # User edited a flag/value/kind in the IBC or dielectric table; the
        # segment dropdowns now need to mirror the new labels. Accepts any
        # signal payload (QTableWidgetItem from itemChanged, int from
        # currentIndexChanged) since we don't use it.
        self._refresh_segment_dropdowns()

    def _apply_neg_mat_editability(self, row: int, seg_type: str) -> None:
        # neg_mat only applies to TYPE 5 (dielectric/dielectric interface).
        widget = self.table.cellWidget(row, 5)
        if widget is None:
            return
        widget.setEnabled(str(seg_type).strip() == "5")

    def _on_main_table_item_changed(self, item: QTableWidgetItem):
        if self._populating:
            return
        row = item.row()
        col = item.column()
        if row < 0 or row >= len(self.segments):
            return

        seg = self.segments[row]
        text = item.text().strip()

        # Cols 1 (Type), 3 (IBC), 4 (pos_mat), 5 (neg_mat) are comboboxes
        # whose updates flow through _on_segment_combo_changed instead.
        if col == 0:
            seg.name = text
        elif col == 2:
            props = self._ensure_prop_len(seg.properties, 5)
            props[1] = text

        if row == self._selected_row:
            self._update_status_label(row)

    def _on_table_selection_changed(self):
        if self._syncing_selection:
            return
        row = self.table.currentRow()
        self._apply_selection(row)

    def _apply_selection(self, row: int):
        self._selected_row = row if (row is not None and row >= 0) else None
        self._refresh_segment_styles()
        self._update_status_label(row if row is not None else -1)
        self.canvas.draw_idle()

    def _on_plot_pick(self, event):
        line = getattr(event, "artist", None)
        if not line:
            return
        try:
            row = self.segment_lines.index(line)
        except ValueError:
            return
        self._syncing_selection = True
        try:
            self.table.selectRow(row)
            self._apply_selection(row)
        finally:
            self._syncing_selection = False

    def _on_plot_button_press(self, event):
        if event.inaxes != self.canvas.ax:
            return
        modifier_select = event.button == 1 and (event.key in ("control", "shift"))
        if event.button == 3 or modifier_select:
            idx = self._hit_test(event)
            if idx is not None:
                self._syncing_selection = True
                try:
                    self.table.selectRow(idx)
                    self._apply_selection(idx)
                finally:
                    self._syncing_selection = False

    def _hit_test(self, event) -> Optional[int]:
        for i in reversed(range(len(self.segment_lines))):
            line = self.segment_lines[i]
            contains, _ = line.contains(event)
            if contains:
                return i
        return None

    def _on_plot_scroll(self, event):
        if event.inaxes != self.canvas.ax or event.xdata is None or event.ydata is None:
            return
        base_scale = 1.2 if event.button == "up" else (1 / 1.2)
        self._zoom_at(event.xdata, event.ydata, base_scale)

    def _zoom_at(self, x: float, y: float, scale: float):
        ax = self.canvas.ax
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        w = (xlim[1] - xlim[0]) / scale
        h = (ylim[1] - ylim[0]) / scale
        ax.set_xlim(x - w / 2, x + w / 2)
        ax.set_ylim(y - h / 2, y + h / 2)
        self.canvas.draw_idle()

    def _refresh_segment_styles(self):
        for i, line in enumerate(self.segment_lines):
            if self._selected_row is not None and i == self._selected_row:
                line.set_color("pink")
                line.set_linewidth(2.5)
                line.set_zorder(10)
                continue
            if i in self.issue_rows:
                line.set_color("crimson")
                line.set_linewidth(2.2)
                line.set_zorder(8)
                continue
            base = self.segment_base_colors[i] if i < len(self.segment_base_colors) else "gray"
            line.set_color(base)
            line.set_linewidth(1.5)
            line.set_zorder(1)

    def _clear_normals(self):
        for art in self.normal_artists:
            try:
                art.remove()
            except Exception:
                pass
        self.normal_artists = []

    def _segment_primitives(self, seg: Segment) -> List[Tuple[float, float, float, float]]:
        count = min(len(seg.x), len(seg.y))
        n_pairs = count // 2
        out: List[Tuple[float, float, float, float]] = []
        for i in range(n_pairs):
            idx = 2 * i
            out.append((seg.x[idx], seg.y[idx], seg.x[idx + 1], seg.y[idx + 1]))
        return out

    def _segment_plot_xy(self, seg: Segment) -> Tuple[List[float], List[float]]:
        primitives = self._segment_primitives(seg)
        if not primitives:
            return list(seg.x), list(seg.y)

        xs: List[float] = []
        ys: List[float] = []
        for i, (x1, y1, x2, y2) in enumerate(primitives):
            if i == 0:
                xs.append(x1)
                ys.append(y1)
            xs.append(x2)
            ys.append(y2)

        if not xs or not ys:
            return list(seg.x), list(seg.y)
        return xs, ys

    # ── Material side semantics (drawing convention) ──────────────────────
    # TYPE 1 sheet:  air | air        TYPE 2: air | PEC
    # TYPE 3:        air | diel N     TYPE 4: diel N | PEC
    # TYPE 5:        diel N | diel M
    # "front" = the side the normal (left of travel) points into.

    _AIR_COLOR = "#4da6ff"
    _PEC_COLOR = "#6e6e6e"
    _SHEET_COLOR = "#d4a017"
    _DIEL_COLORS = ["#2e9e4f", "#7f5bd4", "#0fa3a3", "#c46a1b", "#b33f8e", "#8a9a1a"]

    def _diel_color(self, flag: int) -> str:
        return self._DIEL_COLORS[(max(int(flag), 1) - 1) % len(self._DIEL_COLORS)]

    def _segment_side_materials(self, seg: Segment) -> Tuple[str, str, str, str]:
        """Return (front_label, front_color, back_label, back_color)."""

        props = self._ensure_prop_len(seg.properties, 5)
        seg_type = self._parse_int_token(props[0], 2)
        pos_mat = self._parse_int_token(props[3], 0)
        neg_mat = self._parse_int_token(props[4], 0)
        if seg_type == 1:
            return "air", self._SHEET_COLOR, "air", self._SHEET_COLOR
        if seg_type == 2:
            return "air", self._AIR_COLOR, "PEC", self._PEC_COLOR
        if seg_type == 3:
            return "air", self._AIR_COLOR, f"d{pos_mat}", self._diel_color(pos_mat)
        if seg_type == 4:
            return f"d{pos_mat}", self._diel_color(pos_mat), "PEC", self._PEC_COLOR
        if seg_type == 5:
            return f"d{pos_mat}", self._diel_color(pos_mat), f"d{neg_mat}", self._diel_color(neg_mat)
        return "?", "magenta", "?", "magenta"

    def _render_normals(self):
        self._clear_normals()
        if not self.chk_show_normals.isChecked():
            return
        if not self.segments:
            return

        all_x = [x for seg in self.segments for x in seg.x]
        all_y = [y for seg in self.segments for y in seg.y]
        if not all_x or not all_y:
            return
        diag = max(((max(all_x) - min(all_x)) ** 2 + (max(all_y) - min(all_y)) ** 2) ** 0.5, 1.0)
        arrow_len = 0.04 * diag
        tick_len = 0.018 * diag
        ax = self.canvas.ax

        for row, seg in enumerate(self.segments):
            front_label, front_color, back_label, back_color = self._segment_side_materials(seg)
            issue = row in self.issue_rows
            arrow_color = "crimson" if issue else front_color
            primitives = self._segment_primitives(seg)
            for x1, y1, x2, y2 in primitives:
                dx = x2 - x1
                dy = y2 - y1
                length = (dx * dx + dy * dy) ** 0.5
                if length <= 1e-12:
                    continue
                nx = -dy / length
                ny = dx / length
                mx = 0.5 * (x1 + x2)
                my = 0.5 * (y1 + y2)
                # Arrow into the "front" material (normal side).
                ann = ax.annotate(
                    "",
                    xy=(mx + nx * arrow_len, my + ny * arrow_len),
                    xytext=(mx, my),
                    arrowprops={"arrowstyle": "-|>", "color": arrow_color, "lw": 0.9, "alpha": 0.85},
                    zorder=12,
                )
                self.normal_artists.append(ann)
                # Short tick into the "back" material.
                (tick,) = ax.plot(
                    [mx, mx - nx * tick_len], [my, my - ny * tick_len],
                    color=back_color, lw=2.2, alpha=0.85, zorder=12,
                    solid_capstyle="butt",
                )
                self.normal_artists.append(tick)

            # One label per segment at the middle primitive: "front | back".
            if primitives:
                x1, y1, x2, y2 = primitives[len(primitives) // 2]
                dx, dy = x2 - x1, y2 - y1
                length = max((dx * dx + dy * dy) ** 0.5, 1e-12)
                nx, ny = -dy / length, dx / length
                mx, my = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
                txt = ax.annotate(
                    f"{front_label} | {back_label}",
                    xy=(mx + nx * (arrow_len * 1.35), my + ny * (arrow_len * 1.35)),
                    ha="center", va="center", fontsize=7,
                    color="crimson" if issue else "#222222",
                    bbox={"boxstyle": "round,pad=0.18", "fc": "white",
                          "ec": arrow_color, "lw": 0.6, "alpha": 0.85},
                    zorder=13,
                )
                self.normal_artists.append(txt)

    def _on_show_normals_toggled(self, checked: bool):
        _ = checked
        self._render_normals()
        self.canvas.draw_idle()

    # ── Material region fills ─────────────────────────────────────────────

    def _clear_fills(self):
        for art in self.fill_artists:
            try:
                art.remove()
            except Exception:
                pass
        self.fill_artists = []

    def _fill_loops(self) -> List[Dict[str, Any]]:
        """
        Stitch segment chains into closed loops and infer the enclosed material.

        Returns dicts: {points, label, color, consistent, depth, rows}.
        Interior side rule: for a CCW loop the interior is on the LEFT of
        travel (= the normal side); a chain traversed backwards contributes
        its opposite side.  All member chains of a loop must imply the same
        interior material, otherwise the loop is flagged inconsistent.
        """

        from geometry_io import _chain_area2, _chain_is_closed, _point_in_polygon

        chains: List[Dict[str, Any]] = []
        for row, seg in enumerate(self.segments):
            xs, ys = self._segment_plot_xy(seg)
            pts = list(zip(xs, ys))
            if len(pts) < 2:
                continue
            chains.append({"row": row, "pts": pts})

        all_x = [p[0] for c in chains for p in c["pts"]]
        all_y = [p[1] for c in chains for p in c["pts"]]
        if not all_x:
            return []
        diag = max(((max(all_x) - min(all_x)) ** 2 + (max(all_y) - min(all_y)) ** 2) ** 0.5, 1.0)
        tol = max(1e-12, 1e-9 * diag)

        def key(p):
            return (round(p[0] / tol), round(p[1] / tol))

        # loops as lists of (chain, forward_flag)
        loops: List[List[Tuple[Dict[str, Any], bool]]] = []
        open_chains = []
        for ch in chains:
            if _chain_is_closed(ch["pts"], tol):
                loops.append([(ch, True)])
            else:
                open_chains.append(ch)

        # Stitch open chains at shared endpoints (degree-2 nodes only),
        # allowing either joining orientation — direction is tracked per member.
        ends: Dict[Tuple[int, int], List[Tuple[int, str]]] = {}
        for i, ch in enumerate(open_chains):
            ends.setdefault(key(ch["pts"][0]), []).append((i, "start"))
            ends.setdefault(key(ch["pts"][-1]), []).append((i, "end"))
        links: Dict[Tuple[int, str], Tuple[int, str]] = {}
        for members in ends.values():
            if len(members) == 2 and members[0][0] != members[1][0]:
                links[members[0]] = members[1]
                links[members[1]] = members[0]

        used: Set[int] = set()
        for i, ch in enumerate(open_chains):
            if i in used:
                continue
            member_list = [(ch, True)]
            used.add(i)
            cur, exit_end = i, "end"
            closed_ok = False
            while True:
                nxt = links.get((cur, exit_end))
                if nxt is None:
                    break
                j, joined_at = nxt
                if j == i:
                    closed_ok = True
                    break
                if j in used:
                    break
                forward = joined_at == "start"
                member_list.append((open_chains[j], forward))
                used.add(j)
                cur, exit_end = j, "end" if forward else "start"
            if closed_ok and len(member_list) >= 2:
                loops.append(member_list)

        # Build loop polygons + material inference.
        out: List[Dict[str, Any]] = []
        for members in loops:
            pts: List[Tuple[float, float]] = []
            for ch, forward in members:
                p = ch["pts"] if forward else list(reversed(ch["pts"]))
                pts.extend(p if not pts else p[1:])
            if not _chain_is_closed(pts, tol):
                continue
            area2 = _chain_area2(pts)
            if abs(area2) <= 0.0:
                continue
            ccw = area2 > 0.0
            labels = set()
            label_color: Dict[str, str] = {}
            rows = []
            for ch, forward in members:
                seg = self.segments[ch["row"]]
                rows.append(ch["row"])
                fl, fc, bl, bc = self._segment_side_materials(seg)
                # Interior = left of travel for CCW loops; traversal reversed
                # flips which drawn side faces the interior.
                front_is_interior = (ccw == forward)
                lab, col = (fl, fc) if front_is_interior else (bl, bc)
                labels.add(lab)
                label_color[lab] = col
            consistent = len(labels) == 1
            lab = labels.pop() if consistent else "?"
            out.append({
                "points": pts,
                "label": lab,
                "color": label_color.get(lab, "red"),
                "consistent": consistent,
                "rows": rows,
            })

        # Containment depth (for paint order: outer first, inner over it).
        for i, loop in enumerate(out):
            rep = (0.5 * (loop["points"][0][0] + loop["points"][1][0]),
                   0.5 * (loop["points"][0][1] + loop["points"][1][1]))
            loop["depth"] = sum(
                1 for j, other in enumerate(out)
                if j != i and _point_in_polygon(rep[0], rep[1], other["points"])
            )
        return out

    def _render_fills(self):
        self._clear_fills()
        if not self.chk_fill_materials.isChecked() or not self.segments:
            return
        from matplotlib.patches import Polygon as MplPolygon

        ax = self.canvas.ax
        for loop in sorted(self._fill_loops(), key=lambda d: d["depth"]):
            if not loop["consistent"]:
                patch = MplPolygon(
                    loop["points"], closed=True, facecolor="none",
                    edgecolor="red", hatch="//", lw=0.0, alpha=0.5,
                    zorder=1 + 0.01 * loop["depth"],
                )
            else:
                lab = loop["label"]
                if lab == "air":
                    face = "white"
                else:
                    # Opaque light tint: deeper (nested) regions must fully
                    # cover their parent fill instead of alpha-blending with it.
                    from matplotlib.colors import to_rgb
                    r, g, b = to_rgb(loop["color"])
                    mix = 0.72
                    face = (r + (1 - r) * mix, g + (1 - g) * mix, b + (1 - b) * mix)
                patch = MplPolygon(
                    loop["points"], closed=True, facecolor=face,
                    edgecolor="none", alpha=1.0,
                    zorder=1 + 0.01 * loop["depth"],
                )
            ax.add_patch(patch)
            self.fill_artists.append(patch)
            if loop["consistent"] and loop["label"] != "air":
                xs = [p[0] for p in loop["points"]]
                ys = [p[1] for p in loop["points"]]
                txt = ax.annotate(
                    loop["label"], xy=(sum(xs) / len(xs), sum(ys) / len(ys)),
                    ha="center", va="center", fontsize=8, color=loop["color"],
                    alpha=0.9, zorder=1.5,
                )
                self.fill_artists.append(txt)

    def _on_fill_materials_toggled(self, checked: bool):
        _ = checked
        self._render_fills()
        self.canvas.draw_idle()

    # ── IBCS resolution + visualization ───────────────────────────────────

    def _ibcs_lookup(self) -> Dict[int, Dict[str, Any]]:
        """Build a {flag: info} map from the live IBCS table.

        info keys: 'kind' ('linear'/'cosine'/'exp'/'tabulated'/'undefined'),
                   'z_start' (complex), 'z_end' (complex), 'raw' (token row).
        """
        rows = self._read_small_table(self.table_ibc)
        out: Dict[int, Dict[str, Any]] = {}
        for row in rows:
            if not row:
                continue
            flag = self._parse_int_token(row[0], 0)
            if flag <= 0:
                continue
            if is_tabulated_row(row):
                out[flag] = {"kind": "tabulated", "z_start": None, "z_end": None, "raw": row}
                continue
            if is_ibc_inline_row(row):
                kind = str(row[1]).strip().lower()
                r_s = self._parse_float_token(row[2], 0.0)
                x_s = self._parse_float_token(row[3], 0.0)
                if kind == "constant":
                    z_s = complex(r_s, x_s)
                    z_e = z_s
                else:
                    r_e = self._parse_float_token(row[4], 0.0)
                    x_e = self._parse_float_token(row[5], 0.0)
                    z_s = complex(r_s, x_s)
                    z_e = complex(r_e, x_e)
                out[flag] = {
                    "kind": kind,
                    "z_start": z_s,
                    "z_end": z_e,
                    "raw": row,
                }
                continue
            out[flag] = {"kind": "undefined", "z_start": None, "z_end": None, "raw": row}
        return out

    def _format_z(self, z: Optional[complex]) -> str:
        if z is None:
            return "?"
        return f"{z.real:g}{'+' if z.imag >= 0 else '-'}{abs(z.imag):g}j Ω"

    def _resolve_segment_bc(self, seg: Segment, lookup: Optional[Dict[int, Dict[str, Any]]] = None) -> str:
        """Human-readable resolved boundary condition for a segment."""
        props = list(seg.properties)
        seg_type = self._parse_int_token(props[0] if props else "", -1)
        ibc = self._parse_int_token(props[2] if len(props) >= 3 else "", 0)
        pos_mat = self._parse_int_token(props[3] if len(props) >= 4 else "", 0)
        neg_mat = self._parse_int_token(props[4] if len(props) >= 5 else "", 0)

        # Materialize the base-type description first, then append any IBC.
        if seg_type == 1:
            base = "TYPE 1 · free-floating sheet"
        elif seg_type == 2:
            base = "TYPE 2 · PEC" if ibc == 0 else f"TYPE 2 · IBC-coated PEC"
        elif seg_type in (3, 4, 5):
            dstr = f"pos_mat={pos_mat}" + (f", neg_mat={neg_mat}" if seg_type == 5 else "")
            base = f"TYPE {seg_type} · dielectric interface ({dstr})"
        else:
            base = f"TYPE {seg_type}"

        if ibc == 0:
            return base
        lut = lookup if lookup is not None else self._ibcs_lookup()
        info = lut.get(ibc)
        if info is None:
            return f"{base}  ·  IBC {ibc} (NOT DEFINED)"
        kind = info["kind"]
        if kind == "tabulated":
            return f"{base}  ·  IBC {ibc} → tabulated (mat.{ibc})"
        if kind == "undefined":
            return f"{base}  ·  IBC {ibc} (malformed row)"
        z1, z2 = info["z_start"], info["z_end"]
        if z1 == z2:
            return f"{base}  ·  IBC {ibc} → constant {self._format_z(z1)}"
        return (
            f"{base}  ·  IBC {ibc} → taper({kind})  "
            f"start {self._format_z(z1)}  →  end {self._format_z(z2)}"
        )

    def _z_to_color(self, z: Optional[complex], z_ref_mag: float) -> Tuple[float, float, float]:
        """Map an impedance value to an RGB colour.

        * |Z| near zero  → near-black (PEC-like)
        * |Z| near 377 Ω → mid-blue (free-space-like)
        * |Z| large      → light blue / washed out
        Reactance tints warm (|X| large → toward magenta).
        """
        if z is None:
            return (0.55, 0.55, 0.55)  # neutral grey for unknown/tabulated
        mag = abs(z)
        # Normalize to [0, 1] roughly, with free-space as the mid-point.
        t = min(1.0, mag / max(1.0, 2.0 * 377.0))
        # Base blue ramp from black (PEC) to light blue (high Z).
        r = 0.05 + 0.75 * t
        g = 0.10 + 0.55 * t
        b = 0.25 + 0.70 * (1.0 - abs(t - 0.5) * 2.0)  # peaks near free-space
        # Reactive component adds a warm tint proportional to |X|/|Z|.
        if mag > 1e-12:
            react_frac = min(1.0, abs(z.imag) / mag)
            r = min(1.0, r + 0.25 * react_frac)
        return (r, g, b)

    def _clear_impedance_overlay(self):
        for artist in self.impedance_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self.impedance_artists = []

    def _render_impedance_overlay(self):
        self._clear_impedance_overlay()
        if not self.chk_show_impedance.isChecked() or not self.segments:
            return
        ax = self.canvas.ax
        all_x = [x for seg in self.segments for x in seg.x]
        all_y = [y for seg in self.segments for y in seg.y]
        if not all_x or not all_y:
            return
        diag = max(((max(all_x) - min(all_x)) ** 2 + (max(all_y) - min(all_y)) ** 2) ** 0.5, 1.0)
        marker_size = max(4.0, 0.015 * diag * 100)  # matplotlib scatter "s"

        lookup = self._ibcs_lookup()

        for seg in self.segments:
            primitives = self._segment_primitives(seg)
            if not primitives:
                continue
            props = list(seg.properties)
            seg_type = self._parse_int_token(props[0] if props else "", -1)
            ibc = self._parse_int_token(props[2] if len(props) >= 3 else "", 0)

            # Choose endpoint colours by resolved Z.
            if seg_type in (1, 2, 3, 4, 5) and ibc != 0 and ibc in lookup:
                info = lookup[ibc]
                if info["kind"] == "tabulated":
                    c_start = c_end = (0.95, 0.55, 0.10)  # orange for mat.*
                else:
                    c_start = self._z_to_color(info["z_start"], 377.0)
                    c_end = self._z_to_color(info["z_end"], 377.0)
            elif seg_type in (1, 2, 3, 4, 5) and ibc == 0:
                # PEC — near-black
                c_start = c_end = (0.05, 0.05, 0.08)
            else:
                c_start = c_end = (0.55, 0.55, 0.55)

            # Arc-length parameter along the whole segment (s ∈ [0, 1]).
            seg_lens = []
            for x1, y1, x2, y2 in primitives:
                seg_lens.append(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
            total_len = sum(seg_lens) or 1.0
            cum = 0.0
            SAMPLES_PER_PRIM = 12  # segments of the gradient polyline per primitive
            for (x1, y1, x2, y2), L in zip(primitives, seg_lens):
                s0 = cum / total_len
                s1 = (cum + L) / total_len
                cum += L
                # Densify this primitive into short coloured segments so a taper
                # reads as a smooth gradient.
                for k in range(SAMPLES_PER_PRIM):
                    u0 = k / SAMPLES_PER_PRIM
                    u1 = (k + 1) / SAMPLES_PER_PRIM
                    px0 = x1 + u0 * (x2 - x1); py0 = y1 + u0 * (y2 - y1)
                    px1 = x1 + u1 * (x2 - x1); py1 = y1 + u1 * (y2 - y1)
                    # Blend parameter along whole segment for colour lookup.
                    s_mid = s0 + 0.5 * (u0 + u1) * (s1 - s0)
                    cr = c_start[0] + s_mid * (c_end[0] - c_start[0])
                    cg = c_start[1] + s_mid * (c_end[1] - c_start[1])
                    cb = c_start[2] + s_mid * (c_end[2] - c_start[2])
                    line = ax.plot(
                        [px0, px1], [py0, py1],
                        color=(cr, cg, cb), lw=4.0, solid_capstyle="butt",
                        alpha=0.7, zorder=8,
                    )
                    self.impedance_artists.extend(line)

            # Start / end markers (green → red) for drawn direction.
            sx, sy, _, _ = primitives[0]
            _, _, ex, ey = primitives[-1]
            m_start = ax.scatter([sx], [sy], s=marker_size, marker="o",
                                  facecolor="#1f9e3c", edgecolor="black", lw=0.6, zorder=13)
            m_end = ax.scatter([ex], [ey], s=marker_size, marker="o",
                                facecolor="#d93023", edgecolor="black", lw=0.6, zorder=13)
            self.impedance_artists.append(m_start)
            self.impedance_artists.append(m_end)

    def _on_show_impedance_toggled(self, checked: bool):
        _ = checked
        self._render_impedance_overlay()
        self.canvas.draw_idle()

    def _update_status_label(self, row: int):
        if row < 0 or row >= len(self.segments):
            self.lbl_status.setText("")
            return
        seg = self.segments[row]
        self.lbl_status.setText(f"{seg.name}  ·  {self._resolve_segment_bc(seg)}")

    def _parse_int_token(self, token: str, default: int = 0) -> int:
        text = (token or "").strip().lower()
        if not text:
            return default
        if text.startswith("mat."):
            text = text.split("mat.", 1)[1]
        try:
            return int(float(text))
        except ValueError:
            return default

    def _parse_float_token(self, token: str, default: float = 0.0) -> float:
        text = (token or "").strip()
        if not text:
            return default
        try:
            return float(text)
        except ValueError:
            return default

    def _find_mat_file(self, flag: int) -> str:
        name = f"mat.{flag}"
        base_dir = os.path.dirname(self.loaded_path) if self.loaded_path else ""
        candidates = [os.path.join(base_dir, name), os.path.join(os.getcwd(), name)]
        for path in candidates:
            if path and os.path.isfile(path):
                return path
        return ""

    def _point_key(self, x: float, y: float, tol: float) -> Tuple[int, int]:
        inv = 1.0 / max(tol, 1e-12)
        return int(round(float(x) * inv)), int(round(float(y) * inv))

    def _segments_intersect(
        self,
        a1: Tuple[float, float],
        a2: Tuple[float, float],
        b1: Tuple[float, float],
        b2: Tuple[float, float],
        tol: float,
    ) -> bool:
        ax1, ay1 = a1
        ax2, ay2 = a2
        bx1, by1 = b1
        bx2, by2 = b2

        min_ax, max_ax = min(ax1, ax2), max(ax1, ax2)
        min_ay, max_ay = min(ay1, ay2), max(ay1, ay2)
        min_bx, max_bx = min(bx1, bx2), max(bx1, bx2)
        min_by, max_by = min(by1, by2), max(by1, by2)
        if max_ax < min_bx - tol or max_bx < min_ax - tol:
            return False
        if max_ay < min_by - tol or max_by < min_ay - tol:
            return False

        def orient(px: float, py: float, qx: float, qy: float, rx: float, ry: float) -> float:
            return (qx - px) * (ry - py) - (qy - py) * (rx - px)

        def on_seg(px: float, py: float, qx: float, qy: float, rx: float, ry: float) -> bool:
            return (
                min(px, qx) - tol <= rx <= max(px, qx) + tol
                and min(py, qy) - tol <= ry <= max(py, qy) + tol
            )

        o1 = orient(ax1, ay1, ax2, ay2, bx1, by1)
        o2 = orient(ax1, ay1, ax2, ay2, bx2, by2)
        o3 = orient(bx1, by1, bx2, by2, ax1, ay1)
        o4 = orient(bx1, by1, bx2, by2, ax2, ay2)

        if (o1 > tol and o2 < -tol or o1 < -tol and o2 > tol) and (
            o3 > tol and o4 < -tol or o3 < -tol and o4 > tol
        ):
            return True

        if abs(o1) <= tol and on_seg(ax1, ay1, ax2, ay2, bx1, by1):
            return True
        if abs(o2) <= tol and on_seg(ax1, ay1, ax2, ay2, bx2, by2):
            return True
        if abs(o3) <= tol and on_seg(bx1, by1, bx2, by2, ax1, ay1):
            return True
        if abs(o4) <= tol and on_seg(bx1, by1, bx2, by2, ax2, ay2):
            return True
        return False

    def validate_geometry(self):
        ibcs_rows = self._read_small_table(self.table_ibc)
        dielectric_rows = self._read_small_table(self.table_diel)
        diel_flags = {self._parse_int_token(row[0], 0) for row in dielectric_rows if row}
        # IBCS flag set — only includes rows whose shape is actually usable.
        # A row like ['5'] (flag only, no impedance) must NOT be considered
        # "defined" or downstream segments referencing flag=5 pass silently.
        ibc_flags: Set[int] = set()

        findings: List[Tuple[str, int, str]] = []
        issue_rows: Set[int] = set()

        for ibc_idx, row in enumerate(ibcs_rows):
            if not row:
                continue
            flag = self._parse_int_token(row[0], 0)
            label = f"IBCS row {ibc_idx + 1} (flag {flag})"

            if flag > 50:
                # Tabulated via mat.<flag> — row shape is just [flag].
                ibc_flags.add(flag)
                mat_path = self._find_mat_file(flag)
                if mat_path:
                    try:
                        with open(mat_path, 'r') as f:
                            for lineno, raw in enumerate(f, start=1):
                                line = raw.split('#', 1)[0].strip()
                                if not line:
                                    continue
                                toks = line.split()
                                if len(toks) < 3:
                                    continue
                                vals = [float(toks[i]) for i in range(3)]
                                if not all(math.isfinite(v) for v in vals):
                                    findings.append((
                                        "ERROR",
                                        -1,
                                        f"mat.{flag} line {lineno}: impedance table contains non-finite value(s).",
                                    ))
                                    break
                    except Exception:
                        pass
                continue

            # flag <= 50 — inline 6-token form:
            #   flag  kind  R_start  X_start  R_end  X_end
            # kind is one of IBC_KINDS; for "constant", R_end/X_end are ignored
            # placeholders (write 0 0). Mirrors geometry_io._validate_ibc_row.
            if len(row) != 6:
                findings.append((
                    "ERROR", -1,
                    f"{label}: expected 6 tokens (flag kind R_start X_start R_end X_end), got {len(row)}.",
                ))
                continue

            kind = str(row[1]).strip().lower()
            if kind not in IBC_KINDS:
                findings.append((
                    "ERROR", -1,
                    f"{label}: IBC kind must be one of {'/'.join(IBC_KINDS)}, got {row[1]!r}.",
                ))
                continue

            z_parts = [self._parse_float_token(row[i], float('nan')) for i in (2, 3, 4, 5)]
            if not all(math.isfinite(v) for v in z_parts):
                findings.append((
                    "ERROR", -1,
                    f"{label}: impedance endpoints contain non-finite or unparsable value(s): "
                    f"R_start={row[2]!r}, X_start={row[3]!r}, R_end={row[4]!r}, X_end={row[5]!r}.",
                ))
                continue

            # exp taper cannot have zero endpoints (log undefined).
            if kind == "exp":
                # |Z| = 0 iff both R and X are zero.
                z1_mag = z_parts[0] ** 2 + z_parts[1] ** 2
                z2_mag = z_parts[2] ** 2 + z_parts[3] ** 2
                if z1_mag == 0.0 or z2_mag == 0.0:
                    findings.append((
                        "WARN", -1,
                        f"{label}: exp taper endpoints should be nonzero; solver will floor them. "
                        f"Prefer linear or cosine for PEC-limit transitions.",
                    ))

            ibc_flags.add(flag)

        for diel_idx, row in enumerate(dielectric_rows):
            flag = self._parse_int_token(row[0] if row else '', 0)
            if flag <= 50 and len(row) >= 5:
                vals = [self._parse_float_token(tok, 0.0) for tok in row[1:5]]
                if not all(math.isfinite(v) for v in vals):
                    findings.append((
                        "ERROR",
                        -1,
                        f"Dielectrics row {diel_idx + 1}: material values contain non-finite token(s).",
                    ))
            if flag > 50:
                mat_path = self._find_mat_file(flag)
                if mat_path:
                    try:
                        with open(mat_path, 'r') as f:
                            for lineno, raw in enumerate(f, start=1):
                                line = raw.split('#', 1)[0].strip()
                                if not line:
                                    continue
                                toks = line.split()
                                if len(toks) < 5:
                                    continue
                                vals = [float(toks[i]) for i in range(5)]
                                if not all(math.isfinite(v) for v in vals):
                                    findings.append((
                                        "ERROR",
                                        -1,
                                        f"mat.{flag} line {lineno}: dielectric table contains non-finite value(s).",
                                    ))
                                    break
                    except Exception:
                        pass

        all_points = [(x, y) for seg in self.segments for x, y in zip(seg.x, seg.y)]
        if all_points:
            xs = [p[0] for p in all_points]
            ys = [p[1] for p in all_points]
            diag = max(((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5, 1.0)
        else:
            diag = 1.0
        tol = max(1e-8, 1e-6 * diag)

        for row, seg in enumerate(self.segments):
            props = self._ensure_prop_len(seg.properties, 5)
            seg_type = self._parse_int_token(props[0], -1)
            n_panels = self._parse_int_token(props[1], 0)
            ibc = self._parse_int_token(props[2], 0)
            pos_mat = self._parse_int_token(props[3], 0)
            neg_mat = self._parse_int_token(props[4], 0)
            primitives = self._segment_primitives(seg)
            label = f"Row {row + 1} '{seg.name}'"

            if seg_type < 1 or seg_type > 5:
                findings.append(("ERROR", row, f"{label}: invalid TYPE '{props[0]}', expected 1..5."))
                issue_rows.add(row)

            if n_panels == 0:
                findings.append(("WARN", row, f"{label}: N should be a non-zero integer; positive N means explicit panel count and negative N means wavelength-based meshing. Current value is '{props[1]}'."))
                issue_rows.add(row)

            if not primitives:
                findings.append(("ERROR", row, f"{label}: no line primitives found."))
                issue_rows.add(row)
                continue

            for i, (x1, y1, x2, y2) in enumerate(primitives):
                length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                if length <= tol:
                    findings.append(("ERROR", row, f"{label}: primitive {i + 1} has near-zero length."))
                    issue_rows.add(row)

            for i in range(len(primitives) - 1):
                _, _, ex, ey = primitives[i]
                nx1, ny1, nx2, ny2 = primitives[i + 1]
                d_start = ((ex - nx1) ** 2 + (ey - ny1) ** 2) ** 0.5
                d_end = ((ex - nx2) ** 2 + (ey - ny2) ** 2) ** 0.5
                if d_start > tol:
                    if d_end <= tol:
                        findings.append(
                            ("WARN", row, f"{label}: primitive {i + 2} appears reversed relative to previous one.")
                        )
                    else:
                        findings.append(("WARN", row, f"{label}: primitive {i + 1} and {i + 2} are not connected."))
                    issue_rows.add(row)

            sx, sy, _, _ = primitives[0]
            _, _, ex, ey = primitives[-1]
            closed = (((sx - ex) ** 2 + (sy - ey) ** 2) ** 0.5) <= tol

            if closed:
                points = [(sx, sy)] + [(x2, y2) for _, _, x2, y2 in primitives]
                area2 = 0.0
                for i in range(len(points) - 1):
                    x0, y0 = points[i]
                    x1, y1 = points[i + 1]
                    area2 += x0 * y1 - x1 * y0
                orient = "CCW" if area2 > 0 else "CW"
                findings.append(("INFO", row, f"{label}: closed chain, orientation {orient}."))
                # Winding correctness (nesting-aware) is checked globally by
                # check_orientation_consistency below — no blanket CCW warning
                # here (a nested void is legitimately CCW, TYPE 5 winding is a
                # labeling choice, etc.).
            else:
                # An open chain is fine when both free ends land on another
                # segment's endpoint (the boundary continues there) — only a
                # genuinely dangling end is worth a warning for body types.
                other_end_keys: Set[Tuple[int, int]] = set()
                for other_row, other_seg in enumerate(self.segments):
                    if other_row == row:
                        continue
                    other_prims = self._segment_primitives(other_seg)
                    if not other_prims:
                        continue
                    osx, osy, _, _ = other_prims[0]
                    _, _, oex, oey = other_prims[-1]
                    other_end_keys.add(self._point_key(osx, osy, tol))
                    other_end_keys.add(self._point_key(oex, oey, tol))
                start_connected = self._point_key(sx, sy, tol) in other_end_keys
                end_connected = self._point_key(ex, ey, tol) in other_end_keys
                if start_connected and end_connected:
                    findings.append((
                        "INFO", row,
                        f"{label}: open chain; both ends continue into other segments.",
                    ))
                else:
                    findings.append(("WARN", row, f"{label}: open chain (start/end do not close)."))
                    if seg_type in {3, 4, 5}:
                        issue_rows.add(row)

            if ibc > 0 and ibc not in ibc_flags:
                findings.append(("ERROR", row, f"{label}: IBC flag {ibc} is referenced but not defined in IBCS."))
                issue_rows.add(row)

            if ibc > 50 and not self._find_mat_file(ibc):
                findings.append(("ERROR", row, f"{label}: IBC flag {ibc} expects missing file mat.{ibc}."))
                issue_rows.add(row)

            if seg_type in {3, 4, 5} and pos_mat <= 0:
                findings.append(("ERROR", row, f"{label}: TYPE {seg_type} requires pos_mat > 0."))
                issue_rows.add(row)
            if pos_mat > 0 and pos_mat not in diel_flags:
                findings.append(
                    ("ERROR", row, f"{label}: dielectric flag pos_mat={pos_mat} is referenced but not defined.")
                )
                issue_rows.add(row)
            if pos_mat > 50 and not self._find_mat_file(pos_mat):
                findings.append(("ERROR", row, f"{label}: pos_mat={pos_mat} expects missing file mat.{pos_mat}."))
                issue_rows.add(row)

            if seg_type == 5 and neg_mat <= 0:
                findings.append(("ERROR", row, f"{label}: TYPE 5 requires neg_mat > 0."))
                issue_rows.add(row)
            if neg_mat > 0 and neg_mat not in diel_flags:
                findings.append(
                    ("ERROR", row, f"{label}: dielectric flag neg_mat={neg_mat} is referenced but not defined.")
                )
                issue_rows.add(row)
            if neg_mat > 50 and not self._find_mat_file(neg_mat):
                findings.append(("ERROR", row, f"{label}: neg_mat={neg_mat} expects missing file mat.{neg_mat}."))
                issue_rows.add(row)
            if seg_type in {1, 2, 3, 4} and neg_mat != 0:
                findings.append(("WARN", row, f"{label}: TYPE {seg_type} typically uses neg_mat=0."))
                issue_rows.add(row)

        # Global topology checks across segments (not just within each row).
        global_primitives: List[Tuple[int, int, Tuple[float, float, float, float], str]] = []
        row_type: Dict[int, int] = {}
        for row, seg in enumerate(self.segments):
            props = self._ensure_prop_len(seg.properties, 5)
            seg_type = self._parse_int_token(props[0], -1)
            row_type[row] = seg_type
            for pidx, prim in enumerate(self._segment_primitives(seg)):
                global_primitives.append((row, pidx, prim, seg.name))

        endpoint_hits: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = {}
        for row, pidx, (x1, y1, x2, y2), _name in global_primitives:
            k1 = self._point_key(x1, y1, tol)
            k2 = self._point_key(x2, y2, tol)
            endpoint_hits.setdefault(k1, []).append((row, pidx, 0))
            endpoint_hits.setdefault(k2, []).append((row, pidx, 1))

        for _key, hits in endpoint_hits.items():
            incident_rows = sorted({h[0] for h in hits})
            if len(hits) == 1:
                row = hits[0][0]
                if row_type.get(row, -1) in {2, 3, 4, 5}:
                    findings.append(
                        ("WARN", row, f"Row {row + 1}: dangling endpoint not connected to any other primitive.")
                    )
                    issue_rows.add(row)
            if len(hits) > 6:
                row = incident_rows[0]
                findings.append(
                    (
                        "WARN",
                        row,
                        f"Row {row + 1}: high-degree node with {len(hits)} incident primitive endpoints "
                        "(possible non-manifold junction).",
                    )
                )
                issue_rows.add(row)

        max_intersections = 30
        found_intersections = 0
        n_prims = len(global_primitives)
        stop_intersections = False
        for i in range(n_prims):
            if stop_intersections:
                break
            row_i, pidx_i, prim_i, name_i = global_primitives[i]
            x1, y1, x2, y2 = prim_i
            k_i0 = self._point_key(x1, y1, tol)
            k_i1 = self._point_key(x2, y2, tol)
            for j in range(i + 1, n_prims):
                row_j, pidx_j, prim_j, name_j = global_primitives[j]
                u1, v1, u2, v2 = prim_j
                k_j0 = self._point_key(u1, v1, tol)
                k_j1 = self._point_key(u2, v2, tol)

                shared_endpoint = k_i0 in {k_j0, k_j1} or k_i1 in {k_j0, k_j1}
                if shared_endpoint:
                    continue
                if row_i == row_j and abs(pidx_i - pidx_j) <= 1:
                    continue

                if not self._segments_intersect((x1, y1), (x2, y2), (u1, v1), (u2, v2), tol):
                    continue

                findings.append(
                    (
                        "ERROR",
                        row_i,
                        (
                            f"Rows {row_i + 1} ('{name_i}') and {row_j + 1} ('{name_j}') have a non-endpoint "
                            "primitive intersection."
                        ),
                    )
                )
                issue_rows.add(row_i)
                issue_rows.add(row_j)
                found_intersections += 1
                if found_intersections >= max_intersections:
                    findings.append(
                        (
                            "WARN",
                            row_i,
                            f"Intersection reporting truncated after {max_intersections} findings.",
                        )
                    )
                    stop_intersections = True
                    break

        # Winding / air-side consistency (shared with the solver preflight,
        # which raises on the same conditions). Nesting-aware: a top-level
        # body must be CW, a nested void CCW; chained air-sided segments must
        # run head-to-tail so their air sides agree.
        chain_specs: List[ChainSpec] = []
        for row, seg in enumerate(self.segments):
            props = self._ensure_prop_len(seg.properties, 5)
            xs, ys = self._segment_plot_xy(seg)
            chain_specs.append(ChainSpec(
                name=seg.name or f"segment_{row + 1}",
                seg_type=self._parse_int_token(props[0], 2),
                pos_mat=self._parse_int_token(props[3], 0),
                points=list(zip(xs, ys)),
            ))
        for severity, chain_idx, message in check_orientation_consistency(chain_specs):
            findings.append((severity, chain_idx, message))
            if severity == "ERROR" and 0 <= chain_idx < len(self.segments):
                issue_rows.add(chain_idx)

        self.issue_rows = issue_rows
        self._refresh_segment_styles()
        self._render_normals()
        self._render_fills()
        self.canvas.draw_idle()

        errors = [msg for level, _, msg in findings if level == "ERROR"]
        warns = [msg for level, _, msg in findings if level == "WARN"]
        infos = [msg for level, _, msg in findings if level == "INFO"]

        summary = (
            f"Validation complete: {len(errors)} error(s), {len(warns)} warning(s), {len(infos)} info message(s)."
        )
        detail_lines = errors + warns + infos
        if detail_lines:
            max_lines = 30
            shown = detail_lines[:max_lines]
            detail_text = "\n".join(shown)
            if len(detail_lines) > max_lines:
                detail_text += f"\n... ({len(detail_lines) - max_lines} additional message(s))"
            message = summary + "\n\n" + detail_text
        else:
            message = summary + "\nNo issues found."

        if errors or warns:
            QMessageBox.warning(self, "Geometry Validation", message)
        else:
            QMessageBox.information(self, "Geometry Validation", message)

    def save_geo(self):
        default_name = f"geometry_out{self._last_ext}"
        fname, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Geometry File", default_name, "Geometry Files (*.geo);;All Files (*)"
        )
        if not fname:
            return
        fname = self._ensure_extension(fname, selected_filter)
        self._last_ext = os.path.splitext(fname)[1].lower()
        ibcs_rows = self._read_small_table(self.table_ibc)
        dielectric_rows = self._read_small_table(self.table_diel)
        try:
            text = build_geometry_text(self.title, self.segments, ibcs_rows, dielectric_rows)
        except ValueError as e:
            QMessageBox.warning(self, "Warning", str(e))
            return

        try:
            with open(fname, "w") as f:
                f.write(text)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save file: {e}")
            return
        self.loaded_path = os.path.abspath(fname)
        QMessageBox.information(self, "Saved", f"Geometry saved to {fname}")

    def _read_small_table(self, table: QTableWidget) -> List[List[str]]:
        rows: List[List[str]] = []
        for r in range(table.rowCount()):
            tokens: List[str] = []
            for c in range(table.columnCount()):
                widget = table.cellWidget(r, c)
                if isinstance(widget, QComboBox):
                    data = widget.currentData()
                    val = str(data) if data is not None else widget.currentText().strip()
                else:
                    item = table.item(r, c)
                    val = item.text().strip() if item else ""
                tokens.append(val)
            while tokens and tokens[-1] == "":
                tokens.pop()
            if tokens:
                rows.append(tokens)
        return rows

    def _set_equal_column_widths(self, table: QTableWidget, enabled: bool = True):
        header = table.horizontalHeader()
        if not header:
            return
        if enabled:
            header.setSectionResizeMode(QHeaderView.Stretch)
        else:
            header.setSectionResizeMode(QHeaderView.Interactive)

    def _ensure_extension(self, fname: str, selected_filter: str) -> str:
        root, ext = os.path.splitext(fname)
        ext = ext.lower()
        if ext in (".geo", ".txt"):
            return fname
        filt = (selected_filter or "").lower()
        if ".geo" in filt:
            return root + ".geo"
        if ".txt" in filt:
            return root + ".txt"
        return root + ".geo"

    def get_geometry_snapshot(self) -> Dict[str, Any]:
        ibcs_rows = self._read_small_table(self.table_ibc)
        dielectric_rows = self._read_small_table(self.table_diel)
        snapshot = build_geometry_snapshot(
            self.title,
            self.segments,
            ibcs_rows,
            dielectric_rows,
        )
        snapshot["source_path"] = self.loaded_path
        return snapshot
