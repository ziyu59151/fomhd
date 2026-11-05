# Native parity oracle exporter

The exporter is retained unchanged from commit d9390ac. The numerical solver
baseline is48beca7; this commit only restores the exporter and its CMake target.
Construct the declared rectangle, species, harmonic field and optional stress
seed; evaluate and export the native stage and one midpoint step, together with
geometry, operators, fields, diagnostics-related arrays and parameter metadata.
Refuse an existing output path. Python tests consume these arrays as independent
reference data; they never generate replacement values under a native label.

Build: cmake --preset cpu-release, then cmake --build --preset cpu-release
--target fomhd_export_native. Run the exporter with a new NPZ path and optional
POINTS, K, VELOCITY, STRESS positional values. The default4/K1/velocity0.01
exports141 numerical arrays and one metadata record.
