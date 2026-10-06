# Person A — data, matrix representation, RREF / LU

```bash
python generate_dataset.py                 # --seed <int|word> (default 42), --n-rx <k> (default 3)
python person_A.py                         # add --show to open the plot window
```

* `generate_dataset.py` — builds `uav_sensor_data.csv` and `noise_sigma.csv` **from a seed** (not committed; recreated on demand).
  Same seed + same `--n-rx` ⇒ identical file on any laptop; compare the printed *dataset fingerprint*.
* `person_A.py` — sensor matrix (rows = timestamps, columns = the 19 measurement channels), quadratic motion model
  `position(t) = c0 + c1 t + c2 t²`, 3×3 calibration system at t = 0, 5, 10 s solved by **RREF** and by **LU** (`A = P L U`),
  comparison of the two, plot of noisy GPS vs the calibration model.
* `legacy/` — the original v1 code and README (9 independent-noise channels). Kept for reference; see `../docs/DATASET_FIX.md`.

Dataset design: every clean channel is a linear function of the trajectory coefficients (clean rank 3); channels are 3 GPS
receivers per axis, barometer, velocity, accelerometer and gyro, each with independent noise. Column names of the v1 nine are unchanged.

30-second explanation: *"We simulate noisy sensors that all measure the same underlying quadratic motion. Each row is a
timestamp, each column a sensor channel. A quadratic model turns three calibration readings into a 3×3 system Ac = b; RREF
and LU both solve it, and since A is the same for x, y and z the LU factors are reused. These feed the rank/basis stage."*
RREF and LU solve the calibration model; they are not the noise filter — projection, least squares and PCA are.
