# PERSON A — UAV Sensor Fusion

## What you are responsible for

Your part is:

**Sensor data → Matrix → Motion model Ax=b → RREF → LU → coefficients**

The official PES workflow requires Matrix Representation and Matrix Simplification
(Gaussian Elimination / RREF / LU). The project evaluation also asks you to explain
the output of each component.

## Files

- `person_A.py` — complete executable code
- `uav_sensor_data.csv` — simulated noisy UAV sensor dataset
- `person_A_coefficients.csv` — generated after running the code
- `person_A_trajectory.png` — generated after running the code
- `requirements.txt` — Python packages

## Dataset

101 timestamps from 0 to 10 seconds.

Sensor channels:
- GPS X, Y, Z
- IMU acceleration X, Y, Z
- IMU gyro X, Y, Z

The dataset also contains hidden `true_x`, `true_y`, `true_z` columns for later
team validation. These are not used by Person A's calibration solve.

## Mathematical model

For each spatial coordinate:

    position(t) = c0 + c1*t + c2*t^2

So:

    A c = b

where:
- A = [1, t, t²] for each calibration timestamp
- c = [c0, c1, c2] = unknown trajectory coefficients
- b = observed GPS positions

Person A uses three calibration timestamps (0, 5, 10 seconds), making A a 3×3
system.

## RREF

The code forms:

    [A | b]

and reduces it to RREF. The final right-hand column gives the trajectory
coefficients.

## LU

The code computes:

    A = P L U

and solves:

    P L U c = b

using forward and backward substitution.

The same LU factors are reused for X, Y and Z because A is the same for all three
coordinates.

## Demo order

1. Show the sensor matrix.
2. Show A and b.
3. Show `[A | b]`.
4. Show RREF.
5. Show P, L, U.
6. Show RREF vs LU coefficients.
7. Show the 3D noisy-GPS vs calibration-model plot.

## 30-second explanation

"We start with noisy UAV sensor measurements. Each row represents a timestamp and
each column represents a sensor channel. We model each position coordinate with a
quadratic function, which converts the calibration measurements into a linear
system Ac=b. We use RREF to simplify the augmented matrix and obtain the unknown
trajectory coefficients. We also factor A using LU decomposition and solve the same
system. Since A is common to X, Y and Z, its LU factors can be reused. These
coefficients are passed to the next stages of the project."

## If asked why this is not the final noise reduction

"RREF and LU solve the calibration model; they are not themselves the final noise
filter. Projection, least squares and PCA in the later stages perform the main
trajectory fitting and noise-reduction work."

## If asked why three calibration points

"We use three timestamps because the quadratic model has three unknown coefficients,
c0, c1 and c2, giving a square 3×3 calibration system. The later least-squares
stage uses the full dataset."
