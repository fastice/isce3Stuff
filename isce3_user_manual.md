# ISCE3 User Manual
### InSAR Scientific Computing Environment — Version 3

---

> **Official Resources**
> - Repository: [github.com/isce-framework/isce3](https://github.com/isce-framework/isce3)
> - Documentation: [isce-framework.github.io/isce3](https://isce-framework.github.io/isce3/)
> - License: Apache 2.0

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Architecture Overview](#2-architecture-overview)
3. [Installation](#3-installation)
   - 3.1 [Via conda (Recommended)](#31-via-conda-recommended)
   - 3.2 [Building from Source](#32-building-from-source)
   - 3.3 [Verifying the Installation](#33-verifying-the-installation)
4. [Core Concepts and Coordinate Systems](#4-core-concepts-and-coordinate-systems)
   - 4.1 [SAR Imaging Geometry](#41-sar-imaging-geometry)
   - 4.2 [Radar Grid and Range-Doppler Coordinates](#42-radar-grid-and-range-doppler-coordinates)
   - 4.3 [Map Coordinates and Projections](#43-map-coordinates-and-projections)
   - 4.4 [Look Side Convention](#44-look-side-convention)
5. [Core Modules (`isce3.core`)](#5-core-modules-isce3core)
   - 5.1 [Orbit](#51-orbit)
   - 5.2 [Ellipsoid](#52-ellipsoid)
   - 5.3 [DateTime and TimeDelta](#53-datetime-and-timedelta)
   - 5.4 [LUT1d and LUT2d](#54-lut1d-and-lut2d)
   - 5.5 [Poly1d and Poly2d](#55-poly1d-and-poly2d)
   - 5.6 [Interpolation](#56-interpolation)
   - 5.7 [Attitude](#57-attitude)
6. [Geometry Module (`isce3.geometry`)](#6-geometry-module-isce3geometry)
   - 6.1 [rdr2geo (Topo / Forward Geometry)](#61-rdr2geo-topo--forward-geometry)
   - 6.2 [geo2rdr (Inverse Geometry)](#62-geo2rdr-inverse-geometry)
   - 6.3 [Geocode](#63-geocode)
   - 6.4 [DEMs and Height References](#64-dems-and-height-references)
7. [I/O Module (`isce3.io`)](#7-io-module-isce3io)
   - 7.1 [Raster](#71-raster)
   - 7.2 [HDF5 and NISAR Product Files](#72-hdf5-and-nisar-product-files)
8. [Signal Processing Module (`isce3.signal`)](#8-signal-processing-module-isce3signal)
   - 8.1 [Cross-Multiplication (crossmul)](#81-cross-multiplication-crossmul)
   - 8.2 [Filtering](#82-filtering)
   - 8.3 [Resampling](#83-resampling)
   - 8.4 [Backprojection](#84-backprojection)
   - 8.5 [Split-Spectrum](#85-split-spectrum)
9. [Phase Unwrapping (`isce3.unwrap`)](#9-phase-unwrapping-isce3unwrap)
10. [Image Offsets and Coregistration](#10-image-offsets-and-coregistration)
11. [Radiometric Terrain Correction (RTC)](#11-radiometric-terrain-correction-rtc)
12. [NISAR Standard Workflows](#12-nisar-standard-workflows)
    - 12.1 [Product Hierarchy](#121-product-hierarchy)
    - 12.2 [RSLC — Range-Doppler Single Look Complex](#122-rslc--range-doppler-single-look-complex)
    - 12.3 [GSLC — Geocoded SLC](#123-gslc--geocoded-slc)
    - 12.4 [GCOV — Geocoded Covariance](#124-gcov--geocoded-covariance)
    - 12.5 [RIFG / RUNW / GUNW — Interferograms](#125-rifg--runw--gunw--interferograms)
    - 12.6 [ROFF / GOFF — Pixel Offsets](#126-roff--goff--pixel-offsets)
13. [Running Workflows with Runconfig YAML](#13-running-workflows-with-runconfig-yaml)
14. [GPU Acceleration](#14-gpu-acceleration)
15. [Parallel Processing (OpenMP)](#15-parallel-processing-openmp)
16. [Data Formats Reference](#16-data-formats-reference)
17. [Ecosystem and Related Tools](#17-ecosystem-and-related-tools)
18. [Troubleshooting](#18-troubleshooting)
19. [Glossary](#19-glossary)

---

## 1. Introduction

ISCE3 (InSAR Scientific Computing Environment, version 3) is an open-source library for processing spaceborne and airborne Synthetic Aperture Radar (SAR) data. It is the product of a ground-up redesign of ISCE2, developed primarily at NASA's Jet Propulsion Laboratory (JPL) and released under the **Apache 2.0 license**.

ISCE3 serves as the operational processing platform for the **NASA-ISRO SAR (NISAR) mission**, while being architected as a general-purpose SAR processing framework suitable for any sensor or mission.

### Key Characteristics

**Sensor-neutral core.** ISCE3's core modules are designed to be agnostic to the specific satellite or radar system, making it possible to process data from a wide variety of sensors (Sentinel-1, ALOS-2, NISAR, UAVSAR, etc.) with the same foundational library.

**C++/CUDA backend with Python API.** All compute-intensive algorithms are implemented in high-performance C++ or CUDA (for GPU-accelerated paths). These algorithms are exposed to Python via `pybind11` bindings, so users interact with the library entirely through Python.

**Modular design.** The library is organized into well-separated modules for geometry, signal processing, I/O, phase unwrapping, and more. Users can assemble custom processing pipelines by combining modules as needed.

**HDF5-based data model.** ISCE3 adopts the HDF5 file format for all standard data products, providing a hierarchical, self-describing structure for radar data and metadata.

**InSAR and PolSAR.** Beyond standard InSAR processing, ISCE3 supports polarimetric SAR (PolSAR) analysis, including geocoded covariance matrix generation (GCOV).

---

## 2. Architecture Overview

```
isce3/
├── cxx/isce3/           # C++ source code
│   ├── core/            # Fundamental data structures (orbit, ellipsoid, LUTs, ...)
│   ├── geometry/        # Forward/inverse geometry, geocoding
│   ├── signal/          # SAR signal processing algorithms
│   ├── io/              # Raster and HDF5 I/O
│   ├── unwrap/          # Phase unwrapping
│   ├── polsar/          # Polarimetric SAR
│   ├── cuda/            # GPU-accelerated versions of core algorithms
│   └── ...
├── python/              # Python-level wrappers, workflows, and utilities
│   ├── packages/isce3/  # Python package (mirrors C++ module structure)
│   └── ...
├── tests/               # Unit and integration tests
├── doc/                 # Documentation sources (Sphinx + Doxygen)
└── CMakeLists.txt       # Build system entry point
```

The Python package is installed as `isce3` and mirrors the C++ namespace hierarchy:

| Python module | C++ namespace | Description |
|---|---|---|
| `isce3.core` | `isce3::core` | Orbits, ellipsoids, time, LUTs, polynomials |
| `isce3.geometry` | `isce3::geometry` | Forward/inverse geometry, geocoding, DEMs |
| `isce3.signal` | `isce3::signal` | Crossmul, filtering, resampling, backprojection |
| `isce3.io` | `isce3::io` | Raster I/O, HDF5 product access |
| `isce3.unwrap` | `isce3::unwrap` | Phase unwrapping (ICU, SNAPHU) |
| `isce3.polsar` | `isce3::polsar` | Polarimetric decompositions, covariance |
| `isce3.cuda.*` | `isce3::cuda::*` | GPU-accelerated equivalents (optional) |

---

## 3. Installation

### 3.1 Via conda (Recommended)

ISCE3 is distributed as a `conda` package via the **conda-forge** channel. This is the simplest and most reliable installation path, as all binary dependencies are pre-built and versioned.

**Step 1 — Install Miniconda or Anaconda**

If you do not already have a conda installation, download Miniconda from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html).

**Step 2 — Create a dedicated environment**

It is strongly recommended to install ISCE3 in an isolated conda environment to avoid dependency conflicts.

```bash
conda create -n isce3 python=3.10
conda activate isce3
```

**Step 3a — CPU-only installation**

```bash
conda install -c conda-forge isce3
```

**Step 3b — GPU-accelerated installation (CUDA)**

If you have an NVIDIA GPU with CUDA support and the appropriate driver installed:

```bash
conda install -c conda-forge isce3-cuda
```

> **Note:** The CUDA variant requires a compatible NVIDIA GPU and driver. The CUDA toolkit itself is bundled in the conda package; you only need the NVIDIA driver on the host machine. Check `nvidia-smi` to confirm the driver is present and note the maximum supported CUDA version.

**Checking available versions**

```bash
conda search -c conda-forge isce3
```

### 3.2 Building from Source

Building from source is required if you need a development build, want to modify the C++ core, or are working on a platform not covered by the conda packages.

#### Prerequisites

| Dependency | Minimum Version | Notes |
|---|---|---|
| C++ compiler | GCC 6 / Clang 6 | C++17 support required |
| CMake | 3.12 | Build system |
| Python | 3.7 | With development headers |
| NumPy | Latest stable | Python numerical array library |
| pybind11 | 2.6 | C++/Python binding generator |
| HDF5 | 1.10.2 | With C++ and parallel support |
| h5py | Latest stable | Python HDF5 bindings |
| GDAL | 2.3 | With Python bindings |
| FFTW | 3 | Fast Fourier Transform library |
| ruamel.yaml | Any | YAML configuration parsing |
| OpenMP | — | For multi-threaded CPU processing |
| CUDA Toolkit | 10+ | **Optional**, for GPU acceleration |

#### Clone and Configure

```bash
git clone https://github.com/isce-framework/isce3.git
cd isce3
mkdir build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/path/to/install \
  -DWITH_CUDA=OFF          # Set to ON to enable GPU support
```

#### Build and Install

```bash
cmake --build . --parallel $(nproc)
cmake --install .
```

#### Environment Setup

After installation, add the install directory to your environment:

```bash
export PYTHONPATH=/path/to/install/packages:$PYTHONPATH
export PATH=/path/to/install/bin:$PATH
```

#### Building with CUDA

```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/path/to/install \
  -DWITH_CUDA=ON \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
```

### 3.3 Verifying the Installation

After installation, verify that ISCE3 is importable and check the version:

```python
import isce3
print(isce3.__version__)
```

A basic sanity check for geometry:

```python
import isce3

# Create a WGS84 ellipsoid
ellipsoid = isce3.core.Ellipsoid()
print(f"Semi-major axis: {ellipsoid.a} m")
print(f"Eccentricity squared: {ellipsoid.e2}")
```

You can also run the test suite (requires a source build):

```bash
cd build
ctest --output-on-failure
```

---

## 4. Core Concepts and Coordinate Systems

Understanding ISCE3's handling of geometry is essential for working with the library. This section explains the two primary coordinate systems and how ISCE3 converts between them.

### 4.1 SAR Imaging Geometry

A SAR sensor illuminates a swath of terrain from a moving platform. Each point on the ground is characterized by:

- **Azimuth time** — the time at which the sensor antenna footprint passes over the target.
- **Slant range** — the distance from the antenna to the target at the moment of imaging.

Together, (azimuth time, slant range) define the **radar (or range-Doppler) coordinate system**.

ISCE3 uses the **Zero-Doppler convention** for azimuth time, consistent with ESA sensors (Sentinel-1, ERS, ENVISAT, TerraSAR-X, COSMO-SkyMed) and JAXA ALOS-2 PALSAR L1.1 products. In this convention, the azimuth time of a given target corresponds to the moment of closest approach between the satellite and the target (i.e., when the Doppler centroid is zero).

### 4.2 Radar Grid and Range-Doppler Coordinates

The radar grid is defined by:

- **Sensing start time** — UTC time of the first azimuth line.
- **PRF (Pulse Repetition Frequency)** — determines the azimuth sampling rate.
- **Starting slant range** — distance to the first range sample.
- **Range sampling rate** — speed-of-light / (2 × range bandwidth).
- **Number of azimuth lines and range samples** — image dimensions.
- **Look side** — whether the sensor looks to the LEFT or RIGHT of the flight path.
- **Wavelength / center frequency** — the radar carrier frequency.

### 4.3 Map Coordinates and Projections

Geocoded products are expressed in a geographic or projected coordinate reference system (CRS), such as:

- **WGS84 geographic** (EPSG:4326) — latitude, longitude, ellipsoidal height.
- **UTM zones** — meter-based projected coordinates.
- **Polar stereographic** — used in polar regions.

ISCE3 uses **GDAL** for all CRS handling, so any EPSG-registered projection can be used. DEMs provided as input must be co-registered to the **WGS84 ellipsoid** (heights above WGS84, not geoid-referenced heights).

### 4.4 Look Side Convention

ISCE3 defines look side using the `isce3.core.LookSide` enumeration:

```python
import isce3

# Right-looking (most common for spaceborne SAR)
look = isce3.core.LookSide.Right

# Left-looking
look = isce3.core.LookSide.Left
```

---

## 5. Core Modules (`isce3.core`)

The `isce3.core` module provides the fundamental data structures used throughout the library. These classes encapsulate sensor metadata and mathematical utilities.

### 5.1 Orbit

The `Orbit` class stores a time series of satellite state vectors (position and velocity in an Earth-Centered Earth-Fixed (ECEF) coordinate system) and provides methods for interpolating the satellite position and velocity at arbitrary times.

```python
import isce3
import numpy as np
from datetime import datetime, timezone

# Create an orbit from state vectors
# Each state vector: (time, position [x,y,z], velocity [vx,vy,vz])
times = [
    isce3.core.DateTime("2021-01-01T00:00:00.000000000"),
    isce3.core.DateTime("2021-01-01T00:00:10.000000000"),
    isce3.core.DateTime("2021-01-01T00:00:20.000000000"),
]

# Positions and velocities in ECEF (meters, meters/second)
positions = [
    [-2412125.5, -4898625.0, 4786555.5],
    [-2411850.0, -4900100.0, 4785950.0],
    [-2411575.0, -4901575.0, 4785345.0],
]
velocities = [
    [2750.0, -1480.0, -610.0],
    [2750.5, -1479.5, -610.5],
    [2751.0, -1479.0, -611.0],
]

orbit = isce3.core.Orbit(
    state_vectors=[
        isce3.core.StateVector(t, pos, vel)
        for t, pos, vel in zip(times, positions, velocities)
    ]
)

# Interpolate at a given time
t_interp = isce3.core.DateTime("2021-01-01T00:00:05.000000000")
pos, vel = orbit.interpolate(t_interp)
print(f"Position: {pos}")
print(f"Velocity: {vel}")
```

Supported interpolation methods include Hermite (cubic) and Legendre polynomial interpolation.

### 5.2 Ellipsoid

The `Ellipsoid` class represents a reference ellipsoid for geodetic computations. By default, ISCE3 uses WGS84.

```python
import isce3

# Default WGS84 ellipsoid
ellipsoid = isce3.core.Ellipsoid()

# Semi-major axis (equatorial radius, meters)
print(ellipsoid.a)       # 6378137.0

# Eccentricity squared
print(ellipsoid.e2)

# Convert ECEF to lat/lon/height
lat, lon, height = ellipsoid.xyz_to_lon_lat([
    -2412125.5, -4898625.0, 4786555.5
])

# Convert lat/lon/height to ECEF
xyz = ellipsoid.lon_lat_to_xyz([lon, lat, height])
```

Custom ellipsoids can be instantiated by providing semi-major axis `a` and eccentricity squared `e2`.

### 5.3 DateTime and TimeDelta

ISCE3 uses its own high-precision time classes to avoid floating-point precision loss in UTC timestamps.

```python
import isce3

# Create a DateTime from an ISO 8601 string (nanosecond precision)
t = isce3.core.DateTime("2021-06-15T10:30:00.123456789")

# Arithmetic with TimeDelta
dt = isce3.core.TimeDelta(seconds=5.5)
t2 = t + dt

# Difference between two DateTimes
diff = t2 - t
print(diff.total_seconds())  # 5.5

# Format as ISO string
print(str(t))
```

### 5.4 LUT1d and LUT2d

Lookup tables store slowly varying quantities that need to be sampled efficiently at arbitrary coordinates. `LUT1d` is a one-dimensional lookup table; `LUT2d` is two-dimensional (indexing by azimuth time and slant range).

```python
import isce3
import numpy as np

# LUT2d: used for Doppler, carrier phase correction, etc.
# Axes: azimuth (rows) and range (columns)
az_start = 0.0     # seconds
az_spacing = 1.0
rg_start = 800000.0  # meters
rg_spacing = 10.0

data = np.zeros((100, 500), dtype=np.float64)  # your values here

lut = isce3.core.LUT2d(
    x_start=rg_start,
    x_spacing=rg_spacing,
    y_start=az_start,
    y_spacing=az_spacing,
    data=data
)

# Evaluate at a specific (azimuth_time, slant_range)
value = lut.eval(az_time=10.5, rg=810000.0)
```

LUT2d objects are commonly used to store:

- Doppler centroid frequency as a function of (azimuth time, slant range)
- Carrier phase offsets
- Ionospheric range delay
- Tropospheric delay

### 5.5 Poly1d and Poly2d

Polynomial representations are used for quantities that vary smoothly along one or two dimensions.

```python
import isce3

# Poly1d: 1D polynomial
# coefficients in ascending order (constant, x, x^2, ...)
poly1 = isce3.core.Poly1d(
    order=2,
    mean=0.0,
    norm=1.0,
    coeffs=[1.0, 2.0, 3.0]  # 1 + 2x + 3x^2
)
print(poly1.eval(2.0))  # 1 + 4 + 12 = 17
```

### 5.6 Interpolation

`isce3.core` provides several interpolation kernels used throughout the library:

| Kernel | Class | Description |
|---|---|---|
| Bilinear | `isce3.core.BilinearInterpolator` | Fast, 2×2 stencil |
| Bicubic | `isce3.core.BicubicInterpolator` | Smooth, 4×4 stencil |
| Sinc | `isce3.core.Sinc2dInterpolator` | Band-limited, configurable kernel length |
| Nearest neighbor | `isce3.core.NearestNeighborInterpolator` | Fastest, no smoothing |

### 5.7 Attitude

The `Attitude` class stores a time series of platform orientations (roll, pitch, yaw or quaternions) and provides interpolation.

```python
import isce3

# Euler angle representation (radians)
attitude = isce3.core.Attitude(
    time=[...],
    euler_angles=[...]
)
rpy = attitude.interpolate(t)  # returns (roll, pitch, yaw)
```

---

## 6. Geometry Module (`isce3.geometry`)

The geometry module implements the mathematical transformations between radar (range-Doppler) and map (geographic) coordinate systems. These transformations are the foundation of geocoding and coregistration workflows.

### 6.1 rdr2geo (Topo / Forward Geometry)

**Forward geometry** maps a point in radar coordinates (azimuth time, slant range) to map coordinates (latitude, longitude, height). It requires a DEM to resolve the height ambiguity.

This is implemented via `isce3.geometry.rdr2geo` (Python convenience function) or the C++ `isce3::geometry::Topo` class.

```python
import isce3
import numpy as np

# Set up radar geometry
ellipsoid = isce3.core.Ellipsoid()
orbit = ...      # isce3.core.Orbit object
doppler = isce3.core.LUT2d()   # zero Doppler (default)

# Radar grid parameters
wavelength = 0.056  # meters (C-band ~5.6 cm)
look_side  = isce3.core.LookSide.Right

# Call rdr2geo
lon, lat, height = isce3.geometry.rdr2geo(
    azimuth_time=10.5,    # seconds from orbit reference
    slant_range=850000.0, # meters
    orbit=orbit,
    ellipsoid=ellipsoid,
    doppler=doppler,
    wavelength=wavelength,
    side=look_side,
    threshold=1e-8,       # convergence threshold (meters)
    maxiter=50,           # maximum Newton iterations
    extraiter=10
)
print(f"Lat={lat:.6f}, Lon={lon:.6f}, H={height:.2f}")
```

For processing a full image block, the `Topo` class operates on the entire radar grid at once and writes output layers (latitude, longitude, height, local incidence angle, heading, slope, shadow/layover masks) to raster files.

### 6.2 geo2rdr (Inverse Geometry)

**Inverse geometry** maps a geographic point (latitude, longitude, height) back to radar coordinates (azimuth time, slant range). This is used for image coregistration and DEM-based simulation.

```python
import isce3

az_time, slant_range = isce3.geometry.geo2rdr(
    lon=lon,
    lat=lat,
    height=height,
    orbit=orbit,
    ellipsoid=ellipsoid,
    doppler=doppler,
    wavelength=wavelength,
    side=look_side,
    threshold=1e-8,
    maxiter=50,
    delta_range=1e-8  # numerical Jacobian step (meters)
)
```

### 6.3 Geocode

The `Geocode` class performs geocoding — projecting a radar-coordinate raster into a geographic map grid — and is the workhorse of `GSLC` and `GCOV` product generation.

Two geocoding strategies are available:

**Interpolation-based geocoding** (`GeocodeFloat32`, `GeocodeFloat64`, `GeocodeComplex64`, etc.) — For each output map pixel, the corresponding radar pixel is located via `geo2rdr` and the value is interpolated from the input image.

**Area-based geocoding** — Accounts for the fractional area of each radar pixel that maps to each output map pixel, preserving radiometric accuracy. Used for backscatter / GCOV products.

```python
import isce3

# Geocode a single-band floating-point raster
geocode = isce3.geocode.GeocodeFloat32()

geocode.orbit        = orbit
geocode.ellipsoid    = ellipsoid
geocode.doppler      = doppler
geocode.threshold    = 1e-8
geocode.maxiter      = 50

# Set the output grid (map CRS, extent, spacing)
geocode.geogrid(
    x_start=-120.0,     # West longitude (degrees)
    y_start=35.0,       # South latitude (degrees)
    x_end=-118.0,       # East longitude (degrees)
    y_end=37.0,         # North latitude (degrees)
    x_spacing=0.0001,   # degrees per pixel
    y_spacing=0.0001,
    epsg=4326
)

geocode.geocode(
    radar_grid=radar_grid,   # isce3.product.RadarGridParameters
    input_raster=input_raster,
    output_raster=output_raster,
    dem_raster=dem_raster,
    output_mode=isce3.geocode.GeocodeOutputMode.Interp
)
```

### 6.4 DEMs and Height References

DEMs used with ISCE3 must:

- Be referenced to the **WGS84 ellipsoid** (not mean sea level / EGM96 / EGM2008). Convert geoid-referenced DEMs with a geoid undulation model before use.
- Be in a **GDAL-readable format** (GeoTIFF recommended).
- Have a defined EPSG projection code.
- Cover the full area of interest with some margin.

Commonly used global DEMs compatible with ISCE3:

| DEM | Resolution | Notes |
|---|---|---|
| Copernicus DEM (GLO-30) | 30 m | Ellipsoid-referenced, excellent global coverage |
| NASADEM | 30 m | Ellipsoid-referenced, Shuttle Radar Topography Mission (SRTM) based |
| SRTM v3 | 30 m | Geoid-referenced (EGM96) — requires conversion |
| ALOS World 3D (AW3D30) | 30 m | Ellipsoid-referenced |

---

## 7. I/O Module (`isce3.io`)

### 7.1 Raster

The `Raster` class wraps GDAL raster datasets and is the primary way ISCE3 reads and writes image data.

```python
import isce3

# Open an existing raster for reading
raster_in = isce3.io.Raster("/path/to/input.tif")
print(f"Width:  {raster_in.width}")
print(f"Length: {raster_in.length}")
print(f"Bands:  {raster_in.num_bands}")
print(f"EPSG:   {raster_in.get_epsg()}")

# Create a new output raster
raster_out = isce3.io.Raster(
    "/path/to/output.tif",
    width=1000,
    length=500,
    num_bands=1,
    dtype=isce3.io.gdal_dtype.Float32,
    driver_name="GTiff"
)

# Read a band as a numpy array
import numpy as np
data = np.zeros((raster_in.length, raster_in.width), dtype=np.float32)
raster_in.get_array(band=1, data=data)

# Write data to an output raster
raster_out.set_array(band=1, data=data)
```

Supported raster drivers include GeoTIFF, VRT, HDF5, ENVI, and any other format supported by the system GDAL installation.

### 7.2 HDF5 and NISAR Product Files

NISAR standard products are stored in HDF5 files with a hierarchical group structure. ISCE3 provides convenience classes for reading and writing these products.

The typical HDF5 group structure for a NISAR RSLC product:

```
/science/
  LSAR/                       # or SSAR for S-band
    RSLC/
      metadata/
        orbit/                # Orbit state vectors
        attitude/             # Platform attitude
        processingInformation/
      swaths/
        frequencyA/           # L-band frequency A
          HH/                 # Polarization channel
            (complex SLC data)
        frequencyB/
          ...
```

Accessing product data with `h5py` (complementary to ISCE3's own bindings):

```python
import h5py
import numpy as np

with h5py.File("NISAR_RSLC.h5", "r") as f:
    # Read SLC data
    slc = f["/science/LSAR/RSLC/swaths/frequencyA/HH"][:]

    # Read orbit state vectors
    pos = f["/science/LSAR/RSLC/metadata/orbit/position"][:]
    vel = f["/science/LSAR/RSLC/metadata/orbit/velocity"][:]
    time = f["/science/LSAR/RSLC/metadata/orbit/time"][:]
```

---

## 8. Signal Processing Module (`isce3.signal`)

### 8.1 Cross-Multiplication (crossmul)

Cross-multiplication forms the complex interferogram by multiplying the reference SLC by the complex conjugate of the secondary SLC. It is the central operation in InSAR processing.

```python
import isce3

# Form interferogram between reference and secondary SLCs
crossmul = isce3.signal.Crossmul()

crossmul.crossmul(
    referenceSLC=ref_raster,    # isce3.io.Raster
    secondarySLC=sec_raster,    # isce3.io.Raster
    interferogram=ifgram_raster,
    coherence=coh_raster,
    rangeLooks=5,
    azimuthLooks=5,
    refDoppler=ref_doppler,     # LUT2d
    secDoppler=sec_doppler,     # LUT2d
    wavelength=wavelength
)
```

Multi-looking (spatial averaging) during cross-multiplication reduces speckle noise and decorrelation, improving phase quality at the cost of spatial resolution.

### 8.2 Filtering

ISCE3 provides range and azimuth spectral filters for:

- **Range spectral filtering** — filters reference and secondary images to the common range spectral overlap, reducing noise from spatial baseline decorrelation.
- **Azimuth spectral filtering** — removes Doppler centroid differences between acquisitions.
- **Adaptive interferogram filtering** — Goldstein-type filter to enhance interferometric phase quality.

```python
import isce3

# Range bandwidth filter
filter_obj = isce3.signal.Filter()
filter_obj.bandpassFilter(
    input_raster=raster,
    output_raster=filtered_raster,
    rng_bw=bandwidth_hz,
    center_freq=center_freq_hz
)
```

### 8.3 Resampling

Resampling an SLC from its native radar grid to a coregistered grid aligned with a reference image. This is essential for InSAR pair formation.

```python
import isce3

resamp = isce3.image.ResampSlc(radar_grid=radar_grid, doppler=doppler, wavelength=wavelength)

resamp.resamp(
    inputFilename=secondary_slc_path,
    outputFilename=coregistered_slc_path,
    rgoffFilename=range_offset_path,    # range pixel offsets
    azoffFilename=azimuth_offset_path,  # azimuth pixel offsets
    num_lines=lines,
    num_subswaths=1,
)
```

The offset files that drive resampling can come from geometric coregistration (geo2rdr) or cross-correlation matching.

### 8.4 Backprojection

Backprojection is a time-domain SAR focusing algorithm. Unlike frequency-domain methods (Range-Doppler, Omega-K), backprojection is geometrically exact and particularly suited for circular or highly non-linear trajectories (e.g., airborne platforms).

```python
import isce3

# Focus raw SAR data using backprojection
isce3.focus.backproject(
    output=output_raster,
    input=raw_raster,
    orbit=orbit,
    doppler=doppler,
    dem=dem_raster,
    ellipsoid=ellipsoid,
    fc=center_freq,
    dt=prf_inverse,
    dr=range_spacing
)
```

### 8.5 Split-Spectrum

Split-spectrum processing separates the radar signal into sub-bands in the range direction to estimate and correct the ionospheric phase delay.

```python
import isce3

split = isce3.splitspectrum.SplitSpectrum()
split.splitspectrum(
    slc_raster=slc_raster,
    low_band_raster=low_raster,
    high_band_raster=high_raster,
    range_bandwidth=bandwidth,
    center_frequency=fc
)
```

The sub-band SLCs are then independently processed through InSAR, and the differential phase is used to estimate total electron content (TEC) for ionospheric correction.

---

## 9. Phase Unwrapping (`isce3.unwrap`)

Phase unwrapping converts the wrapped interferometric phase (restricted to [−π, π]) into an absolute phase map. ISCE3 includes two unwrapping engines:

### ICU (Integrated Correlation and Unwrapping)

ICU is a fast branch-cut algorithm suitable for moderate-complexity interferograms. It uses coherence as a reliability measure to guide the unwrapping.

```python
import isce3

icu = isce3.unwrap.ICU(
    buffer_lines=3700,
    overlap=200,
    use_phase_gradient_neut=False,
    neut_phase_grad_win_size=5
)

icu.unwrap(
    wrapped_igram=ifgram_raster,
    coherence=coh_raster,
    unwrapped_igram=unw_raster,
    coherence_threshold=0.05
)
```

### SNAPHU

SNAPHU (Statistical-cost, Network-flow Algorithm for Phase Unwrapping) is a minimum-cost network flow algorithm developed by Chen and Zebker. It is more robust than ICU for complex topography or low-coherence scenes, at higher computational cost.

ISCE3 interfaces with SNAPHU via the `isce3.unwrap.snaphu` module:

```python
import isce3

snaphu = isce3.unwrap.snaphu.Snaphu(
    cost_mode="SMOOTH",   # or "DEFO" for deformation, "TOPO" for topography
    init_method="MCF"     # Minimum Cost Flow initialization
)

snaphu.unwrap(
    wrapped_igram=ifgram_raster,
    coherence=coh_raster,
    unwrapped_igram=unw_raster,
    nlooks=nlooks,
    cost_threshold=100
)
```

**SNAPHU cost modes:**

| Mode | Description |
|---|---|
| `SMOOTH` | Assumes slowly varying phase (smooth deformation) |
| `DEFO` | Optimized for deformation signals (e.g., earthquake, subsidence) |
| `TOPO` | Optimized for topographic phase |

---

## 10. Image Offsets and Coregistration

Dense pixel offset maps are used both for coregistration of SLC pairs and for measuring surface displacement (azimuth offsets capture along-track motion; range offsets capture range-direction motion).

ISCE3 supports two offset estimation approaches:

**Geometric offsets** — computed analytically from geo2rdr using orbit and DEM information. Fast and noise-free, but only as accurate as the orbits and DEM.

```python
import isce3

# Compute offsets from geometry
isce3.geometry.compute_geo2rdr_offsets(
    radar_grid=radar_grid,
    orbit=orbit,
    doppler=doppler,
    ellipsoid=ellipsoid,
    dem=dem_raster,
    range_offset_raster=rg_off,
    azimuth_offset_raster=az_off
)
```

**Correlation-based offsets (ampcor)** — cross-correlates amplitude patches between reference and secondary images. Provides sub-pixel accuracy and captures any systematic bias in geometric offsets (e.g., ionospheric distortion).

```python
import isce3

ampcor = isce3.matchtemplate.AmpcorNormSqCorr(
    refSlc=ref_raster,
    secSlc=sec_raster,
    ...
)
ampcor.runAmpcor()
```

In production workflows, both methods are combined: geometric offsets provide the initial estimate, and ampcor refines them.

---

## 11. Radiometric Terrain Correction (RTC)

Radiometric Terrain Correction normalizes SAR backscatter for the local terrain slope, removing the brightness modulation caused by variable radar incidence angle across rugged terrain.

ISCE3 implements RTC as part of the GCOV (Geocoded Covariance) product generation workflow.

```python
import isce3

rtc = isce3.geometry.RtcAlgorithm.RtcAreaProjection  # preferred for accuracy

# Alternatively: RtcDavidSmallMethod (faster, approximation)
```

RTC output modes:

- **Beta-nought** — raw backscatter (not terrain corrected)
- **Sigma-nought** — backscatter normalized by sin(incidence angle)
- **Gamma-nought** — backscatter normalized by cos(local slope), the most terrain-invariant measure

---

## 12. NISAR Standard Workflows

ISCE3 ships with a complete set of end-to-end workflows designed to produce the standard NISAR Level 1 and Level 2 data products. These workflows are invoked via command-line scripts and configured through YAML runconfig files.

### 12.1 Product Hierarchy

```
L0B (raw telemetry)
  │
  └─► RSLC (Range-Doppler SLC)
        │
        ├─► GSLC  (Geocoded SLC)
        ├─► GCOV  (Geocoded Covariance / backscatter)
        └─► RIFG  (Range-Doppler Interferogram, wrapped)
              │
              ├─► RUNW  (Range-Doppler Unwrapped)
              │     └─► GUNW  (Geocoded Unwrapped Interferogram)
              └─► ROFF  (Range-Doppler Offsets)
                    └─► GOFF  (Geocoded Offsets)
```

### 12.2 RSLC — Range-Doppler Single Look Complex

RSLC is a focused, complex-valued SAR image in range-Doppler coordinates. It is the primary Level-1 output and the input to all downstream products.

**Run command:**

```bash
python -m nisar.workflows.focus \
  --run-config-path rslc_runconfig.yaml
```

Key processing steps:

1. Ingest L0B raw data
2. Range pulse compression (matched filtering)
3. Range migration correction
4. Azimuth compression
5. Doppler centroid estimation
6. RFI (Radio Frequency Interference) suppression (optional)
7. Output RSLC HDF5 product

### 12.3 GSLC — Geocoded SLC

GSLC resamples the RSLC into a regular geographic map grid while preserving the complex phase. It is used for polarimetric analysis and change detection.

**Run command:**

```bash
python -m nisar.workflows.gslc \
  --run-config-path gslc_runconfig.yaml
```

### 12.4 GCOV — Geocoded Covariance

GCOV computes the polarimetric covariance matrix from multi-polarization SLCs and geocodes it. It includes radiometric terrain correction (RTC) and is the primary product for backscatter analysis.

**Run command:**

```bash
python -m nisar.workflows.gcov \
  --run-config-path gcov_runconfig.yaml
```

Output HDF5 structure includes:

- Geocoded covariance elements (HH, VV, HV, VH) in gamma-nought
- RTC area normalization factor
- Local incidence angle
- Shadow/layover mask

### 12.5 RIFG / RUNW / GUNW — Interferograms

The InSAR workflow generates wrapped (RIFG), unwrapped range-Doppler (RUNW), and unwrapped geocoded (GUNW) interferograms.

**InSAR workflow run command:**

```bash
python -m nisar.workflows.insar \
  --run-config-path insar_runconfig.yaml
```

The InSAR workflow performs:

1. SLC coregistration (geometric + dense offsets)
2. Range and azimuth spectral filtering
3. Cross-multiplication → RIFG
4. Coherence estimation
5. Phase unwrapping (ICU or SNAPHU) → RUNW
6. Geocoding of unwrapped phase → GUNW
7. Ionospheric correction (split-spectrum, optional)
8. Solid earth tide correction (optional)

### 12.6 ROFF / GOFF — Pixel Offsets

Pixel offset products measure range and azimuth surface displacement by cross-correlating SAR amplitude images.

**Run command:**

```bash
python -m nisar.workflows.offsets \
  --run-config-path offsets_runconfig.yaml
```

---

## 13. Running Workflows with Runconfig YAML

All NISAR workflows are configured through YAML files. Each product type has its own schema, but shares a common structure.

### General Runconfig Structure

```yaml
runconfig:
  name: insar_workflow

  groups:
    pge_name_group:
      pge_name: InSarPge

    input_file_group:
      reference_rslc_file: /path/to/reference.h5
      secondary_rslc_file: /path/to/secondary.h5

    dynamic_ancillary_file_group:
      dem_file: /path/to/dem.tif

    product_path_group:
      product_path: /path/to/output/
      scratch_path: /path/to/scratch/
      sas_output_file: /path/to/output/insar_product.h5

    primary_executable:
      product_type: RUNW

    processing:
      # Subswaths to process (1-indexed)
      process_subswaths: [1]

      # Frequencies and polarizations
      process_freqs:
        - freq: A
          pols: [HH, VV]

      # Coregistration
      coarse_offsets:
        enabled: true
        window_range: 256
        window_azimuth: 256
        skip_range: 32
        skip_azimuth: 32

      dense_offsets:
        enabled: false

      # Interferogram formation
      interferogram:
        range_looks: 11
        azimuth_looks: 3
        flatten: true

      # Phase unwrapping
      unwrap:
        algorithm: ICU       # or SNAPHU
        run_snaphu: false
        snaphu:
          cost_type: SMOOTH

      # Geocoding
      geocode:
        algorithm: interp
        epsg: 4326
```

### Generating a Default Runconfig

Each workflow module provides a `--generate-runconfig` flag to produce a template runconfig with all options and their defaults:

```bash
python -m nisar.workflows.insar --generate-runconfig > insar_runconfig.yaml
```

Edit the generated file, replacing placeholder paths and adjusting processing parameters as needed.

### Validating a Runconfig

```bash
python -m nisar.workflows.insar --validate-runconfig insar_runconfig.yaml
```

---

## 14. GPU Acceleration

ISCE3 includes CUDA-accelerated implementations of the most compute-intensive modules. GPU processing is available when the `isce3-cuda` conda package is installed (or when the source build is compiled with `-DWITH_CUDA=ON`).

### Modules with GPU Acceleration

| Module | CPU class | GPU class |
|---|---|---|
| Geocoding | `isce3.geocode.GeocodeFloat32` | `isce3.cuda.geocode.GeocodeFloat32` |
| Crossmul | `isce3.signal.Crossmul` | `isce3.cuda.signal.Crossmul` |
| Ampcor (offset tracking) | `isce3.matchtemplate.AmpcorNormSqCorr` | `isce3.cuda.matchtemplate.PyCuAmpcor` |
| Backprojection | `isce3.focus.backproject` | `isce3.cuda.focus.backproject` |
| Resamp | `isce3.image.ResampSlc` | `isce3.cuda.image.ResampSlc` |

The GPU APIs are designed to be drop-in replacements for their CPU equivalents, with the same interface. Workflow scripts automatically select the GPU path when a CUDA-capable device is present and the GPU package is installed.

### Checking GPU Availability

```python
import isce3

# Check if CUDA is available
print(isce3.cuda.have_cuda())

# List available GPU devices
isce3.cuda.print_cuda_device_info()
```

### Controlling GPU Device

```python
import isce3

# Use a specific GPU device (for multi-GPU systems)
isce3.cuda.set_device(device_id=0)
```

---

## 15. Parallel Processing (OpenMP)

CPU modules use **OpenMP** for multi-threaded parallel processing. The number of threads is controlled via the standard OpenMP environment variable:

```bash
export OMP_NUM_THREADS=16
python -m nisar.workflows.gcov --run-config-path gcov_runconfig.yaml
```

If not set, OpenMP defaults to the number of physical CPU cores available.

Some workflow scripts also accept a `--threads` argument or equivalent runconfig parameter. Refer to the specific workflow documentation for details.

---

## 16. Data Formats Reference

### HDF5 Product Format

All NISAR standard products use HDF5 with a prescribed internal path structure. Datasets within the HDF5 file are stored as:

- Complex32 (two float16 values) or Complex64 (two float32 values) for SLC data
- Float32 for real-valued layers (coherence, offsets, unwrapped phase, backscatter)
- Int8 or Int16 for mask layers

Standard metadata datasets include:

| Path | Content |
|---|---|
| `.../metadata/orbit/position` | ECEF position [x,y,z] in meters |
| `.../metadata/orbit/velocity` | ECEF velocity [vx,vy,vz] in m/s |
| `.../metadata/orbit/time` | UTC times (seconds from reference) |
| `.../metadata/processingInformation/parameters/...` | Range/azimuth bandwidth, center frequency, wavelength |
| `.../swaths/frequencyA/slantRange` | Slant range vector (meters) |
| `.../swaths/frequencyA/zeroDopplerTime` | Zero-Doppler azimuth time vector (seconds) |

### Raster Formats

ISCE3 accepts any GDAL-readable raster as input. Recommended output formats:

| Format | Extension | Use case |
|---|---|---|
| GeoTIFF | `.tif` | General purpose geocoded outputs |
| ENVI | `.bin` / `.hdr` | Legacy compatibility, intermediate files |
| VRT | `.vrt` | Virtual mosaics and resampled views |
| HDF5 | `.h5` | Standard NISAR products |

### Coordinate Reference Systems

ISCE3 uses **EPSG codes** for all geographic coordinate reference systems via GDAL's PROJ library. Commonly used EPSG codes:

| EPSG | Description |
|---|---|
| 4326 | WGS84 geographic (lat/lon) |
| 32601–32660 | UTM zones, North hemisphere |
| 32701–32760 | UTM zones, South hemisphere |
| 3031 | Antarctic Polar Stereographic |
| 3413 | Arctic Polar Stereographic (NSIDC) |

---

## 17. Ecosystem and Related Tools

ISCE3 is the foundation for a growing ecosystem of higher-level tools and mission-specific processors:

### OPERA Project

The [OPERA (Observational Products for End-Users from Remote Sensing Analysis)](https://www.jpl.nasa.gov/go/opera) project uses ISCE3 to generate operational analysis-ready products from Sentinel-1 and NISAR data:

- **DIST-ALERT / DIST-ANN** — Surface disturbance detection from GCOV time series
- **DISP-S1** — Surface displacement from Sentinel-1 InSAR
- **CSLC-S1** — Coregistered SLC for Sentinel-1
- **RTC-S1** — Radiometrically terrain-corrected Sentinel-1 backscatter

### PLAnT-ISCE3

[PLAnT-ISCE3](https://github.com/isce-framework/plant-isce3) (Polarimetric Interferometric Lab and Analysis Tool) provides Python scripts for PolSAR analysis built on top of ISCE3.

### NISAR Quality Assurance (nisarqa)

[nisarqa](https://github.com/isce-framework/nisarqa) is the quality assurance tool for NISAR products. It validates product files and generates QA reports:

```bash
# Generate a default runconfig for QA of RSLC
nisarqa dumpconfig rslc > rslc_qa.yaml

# Run QA on an RSLC product
nisarqa rslc --run-config-path rslc_qa.yaml
```

### MultiRTC

[MultiRTC](https://github.com/MultiSAR/MultiRTC) is a Python library for creating ISCE3-based RTC products from multiple SAR missions (Sentinel-1, ALOS-2, NISAR, etc.).

### SDS On-Demand Tutorials

[sds-ondemand](https://github.com/isce-framework/sds-ondemand) provides Jupyter notebook tutorials for NISAR science processing on cloud platforms.

---

## 18. Troubleshooting

### ImportError when importing isce3

**Symptom:** `ImportError: No module named 'isce3'`

**Fixes:**
- Confirm the correct conda environment is activated: `conda activate isce3`
- Verify the package is installed: `conda list isce3`
- If building from source, check that `PYTHONPATH` points to the correct install prefix

### HDF5 version conflicts

**Symptom:** HDF5 library version mismatch warnings or crashes on file open.

**Fix:** Pin HDF5 to a compatible version:
```bash
conda install -c conda-forge hdf5=1.12
```

### Conda environment solver timeout

**Symptom:** Conda hangs or takes very long resolving the ISCE3 environment.

**Fix:** Use `mamba` (a faster conda solver):
```bash
conda install -c conda-forge mamba
mamba install -c conda-forge isce3
```

### GPU out-of-memory errors

**Symptom:** `CUDA error: out of memory` during geocoding or ampcor.

**Fixes:**
- Reduce block size / tile size in the runconfig
- Process in smaller chunks (adjust `block_size_y` in the geocoding runconfig)
- Use a GPU with more VRAM

### Phase unwrapping diverges or fails

**Symptom:** SNAPHU or ICU produces clearly wrong output or exits with an error.

**Fixes:**
- Check coherence map — very low coherence scenes may be unwrappable
- Increase `coherence_threshold` to mask low-coherence pixels before unwrapping
- Switch to `DEFO` cost mode for deformation scenes
- Increase SNAPHU `tile_nrow` / `tile_ncol` to process in tiles for large images

### DEM not covering the scene

**Symptom:** `RuntimeError: DEM does not cover the radar scene extent`

**Fix:** Ensure the DEM bounding box fully encompasses the SAR scene footprint with extra margin (at least 0.1 degrees in each direction for a spaceborne SAR scene).

---

## 19. Glossary

| Term | Definition |
|---|---|
| **Azimuth** | Along-track direction (parallel to satellite flight path) |
| **Range / Slant range** | Cross-track distance from antenna to target |
| **SLC** | Single Look Complex — focused SAR image with preserved phase |
| **InSAR** | Interferometric SAR — technique measuring phase difference between two SAR acquisitions |
| **Interferogram** | Complex image of phase differences between two SLCs |
| **Coherence** | Measure of phase quality (0–1); low coherence → poor interferometric phase |
| **Phase unwrapping** | Converting wrapped (−π,π) phase to continuous absolute phase |
| **Geocoding** | Projecting a radar-grid image to a map coordinate grid |
| **RTC** | Radiometric Terrain Correction — normalizing backscatter for local slope |
| **DEM** | Digital Elevation Model — raster of terrain heights |
| **ECEF** | Earth-Centered Earth-Fixed Cartesian coordinate system |
| **Zero-Doppler** | Azimuth convention: each pixel timed at closest satellite approach |
| **PRF** | Pulse Repetition Frequency — radar pulse rate (determines azimuth sampling) |
| **LUT** | Lookup Table — gridded array for fast interpolation of slowly-varying quantities |
| **NISAR** | NASA-ISRO SAR mission (L- and S-band, dual polarization) |
| **RSLC** | Range-Doppler Single Look Complex (NISAR Level-1 product) |
| **GSLC** | Geocoded SLC (NISAR Level-1 product) |
| **GCOV** | Geocoded Covariance matrix (NISAR Level-2 backscatter product) |
| **RIFG** | Range-Doppler Interferogram — wrapped (NISAR Level-2) |
| **RUNW** | Range-Doppler Unwrapped interferogram (NISAR Level-2) |
| **GUNW** | Geocoded Unwrapped interferogram (NISAR Level-2) |
| **ROFF / GOFF** | Range-Doppler / Geocoded pixel offsets (NISAR Level-2) |
| **TEC** | Total Electron Content — integral ionospheric electron density |
| **SNAPHU** | Statistical-cost, Network-flow Algorithm for Phase Unwrapping |
| **ICU** | Integrated Correlation and Unwrapping (branch-cut unwrapper) |
| **OpenMP** | Open Multi-Processing — API for shared-memory parallel programming in C++ |
| **CUDA** | NVIDIA GPU parallel computing platform |
| **pybind11** | C++11-based library for creating Python bindings of C++ code |

---

*This manual was compiled from the ISCE3 GitHub repository, official documentation, and associated publications. For the latest information, always refer to the [official documentation](https://isce-framework.github.io/isce3/) and the [repository README](https://github.com/isce-framework/isce3/blob/develop/README.md).*
