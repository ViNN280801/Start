# Start

## Description

**Start** is a sophisticated software prototype designed for optimizing technological processes in vacuum-plasma installations. It focuses on the precise modeling of particle trajectories, gas discharge simulations, and plasma chemical reactions within these installations. The software aims to provide accurate parameters for vacuum-plasma processes, even for non-standard geometries, thereby improving the efficiency and quality of thin-film deposition on substrates.

## Features

- **Particle Trajectory Calculation:** Simulates the movement of particles within the vacuum chamber using advanced algorithms.
- **Scattering Models:** Includes models for hard spheres (HS), variable hard spheres (VHS), and variable soft spheres (VSS).
- **Deposition Process Simulation:** Models the deposition of particles on substrates, taking into account various material interactions.
- **Optimization:** Optimizes the technological parameters for the deposition process to achieve the best possible outcomes.
- **User Interface:** A user-friendly interface developed in Python for ease of use and visualization.

## Dependencies

- **Programming Languages:** C++, Python
- **C++ Libraries and Tools:**
  - [GCC 13](https://gcc.gnu.org/gcc-13/) (GNU Compiler Collection)
  - [HDF5](https://github.com/HDFGroup/hdf5)
  - [CGAL](https://www.cgal.org/)
  - [Gmsh](https://gmsh.info/)
  - [Boost](https://www.boost.org/)
  - [GMP](https://gmplib.org/)
  - TBB
  - [json](https://github.com/nlohmann/json)
  - [OpenMP](https://www.openmp.org/) (optional)
  - [MPI](https://docs.open-mpi.org/en/v5.0.x/installing-open-mpi/quickstart.html) (optional)
  - [CUDA](https://developer.nvidia.com/cuda-toolkit) (optional)
  - [Trilinos](https://trilinos.github.io/)


- **Python Libraries:**
  - [Python 3.7](https://www.python.org/downloads/)
  - [PyQt5](https://pypi.org/project/PyQt5/)
  - [VTK](https://vtk.org/download/)
  - [json](https://pypi.org/project/nlohmann-json/)

The program uses C++20 features, so ensure your compiler supports this standard.

### HDF5 installation example

To install the necessary dependencies and run the software on Debian-based Linux distributions, follow these steps:

- Install Python 3.7
- Install pip dependency

```bash
pip install h5py
```

- Install development package

```bash
sudo apt-get update
sudo apt-get install -y libhdf5-dev
```

- Add the following environment variables to your **~/.bashrc** to ensure the HDF5 libraries are correctly linked:

```bash
export HDF5_LIBRARY_DIRS=/usr/lib/x86_64-linux-gnu/hdf5/serial
export HDF5_INCLUDE_DIRS=/usr/include/hdf5/serial
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/hdf5/openmpi:/usr/lib/x86_64-linux-gnu/hdf5/serial:$LD_LIBRARY_PATH
```

## Technical Specifications

- **Grid Size:** 1 to 15 mm
- **Calculation Time:**
  - Local PC: 1 to 5 hours
  - Computational Cluster: 1 hour to 10 days
- **Optimization Time:** 1 to 24 hours

## Installation

To install the necessary dependencies and run the software, follow these steps:

1. Install Python 3.7 or later.
2. Install required Python libraries:

```bash
pip install PyQt5 h5py gmsh vtk
```

or just install them from **requirements.txt**:

```bash
pip install -r requirements.txt
```

## Usage

### Example: C++ Code Snippet

```cpp
#include "Particle.h"
#include "RealNumberGenerator.h"
#include "VolumeCreator.h"

int main() {
    // Initialize particle generator
    RealNumberGenerator rng;
    Particle particle = rng.generateParticle();

    // Create volume
    VolumeCreator volume;
    volume.createMesh(particle);

    // Update particle position
    particle.updatePosition(0.01);

    return 0;
}
```

This snippet demonstrates how to initialize a particle generator, create a mesh volume, and update the particle's position using the **updatePosition** function.

### Example: Python Code Snippet

```python
from PyQt5.QtGui import QDoubleValidator

class CustomSignedDoubleValidator(QDoubleValidator):
    def __init__(self, bottom, top, decimals, parent=None):
        super().__init__(bottom, top, decimals, parent)
        self.decimals = decimals

    def validate(self, input_str, pos):
        # Replace comma with dot for uniform interpretation
        input_str = input_str.replace(',', '.')

        # Allow empty input
        if not input_str:
            return self.Intermediate, input_str, pos

        # Allow '-' if it's the first character
        if input_str == '-':
            return self.Intermediate, input_str, pos

        # Check if the input string is a valid number
        try:
            value = float(input_str)

            # Allow zero value regardless of format
            if value == 0:
                return self.Acceptable, input_str, pos

            # Check if the value is within the valid range
            if self.bottom() <= value <= self.top():
                parts = input_str.split('.')
                if len(parts) == 2 and len(parts[1]) <= self.decimals:
                    return self.Acceptable, input_str, pos
                elif len(parts) == 1:
                    return self.Acceptable, input_str, pos

            return self.Invalid, input_str, pos

        except ValueError:
            return self.Invalid, input_str, pos
```

This snippet shows how to create a custom validator for a signed double input in PyQt5.

### Parameter `FEM Accuracy`

FEM accuracy is the parameter that manages count of cubature points for the more accurate integration. The following table shows relationship beetween parameters `FEM accuracy` and `Count of cubature points`.

| FEM accuracy | Count of cubature points |
| :----------: | :----------------------: |
|      1       |            1             |
|      2       |            4             |
|      3       |            5             |
|      4       |            11            |
|      5       |            14            |
|      6       |            24            |
|      7       |            31            |
|      8       |            43            |
|      9       |           126            |
|      10      |           126            |
|      11      |           126            |
|      12      |           210            |
|      13      |           210            |
|      14      |           330            |
|      15      |           330            |
|      16      |           495            |
|      17      |           495            |
|      18      |           715            |
|      19      |           715            |
|      20      |           1001           |

### Variables to set within `.bashrc` to work with OpenMP and CUDA

#### OpenMP Environment Variables

`OMP_PROC_BIND=spread`: This setting controls how OpenMP threads are distributed across the cores. By setting it to spread, OpenMP threads are distributed across as many processors as possible, which can help maximize memory bandwidth usage and avoid oversubscribing a single core. This setting is optimal for multi-threaded parallelism.

`OMP_PLACES=threads`: This variable specifies the places on which OpenMP threads can execute. Setting it to threads means that each OpenMP thread is bound to a separate processing unit (core), which further enhances the parallel efficiency of the program by distributing the workload across available CPU cores.

`export OMP_PROC_BIND=spread`

`export OMP_PLACES=threads`

or

`vim ~/.bashrc`
and write down the vars within, then write `:wq`.
