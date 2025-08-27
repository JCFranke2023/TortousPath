# TortousPath: PECVD Barrier Coating Permeation Simulation

## Overview

This simulation models water vapor permeation through multilayer PECVD barrier coatings with periodic crack patterns on PET substrates. The model investigates how crack geometry affects barrier performance through transient diffusion analysis.

## Physical System

**Materials:**
- Substrate: PET (polyethylene terephthalate)
- Barrier layers: Impermeable PECVD coatings with through-thickness cracks
- Interlayers: Permeable silicon organic layers (adhesion promoter, interlayer, topcoat)

**Test Conditions:**
- Water vapor at 37°C, 100% relative humidity
- Transient permeation analysis

## Model Features

**Geometry:**
- Periodic crack patterns with controllable spacing and width
- Crack offset between layers for tortuous path modeling
- Representative unit cell with periodic boundary conditions
- Support for single-sided and double-sided coating configurations

**Parameters:**
- `c`: Crack width (nm to μm)
- `d`: Crack spacing (μm to mm)  
- `o`: Crack offset between layers (0-50% of spacing)
- Layer thicknesses: h0-h4 (barrier and interlayer thicknesses)

**Outputs:**
- Breakthrough time
- Steady-state flux
- Lag time
- Transient concentration profiles

## Model Approach

1. **Single-sided model**: Establishes substrate thickness requirements
2. **Double-sided model**: Uses reduced substrate thickness with symmetry boundary conditions
3. **Parametric studies**: Systematic variation of crack geometry parameters

## Files Structure

```
├── README.md
├── geometry_generator.py    # Creates ABAQUS geometry and mesh
├── material_properties.py   # Material parameters and diffusivities  
├── simulation_runner.py     # ABAQUS job control and parameter sweeps
├── post_processor.py        # Results extraction and analysis
└── parameter_sets/          # Configuration files for different studies
```

## Usage

```python
# Run single parameter set
python simulation_runner.py --config parameter_sets/baseline.json

# Run parameter sweep  
python simulation_runner.py --sweep parameter_sets/crack_width_study.json
```

## Dependencies

- ABAQUS CAE/Standard
- Python 3.x
- NumPy, Matplotlib (for post-processing)

## Citation

[Add publication details when available]