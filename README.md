# TortuousPath: PECVD Barrier Coating Permeation Simulation

## Overview

TortuousPath is a comprehensive finite element simulation framework for modeling water vapor permeation through multilayer PECVD (Plasma-Enhanced Chemical Vapor Deposition) barrier coatings with periodic crack defects. The simulation uses ABAQUS to solve transient mass diffusion equations and evaluate how crack geometry affects barrier performance.

## Purpose

This simulation framework addresses a critical challenge in flexible electronics and packaging: understanding how microscopic defects in barrier coatings affect macroscopic permeation rates. PECVD coatings often develop periodic crack patterns due to mechanical stress, thermal cycling, or manufacturing defects. By modeling these cracks as periodic structures with controllable geometry, we can:

- Predict water vapor transmission rates (WVTR) through damaged barriers
- Optimize coating designs for fault tolerance
- Understand the "tortuous path" effect when cracks are offset between layers
- Validate analytical permeation models with numerical simulations

## Physical System

### Materials Stack (7 layers, bottom to top)
1. **Dense PET substrate** (500 nm) - Modified PET with D = PET/23, S = PET×23
2. **PET layer** (500 nm) - Standard polyethylene terephthalate
3. **Adhesion promoter** (50 nm) - Silicon organic interlayer
4. **Barrier 1** (50 nm) - PECVD inorganic coating with through-thickness cracks
5. **Interlayer** (50 nm) - Silicon organic spacer
6. **Barrier 2** (50 nm) - PECVD inorganic coating with offset cracks
7. **Top coat** (50 nm) - Protective silicon organic layer

### Test Conditions
- Temperature: 37°C (310.15 K)
- Relative humidity: 100% RH at top surface
- Transient diffusion analysis
- Periodic boundary conditions on lateral edges

### Key Parameters
- **c**: Crack width (10-10,000 nm)
- **d**: Crack spacing (100-1,000,000 nm)
- **o**: Crack offset fraction between barrier layers (0-1)
  - o=0: Aligned cracks (worst case)
  - o=0.5: Maximum offset (best case)

## Installation

### Prerequisites
- ABAQUS CAE/Standard (2019 or later)
- Python 3.7+ with ABAQUS Python environment
- NumPy, Matplotlib (for post-processing)
- Optional: Pandas for campaign analysis

### Setup

Clone repository:

    git clone https://github.com/yourusername/tortuouspath.git
    cd tortuouspath

Create directory structure:

    python batch_runner.py --setup

Create material templates:

    python material_properties.py --create-templates

Create example campaign templates:

    python config_manager.py templates

## Directory Structure

    tortuouspath/
    ├── Core Scripts
    │   ├── batch_runner.py           # Main execution controller
    │   ├── simulation_runner.py      # ABAQUS simulation interface
    │   ├── geometry_generator.py     # 2D unit cell geometry creation
    │   └── material_properties.py    # Material property database
    │
    ├── Analysis Tools
    │   ├── odb_extractor.py         # Extract data from ABAQUS output
    │   ├── flux_postprocessor.py    # Calculate flux and permeation metrics
    │   └── manage_files.py          # File organization utility
    │
    ├── Campaign Management
    │   ├── config_manager.py        # Parameter configuration manager
    │   └── campaign_runner.py       # Multi-simulation campaign executor
    │
    ├── Output Structure
    │   ├── simulations/              # Individual simulation results
    │   │   └── {simulation_name}/
    │   │       ├── models/           # CAE and ODB files
    │   │       ├── abaqus_files/     # ABAQUS working files
    │   │       ├── extracted_data/   # Raw extraction results
    │   │       ├── analysis/         # Processed results and plots
    │   │       └── summary/          # Reports and metrics
    │   │
    │   ├── batch_runs/               # Campaign configurations
    │   │   └── {campaign_name}/
    │   │       ├── campaign_config.json
    │   │       └── summary_analysis/
    │   │
    │   └── templates/                # Reusable configurations
    │       ├── material_sets/
    │       └── campaign_templates/

## Usage

### Quick Start: Single Simulation

Run a single simulation with default parameters:

    python batch_runner.py

Specify parameters:

    python batch_runner.py --name my_test --crack_width 200 --crack_spacing 20000 --crack_offset 0.25

Run only up to a specific stage:

    python batch_runner.py --name test_geom --stage 1  # Geometry only
    python batch_runner.py --name test_sim --stage 2   # Simulate and organize
    python batch_runner.py --name test_extract --stage 3  # Extract ODB data
    python batch_runner.py --name test_full --stage 4  # Complete analysis

### Parameter Sweep Campaign

**Step 1: Create a campaign configuration**

Use a template:

    python config_manager.py campaign crack_width_study --template crack_width_sweep

Or create custom configuration file `my_sweep.json`:

    {
      "name": "offset_study",
      "type": "grid",
      "parameters": {
        "crack_width": {"default": 100},
        "crack_spacing": {"default": 10000},
        "crack_offset": {"values": [0, 0.1, 0.25, 0.5]}
      }
    }

Then run:

    python config_manager.py campaign offset_study --config my_sweep.json

**Step 2: Run the campaign**

Sequential execution (recommended for ABAQUS):

    python campaign_runner.py offset_study

Check status:

    python campaign_runner.py offset_study --status

Resume interrupted campaign:

    python campaign_runner.py offset_study --no-retry

### Material Properties

View available material sets:

    python material_properties.py --set pure_PET     # All materials same as PET
    python material_properties.py --set realistic    # Realistic barrier properties
    python material_properties.py --set high_barrier # Enhanced barrier performance

Validate properties:

    python material_properties.py --validate

Save custom material set:

    python material_properties.py --save my_materials.json

### File Management

Check simulation status:

    python manage_files.py status --simulation my_test

Find key files:

    python manage_files.py find --simulation my_test --job Job_c100_s10000_o25

Clean up workspace:

    python manage_files.py cleanup

## Analysis Outputs

### Key Metrics
- **Steady-state flux** [g/(m²·day)]: Long-time permeation rate
- **Breakthrough time** [hours]: Time to reach 10% of steady-state
- **Lag time** [hours]: Diffusion time scale from Fick's law
- **Effective diffusivity**: Calculated from lag time
- **Barrier improvement factor**: Ratio vs. uncoated substrate

### Generated Files
- `*_processed_flux.csv`: Time-dependent flux data
- `*_metrics.json`: Calculated permeation metrics
- `*_analysis.png`: Multi-panel diagnostic plots
- `campaign_summary.json`: Aggregate results for parameter sweeps

## Model Physics

### Governing Equation

The simulation solves Fick's second law in 2D:

    ∂C/∂t = ∇·(D∇C)

where:
- C: Water vapor concentration [mol/nm³]
- D: Diffusivity [nm²/s]
- t: Time [s]

### Boundary Conditions
- **Top surface**: Fixed concentration (saturated vapor)
- **Bottom surface**: Zero concentration (perfect sink)
- **Lateral edges**: Periodic boundary conditions

### Crack Modeling
Cracks are modeled as rectangular regions with high diffusivity (air properties) cutting through barrier layers. The offset between crack arrays in different barriers creates a tortuous diffusion path that reduces overall permeation.

## Material Sets

### pure_PET (Validation)
- All materials: D = 1×10⁵ nm²/s, S = 2.63×10⁻²⁸ mol/(nm³·Pa)
- Dense PET: D = 4.35×10³ nm²/s, S = 6.05×10⁻²⁷ mol/(nm³·Pa)

### realistic (Default)
- PET: D = 1.44×10¹¹ nm²/s
- Barriers: D = 1×10⁻² nm²/s (effectively impermeable)
- Air (cracks): D = 2.4×10¹³ nm²/s

## Validation

The framework includes several validation cases:

1. **Pure diffusion**: No cracks, analytical solution available
2. **Aligned cracks**: Maximum permeation, validates crack implementation
3. **Crack fraction scaling**: Permeation vs. crack area fraction
4. **Tortuous path**: Offset effect on barrier improvement

## Troubleshooting

### Common Issues

**ODB file not created:**
- Check ABAQUS license availability
- Verify material properties are positive
- Check mesh quality near cracks

**Zero flux results:**
- Verify boundary conditions in simulation_runner.py
- Check material diffusivities aren't too small
- Ensure simulation time is sufficient

**Memory errors:**
- Reduce mesh density in geometry_generator.py
- Use coarser crack discretization
- Run on high-memory node

### Debug Mode

Verbose output:

    python batch_runner.py --name debug_run --verbose

Check intermediate files:

    ls simulations/debug_run/abaqus_files/logs/

## Example Workflows

### Workflow 1: Crack Width Sensitivity Study

Create sweep configuration:

    {
      "name": "width_sensitivity",
      "type": "grid",
      "parameters": {
        "crack_width": {
          "range": {"min": 10, "max": 1000, "steps": 7},
          "scale": "log"
        },
        "crack_spacing": {"default": 10000},
        "crack_offset": {"default": 0.25}
      }
    }

Run campaign:

    python config_manager.py campaign width_study --config width_sweep.json
    python campaign_runner.py width_study

### Workflow 2: Full Factorial Design

Create 3×3×3 factorial:

    python config_manager.py campaign factorial --template full_factorial_3x3x3
    python campaign_runner.py factorial

### Workflow 3: Single Validation Run

    python batch_runner.py --name validation_pure_pet \
        --crack_width 100 --crack_spacing 10000 --crack_offset 0 \
        --stage 4

## Python API Usage

For integration into other workflows:

    from simulation_runner import SimulationRunner
    from material_properties import MaterialProperties
    
    # Set up materials
    materials = MaterialProperties(material_set='realistic')
    
    # Create and run simulation
    runner = SimulationRunner()
    parameters = {
        'crack_width': 150,
        'crack_spacing': 15000,
        'crack_offset': 0.3
    }
    job_name = runner.run_single_simulation(parameters)
    
    # Process results
    from flux_postprocessor import FluxPostProcessor
    processor = FluxPostProcessor()
    results = processor.process_nnc_data(f'extracted_data/{job_name}_flux.csv')

## Performance Considerations

### Mesh Density
- Default: Element size = min(crack_width/2, crack_spacing/20)
- Fine mesh near cracks critical for accuracy
- Typical mesh: 10,000-50,000 elements

### Computation Time
- Single simulation: 1-5 minutes
- Parameter sweep (27 points): 30-120 minutes
- Scales with mesh density and time steps

### Memory Requirements
- Typical: 2-4 GB per simulation
- Large meshes: up to 8 GB
- ODB files: 10-100 MB each

## Citation

If you use this code in your research, please cite:

    [Publication details to be added]

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please ensure:
- Code follows existing style conventions
- New features include appropriate tests
- Documentation is updated accordingly
- All simulations validate against known solutions

## Contact

For questions or support, please open an issue on GitHub or contact [your contact information].

## Acknowledgments

This work was supported by [funding source]. The authors thank [collaborators] for valuable discussions on barrier coating mechanics.
