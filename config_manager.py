#!/usr/bin/env python
"""
Configuration manager for PECVD barrier coating simulations
Handles parameter templates, validation, and campaign generation
"""

import json
import yaml
import sys
import itertools
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

class ConfigManager:
    """Manage simulation configurations and parameter sweeps"""
    
    def __init__(self, templates_dir='templates'):
        """
        Initialize configuration manager
        
        Args:
            templates_dir: Directory containing configuration templates
        """
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Create template subdirectories
        (self.templates_dir / 'material_sets').mkdir(exist_ok=True)
        (self.templates_dir / 'geometry_configs').mkdir(exist_ok=True)
        (self.templates_dir / 'analysis_settings').mkdir(exist_ok=True)
        (self.templates_dir / 'campaign_templates').mkdir(exist_ok=True)
        
        # Default parameter ranges and constraints
        self.parameter_constraints = {
            'crack_width': {
                'min': 10.0,      # nm
                'max': 10000.0,   # nm (10 μm)
                'unit': 'nm',
                'description': 'Crack width'
            },
            'crack_spacing': {
                'min': 100.0,      # nm
                'max': 1000000.0,  # nm (1 mm)
                'unit': 'nm',
                'description': 'Crack spacing'
            },
            'crack_offset': {
                'min': 0.0,
                'max': 1.0,
                'unit': 'fraction',
                'description': 'Crack offset fraction between layers'
            }
        }
    
    def create_single_config(self, simulation_name: str, parameters: Dict[str, float],
                            metadata: Optional[Dict] = None) -> Dict:
        """
        Create configuration for a single simulation
        
        Args:
            simulation_name: Name for this simulation
            parameters: Crack parameters (width, spacing, offset)
            metadata: Additional metadata
        
        Returns:
            Complete configuration dictionary
        """
        # Validate parameters
        self.validate_parameters(parameters)
        
        # Generate job name
        job_name = self.generate_job_name(parameters)
        
        config = {
            'simulation_name': simulation_name,
            'job_name': job_name,
            'timestamp': datetime.now().isoformat(),
            'parameters': parameters,
            'metadata': metadata or {},
            'derived_metrics': self._calculate_derived_metrics(parameters)
        }
        
        return config
    
    def create_parameter_sweep(self, campaign_name: str, sweep_config: Dict) -> List[Dict]:
        """
        Create configurations for a parameter sweep campaign
        
        Args:
            campaign_name: Name for this campaign
            sweep_config: Dictionary defining the sweep
        
        Returns:
            List of configurations for each simulation
        """
        sweep_type = sweep_config.get('type', 'grid')
        
        if sweep_type == 'grid':
            return self._create_grid_sweep(campaign_name, sweep_config)
        elif sweep_type == 'list':
            return self._create_list_sweep(campaign_name, sweep_config)
        elif sweep_type == 'random':
            return self._create_random_sweep(campaign_name, sweep_config)
        else:
            raise ValueError(f"Unknown sweep type: {sweep_type}")
    
    def _create_grid_sweep(self, campaign_name: str, sweep_config: Dict) -> List[Dict]:
        """Create grid-based parameter sweep"""
        parameters = sweep_config.get('parameters', {})
        
        # Create parameter value lists
        param_values = {}
        for param_name, param_config in parameters.items():
            if 'values' in param_config:
                param_values[param_name] = param_config['values']
            elif 'range' in param_config:
                # Generate values from range
                range_spec = param_config['range']
                if param_config.get('scale') == 'log':
                    import numpy as np
                    values = np.logspace(
                        np.log10(range_spec['min']),
                        np.log10(range_spec['max']),
                        range_spec.get('steps', 5)
                    ).tolist()
                else:
                    # Linear spacing
                    step = (range_spec['max'] - range_spec['min']) / (range_spec.get('steps', 5) - 1)
                    values = [range_spec['min'] + i * step 
                             for i in range(range_spec.get('steps', 5))]
                param_values[param_name] = values
            else:
                # Use default value
                param_values[param_name] = [param_config.get('default', 
                                           self.get_default_value(param_name))]
        
        # Generate all combinations
        configs = []
        param_names = list(param_values.keys())
        
        for i, combination in enumerate(itertools.product(*param_values.values())):
            params = dict(zip(param_names, combination))
            
            # Add fixed parameters
            for param_name in ['crack_width', 'crack_spacing', 'crack_offset']:
                if param_name not in params:
                    params[param_name] = self.get_default_value(param_name)
            
            sim_name = f"{campaign_name}_run_{i+1:03d}"
            config = self.create_single_config(sim_name, params, 
                                              metadata={'campaign': campaign_name,
                                                       'run_index': i+1})
            configs.append(config)
        
        return configs
    
    def _create_list_sweep(self, campaign_name: str, sweep_config: Dict) -> List[Dict]:
        """Create sweep from explicit list of parameter sets"""
        configs = []
        
        parameter_sets = sweep_config.get('parameter_sets', [])
        for i, params in enumerate(parameter_sets):
            sim_name = f"{campaign_name}_run_{i+1:03d}"
            
            # Ensure all required parameters are present
            full_params = {
                'crack_width': params.get('crack_width', self.get_default_value('crack_width')),
                'crack_spacing': params.get('crack_spacing', self.get_default_value('crack_spacing')),
                'crack_offset': params.get('crack_offset', self.get_default_value('crack_offset'))
            }
            
            config = self.create_single_config(sim_name, full_params,
                                              metadata={'campaign': campaign_name,
                                                       'run_index': i+1})
            configs.append(config)
        
        return configs
    
    def _create_random_sweep(self, campaign_name: str, sweep_config: Dict) -> List[Dict]:
        """Create random parameter sampling sweep"""
        import random
        
        n_samples = sweep_config.get('n_samples', 10)
        seed = sweep_config.get('seed')
        if seed:
            random.seed(seed)
        
        configs = []
        parameters = sweep_config.get('parameters', {})
        
        for i in range(n_samples):
            params = {}
            
            for param_name in ['crack_width', 'crack_spacing', 'crack_offset']:
                if param_name in parameters:
                    param_config = parameters[param_name]
                    if 'distribution' in param_config:
                        dist = param_config['distribution']
                        if dist == 'uniform':
                            value = random.uniform(param_config['min'], param_config['max'])
                        elif dist == 'log_uniform':
                            import numpy as np
                            log_min = np.log10(param_config['min'])
                            log_max = np.log10(param_config['max'])
                            value = 10 ** random.uniform(log_min, log_max)
                        else:
                            value = self.get_default_value(param_name)
                    else:
                        value = self.get_default_value(param_name)
                else:
                    value = self.get_default_value(param_name)
                
                params[param_name] = value
            
            sim_name = f"{campaign_name}_run_{i+1:03d}"
            config = self.create_single_config(sim_name, params,
                                              metadata={'campaign': campaign_name,
                                                       'run_index': i+1,
                                                       'sampling': 'random',
                                                       'seed': seed})
            configs.append(config)
        
        return configs
    
    def validate_parameters(self, parameters: Dict[str, float]) -> bool:
        """
        Validate simulation parameters
        
        Raises:
            ValueError: If parameters are invalid
        """
        for param_name, value in parameters.items():
            if param_name in self.parameter_constraints:
                constraints = self.parameter_constraints[param_name]
                if value < constraints['min'] or value > constraints['max']:
                    raise ValueError(
                        f"{param_name} = {value} is outside valid range "
                        f"[{constraints['min']}, {constraints['max']}] {constraints['unit']}"
                    )
        
        # Check physical constraints
        if parameters.get('crack_width', 0) >= parameters.get('crack_spacing', 1):
            raise ValueError("Crack width must be less than crack spacing")
        
        return True
    
    def generate_job_name(self, parameters: Dict[str, float]) -> str:
        """Generate standard job name from parameters"""
        c = parameters.get('crack_width', 100)
        s = parameters.get('crack_spacing', 10000)
        o = parameters.get('crack_offset', 0.25)
        
        return f"Job_c{c:.0f}_s{s:.0f}_o{int(o*100)}"
    
    def get_default_value(self, param_name: str) -> float:
        """Get default value for a parameter"""
        defaults = {
            'crack_width': 100.0,      # nm
            'crack_spacing': 10000.0,  # nm
            'crack_offset': 0.25       # fraction
        }
        return defaults.get(param_name, 0.0)
    
    def _calculate_derived_metrics(self, parameters: Dict[str, float]) -> Dict:
        """Calculate derived metrics from parameters"""
        c = parameters.get('crack_width', 100)
        s = parameters.get('crack_spacing', 10000)
        o = parameters.get('crack_offset', 0.25)
        
        return {
            'crack_fraction': c / s,
            'crack_density': 1000000 / s,  # cracks per mm
            'effective_offset_nm': o * s,
            'tortuosity_factor': 1 + 2 * o  # Simplified estimate
        }
    
    def save_config(self, config: Dict, filepath: Path):
        """Save configuration to JSON file"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_config(self, filepath: Path) -> Dict:
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def save_campaign(self, campaign_name: str, configs: List[Dict], 
                     output_dir: str = 'batch_runs'):
        """
        Save campaign configuration and individual simulation configs
        
        Args:
            campaign_name: Name of the campaign
            configs: List of simulation configurations
            output_dir: Output directory
        """
        campaign_dir = Path(output_dir) / campaign_name
        campaign_dir.mkdir(parents=True, exist_ok=True)
        
        # Save campaign summary
        campaign_summary = {
            'campaign_name': campaign_name,
            'timestamp': datetime.now().isoformat(),
            'n_simulations': len(configs),
            'simulations': []
        }
        
        # Save individual configs and build simulation list
        for config in configs:
            sim_name = config['simulation_name']
            
            # Save individual config
            config_file = campaign_dir / f"{sim_name}_config.json"
            self.save_config(config, config_file)
            
            # Add to campaign summary
            campaign_summary['simulations'].append({
                'simulation_name': sim_name,
                'job_name': config['job_name'],
                'parameters': config['parameters'],
                'config_file': str(config_file.relative_to(campaign_dir))
            })
        
        # Save campaign summary
        summary_file = campaign_dir / 'campaign_config.json'
        with open(summary_file, 'w') as f:
            json.dump(campaign_summary, f, indent=2)
        
        # Create simulation list for batch processing
        list_file = campaign_dir / 'simulation_list.txt'
        with open(list_file, 'w') as f:
            f.write(f"# Campaign: {campaign_name}\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write(f"# Total simulations: {len(configs)}\n\n")
            
            for config in configs:
                f.write(f"{config['simulation_name']},{config['job_name']}\n")
        
        print(f"Campaign saved to: {campaign_dir}")
        print(f"  Configurations: {len(configs)} files")
        print(f"  Campaign summary: {summary_file}")
        print(f"  Simulation list: {list_file}")
        
        return campaign_dir
    
    def load_template(self, template_name: str, template_type: str = 'campaign_templates') -> Dict:
        """Load a configuration template"""
        template_file = self.templates_dir / template_type / f"{template_name}.json"
        
        if not template_file.exists():
            # Try YAML
            template_file = self.templates_dir / template_type / f"{template_name}.yaml"
            if template_file.exists():
                with open(template_file, 'r') as f:
                    return yaml.safe_load(f)
        
        if template_file.exists():
            with open(template_file, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"Template not found: {template_name}")
    
    def save_template(self, template_data: Dict, template_name: str, 
                     template_type: str = 'campaign_templates'):
        """Save a configuration template"""
        template_file = self.templates_dir / template_type / f"{template_name}.json"
        template_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(template_file, 'w') as f:
            json.dump(template_data, f, indent=2)
        
        print(f"Template saved: {template_file}")


def create_example_templates():
    """Create example configuration templates"""
    manager = ConfigManager()
    
    # Example 1: Crack width sweep
    width_sweep = {
        'name': 'crack_width_sweep',
        'description': 'Sweep crack width with fixed spacing and offset',
        'type': 'grid',
        'parameters': {
            'crack_width': {
                'range': {'min': 10, 'max': 1000, 'steps': 7},
                'scale': 'log'
            },
            'crack_spacing': {'default': 10000},
            'crack_offset': {'default': 0.25}
        }
    }
    manager.save_template(width_sweep, 'crack_width_sweep')
    
    # Example 2: Full factorial
    factorial = {
        'name': 'full_factorial',
        'description': 'Full factorial design with 3 levels per parameter',
        'type': 'grid',
        'parameters': {
            'crack_width': {'values': [50, 200, 500]},
            'crack_spacing': {'values': [5000, 20000, 50000]},
            'crack_offset': {'values': [0, 0.25, 0.5]}
        }
    }
    manager.save_template(factorial, 'full_factorial_3x3x3')
    
    # Example 3: Random sampling
    random_sweep = {
        'name': 'random_exploration',
        'description': 'Random parameter exploration',
        'type': 'random',
        'n_samples': 20,
        'seed': 42,
        'parameters': {
            'crack_width': {
                'distribution': 'log_uniform',
                'min': 10,
                'max': 1000
            },
            'crack_spacing': {
                'distribution': 'log_uniform',
                'min': 1000,
                'max': 100000
            },
            'crack_offset': {
                'distribution': 'uniform',
                'min': 0,
                'max': 0.5
            }
        }
    }
    manager.save_template(random_sweep, 'random_exploration')
    
    print("Example templates created in templates/campaign_templates/")


def main():
    """Command line interface for config manager"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Manage simulation configurations')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Create single config
    single_parser = subparsers.add_parser('single', help='Create single configuration')
    single_parser.add_argument('name', help='Simulation name')
    single_parser.add_argument('--crack_width', type=float, default=100)
    single_parser.add_argument('--crack_spacing', type=float, default=10000)
    single_parser.add_argument('--crack_offset', type=float, default=0.25)
    single_parser.add_argument('--output', help='Output file')
    
    # Create campaign
    campaign_parser = subparsers.add_parser('campaign', help='Create parameter sweep campaign')
    campaign_parser.add_argument('name', help='Campaign name')
    campaign_parser.add_argument('--template', help='Template name to use')
    campaign_parser.add_argument('--config', help='Config file for sweep')
    
    # Create templates
    template_parser = subparsers.add_parser('templates', help='Create example templates')
    
    args = parser.parse_args()
    
    manager = ConfigManager()
    
    if args.command == 'single':
        params = {
            'crack_width': args.crack_width,
            'crack_spacing': args.crack_spacing,
            'crack_offset': args.crack_offset
        }
        
        config = manager.create_single_config(args.name, params)
        
        if args.output:
            manager.save_config(config, Path(args.output))
        else:
            print(json.dumps(config, indent=2))
    
    elif args.command == 'campaign':
        if args.template:
            sweep_config = manager.load_template(args.template)
        elif args.config:
            with open(args.config, 'r') as f:
                sweep_config = json.load(f)
        else:
            print("ERROR: Must specify --template or --config")
            return 1
        
        configs = manager.create_parameter_sweep(args.name, sweep_config)
        manager.save_campaign(args.name, configs)
        
        print(f"\nCreated {len(configs)} configurations")
    
    elif args.command == 'templates':
        create_example_templates()
    
    else:
        parser.print_help()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
