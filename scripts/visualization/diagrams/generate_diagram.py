#!/usr/bin/env python3
"""
Autoencoder Architecture Diagram Generator

Generates publication-ready SVG diagrams from the actual PyTorch autoencoder implementation.
Automatically captures layer structure, shapes, and parameter counts to create accurate
architectural visualizations.

Usage:
    python generate_diagram.py --output-dir out
    python generate_diagram.py --output-dir out --latent-dim 64
    python generate_diagram.py --output-dir out --format mermaid
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import traceback

try:
    import torch
    import torchinfo
    import graphviz
    from colorama import init, Fore, Style
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Install with: pip install torch torchinfo graphviz colorama")
    sys.exit(1)

# Initialize colorama for cross-platform colored output
init(autoreset=True)

# Add the project root to path to import models
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from models.autoencoder import Autoencoder
except ImportError as e:
    print(f"{Fore.RED}Error: Could not import Autoencoder model.")
    print(f"Make sure you're running from the project root and models/ is accessible.")
    print(f"Import error: {e}{Style.RESET_ALL}")
    sys.exit(1)


class DiagramGenerator:
    """Generates architecture diagrams from PyTorch models."""
    
    def __init__(self, output_dir: Path, latent_dim: int = 32):
        self.output_dir = Path(output_dir)
        self.latent_dim = latent_dim
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Color scheme for different sections
        self.colors = {
            'encoder': '#87CEEB',  # light-sky-blue
            'latent': '#FFD700',   # gold
            'decoder': '#90EE90'   # light-green
        }
        
    def instantiate_model(self) -> torch.nn.Module:
        """Instantiate the autoencoder model with specified parameters."""
        print(f"{Fore.CYAN}Instantiating autoencoder model (latent_dim={self.latent_dim})...")
        
        try:
            model = Autoencoder(latent_dim=self.latent_dim, out_shape=(256, 256))
            model.eval()  # Set to evaluation mode
            print(f"{Fore.GREEN}✓ Model instantiated successfully")
            return model
        except Exception as e:
            print(f"{Fore.RED}Error instantiating model: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    def generate_summary(self, model: torch.nn.Module) -> torchinfo.ModelStatistics:
        """Generate detailed model summary using torchinfo."""
        print(f"{Fore.CYAN}Generating model summary...")
        
        try:
            # Create dummy input for 256x256 single-channel image
            input_size = (1, 1, 256, 256)  # (batch_size, channels, height, width)
            
            summary = torchinfo.summary(
                model,
                input_size=input_size,
                verbose=0,
                col_names=["input_size", "output_size", "num_params", "params_percent"],
                row_settings=["var_names"]
            )
            
            print(f"{Fore.GREEN}✓ Model summary generated")
            return summary
        except Exception as e:
            print(f"{Fore.RED}Error generating model summary: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    def save_summary(self, summary: torchinfo.ModelStatistics) -> None:
        """Save the torchinfo summary to text file."""
        summary_path = self.output_dir / "autoencoder_summary.txt"
        
        try:
            with open(summary_path, 'w') as f:
                f.write(str(summary))
            print(f"{Fore.GREEN}✓ Summary saved to {summary_path}")
        except Exception as e:
            print(f"{Fore.RED}Error saving summary: {e}")
    
    def classify_layer(self, layer_name: str) -> str:
        """Classify layer into encoder, latent, or decoder section."""
        layer_lower = layer_name.lower()
        
        if any(term in layer_lower for term in ['encoder', 'conv_input', 'conv_pre', 'resnet', 'conv_post', 'conv_final', 'adaptive_pool']):
            return 'encoder'
        elif any(term in layer_lower for term in ['embedding', 'linear']):
            return 'latent'
        elif any(term in layer_lower for term in ['decoder', 'resnet_up', 'upsample']):
            return 'decoder'
        else:
            # Default classification based on position
            return 'encoder'  # Most layers are encoder layers
    
    def parse_summary_layers(self, summary: torchinfo.ModelStatistics) -> List[Dict]:
        """Parse torchinfo summary into layer information."""
        print(f"{Fore.CYAN}Parsing layer information...")
        
        layers = []
        
        try:
            # Extract layer information from summary
            for layer_info in summary.summary_list:
                if hasattr(layer_info, 'class_name') and hasattr(layer_info, 'output_size'):
                    # Use class_name and var_name for meaningful layer names
                    class_name = getattr(layer_info, 'class_name', 'Unknown')
                    var_name = getattr(layer_info, 'var_name', '')
                    
                    # Create meaningful layer name
                    if var_name and var_name != class_name:
                        layer_name = f"{class_name} ({var_name})"
                    else:
                        layer_name = class_name
                    
                    output_size = layer_info.output_size
                    num_params = getattr(layer_info, 'num_params', 0)
                    
                    # Skip activation layers with no parameters for cleaner diagram
                    if num_params == 0 and class_name in ['ReLU', 'Sigmoid', 'Tanh']:
                        continue
                    
                    # Format output size
                    if isinstance(output_size, (list, tuple)) and len(output_size) > 0:
                        if isinstance(output_size[0], (list, tuple)):
                            size_str = f"{output_size[0]}"
                        else:
                            size_str = f"{output_size}"
                    else:
                        size_str = str(output_size)
                    
                    layers.append({
                        'name': layer_name,
                        'output_size': size_str,
                        'num_params': num_params,
                        'section': self.classify_layer(layer_name)
                    })
            
            print(f"{Fore.GREEN}✓ Parsed {len(layers)} layers")
            return layers
            
        except Exception as e:
            print(f"{Fore.RED}Error parsing summary: {e}")
            # Fallback: create a simplified layer structure
            return self.create_fallback_layers()
    
    def create_fallback_layers(self) -> List[Dict]:
        """Create a fallback layer structure when parsing fails."""
        print(f"{Fore.YELLOW}Creating fallback layer structure...")
        
        return [
            {'name': 'Input', 'output_size': '[1, 1, 256, 256]', 'num_params': 0, 'section': 'encoder'},
            {'name': 'Conv Input (1→64)', 'output_size': '[1, 64, 256, 256]', 'num_params': 640, 'section': 'encoder'},
            {'name': 'Conv Pre (64→128)', 'output_size': '[1, 128, 256, 256]', 'num_params': 73856, 'section': 'encoder'},
            {'name': 'ResNet Block 1', 'output_size': '[1, 128, 64, 64]', 'num_params': 443520, 'section': 'encoder'},
            {'name': 'ResNet Block 2', 'output_size': '[1, 128, 16, 16]', 'num_params': 443520, 'section': 'encoder'},
            {'name': 'ResNet Block 3', 'output_size': '[1, 128, 4, 4]', 'num_params': 443520, 'section': 'encoder'},
            {'name': 'Conv Post (128→64)', 'output_size': '[1, 64, 4, 4]', 'num_params': 73792, 'section': 'encoder'},
            {'name': 'Conv Final (64→1)', 'output_size': '[1, 1, 4, 4]', 'num_params': 577, 'section': 'encoder'},
            {'name': 'Adaptive Pool', 'output_size': '[1, 1, 4, 4]', 'num_params': 0, 'section': 'encoder'},
            {'name': f'Embedding (16→{self.latent_dim})', 'output_size': f'[1, {self.latent_dim}]', 'num_params': 16 * self.latent_dim + self.latent_dim, 'section': 'latent'},
            {'name': f'Linear ({self.latent_dim}→2048)', 'output_size': '[1, 2048]', 'num_params': self.latent_dim * 2048 + 2048, 'section': 'decoder'},
            {'name': 'Reshape (128×4×4)', 'output_size': '[1, 128, 4, 4]', 'num_params': 0, 'section': 'decoder'},
            {'name': 'ResNet Up Block 1', 'output_size': '[1, 128, 16, 16]', 'num_params': 443520, 'section': 'decoder'},
            {'name': 'ResNet Up Block 2', 'output_size': '[1, 128, 64, 64]', 'num_params': 443520, 'section': 'decoder'},
            {'name': 'ResNet Up Block 3', 'output_size': '[1, 128, 256, 256]', 'num_params': 443520, 'section': 'decoder'},
            {'name': 'Final Conv (128→1)', 'output_size': '[1, 1, 256, 256]', 'num_params': 1153, 'section': 'decoder'},
        ]
    
    def generate_dot_graph(self, layers: List[Dict]) -> str:
        """Generate Graphviz DOT representation of the architecture."""
        print(f"{Fore.CYAN}Generating DOT graph...")
        
        dot_lines = [
            'digraph AutoencoderArchitecture {',
            '    rankdir=LR;',
            '    nodesep=0.25;',
            '    ranksep=0.5;',
            '    node [shape=box, style=filled, fontname="Arial", fontsize=10];',
            '    edge [fontname="Arial", fontsize=8];',
            ''
        ]
        
        # Add legend
        dot_lines.extend([
            '    subgraph cluster_legend {',
            '        label="Legend";',
            '        style=dashed;',
            '        fontname="Arial";',
            '        fontsize=12;',
            f'        legend_encoder [label="Encoder", fillcolor="{self.colors["encoder"]}", style=filled];',
            f'        legend_latent [label="Latent", fillcolor="{self.colors["latent"]}", style=filled];',
            f'        legend_decoder [label="Decoder", fillcolor="{self.colors["decoder"]}", style=filled];',
            '        legend_encoder -> legend_latent -> legend_decoder [style=invis];',
            '    }',
            ''
        ])
        
        # Add nodes
        for i, layer in enumerate(layers):
            node_id = f"layer_{i}"
            color = self.colors[layer['section']]
            
            # Format parameter count
            if layer['num_params'] > 1000000:
                param_str = f"{layer['num_params']/1000000:.1f}M"
            elif layer['num_params'] > 1000:
                param_str = f"{layer['num_params']/1000:.1f}K"
            else:
                param_str = str(layer['num_params'])
            
            label = f"{layer['name']}\\nShape: {layer['output_size']}\\nParams: {param_str}"
            
            dot_lines.append(f'    {node_id} [label="{label}", fillcolor="{color}"];')
        
        dot_lines.append('')
        
        # Add edges (connections between layers)
        for i in range(len(layers) - 1):
            dot_lines.append(f'    layer_{i} -> layer_{i+1};')
        
        dot_lines.append('}')
        
        dot_content = '\n'.join(dot_lines)
        print(f"{Fore.GREEN}✓ DOT graph generated")
        return dot_content
    
    def save_dot_file(self, dot_content: str) -> Path:
        """Save DOT content to file."""
        dot_path = self.output_dir / "autoencoder_arch.dot"
        
        try:
            with open(dot_path, 'w') as f:
                f.write(dot_content)
            print(f"{Fore.GREEN}✓ DOT file saved to {dot_path}")
            return dot_path
        except Exception as e:
            print(f"{Fore.RED}Error saving DOT file: {e}")
            sys.exit(1)
    
    def render_svg(self, dot_path: Path) -> Path:
        """Render DOT file to SVG."""
        print(f"{Fore.CYAN}Rendering SVG...")
        
        svg_path = self.output_dir / "autoencoder_arch.svg"
        
        try:
            # Use graphviz to render DOT to SVG
            with open(dot_path, 'r') as f:
                dot_content = f.read()
            
            graph = graphviz.Source(dot_content)
            graph.render(str(svg_path.with_suffix('')), format='svg', cleanup=True)
            
            print(f"{Fore.GREEN}✓ SVG rendered to {svg_path}")
            return svg_path
        except Exception as e:
            print(f"{Fore.RED}Error rendering SVG: {e}")
            print("Make sure Graphviz is installed on your system:")
            print("  - Ubuntu/Debian: sudo apt install graphviz")
            print("  - macOS: brew install graphviz")
            print("  - Windows: choco install graphviz")
            sys.exit(1)
    
    def generate_mermaid(self, layers: List[Dict]) -> str:
        """Generate Mermaid diagram representation."""
        print(f"{Fore.CYAN}Generating Mermaid diagram...")
        
        mermaid_lines = [
            'flowchart LR',
            '    %% Autoencoder Architecture',
            ''
        ]
        
        # Add nodes with styling
        for i, layer in enumerate(layers):
            node_id = f"L{i}"
            
            # Format parameter count
            if layer['num_params'] > 1000000:
                param_str = f"{layer['num_params']/1000000:.1f}M"
            elif layer['num_params'] > 1000:
                param_str = f"{layer['num_params']/1000:.1f}K"
            else:
                param_str = str(layer['num_params'])
            
            label = f"{layer['name']}<br/>Shape: {layer['output_size']}<br/>Params: {param_str}"
            mermaid_lines.append(f'    {node_id}["{label}"]')
        
        mermaid_lines.append('')
        
        # Add connections
        for i in range(len(layers) - 1):
            mermaid_lines.append(f'    L{i} --> L{i+1}')
        
        mermaid_lines.extend([
            '',
            '    %% Styling',
            '    classDef encoder fill:#87CEEB',
            '    classDef latent fill:#FFD700',
            '    classDef decoder fill:#90EE90',
            ''
        ])
        
        # Apply styling
        encoder_nodes = [f"L{i}" for i, layer in enumerate(layers) if layer['section'] == 'encoder']
        latent_nodes = [f"L{i}" for i, layer in enumerate(layers) if layer['section'] == 'latent']
        decoder_nodes = [f"L{i}" for i, layer in enumerate(layers) if layer['section'] == 'decoder']
        
        if encoder_nodes:
            mermaid_lines.append(f'    class {",".join(encoder_nodes)} encoder')
        if latent_nodes:
            mermaid_lines.append(f'    class {",".join(latent_nodes)} latent')
        if decoder_nodes:
            mermaid_lines.append(f'    class {",".join(decoder_nodes)} decoder')
        
        mermaid_content = '\n'.join(mermaid_lines)
        print(f"{Fore.GREEN}✓ Mermaid diagram generated")
        return mermaid_content
    
    def save_mermaid_file(self, mermaid_content: str) -> Path:
        """Save Mermaid content to file."""
        mermaid_path = self.output_dir / "autoencoder_arch.mmd"
        
        try:
            with open(mermaid_path, 'w') as f:
                f.write(mermaid_content)
            print(f"{Fore.GREEN}✓ Mermaid file saved to {mermaid_path}")
            return mermaid_path
        except Exception as e:
            print(f"{Fore.RED}Error saving Mermaid file: {e}")
            sys.exit(1)
    
    def generate_all(self, format_type: str = 'svg') -> None:
        """Generate all diagram outputs."""
        print(f"{Fore.MAGENTA}=== Autoencoder Architecture Diagram Generator ==={Style.RESET_ALL}")
        print(f"Output directory: {self.output_dir}")
        print(f"Latent dimension: {self.latent_dim}")
        print(f"Format: {format_type}")
        print()
        
        # Step 1: Instantiate model
        model = self.instantiate_model()
        
        # Step 2: Generate summary
        summary = self.generate_summary(model)
        
        # Step 3: Save summary
        self.save_summary(summary)
        
        # Step 4: Parse layers
        layers = self.parse_summary_layers(summary)
        
        # Step 5: Generate diagrams based on format
        if format_type in ['svg', 'all']:
            # Generate and save DOT/SVG
            dot_content = self.generate_dot_graph(layers)
            dot_path = self.save_dot_file(dot_content)
            svg_path = self.render_svg(dot_path)
        
        if format_type in ['mermaid', 'all']:
            # Generate and save Mermaid
            mermaid_content = self.generate_mermaid(layers)
            mermaid_path = self.save_mermaid_file(mermaid_content)
        
        print()
        print(f"{Fore.GREEN}✓ All deliverables generated successfully!")
        print(f"Check {self.output_dir} for output files.")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Generate autoencoder architecture diagrams",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_diagram.py --output-dir out
  python generate_diagram.py --output-dir out --latent-dim 64
  python generate_diagram.py --output-dir out --format mermaid
  python generate_diagram.py --output-dir out --format all
        """
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="out",
        help="Output directory for generated files (default: out)"
    )
    
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=32,
        help="Latent dimension for the autoencoder (default: 32)"
    )
    
    parser.add_argument(
        "--format",
        type=str,
        choices=['svg', 'mermaid', 'all'],
        default='svg',
        help="Output format: svg, mermaid, or all (default: svg)"
    )
    
    args = parser.parse_args()
    
    try:
        generator = DiagramGenerator(
            output_dir=args.output_dir,
            latent_dim=args.latent_dim
        )
        generator.generate_all(format_type=args.format)
        
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Generation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"{Fore.RED}Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()