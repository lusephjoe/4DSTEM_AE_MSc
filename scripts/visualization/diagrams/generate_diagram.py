#!/usr/bin/env python3
"""
Publication-grade Autoencoder Architecture Diagram Generator

Generates publication-ready SVG diagrams from PyTorch autoencoder models at the block level.
Creates a single composite figure with three panels: (a) overview, (b) ResNet down, (c) ResNet up.

Usage:
    python generate_diagram.py --model lightning_model:LitAE
    python generate_diagram.py --from-autoencoder --split
    python generate_diagram.py --figure overview --png
"""

import argparse
import importlib
import json
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import re

try:
    import torch
    import torch.nn as nn
    import yaml
    import svgwrite
    import cairosvg
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Install with: pip install torch pyyaml svgwrite cairosvg")
    sys.exit(1)

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import SVG renderer
try:
    from .svg_renderer import compose_publication_svg
except ImportError:
    # Fallback for module import
    import sys
    from pathlib import Path
    svg_renderer_path = Path(__file__).parent / "svg_renderer.py"
    if svg_renderer_path.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("svg_renderer", svg_renderer_path)
        svg_renderer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(svg_renderer)
        compose_publication_svg = svg_renderer.compose_publication_svg
    else:
        def compose_publication_svg(*args, **kwargs):
            raise ImportError("SVG renderer not available")


class Block:
    """Represents a semantic block in the autoencoder architecture."""
    
    def __init__(self, id: str, group: str, type: str, name: str, 
                 in_shape: Tuple, out_shape: Tuple, modules: List[str] = None):
        self.id = id
        self.group = group  # "encoder", "decoder", "misc"
        self.type = type    # "stem", "resnet_down", "resnet_up", "bottleneck", etc.
        self.name = name
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.modules = modules or []


class ShapeTracker:
    """Tracks tensor shapes through model forward pass using hooks."""
    
    def __init__(self):
        self.shapes = {}
        self.hooks = []
        self.module_names = {}
    
    def register_hooks(self, model: nn.Module):
        """Register forward hooks on all modules to capture shapes."""
        for name, module in model.named_modules():
            self.module_names[id(module)] = name
            hook = module.register_forward_hook(self._hook_fn)
            self.hooks.append(hook)
    
    def _hook_fn(self, module, input, output):
        """Hook function to capture output shapes."""
        module_name = self.module_names.get(id(module), f"module_{id(module)}")
        if isinstance(output, torch.Tensor):
            self.shapes[module_name] = tuple(output.shape)
        elif isinstance(output, (list, tuple)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
            self.shapes[module_name] = tuple(output[0].shape)
    
    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class BlockDetector:
    """Detects and groups model modules into semantic blocks."""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
    
    def _default_config(self) -> Dict:
        """Default block detection configuration."""
        return {
            'block_patterns': {
                'stem': [r'conv_input', r'bn_input', r'relu_input', r'conv_pre', r'bn_pre', r'relu_pre'],
                'resnet_down': [r'resnet\d+', r'ResNetBlock'],
                'bottleneck': [r'adaptive_pool', r'embedding'],
                'dense_reshape': [r'decoder\.linear', r'decoder\.conv_initial'],
                'resnet_up': [r'decoder\.resnet_up\d+', r'ResNetUpBlock'],
                'output_head': [r'decoder\.final_conv']
            },
            'ignore_patterns': [r'relu', r'bn', r'dropout', r'activation']
        }
    
    def detect_blocks(self, model: nn.Module, shapes: Dict[str, Tuple]) -> List[Block]:
        """Detect semantic blocks from model structure and shapes."""
        blocks = []
        
        # Get model structure
        encoder_modules = self._get_encoder_modules(model)
        decoder_modules = self._get_decoder_modules(model)
        
        # Create input block
        input_shape = self._get_input_shape(shapes)
        blocks.append(Block("input", "misc", "input", "Input", input_shape, input_shape))
        
        # Process encoder blocks
        encoder_blocks = self._process_encoder(encoder_modules, shapes)
        blocks.extend(encoder_blocks)
        
        # Process decoder blocks  
        decoder_blocks = self._process_decoder(decoder_modules, shapes)
        blocks.extend(decoder_blocks)
        
        # Create output block
        output_shape = self._get_output_shape(shapes)
        blocks.append(Block("output", "misc", "output", "Output", output_shape, output_shape))
        
        return blocks
    
    def _get_encoder_modules(self, model: nn.Module) -> Dict[str, nn.Module]:
        """Extract encoder module hierarchy."""
        encoder_modules = {}
        if hasattr(model, 'encoder'):
            for name, module in model.encoder.named_modules():
                if name:  # Skip the encoder module itself
                    encoder_modules[f"encoder.{name}"] = module
        elif hasattr(model, 'model') and hasattr(model.model, 'encoder'):
            for name, module in model.model.encoder.named_modules():
                if name:
                    encoder_modules[f"encoder.{name}"] = module
        return encoder_modules
    
    def _get_decoder_modules(self, model: nn.Module) -> Dict[str, nn.Module]:
        """Extract decoder module hierarchy."""
        decoder_modules = {}
        if hasattr(model, 'decoder'):
            for name, module in model.decoder.named_modules():
                if name:
                    decoder_modules[f"decoder.{name}"] = module
        elif hasattr(model, 'model') and hasattr(model.model, 'decoder'):
            for name, module in model.model.decoder.named_modules():
                if name:
                    decoder_modules[f"decoder.{name}"] = module
        return decoder_modules
    
    def _process_encoder(self, modules: Dict[str, nn.Module], shapes: Dict[str, Tuple]) -> List[Block]:
        """Process encoder modules into blocks."""
        blocks = []
        
        # Group stem layers
        stem_shapes = self._get_shapes_for_pattern(shapes, [r'encoder\.conv_input', r'encoder\.conv_pre'])
        if stem_shapes:
            in_shape = stem_shapes[0]
            out_shape = stem_shapes[-1]
            blocks.append(Block("stem", "encoder", "stem", "Encoder Stem (Conv×2)", in_shape, out_shape))
        
        # Detect ResNet blocks
        resnet_patterns = [r'encoder\.resnet1', r'encoder\.resnet2', r'encoder\.resnet3']
        for i, pattern in enumerate(resnet_patterns, 1):
            resnet_shapes = self._get_shapes_for_pattern(shapes, [pattern])
            if resnet_shapes:
                in_shape = resnet_shapes[0] if len(resnet_shapes) > 1 else stem_shapes[-1] if stem_shapes else (1, 128, 256, 256)
                out_shape = resnet_shapes[-1]
                name = f"Down Block E{i}"
                blocks.append(Block(f"E{i}", "encoder", "resnet_down", name, in_shape, out_shape))
        
        # Bottleneck block
        bottleneck_shapes = self._get_shapes_for_pattern(shapes, [r'encoder\.adaptive_pool', r'encoder\.embedding'])
        if bottleneck_shapes:
            in_shape = bottleneck_shapes[0]
            out_shape = bottleneck_shapes[-1]
            latent_dim = out_shape[-1] if len(out_shape) > 1 else 32
            blocks.append(Block("bottleneck", "encoder", "bottleneck", f"z = {latent_dim}", in_shape, out_shape))
        
        return blocks
    
    def _process_decoder(self, modules: Dict[str, nn.Module], shapes: Dict[str, Tuple]) -> List[Block]:
        """Process decoder modules into blocks."""
        blocks = []
        
        # Dense + Reshape block
        dense_shapes = self._get_shapes_for_pattern(shapes, [r'decoder\.linear', r'decoder\.conv_initial'])
        if dense_shapes:
            in_shape = dense_shapes[0]
            out_shape = dense_shapes[-1]
            name = "Dense→Reshape"
            blocks.append(Block("dense_reshape", "decoder", "dense_reshape", name, in_shape, out_shape))
        
        # Detect ResNet Up blocks
        resnet_up_patterns = [r'decoder\.resnet_up1', r'decoder\.resnet_up2', r'decoder\.resnet_up3']
        for i, pattern in enumerate(resnet_up_patterns, 1):
            resnet_shapes = self._get_shapes_for_pattern(shapes, [pattern])
            if resnet_shapes:
                in_shape = resnet_shapes[0]
                out_shape = resnet_shapes[-1]
                name = f"Up Block D{i}"
                blocks.append(Block(f"D{i}", "decoder", "resnet_up", name, in_shape, out_shape))
        
        # Output head
        output_shapes = self._get_shapes_for_pattern(shapes, [r'decoder\.final_conv'])
        if output_shapes:
            in_shape = output_shapes[0]
            out_shape = output_shapes[-1]
            name = "Conv Head → 1ch"
            blocks.append(Block("output_head", "decoder", "output_head", name, in_shape, out_shape))
        
        return blocks
    
    def _get_shapes_for_pattern(self, shapes: Dict[str, Tuple], patterns: List[str]) -> List[Tuple]:
        """Get shapes matching any of the given patterns."""
        matching_shapes = []
        for pattern in patterns:
            for module_name, shape in shapes.items():
                if re.search(pattern, module_name):
                    matching_shapes.append(shape)
        return matching_shapes
    
    def _get_input_shape(self, shapes: Dict[str, Tuple]) -> Tuple:
        """Get model input shape."""
        # Look for encoder input or use default
        for name, shape in shapes.items():
            if 'conv_input' in name and len(shape) == 4:
                return (shape[0], 1, shape[2], shape[3])  # Assume 1 input channel
        return (1, 1, 256, 256)  # Default
    
    def _get_output_shape(self, shapes: Dict[str, Tuple]) -> Tuple:
        """Get model output shape."""
        # Look for final output
        for name, shape in shapes.items():
            if any(pattern in name for pattern in ['final_conv', 'output']):
                return shape
        # Use largest spatial dimensions as fallback
        max_shape = (1, 1, 256, 256)
        for shape in shapes.values():
            if len(shape) == 4 and shape[2] * shape[3] > max_shape[2] * max_shape[3]:
                max_shape = shape
        return max_shape


class ThemeManager:
    """Manages visual themes for diagram generation."""
    
    def __init__(self, theme_config: Dict = None):
        self.config = theme_config or self._default_theme()
    
    def _default_theme(self) -> Dict:
        """Default publication-ready theme."""
        return {
            'palette': {
                'encoder': '#4A90E2',     # Cool blue
                'decoder': '#F5A623',     # Warm orange  
                'io': '#D0D0D0',          # Neutral gray
                'latent': '#BD10E0'       # Purple highlight
            },
            'role_palette': {
                'conv': '#f6d6a8',       # Warm beige for convolution
                'norm': '#a7e1f5',       # Light blue for normalization
                'act': '#d9c7c0',        # Light brown for activation
                'pool': '#b8e2b1',       # Light green for pooling
                'up': '#cfe399',         # Green for upsampling
                'skip': '#3aa57a'        # Dark green for skip connections
            },
            'font': {
                'name': 'Arial',
                'size': 10,
                'meta_size': 9
            },
            'node': {
                'width': 1.6,
                'height': 1.0
            },
            'layout': {
                'rankdir': 'LR',
                'nodesep': 0.35,
                'ranksep': 0.6,
                'splines': 'true'
            }
        }
    
    @classmethod
    def load_theme(cls, theme_path: Path) -> 'ThemeManager':
        """Load theme from YAML file."""
        try:
            with open(theme_path, 'r') as f:
                theme_config = yaml.safe_load(f)
            return cls(theme_config)
        except Exception as e:
            print(f"Warning: Could not load theme {theme_path}: {e}")
            return cls()


class BlockTemplateRenderer:
    """Renders micro-diagrams for ResNet blocks with skip connections."""
    
    def __init__(self, theme: ThemeManager):
        self.theme = theme
    
    def generate_resnet_down_cluster(self) -> str:
        """Generate ResNet MaxPool block cluster for composite figure."""
        spec = [
            ("conv", "128"), ("conv", "128"), ("norm", ""), ("act", ""),
            ("conv", "128"), ("norm", ""), ("act", ""), ("pool", "")
        ]
        return self._generate_cluster("b) ResNet MaxPool Block", spec, has_skip=True, cluster_id="cluster_b")
    
    def generate_resnet_up_cluster(self) -> str:
        """Generate ResNet UpSample block cluster for composite figure."""
        spec = [
            ("conv", "128"), ("conv", "128"), ("norm", ""), ("act", ""),
            ("conv", "128"), ("norm", ""), ("act", ""), ("up", "×2")
        ]
        return self._generate_cluster("c) ResNet UpSample Block", spec, has_skip=True, cluster_id="cluster_c")
    
    def _generate_cluster(self, title: str, spec: List[Tuple[str, str]], 
                         has_skip: bool = False, cluster_id: str = "cluster_block") -> str:
        """Generate cluster content for composite figure."""
        lines = [
            f'    subgraph {cluster_id} {{',
            f'        label="{title}";',
            '        style=dashed;',
            '        color="#666";',
            '        labelloc="t";',
            '        labeljust="l";',
            '        margin="12,10";',
            '        node [shape=box, style="rounded,filled"];',
            ''
        ]
        
        # Generate nodes
        node_ids = []
        
        # Previous layer ghost node
        lines.append('        prev [label="Previous Layer", style="rounded,dashed", '
                    'fillcolor="white", color="#ccc"];')
        node_ids.append('prev')
        
        # Main operation nodes
        for i, (op_type, label) in enumerate(spec):
            node_id = f"{op_type}{i+1}"
            node_ids.append(node_id)
            
            color = self.theme.config['role_palette'].get(op_type, '#f0f0f0')
            
            if op_type == 'up' and label:
                # Special handling for upsample with label
                lines.append(f'        {node_id} [label="UP", xlabel="{label}", '
                           f'fillcolor="{color}"];')
            else:
                display_label = f"{op_type.upper()}"
                if label and op_type in ['conv']:
                    display_label += f"\\n{label}ch"
                lines.append(f'        {node_id} [label="{display_label}", '
                           f'fillcolor="{color}"];')
        
        # Next layer ghost node
        lines.append('        next [label="Next Layer", style="rounded,dashed", '
                    'fillcolor="white", color="#ccc"];')
        node_ids.append('next')
        
        # Skip connection elements if needed
        if has_skip:
            lines.extend([
                '        plus [label="+", shape=circle, width=0.18, height=0.18, '
                'style=filled, fillcolor="#3aa57a"];',
                '        top_anchor [shape=point, label="", width=0.01, height=0.01, style=invis];'
            ])
        
        lines.append('')
        
        # Rank constraints
        lines.append(f'        {{rank=same; {" ".join(node_ids)}}}')
        if has_skip:
            lines.append('        {rank=min; top_anchor plus}')
            # Force plus above conv3 (index 4 in our spec)
            lines.append(f'        {node_ids[5]} -> top_anchor [style=invis, weight=10];')
        
        lines.append('')
        
        # Main path edges
        main_path = " -> ".join(node_ids)
        lines.append(f'        {main_path} [color="#666"];')
        
        # Skip connection if needed
        if has_skip and len(spec) >= 4:
            lines.extend([
                f'        prev -> plus [color="#3aa57a", constraint=false, arrowhead=none, minlen=2];',
                f'        {node_ids[5]} -> plus [color="#3aa57a", constraint=false, arrowhead=none, minlen=1];',
                f'        plus -> next [color="#3aa57a"];'
            ])
        
        lines.append('    }')
        
        return '\n'.join(lines)


class DOTGenerator:
    """Generates Graphviz DOT representation of autoencoder architecture."""
    
    def __init__(self, theme: ThemeManager):
        self.theme = theme
    
    def generate_publication_dot(self, blocks: List[Block], 
                                resnet_down_cluster: str, resnet_up_cluster: str) -> str:
        """Generate composite publication DOT with three panels."""
        lines = self._generate_publication_header()
        lines.extend(self._generate_overview_cluster(blocks))
        lines.append(resnet_down_cluster)
        lines.append(resnet_up_cluster)
        lines.append('}')
        
        return '\n'.join(lines)
    
    def generate_overview_cluster(self, blocks: List[Block]) -> str:
        """Generate overview cluster for composite figure."""
        lines = self._generate_overview_cluster(blocks)
        return '\n'.join(lines)
    
    def _generate_publication_header(self) -> List[str]:
        """Generate header for composite publication figure."""
        font = self.theme.config['font']
        layout = self.theme.config['layout']
        
        return [
            'digraph Figure {',
            '    rankdir=TB;',
            '    newrank=true;',
            '    compound=true;',
            f'    splines={layout["splines"]};',
            f'    nodesep={layout["nodesep"]};',
            f'    ranksep={layout["ranksep"]};',
            f'    node [fontname="{font["name"]}"];',
            '    edge [arrowsize=0.7, penwidth=1.2, color="#888"];',
            ''
        ]
    
    def _generate_overview_cluster(self, blocks: List[Block]) -> List[str]:
        """Generate overview cluster with enhanced styling."""
        lines = [
            '    subgraph cluster_a {',
            '        label="a) Model Structure";',
            '        style=dashed;',
            '        color="#9ec5ff";',
            '        labelloc="t";',
            '        labeljust="l";',
            '        margin="12,10";',
            ''
        ]
        
        # Generate nodes
        for block in blocks:
            node_def = self._format_overview_node(block)
            lines.append(f'        {node_def}')
        
        lines.append('')
        
        # Generate chevrons and edges
        chevron_count = 0
        for i in range(len(blocks) - 1):
            current = blocks[i]
            next_block = blocks[i+1]
            
            # Add chevron between nodes
            chevron_id = f"chevr{chevron_count + 1}"
            chevron_count += 1
            
            lines.append(f'        {chevron_id} [shape=triangle, width=0.12, height=0.12, '
                        f'label="", orientation=90, color="#aaa", fillcolor="#ddd", style=filled];')
            
            # Connect with chevron
            lines.append(f'        {current.id} -> {chevron_id} -> {next_block.id};')
        
        lines.append('')
        
        # Rank constraint for neat single row
        all_node_ids = [b.id for b in blocks]
        lines.append(f'        {{rank=same; {" ".join(all_node_ids)}}}')
        
        lines.append('    }')
        lines.append('')
        
        return lines
    
    def _format_overview_node(self, block: Block) -> str:
        """Format overview node with 3D styling and no stripes."""
        palette = self.theme.config['palette']
        node_config = self.theme.config['node']
        font = self.theme.config['font']
        
        # Determine colors based on group
        if block.group == 'encoder':
            fill_color = palette['encoder']
        elif block.group == 'decoder':
            fill_color = palette['decoder']
        elif block.type == 'bottleneck':
            fill_color = palette['latent']
        else:
            fill_color = palette['io']
        
        # Determine shape and sizing
        if block.type == 'bottleneck':
            shape = 'oval'
            width = 0.8
            height = 0.6
            label = block.name
        else:
            shape = 'box3d'
            width = node_config['width']
            height = node_config['height']
            label = self._format_overview_label(block)
        
        # Generate gradient fill
        gradient_fill = f"white:{fill_color}"
        
        return (f'{block.id} [label=<{label}>, '
                f'fillcolor="{gradient_fill}", gradientangle=90, '
                f'shape={shape}, style="filled,rounded", '
                f'fixedsize=true, width={width}, height={height}];')
    
    def _format_overview_label(self, block: Block) -> str:
        """Format overview label with bold title and smaller meta."""
        if block.type in ['input', 'output']:
            return f"<B>{block.name}</B>"
        
        # Get shape information
        spatial_info = ""
        channel_info = ""
        
        if len(block.out_shape) == 4:
            spatial_info = f"{block.out_shape[2]}×{block.out_shape[3]}"
            channel_info = f"{block.out_shape[1]}ch"
            
            # Add spatial change info for ResNet blocks
            if block.type in ['resnet_down', 'resnet_up'] and len(block.in_shape) == 4:
                in_spatial = f"{block.in_shape[2]}×{block.in_shape[3]}"
                out_spatial = f"{block.out_shape[2]}×{block.out_shape[3]}"
                if block.in_shape[2] != block.out_shape[2]:
                    spatial_info = f"{in_spatial} → {out_spatial}"
        elif len(block.out_shape) == 2:
            channel_info = f"dim = {block.out_shape[1]}"
        
        # Build label
        meta_info = " • ".join(filter(None, [spatial_info, channel_info]))
        
        return (f"<B>{block.name}</B><BR/>"
                f"<FONT POINT-SIZE=\"9\" COLOR=\"#555\">{meta_info}</FONT>")


class ModelLoader:
    """Loads and instantiates autoencoder models."""
    
    @staticmethod
    def load_model(model_path: str, model_kwargs: Dict = None, checkpoint_path: str = None,
                   from_autoencoder: bool = False) -> nn.Module:
        """Load model from import path."""
        model_kwargs = model_kwargs or {}
        
        try:
            if from_autoencoder:
                module_name, class_name = "models.autoencoder", "Autoencoder"
            else:
                if ':' in model_path:
                    module_name, class_name = model_path.split(':')
                else:
                    raise ValueError("Model path must be in format 'module:class'")
            
            # Import module and get class
            module = importlib.import_module(module_name)
            model_class = getattr(module, class_name)
            
            # Instantiate model
            model = model_class(**model_kwargs)
            
            # Extract actual model if Lightning wrapper
            if hasattr(model, 'model'):
                model = model.model
            
            # Load checkpoint if provided
            if checkpoint_path:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                if 'state_dict' in checkpoint:
                    # Lightning checkpoint
                    state_dict = {}
                    for key, value in checkpoint['state_dict'].items():
                        if key.startswith('model.'):
                            state_dict[key[6:]] = value  # Remove 'model.' prefix
                        else:
                            state_dict[key] = value
                    model.load_state_dict(state_dict, strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
            
            model.eval()
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")


class DiagramGenerator:
    """Main diagram generator orchestrating the entire process."""
    
    def __init__(self, output_dir: Path = None, theme: ThemeManager = None):
        self.output_dir = Path(output_dir or ".")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.theme = theme or ThemeManager()
        
        self.shape_tracker = ShapeTracker()
        self.block_detector = BlockDetector()
        self.dot_generator = DOTGenerator(self.theme)
        self.block_renderer = BlockTemplateRenderer(self.theme)
    
    def generate_publication(self, model: nn.Module, input_size: Tuple[int, ...] = (1, 1, 256, 256),
                           output_name: str = "publication") -> Dict[str, Path]:
        """Generate single composite publication figure using SVG renderer."""
        print("🔍 Tracing model architecture...")
        
        # Set deterministic behavior
        torch.manual_seed(42)
        
        # Register hooks and trace model
        self.shape_tracker.register_hooks(model)
        
        try:
            # Create dummy input and run forward pass
            dummy_input = torch.randn(input_size)
            with torch.no_grad():
                _ = model(dummy_input)
            
            print(f"✓ Captured shapes for {len(self.shape_tracker.shapes)} modules")
            
            # Detect blocks
            print("🔗 Detecting semantic blocks...")
            blocks = self.block_detector.detect_blocks(model, self.shape_tracker.shapes)
            print(f"✓ Detected {len(blocks)} blocks")
            
            # Generate SVG using custom renderer
            print("📊 Generating publication SVG...")
            svg_path = self.output_dir / f"{output_name}.svg"
            compose_publication_svg(blocks, out_path=str(svg_path))
            
            output_files = {'svg': svg_path}
            print(f"✓ SVG saved to {svg_path}")
            
            # Optional PNG conversion
            try:
                png_path = self.output_dir / f"{output_name}.png"
                cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), dpi=300)
                output_files['png'] = png_path
                print(f"✓ PNG rendered to {png_path}")
            except Exception as e:
                print(f"⚠ PNG conversion failed: {e}")
            
            return output_files
            
        finally:
            self.shape_tracker.clear_hooks()
    
    def generate_overview(self, model: nn.Module, input_size: Tuple[int, ...] = (1, 1, 256, 256),
                         output_name: str = "overview") -> Dict[str, Path]:
        """Generate overview diagram only."""
        print("🔍 Tracing model architecture...")
        
        # Set deterministic behavior
        torch.manual_seed(42)
        
        # Register hooks and trace model
        self.shape_tracker.register_hooks(model)
        
        try:
            # Create dummy input and run forward pass
            dummy_input = torch.randn(input_size)
            with torch.no_grad():
                _ = model(dummy_input)
            
            print(f"✓ Captured shapes for {len(self.shape_tracker.shapes)} modules")
            
            # Detect blocks
            print("🔗 Detecting semantic blocks...")
            blocks = self.block_detector.detect_blocks(model, self.shape_tracker.shapes)
            print(f"✓ Detected {len(blocks)} blocks")
            
            # Generate overview cluster as standalone
            print("📊 Generating overview DOT representation...")
            overview_cluster = self.dot_generator.generate_overview_cluster(blocks)
            
            # Wrap in digraph for standalone use
            lines = [
                'digraph Overview {',
                '    rankdir=LR;',
                '    compound=true;',
                '    nodesep=0.35;',
                '    ranksep=0.6;',
                '    splines=true;',
                '    node [fontname="Arial"];',
                '    edge [arrowsize=0.7, penwidth=1.2, color="#888"];',
                '',
                overview_cluster,
                '}'
            ]
            
            dot_content = '\n'.join(lines)
            
            return self._save_diagram(dot_content, output_name, include_png=False)
            
        finally:
            self.shape_tracker.clear_hooks()
    
    def generate_blocks(self, output_base: str = "blocks") -> Dict[str, Path]:
        """Generate micro-diagrams for ResNet blocks only."""
        output_files = {}
        
        print("📊 Generating ResNet down block diagram...")
        resnet_down_cluster = self.block_renderer.generate_resnet_down_cluster()
        down_dot = self._wrap_cluster_as_digraph(resnet_down_cluster, "ResNetDown")
        down_files = self._save_diagram(down_dot, f"{output_base}_down", include_png=False)
        output_files.update({f"down_{k}": v for k, v in down_files.items()})
        
        print("📊 Generating ResNet up block diagram...")
        resnet_up_cluster = self.block_renderer.generate_resnet_up_cluster()
        up_dot = self._wrap_cluster_as_digraph(resnet_up_cluster, "ResNetUp")
        up_files = self._save_diagram(up_dot, f"{output_base}_up", include_png=False)
        output_files.update({f"up_{k}": v for k, v in up_files.items()})
        
        return output_files
    
    def generate_split(self, model: nn.Module, input_size: Tuple[int, ...] = (1, 1, 256, 256),
                      output_base: str = "split") -> Dict[str, Path]:
        """Generate all three diagram types separately."""
        output_files = {}
        
        # Generate overview
        overview_files = self.generate_overview(model, input_size, f"{output_base}_overview")
        output_files.update({f"overview_{k}": v for k, v in overview_files.items()})
        
        # Generate block diagrams
        block_files = self.generate_blocks(output_base)
        output_files.update(block_files)
        
        return output_files
    
    def _wrap_cluster_as_digraph(self, cluster_content: str, graph_name: str) -> str:
        """Wrap cluster content as standalone digraph."""
        lines = [
            f'digraph {graph_name} {{',
            '    rankdir=LR;',
            '    compound=true;',
            '    splines=ortho;',
            '    nodesep=0.3;',
            '    ranksep=0.4;',
            '    node [fontname="Arial"];',
            '    edge [arrowsize=0.6, penwidth=1.1];',
            '',
            cluster_content,
            '}'
        ]
        return '\n'.join(lines)
    
    def _save_diagram(self, dot_content: str, output_name: str, 
                     include_png: bool = False, save_dot: bool = True) -> Dict[str, Path]:
        """Save diagram files and render to SVG."""
        output_files = {}
        
        # Save DOT file
        if save_dot:
            dot_path = self.output_dir / f"{output_name}.dot"
            with open(dot_path, 'w') as f:
                f.write(dot_content)
            output_files['dot'] = dot_path
            print(f"✓ DOT saved to {dot_path}")
        
        # Note: SVG rendering is now handled by the SVG renderer
        # This method is kept for backward compatibility with other modes
        print("Note: Using legacy DOT rendering - consider using SVG mode")
        
        return output_files


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Generate publication-grade autoencoder architecture diagrams",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_diagram.py --model lightning_model:LitAE
  python generate_diagram.py --from-autoencoder --split
  python generate_diagram.py --figure overview --png
        """
    )
    
    # Model specification
    parser.add_argument("--model", type=str, default="models.autoencoder:Autoencoder",
                       help="Model import path (module:class)")
    parser.add_argument("--from-autoencoder", action="store_true",
                       help="Use models.autoencoder:Autoencoder directly")
    parser.add_argument("--model-kwargs", type=str,
                       help="Model kwargs as JSON string")
    parser.add_argument("--ckpt", type=str,
                       help="Checkpoint path to load weights")
    
    # Input configuration
    parser.add_argument("--input-size", type=int, default=256,
                       help="Input image size (assumes square)")
    parser.add_argument("--latent-dim", type=int, default=32,
                       help="Latent dimension")
    
    # Output configuration
    parser.add_argument("--figure", type=str, 
                       choices=['publication', 'overview', 'blocks'], 
                       default='publication',
                       help="Figure type: publication (default SVG), overview, or blocks")
    parser.add_argument("--out", type=str, default="publication",
                       help="Output file base name")
    parser.add_argument("--theme", type=str,
                       help="Theme configuration YAML file")
    
    # Options
    parser.add_argument("--split", action="store_true",
                       help="Generate separate files for overview and blocks")
    parser.add_argument("--no-dot", action="store_true",
                       help="Suppress DOT file output")
    parser.add_argument("--png", action="store_true",
                       help="Also generate PNG output (auto-enabled for publication mode)")
    
    args = parser.parse_args()
    
    try:
        # Determine output directory and base name
        output_path = Path(args.out)
        if output_path.suffix:
            # Remove extension from base name
            output_base = output_path.stem
            output_dir = output_path.parent
        else:
            output_base = args.out
            output_dir = Path(".")
        
        # Load theme
        theme = ThemeManager.load_theme(Path(args.theme)) if args.theme else ThemeManager()
        
        # Parse model kwargs
        model_kwargs = {"latent_dim": args.latent_dim}
        if args.model_kwargs:
            additional_kwargs = json.loads(args.model_kwargs)
            model_kwargs.update(additional_kwargs)
        
        # Generate diagrams based on figure type
        generator = DiagramGenerator(output_dir=output_dir, theme=theme)
        
        if args.figure == 'blocks':
            # Only generate block diagrams
            print("📦 Generating ResNet block diagrams...")
            output_files = generator.generate_blocks(output_base)
        elif args.split:
            # Generate separate files regardless of figure type
            print(f"📦 Loading model...")
            model = ModelLoader.load_model(
                args.model if not args.from_autoencoder else "models.autoencoder:Autoencoder",
                model_kwargs=model_kwargs,
                checkpoint_path=args.ckpt,
                from_autoencoder=args.from_autoencoder
            )
            print(f"✓ Model loaded successfully")
            
            input_size = (1, 1, args.input_size, args.input_size)
            output_files = generator.generate_split(model, input_size=input_size, 
                                                  output_base=output_base)
        else:
            # Load model for overview or publication
            print(f"📦 Loading model...")
            model = ModelLoader.load_model(
                args.model if not args.from_autoencoder else "models.autoencoder:Autoencoder",
                model_kwargs=model_kwargs,
                checkpoint_path=args.ckpt,
                from_autoencoder=args.from_autoencoder
            )
            print(f"✓ Model loaded successfully")
            
            input_size = (1, 1, args.input_size, args.input_size)
            
            if args.figure == 'overview':
                output_files = generator.generate_overview(model, input_size=input_size, 
                                                         output_name=output_base)
            else:  # publication (default)
                output_files = generator.generate_publication(model, input_size=input_size, 
                                                            output_name=output_base)
        
        print(f"\n🎉 Diagram generation complete!")
        print(f"📁 Output directory: {output_dir}")
        for file_type, file_path in output_files.items():
            print(f"  {file_type.upper()}: {file_path}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n❌ Generation interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())