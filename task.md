Migration Requirements: STEMVisualizer to py4DSTEM Backend
Executive Summary
Migrate internal implementation of STEMVisualizer class from custom NumPy-based processing to py4DSTEM backend while maintaining backward compatibility with existing API.
Objectives

Resolve current virtual imaging issues (dark field spots, bright field artifacts)
Maintain 100% API compatibility for existing code
Leverage py4DSTEM's optimized algorithms and validated implementations
Enable future feature expansion with minimal development effort

Additionally:

1. Incorrect Dark Field Detector Geometry
Issue: Current implementation uses rectangular regions instead of annular masks
Impact: Direct beam contamination in dark field signal
Current Code:
pythondef _default_dark_field_region(self):
    # Returns rectangle that includes center/direct beam
    return (center_y - outer_radius, center_y + outer_radius, ...)
2. Missing Annular Mask Implementation
Issue: No proper annular (ring-shaped) detector implementation
Impact: Cannot exclude direct beam while selecting scattered electrons
Current Code: create_virtual_field_image() only handles rectangular regions
3. Data Preprocessing Issues
Issue: Inadequate handling of negative values and outliers
Impact: Hot pixels and reconstruction artifacts appear as bright spots
Required Fixes
Priority 1: Implement Annular Dark Field Detector

 Add create_annular_mask() method with inner/outer radius parameters
 Add create_dark_field_image() method using annular masks
 Update dark field region storage format from (y_min, y_max, x_min, x_max) to (center_y, center_x, inner_radius, outer_radius)