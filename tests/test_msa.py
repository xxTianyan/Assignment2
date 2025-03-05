import os
import numpy as np
import pytest

# Import your classes from your module.
# Adjust the import if your file is named differently.
from msa import Node, Material, Section, Element, Structure


def create_simple_structure():
    """
    Helper function to create a simple cantilever beam structure for testing.
    - Node 1 is fixed.
    - Node 2 is free and loaded in the negative y-direction.
    """
    # Create nodes.
    node1 = Node(1, 0.0, 0.0, 0.0)
    node2 = Node(2, 5.0, 0.0, 0.0)  # 5 meters from node1

    # Set node1 to be fully fixed.
    node1.fixed_dofs = [True, True, True, True, True, True]
    
    # Apply a load at node2 (e.g., 1000 N in negative y-direction).
    node2.loads = [0.0, -1000.0, 0.0, 0.0, 0.0, 0.0]
    
    # Create a material (e.g., steel).
    steel = Material(1, E=210e9, nu=0.3)
    
    # Create a section (dummy values for cross-sectional properties).
    section = Section(1, A=0.01, Iy=1e-6, Iz=2e-6, J=1e-7)
    
    # Create an element connecting node1 and node2.
    element = Element(1, node1, node2, steel, section)
    
    # Assemble the structure.
    structure = Structure()
    structure.add_node(node1)
    structure.add_node(node2)
    structure.add_material(steel)
    structure.add_section(section)
    structure.add_element(element)
    
    return structure, node1, node2


def test_linear_analysis():
    """
    Test linear analysis:
    - Verifies that nodal displacements are computed.
    - Checks that the number of computed displacements matches the total DOFs.
    - Ensures that the free node has nonzero displacement.
    - Checks that reaction forces at the fixed node are computed.
    """
    structure, node1, node2 = create_simple_structure()
    structure.linear_analyze()
    
    # Verify that the global displacement vector is a NumPy array of correct size.
    assert isinstance(structure.u, np.ndarray)
    assert structure.u.shape[0] == structure.num_dofs
    
    # Check that node2 (the free node) displacement is not all zeros.
    assert not np.allclose(node2.displacements, np.zeros(6))
    
    # Check that node1 (the fixed node) reaction forces are not all zeros.
    assert not np.allclose(node1.reforces, np.zeros(6))


def test_buckling_analysis():
    """
    Test buckling analysis:
    - Verifies that the computed elastic critical load (buckling load factor) is positive.
    - Checks that the buckling mode shape is normalized (maximum absolute value equal to 1).
    """
    structure, node1, node2 = create_simple_structure()
    structure.linear_analyze()
    
    lambda_cr, mode_shape = structure.compute_elastic_critical_load()
    
    # The critical load factor should be positive.
    assert lambda_cr > 0
    
    # The mode shape should be normalized.
    np.testing.assert_allclose(np.max(np.abs(mode_shape)), 1.0, atol=1e-6)


def test_post_processing(tmp_path):
    """
    Test post-processing:
    - Runs the post_processing method which should generate a plot file 'test1.png'.
    - Checks that the file exists, then removes it.
    """
    structure, node1, node2 = create_simple_structure()
    structure.linear_analyze()
    structure.post_processing()
    
    # Check if the plot file 'deformed_shape.png' has been created.
    assert os.path.exists('deformed_shape.png')
    
    # Clean up the generated file.
    os.remove('deformed_shape.png')
