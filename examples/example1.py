from msa import Node, Element, Structure
from msa import Material, Section

# Step 1: Create structural model
struct = Structure()
    
# Step 2: Define material properties
# Steel properties (Units: N, mm)
steel = Material(
        id=1,
        E=210e3,    # 210 GPa = 210e3 N/mm²
        G=80e3,     # 80 GPa
        nu=0.3,     # Poisson's ratio
        density=7.85e-6  # kg/mm³
    )
struct.add_material(steel)
    
# Step 3: Define cross-section properties
# Rectangular section 200x400 mm
rect_section = Section(
        id=1,
        A=200*400,          # Area (mm²)
        Iy=(200*400**3)/12, # Strong axis moment of inertia (mm⁴)
        Iz=(400*200**3)/12, # Weak axis moment of inertia (mm⁴)
        Ir=0,
        J=1e6               # Torsional constant (simplified)
    )
struct.add_section(rect_section)
    
# Step 4: Create nodes
# Node 1: Fixed support
node1 = Node(id=1, x=0, y=0, z=5000)
node1.fixed_dofs = [True]*6  # Fully fixed
struct.add_node(node1)
    
# Node 2: Free node (5m span = 5000mm)
node2 = Node(id=2, x=1000, y=1000, z=3000)
node2.loads = [0, 0, -10000, 0, 0, 0]  # -10kN Y-direction load
struct.add_node(node2)

# Node 2: Pinned node (5m span = 5000mm)
node3 = Node(id=3, x=3000, y=1500, z=0)
node3.fixed_dofs = [True, True, True, False, False, False]
struct.add_node(node3)

# Step 5: Create elements
# Create beam element connecting nodes 1,2 and node 2,3
beam_element1= Element(
        id=1,
        node1=struct.nodes[1],
        node2=struct.nodes[2],
        material=steel,
        section=rect_section
    )
struct.add_element(beam_element1)

beam_element2= Element(
        id=2,
        node1=struct.nodes[2],
        node2=struct.nodes[3],
        material=steel,
        section=rect_section
    )
struct.add_element(beam_element2)


# Step 6: Perform analysis
print("Starting analysis...")
struct.analyze()
print("Analysis completed!")
    
# Step 7: Verify results
# Print nodal displacements
print("\n------------ Nodal Displacement Results -----------------")
for node in [node1, node2, node3]:
    print(f"Node {node.id}:")
    print(f"  Ux = {node.displacements[0]:.4e} mm")
    print(f"  Uy = {node.displacements[1]:.4e} mm (main deformation)")
    print(f"  Uz = {node.displacements[2]:.4e} mm")
    print(f"  θx = {node.displacements[3]:.4e} rad")
    print(f"  θy = {node.displacements[4]:.4e} rad")
    print(f"  θz = {node.displacements[5]:.4e} rad")
    
# Print reaction forces
print("\n------------ Reaction Force Results-----------------")
for node in [node1 , node3]:
    print(f"Node {node.id}:")
    print(f"  Fx = {node.forces[0]:.2f} N ")
    print(f"  Fy = {node.forces[1]:.2f} N ")
    print(f"  Fz = {node.forces[2]:.2f} N ")

# Force equilibrium check
total_Fz = node1.forces[2] + node3.forces[2]
print(f"\nZ-direction Force Balance: {total_Fz:.2f} N (should approach 10000)")
    
