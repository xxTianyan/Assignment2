from msa import Node, Element, Structure
from msa import Material, Section
import numpy as np

# Step 1: Create structural model
struct = Structure()
    
# Step 2: Define material properties
# Steel properties
mat= Material(
        id=1,
        E=500,
        G=500/2.6,
        nu=0.3,
    )
struct.add_material(mat)
    
# Step 3: Define cross-section properties
# Rectangular section
circle_section = Section(
        id=1,
        A=np.pi,   
        Iy=np.pi/4,
        Iz=np.pi/4, 
        J=np.pi/2      
    )
struct.add_section(circle_section)
    
# Step 4: Create nodes
# Node 1: Fixed support
node1 = Node(id=1, x=0, y=0, z=0)
node1.fixed_dofs = [False,False,True,False,False,False] # fixed in z
struct.add_node(node1)
    
# Node 2: Free node
node2 = Node(id=2, x=-5, y=1, z=10)
node2.loads = [0.1, -0.05, -0.075, 0, 0, 0] 
struct.add_node(node2)

# Node 3: Free node
node3 = Node(id=3, x=-1, y=5, z=13)
node3.loads = [0, 0, 0, 0.5, -0.1, 0.3] 
struct.add_node(node3)

# Node 4: Fixed node
node4 = Node(id=4, x=-3, y=7, z=11)
node4.fixed_dofs = [True]*6 
struct.add_node(node4)

# Node 5: Pinned node
node5 = Node(id=5, x=6, y=9, z=5)
node5.fixed_dofs =  [True, True, True, False, False, False]
struct.add_node(node5)


# Step 5: Create elements
# Create beam element connecting nodes 1,2 and node 2,3
beam_element1= Element(
        id=1,
        node1=struct.nodes[1],
        node2=struct.nodes[2],
        material=mat,
        section=circle_section,
    )
struct.add_element(beam_element1)

beam_element2= Element(
        id=2,
        node1=struct.nodes[2],
        node2=struct.nodes[3],
        material=mat,
        section=circle_section,
    )
struct.add_element(beam_element2)

beam_element3= Element(
        id=3,
        node1=struct.nodes[3],
        node2=struct.nodes[4],
        material=mat,
        section=circle_section,
    )
struct.add_element(beam_element3)

beam_element4= Element(
        id=4,
        node1=struct.nodes[3],
        node2=struct.nodes[5],
        material=mat,
        section=circle_section,
    )
struct.add_element(beam_element4)


# Step 6: Perform analysis
print("Starting analysis...")
struct.linear_analyze()
print("Analysis completed!")
    
# Step 7: Verify results
# Print nodal displacements
print("\n------------ Nodal Displacement Results -----------------")
for node in [node1, node2, node3, node4, node5]:
    print(f"Node {node.id}:")
    print(f"  Ux = {node.displacements[0]:.4e} ")
    print(f"  Uy = {node.displacements[1]:.4e} ")
    print(f"  Uz = {node.displacements[2]:.4e} ")
    print(f"  θx = {node.displacements[3]:.4e} ")
    print(f"  θy = {node.displacements[4]:.4e} ")
    print(f"  θz = {node.displacements[5]:.4e} ")
    
# Print reaction forces
print("\n------------ Reaction Force Results-----------------")
for node in [node1 , node4, node5]:
    print(f"Node {node.id}:")
    print(f"  Fx = {node.reforces[0]:.4e} ")
    print(f"  Fy = {node.reforces[1]:.4e} ")
    print(f"  Fz = {node.reforces[2]:.4e} ")
    print(f"  Mx = {node.reforces[3]:.4e} ")
    print(f"  My = {node.reforces[4]:.4e} ")
    print(f"  Mz = {node.reforces[5]:.4e} ")

    
