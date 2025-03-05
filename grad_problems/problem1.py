from msa import Structure, Element, Node, Material, Section
import numpy as np

struct = Structure()

mat= Material(
        id=1,
        E=10000,    
        nu=0.3,
    )
struct.add_material(mat) # add material to structure

r = 1
section = Section(
        id=1,
        A=np.pi * r ** 2.0,   
        Iy=np.pi * r ** 4.0 / 4.0,
        Iz=np.pi * r ** 4.0 / 4.0,
        Ir=np.pi * r ** 4.0 / 2.0,
        J=np.pi * r ** 4.0 / 2.0,
    )
struct.add_section(section)

start_point = np.array([0,0,0])
end_point = np.array([25,50,37])

coords = np.linspace(start_point, end_point, 7)
for i, coord in enumerate(coords):
    node = Node(id=i+1, x=coord[0], y=coord[1], z=coord[2])
    if i == 0:
        node.fixed_dofs = [True]*6
    if i == 6:
        node.loads = [0.05, -0.1, 0.23, 0.1, -0.025, -0.08]
    struct.add_node(node)

for i in range(1,7):
    beam_element= Element(
        id=i,
        node1=struct.nodes[i],
        node2=struct.nodes[i+1],
        material=mat,
        section=section,
    )
    struct.add_element(beam_element)

struct.linear_analyze()

print('The reaction forces and moments at Node 0 is:')
print(f'{struct.nodes[1].reforces}')

print('The displacements and rotations at Node 3 is:')
print(f'{struct.nodes[4].displacements}')

print('The displacements and rotations at Node 6 is:')
print(f'{struct.nodes[7].displacements}')


