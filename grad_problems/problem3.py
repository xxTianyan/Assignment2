from msa import Structure, Element, Node, Material, Section
import numpy as np

struct = Structure()

# material 1 and section 1

mat1= Material(
        id=1,
        E=10000,    
        nu=0.3,
    )
r=1
sec1 = Section(
        id=1,
        A=np.pi * r ** 2.0,   
        Iy=np.pi * r ** 4.0 / 4.0,
        Iz=np.pi * r ** 4.0 / 4.0,
        Ir=np.pi * r ** 4.0 / 2.0,
        J=np.pi * r ** 4.0 / 2.0,
    )
struct.add_material(mat1)
struct.add_section(sec1)

# material 2 and section 2
mat2 = Material(
        id=2,
        E=50000,
        nu=0.3
    )
h=1
b=0.5
sec2 = Section(
        id=2,
        A=b*h,
        Iy=b**3*h/12,
        Iz=b*h**3/12,
        Ir=b*h/12*(b**2+h**2),
        J=0.028610026041666667,
    )
struct.add_material(mat2)
struct.add_section(sec2)

# create nodes
L1 = 11.
L2 = 23.
L3 = 15.
L4 = 13.
x0, y0, z0 = 0, 0, 0


node_positions = [
    # Base rectangle (4 corners)
    (x0,         y0,         z0),
    (x0 + L1,    y0,         z0),
    (x0 + L1,    y0 + L2,    z0),
    (x0,         y0 + L2,    z0),

    # Middle rectangle (4 corners)
    (x0,         y0,         z0 + L3),
    (x0 + L1,    y0,         z0 + L3),
    (x0 + L1,    y0 + L2,    z0 + L3),
    (x0,         y0 + L2,    z0 + L3),

    # Top rectangle (4 corners)
    (x0,         y0,         z0 + L3 + L4),
    (x0 + L1,    y0,         z0 + L3 + L4),
    (x0 + L1,    y0 + L2,    z0 + L3 + L4),
    (x0,         y0 + L2,    z0 + L3 + L4),
]

for i, coord in enumerate(node_positions):
        node = Node(id=i+1, x=coord[0], y=coord[1], z=coord[2])
        if i<=3:
            node.fixed_dofs = [True]*6
        if i>7:
            node.loads = [0, 0, -1, 0, 0, 0]
        struct.add_node(node)

# create connectivity and elements
# Suppose we want beam/truss elements along edges of these rectangles
# plus vertical connections, etc.
# Note: node indices start at 1 in many finite element libraries
# but Python indexing is 0-based, so keep track carefully.

connectivity = [
    # vertical elements
    (1, 5), (2, 6), (3, 7), (4, 8),
    (5, 9), (6, 10), (7, 11), (8, 12),

    # horizontal rectangle edges
    (5, 6), (6, 7), (7, 8), (8, 5),
    (9, 10), (10, 11), (11, 12), (12,9)
    ]

for i, con in enumerate(connectivity):
    node1_id = con[0]
    node2_id = con[1]
    if i <= 7:
         beam_element= Element(
              id=i+1,
              node1=struct.nodes[node1_id],
              node2=struct.nodes[node2_id],
              material=mat1,
              section=sec1
              )
    else:
         beam_element= Element(
              id=i+1,
              node1=struct.nodes[node1_id],
              node2=struct.nodes[node2_id],
              material=mat2,
              section=sec2,
              vtemp=np.asarray([0,0,1])
              )
    struct.add_element(beam_element)

struct.linear_analyze()
lambda_cr, buckling_mode = struct.compute_elastic_critical_load()
print(f'The celastic ritical load of the sturcture is {lambda_cr}')

struct.post_processing(scale=-10)

