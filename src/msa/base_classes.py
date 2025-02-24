import numpy as np
from .functions import * 

class Node:
    '''
    Represents a structural node with spatial coordinates and boundary conditions.
    '''
    def __init__(self, id, x, y, z):
        '''
        Args:
            id (int): Unique node identifier
            x (float): X-coordinate in global system
            y (float): Y-coordinate in global system
            z (float): Z-coordinate in global system
        '''
        self.id = id
        self.coords = (x, y, z)
        self.fixed_dofs = [False]*6
        self.loads = [0.0]*6
        self.displacements = [0.0]*6
        self.forces = [0.0]*6

class Material:
    '''
    Contains material properties for structural elements.
    '''
    def __init__(self, id, E, G, nu, density=0.0):
        '''
        Args:
            id (int): Unique material identifier
            E (float): Young's modulus (elastic modulus)
            G (float): Shear modulus
            nu (float): Poisson's ratio
            density (float, optional): Material density. Defaults to 0.0.
        '''
        self.id = id
        self.E = E
        self.G = G
        self.nu = nu
        self.density = density

class Section:
    '''
    Defines cross-sectional properties for structural elements.
    '''
    def __init__(self, id, A, Iy, Iz, J):
        '''
        Args:
            id (int): Unique section identifier
            A (float): Cross-sectional area
            Iy (float): Moment of inertia about local y-axis
            Iz (float): Moment of inertia about local z-axis
            J (float): Torsional constant
        '''
        self.id = id
        self.A = A
        self.Iy = Iy
        self.Iz = Iz
        self.J = J

class Element:
    '''
    Represents a 3D beam element connecting two nodes.
    '''
    def __init__(self, id, node1, node2, material, section):
        '''
        Args:
            id (int): Unique element identifier
            node1 (Node): First end node
            node2 (Node): Second end node
            material (Material): Material properties
            section (Section): Cross-section properties
        '''
        self.id = id
        self.node1 = node1
        self.node2 = node2
        self.L = self.compute_length()
        self.material = material
        self.section = section

    def compute_length(self):
        '''
        Calculate element length from nodal coordinates.
        Returns:
            float: Euclidean distance between node1 and node2
        '''
        x1, y1, z1 = self.node1.coords
        x2, y2, z2 = self.node2.coords
        length = ((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)**0.5
        return length

    def compute_local_stiffness(self):
        """Return 12x12 local stiffness matrix for a 3D beam/truss."""
        # Example for a 3D beam element
        # Implementation depends on the formula from your structural analysis references
        K_local = local_elastic_stiffness_matrix_3D_beam(self.material.E, self.material.nu, self.section.A, 
                                                         self.L, self.section.Iy, self.section.Iz, self.section.J)
        return K_local

    def compute_transformation_matrix(self):
        """Return 12x12 transformation matrix (T) to go from local to global."""
        # 1) Calculate direction cosines from node1.coords to node2.coords
        # 2) Build the 3x3 rotation matrix
        # 3) Expand into 12x12 for 3D beam (translations+rotations)
        x1, y1, z1 = self.node1.coords
        x2, y2, z2 = self.node2.coords
        gamma = rotation_matrix_3D(x1, y1, z1, x2, y2, z2,)
        T = transformation_matrix_3D(gamma)
        return T

    def compute_global_stiffness(self):
        """Return the 12x12 element stiffness matrix in global coordinates."""
        K_local = self.compute_local_stiffness()
        T = self.compute_transformation_matrix()
        # K_global = T^T * K_local * T
        return T.T @ K_local @ T

class Structure:
    '''
    Main class representing the structural system.
    Attributes:
        nodes (dict): {id: Node} mapping
        elements (dict): {id: Element} mapping
        materials (dict): {id: Material} mapping
        sections (dict): {id: Section} mapping
        dof_map (dict): Maps (node_id, local_dof) to global DOF indices
        num_dofs (int): Total number of degrees of freedom
        K (np.ndarray): Global stiffness matrix
        F (np.ndarray): Global force vector
    '''
    def __init__(self):
        self.nodes = {}
        self.elements = {}
        self.materials = {}
        self.sections = {}
        self.dof_map = {}  # Maps (node_id, local_dof) -> global_dof
        self.num_dofs = 0  # The number of dofs of the whole structure
        self.K = None
        self.Load = None
        self.u = None
        self.reaction = None

    def add_node(self, node):
        self.nodes[node.id] = node

    def add_material(self, material):
        self.materials[material.id] = material

    def add_section(self, section):
        self.sections[section.id] = section

    def add_element(self, element):
        self.elements[element.id] = element

    def assign_dof_numbers(self):
        """
        Assign each node's 6 DOFs a global index. 
        We do NOT skip fixed DOFs here; we'll handle them later by zeroing out.
        """
        dof_counter = 0

        # Suppose self.nodes is a dict {node_id: Node}
        # We'll iterate in ascending node_id order for consistency
        for node_id, _ in sorted(self.nodes.items()):
            for local_dof in range(6):  # 0..5 => [u, v, w, rx, ry, rz]
                self.dof_map[(node_id, local_dof)] = dof_counter
                dof_counter += 1

        # Store total DOFs in the model
        self.num_dofs = dof_counter


    def assemble_global_stiffness(self):
        """
        Build the global stiffness matrix by looping over elements.
        For each element, get K_global (12x12 for a 3D beam element),
        then place it into the big K matrix at the appropriate rows/columns.
        """
        # Initialize global stiffness matrix
        Kg = np.zeros((self.num_dofs, self.num_dofs))

        # Loop over all elements in your model
        for ele_id, ele in self.elements.items():
            # For a 3D beam, each node has 6 DOFs
            node1_id = ele.node1.id
            node2_id = ele.node2.id

            # Build a 12-element list of the global DOF indices for this element
            dofs_element = []
            for local_dof in range(6):  # node1’s 6 DOFs
                dofs_element.append(self.dof_map[(node1_id, local_dof)])
            for local_dof in range(6):  # node2’s 6 DOFs
                dofs_element.append(self.dof_map[(node2_id, local_dof)])

            # Get the 12×12 element stiffness in global coords
            Ke_global = ele.compute_global_stiffness()

            # "Scatter" Ke_global into the master Kg
            for i in range(12):
                for j in range(12):
                    Kg[dofs_element[i], dofs_element[j]] += Ke_global[i, j]

        return Kg

    def assemble_external_force(self):
        """
        Build the global force vector (Fg) from external nodal loads.
        For each node, we add the node's loads to the corresponding
        global DOFs as defined by self.dof_map.
        """
        Fg = np.zeros(self.num_dofs)

        # Loop over all nodes
        for node_id, node in self.nodes.items():
            # Each node in 3D beams/frames has 6 DOFs: (u, v, w, rx, ry, rz)
            for local_dof in range(6):
                # Look up the global DOF index from the dof_map
                global_dof = self.dof_map[(node_id, local_dof)]
                Fg[global_dof] += node.loads[local_dof]

        return Fg
        
    def apply_boundary_conditions(self):
        """
        Modify the global stiffness matrix (self.K) and the global load vector (self.F)
        to account for fixed DOFs. This approach:
        - Zeros out the row and column of the fixed DOF in K,
        - Sets the diagonal to 1,
        - Sets the load at that DOF to 0,
        which enforces zero displacement at that DOF.
        """
        # Make copies (or work in-place) depending on your design
        K = self.K.copy()
        F = self.Load.copy()

        # Loop through all nodes
        for node_id, node in self.nodes.items():
            # For each local DOF (0..5), check if it's fixed
            for local_dof, is_fixed in enumerate(node.fixed_dofs):
                if is_fixed:
                    # Look up the global DOF index from self.dof_map
                    global_dof = self.dof_map[(node_id, local_dof)]

                    # 1) Zero out entire row and column
                    K[global_dof, :] = 0.0
                    K[:, global_dof] = 0.0

                    # 2) Set diagonal to 1
                    K[global_dof, global_dof] = 1.0

                    # 3) Set load to 0
                    F[global_dof] = 0.0

        return (K, F)


    def solve_u(self):
        """
        Solve the system of equations [K]{u} = {F} after assembling the
        global stiffness matrix and force vector, and applying boundary conditions.
        Then store the displacement results in each Node.
        """

        K, F = self.apply_boundary_conditions()

        #    Solve the linear system.
        #    Use a standard solver; for large systems, consider a sparse solver.
        try:
            u = np.linalg.solve(K, F)
        except np.linalg.LinAlgError as e:
            raise ValueError("Global stiffness matrix is singular or ill-conditioned. "
                            "Check boundary conditions.") from e

        #    Store the computed displacements in each node.
        for node_id, node in self.nodes.items():
            for local_dof in range(6):
                global_dof = self.dof_map[(node_id, local_dof)]
                node.displacements[local_dof] = u[global_dof]

        return u
    
    def compute_reaction_force(self):
        '''
        After solving for the displacement vector self.u,
        compute the nodal reaction force at every DOF by F = K @ u.
        Then store these forces back into each Node object, force attributes. 
        '''
        # 1) Multiply K by u to get total nodal forces
        F_total = self.K @ self.u  # shape = (num_dofs,)
        reactions = F_total - self.Load

        # 2) Store these forces in each Node
        for node_id, node in self.nodes.items():
            for local_dof in range(6):
                global_dof = self.dof_map[(node_id, local_dof)]
                node.forces[local_dof] = reactions[global_dof]
        return reactions

    def analyze(self):
        '''
        Prepare and solve the structure.
        '''
        self.assign_dof_numbers()
        self.K = self.assemble_global_stiffness()
        self.Load = self.assemble_external_force()
        self.u = self.solve_u()
        self.reaction = self.compute_reaction_force()
