import numpy as np
from .functions import * 
import matplotlib.pyplot as plt
from scipy.linalg import eigh

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
    def __init__(self, id, E, G=0, nu=0, density=0.0):
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
    def __init__(self, id, A, Iy=0, Iz=0, Ir=0, J=0):
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
        self.Ir = Ir
        self.J = J

class Element:
    '''
    Represents a 3D beam element connecting two nodes.
    '''
    def __init__(self, id, node1, node2, material, section, vtemp=None):
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
        self.vtemp = vtemp
        self.T, self.gamma = self.compute_transformation_matrix()

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
    
    def compute_local_geometry_stiffness(self):
        Kg_local = local_geometric_stiffness_matrix_3D_beam(self.L, self.section.A, self.section.Ir, self.node2.loads[0],
                                                            self.node2.loads[3], self.node1.loads[4], self.node1.loads[5],
                                                            self.node2.loads[4], self.node2.loads[5])
        return Kg_local

    def compute_transformation_matrix(self):
        """Return 12x12 transformation matrix (T) to go from local to global."""
        # 1) Calculate direction cosines from node1.coords to node2.coords
        # 2) Build the 3x3 rotation matrix
        # 3) Expand into 12x12 for 3D beam (translations+rotations)
        x1, y1, z1 = self.node1.coords
        x2, y2, z2 = self.node2.coords
        gamma = rotation_matrix_3D(x1, y1, z1, x2, y2, z2, self.vtemp)
        T = transformation_matrix_3D(gamma)
        return T, gamma

    def compute_global_stiffness(self):
        """Return the 12x12 element stiffness matrix in global coordinates."""
        Ke_local = self.compute_local_stiffness()
        T = self.T
        # K_global = T^T * K_local * T
        return T.T @ Ke_local @ T
    
    def compute_global_geometry_stiffness(self):
        """Return the 12x12 element stiffness matrix in global coordinates."""
        Kg_local = self.compute_local_geometry_stiffness()
        T = self.T
        # K_global = T^T * K_local * T
        return T.T @ Kg_local @ T

    def interpolate(self):
        u_local = self.T @ np.array(self.node1.displacements + self.node2.displacements)
        x = np.linspace(0, self.L, 10)
        L = self.L
        xi = x / L  # 归一化坐标 (0 <= xi <= 1)
    
        # 提取节点自由度
        u_x1, u_y1, u_z1, theta_x1, theta_y1, theta_z1 = u_local[0:6]
        u_x2, u_y2, u_z2, theta_x2, theta_y2, theta_z2 = u_local[6:12]
        
        # --- 轴向位移 u_x (线性插值) ---
        N1 = 1 - xi
        N2 = xi
        u = N1 * u_x1 + N2 * u_x2
        
        # --- 横向位移 u_y (三次Hermite插值) ---
        H1 = 1 - 3*xi**2 + 2*xi**3
        H2 = L * xi * (1 - xi)**2
        H3 = 3*xi**2 - 2*xi**3
        H4 = -L * xi**2 * (1 - xi)
        v = H1 * u_y1 + H2 * theta_z1 + H3 * u_y2 + H4 * theta_z2
        
        # --- 横向位移 u_z (三次Hermite插值，注意符号) ---
        G1 = 1 - 3*xi**2 + 2*xi**3
        G2 = -L * xi * (1 - xi)**2  # 注意负号
        G3 = 3*xi**2 - 2*xi**3
        G4 = L * xi**2 * (1 - xi)   # 符号与H4相反
        w = G1 * u_z1 + G2 * theta_y1 + G3 * u_z2 + G4 * theta_y2
        displacement_local = np.vstack((u, v, w))
        displacement_global = np.linalg.inv(self.gamma) @ displacement_local

        return displacement_global

    def plot_internal_force(self):
        reaction = np.array(self.node1.forces + self.node2.forces)
        load = np.array(self.node1.loads + self.node2.loads)
        F_local = self.T @ (reaction + load)
        x = np.array([0, self.L])  # 局部坐标系x轴坐标
        fig, axs = plt.subplots(3, 2, figsize=(15, 10))
        fig.suptitle('Member Internal Forces and Moments in Local Coordinates', fontsize=14)
        
        N1, Vy1, Vz1 = F_local[0], F_local[1], F_local[2]
        Tx1, My1, Mz1 = F_local[3], F_local[4], F_local[5]

        N2, Vy2, Vz2 = F_local[6], F_local[7], F_local[8]
        Tx2, My2, Mz2 = F_local[9], F_local[10], F_local[11]

        # 绘制轴力
        axs[0,0].plot(x, [N1, N2], 'r-o', linewidth=2)
        axs[0,0].set_title('Axial Force (N)')
        axs[0,0].set_xlabel('Position along member')
        axs[0,0].grid(True)
        axs[0,0].axhline(0, color='black', linewidth=0.5)

        # 绘制剪力Y方向
        axs[0,1].plot(x, [Vy1, Vy2], 'b-o', linewidth=2)
        axs[0,1].set_title('Shear Force Y (Vy)')
        axs[0,1].set_xlabel('Position along member')
        axs[0,1].grid(True)
        axs[0,1].axhline(0, color='black', linewidth=0.5)

        # 绘制剪力Z方向
        axs[1,0].plot(x, [Vz1, Vz2], 'g-o', linewidth=2)
        axs[1,0].set_title('Shear Force Z (Vz)')
        axs[1,0].set_xlabel('Position along member')
        axs[1,0].grid(True)
        axs[1,0].axhline(0, color='black', linewidth=0.5)

        # 绘制扭矩
        axs[1,1].plot(x, [Tx1, Tx2], 'm-o', linewidth=2)
        axs[1,1].set_title('Torque (Tx)')
        axs[1,1].set_xlabel('Position along member')
        axs[1,1].grid(True)
        axs[1,1].axhline(0, color='black', linewidth=0.5)

        # 绘制弯矩Y方向
        axs[2,0].plot(x, [My1, My2], 'c-o', linewidth=2)
        axs[2,0].set_title('Bending Moment Y (My)')
        axs[2,0].set_xlabel('Position along member')
        axs[2,0].grid(True)
        axs[2,0].axhline(0, color='black', linewidth=0.5)

        # 绘制弯矩Z方向
        axs[2,1].plot(x, [Mz1, Mz2], 'y-o', linewidth=2)
        axs[2,1].set_title('Bending Moment Z (Mz)')
        axs[2,1].set_xlabel('Position along member')
        axs[2,1].grid(True)
        axs[2,1].axhline(0, color='black', linewidth=0.5)

        plt.tight_layout()
        plt.savefig('test.png', dpi=300)


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
        self.Ke = None
        self.Kg = None
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
        K = np.zeros((self.num_dofs, self.num_dofs))

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
                    K[dofs_element[i], dofs_element[j]] += Ke_global[i, j]

        return K
    

    def assemble_global_geometry_stiffness(self):
        """
        Build the global stiffness matrix by looping over elements.
        For each element, get K_global (12x12 for a 3D beam element),
        then place it into the big K matrix at the appropriate rows/columns.
        """
        # Initialize global stiffness matrix
        K = np.zeros((self.num_dofs, self.num_dofs))

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
            Ke_global = ele.compute_global_geometry_stiffness()

            # "Scatter" Ke_global into the master Kg
            for i in range(12):
                for j in range(12):
                    K[dofs_element[i], dofs_element[j]] += Ke_global[i, j]

        return K

    def assemble_external_force(self):
        """
        Build the global force vector (Fg) from external nodal loads.
        For each node, we add the node's loads to the corresponding
        global DOFs as defined by self.dof_map.
        """
        F = np.zeros(self.num_dofs)

        # Loop over all nodes
        for node_id, node in self.nodes.items():
            # Each node in 3D beams/frames has 6 DOFs: (u, v, w, rx, ry, rz)
            for local_dof in range(6):
                # Look up the global DOF index from the dof_map
                global_dof = self.dof_map[(node_id, local_dof)]
                F[global_dof] += node.loads[local_dof]

        return F
        
    def apply_boundary_conditions(self):
        """
        Modify the global stiffness matrix (self.Ke) and the global load vector (self.F)
        to account for fixed DOFs. This approach:
        - Zeros out the row and column of the fixed DOF in K,
        - Sets the diagonal to 1,
        - Sets the load at that DOF to 0,
        which enforces zero displacement at that DOF.
        """
        # Make copies (or work in-place) depending on your design
        Ke = self.Ke.copy()
        Kg = self.Kg.copy()
        F = self.Load.copy()
        constrain = []
        # Loop through all nodes
        for node_id, node in self.nodes.items():
            # For each local DOF (0..5), check if it's fixed
            for local_dof, is_fixed in enumerate(node.fixed_dofs):
                if is_fixed:
                    # Look up the global DOF index from self.dof_map
                    global_dof = self.dof_map[(node_id, local_dof)]
                    constrain.append(global_dof)

                    # 1) Zero out entire row and column
                    Ke[global_dof, :] = 0.0
                    Ke[:, global_dof] = 0.0

                    Kg[global_dof, :] = 0.0
                    Kg[:, global_dof] = 0.0

                    # 2) Set diagonal to 1
                    Ke[global_dof, global_dof] = 1.0
                    Kg[global_dof, global_dof] = 1.0

                    # 3) Set load to 0
                    F[global_dof] = 0.0

        return (Ke, F, Kg)


    def solve_u(self):
        """
        Solve the system of equations [K]{u} = {F} after assembling the
        global stiffness matrix and force vector, and applying boundary conditions.
        Then store the displacement results in each Node.
        """

        K, F = self.apply_boundary_conditions()[:2]

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
        F_total = self.Ke @ self.u  # shape = (num_dofs,)
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
        self.Ke = self.assemble_global_stiffness()
        self.Kg = self.assemble_global_geometry_stiffness()
        self.Load = self.assemble_external_force()
        self.u = self.solve_u()
        self.reaction = self.compute_reaction_force()

    def plot(self):
        for node_id, node in self.nodes.items():
            pass


    def solve_elastic_critical_load(self):
        Ke, _, Kg= self.apply_boundary_conditions()

        eigenvalues, eigenvectors = eigh(Ke, -Kg) 
        lambda_cr = eigenvalues[-1]  # 最小的正特征值

        return lambda_cr
