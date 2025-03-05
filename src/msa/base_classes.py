import numpy as np
from .functions import * 
import matplotlib.pyplot as plt
from scipy.linalg import eig

class Node:
    """
    Represents a node in the structure with its coordinates, 
    degrees of freedom (DOFs), loads, displacements, and reaction forces.
    
    Attributes:
        id (int): Unique identifier for the node.
        coords (tuple): Coordinates (x, y, z) of the node.
        fixed_dofs (list of bool): Flags for each DOF to indicate if it is fixed (default: all False).
        loads (list of float): External loads applied at the node (6 DOFs).
        displacements (list of float): Computed displacements for each DOF.
        reforces (list of float): Reaction forces at the node.
        F_local (list of float): Local force vector for internal force calculations.
        geo_disp (list of float): Geometric displacements (e.g., for buckling analysis).
    """
    def __init__(self, id, x, y, z):
        self.id = id
        self.coords = (x, y, z)
        self.fixed_dofs = [False]*6
        self.loads = [0.0]*6
        self.displacements = [0.0]*6
        self.reforces = [0.0]*6
        self.F_local = [0.0]*6
        self.geo_disp = [0.0]*6

class Material:
    """
    Defines the material properties used in the stiffness calculations.
    
    Attributes:
        id (int): Unique identifier for the material.
        E (float): Young's modulus.
        G (float): Shear modulus (default: 0).
        nu (float): Poisson's ratio (default: 0).
        density (float): Material density (default: 0.0).
    """
    def __init__(self, id, E, G=0, nu=0, density=0.0):
        self.id = id
        self.E = E
        self.G = G
        self.nu = nu
        self.density = density

class Section:
    """
    Contains the cross-sectional properties of the element.
    
    Attributes:
        id (int): Unique identifier for the section.
        A (float): Cross-sectional area.
        Iy (float): Moment of inertia about the local y-axis.
        Iz (float): Moment of inertia about the local z-axis.
        Ir (float): Warping constant (if applicable).
        J (float): Torsional constant.
    """
    def __init__(self, id, A, Iy=0, Iz=0, Ir=0, J=0):
        self.id = id
        self.A = A
        self.Iy = Iy
        self.Iz = Iz
        self.Ir = Ir
        self.J = J

class Element:
    """
    Represents a 3D beam or truss element connecting two nodes.
    Computes element length, local and global stiffness matrices, and transformation matrices.
    
    Attributes:
        id (int): Unique identifier for the element.
        node1, node2 (Node): The two nodes connected by the element.
        L (float): Length of the element computed from node coordinates.
        material (Material): Associated material.
        section (Section): Associated section.
        vtemp (optional): Vector to help define the local coordinate system orientation.
        T (numpy.ndarray): 12x12 transformation matrix from local to global coordinates.
        Ke_local (numpy.ndarray): 12x12 local stiffness matrix.
    """
    def __init__(self, id, node1, node2, material, section, vtemp=None):
        self.id = id
        self.node1 = node1
        self.node2 = node2
        self.L = self.compute_length()
        self.material = material
        self.section = section
        self.vtemp = vtemp
        self.T = self.compute_transformation_matrix()
        self.Ke_local = self.compute_local_stiffness()
        
    def compute_length(self):
        """
        Compute the length of the element based on node coordinates.
        """
        x1, y1, z1 = self.node1.coords
        x2, y2, z2 = self.node2.coords
        length = ((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)**0.5
        return length

    def compute_local_stiffness(self):
        """
        Compute the 12x12 local stiffness matrix using the material and section properties.
        Relies on the external function: local_elastic_stiffness_matrix_3D_beam.
        """
        ke_local = local_elastic_stiffness_matrix_3D_beam(
            self.material.E, 
            self.material.nu, 
            self.section.A, 
            self.L, 
            self.section.Iy, 
            self.section.Iz, 
            self.section.J
        )
        return ke_local
    
    def compute_local_geometry_stiffness(self):
        """
        Compute the local geometric stiffness matrix (useful for nonlinear buckling analysis).
        Relies on the external function: local_geometric_stiffness_matrix_3D_beam.
        """
        kg_local = local_geometric_stiffness_matrix_3D_beam(
            self.L, 
            self.section.A, 
            self.section.Ir, 
            self.node2.F_local[0],
            self.node2.F_local[3], 
            self.node1.F_local[4], 
            self.node1.F_local[5],
            self.node2.F_local[4], 
            self.node2.F_local[5]
        )
        return kg_local
        
    def compute_transformation_matrix(self):
        """
        Compute the 12x12 transformation matrix to convert local stiffness to global coordinates.
        
        Steps:
          1. Calculate direction cosines from node1 to node2.
          2. Build a 3x3 rotation matrix using the external function rotation_matrix_3D.
          3. Expand the rotation matrix into a 12x12 transformation matrix using transformation_matrix_3D.
        """
        x1, y1, z1 = self.node1.coords
        x2, y2, z2 = self.node2.coords
        gamma = rotation_matrix_3D(x1, y1, z1, x2, y2, z2, self.vtemp)
        T = transformation_matrix_3D(gamma)
        return T

    def compute_global_stiffness(self, type='linear'):
        """
        Compute the global stiffness matrix for the element by transforming the local stiffness matrix.
        
        Args:
            type (str): 'linear' for the elastic stiffness matrix or another string for geometric stiffness.
            
        Returns:
            numpy.ndarray: 12x12 global stiffness matrix.
        """
        T = self.T
        if type == 'linear':
            K_local = self.Ke_local
        else:
            K_local = self.compute_local_geometry_stiffness()
        return T.T @ K_local @ T
        
    def get_local_displacement(self):
        """
        Transform the global displacements of the connected nodes into the local coordinate system.
        
        Returns:
            numpy.ndarray: Local displacement vector (12 elements).
        """
        u_global = np.array(self.node1.displacements + self.node2.displacements)
        u_local = self.T @ u_global
        return u_local
    
    def assign_local_internal_force(self):
        """
        Compute and assign the local internal force vector based on the local stiffness matrix 
        and local displacements. The forces are distributed to the connected nodes.
        """
        u_local = self.get_local_displacement()
        F_local = self.Ke_local @ u_local
        self.node1.F_local = F_local[:6]
        self.node2.F_local = F_local[6:]


class Structure:
    """
    Represents the entire structure, managing nodes, elements, materials, and sections.
    Responsible for assembling global matrices, applying boundary conditions, solving the system,
    and performing post-processing such as visualization and buckling analysis.
    
    Attributes:
        nodes (dict): Maps node IDs to Node objects.
        elements (dict): Maps element IDs to Element objects.
        materials (dict): Maps material IDs to Material objects.
        sections (dict): Maps section IDs to Section objects.
        dof_map (dict): Maps (node_id, local_dof) to global DOF indices.
        num_dofs (int): Total number of degrees of freedom in the structure.
        Ke (numpy.ndarray): Global elastic stiffness matrix.
        Kg (numpy.ndarray): Global geometric stiffness matrix.
        Load (numpy.ndarray): Global load vector.
        u (numpy.ndarray): Global displacement vector.
        reaction (numpy.ndarray): Reaction forces computed at fixed DOFs.
        fixed_dofs (list): Global DOF indices that are fixed.
        free_dofs (list): Global DOF indices that are free.
    """
    def __init__(self):
        self.nodes = {}
        self.elements = {}
        self.materials = {}
        self.sections = {}
        self.dof_map = {}  # Maps (node_id, local_dof) -> global_dof
        self.num_dofs = 0  # Total degrees of freedom in the structure
        self.Ke = None
        self.Kg = None
        self.Load = None
        self.u = None
        self.reaction = None


    def add_node(self, node):
        """Add a Node object to the structure."""
        self.nodes[node.id] = node

    def add_material(self, material):
        """Add a Material object to the structure."""
        self.materials[material.id] = material

    def add_section(self, section):
        """Add a Section object to the structure."""
        self.sections[section.id] = section

    def add_element(self, element):
        """Add an Element object to the structure."""
        self.elements[element.id] = element

    def assign_dof_numbers(self):
        """
        Assign global DOF numbers to each node's local DOFs.
        Populates the dof_map and classifies DOFs as fixed or free.
        """
        dof_counter = 0
        # Iterate in ascending order of node_id for consistency.
        for node_id, node in sorted(self.nodes.items()):
            for local_dof in range(6):  # Each node has 6 DOFs: [u, v, w, rx, ry, rz]
                self.dof_map[(node_id, local_dof)] = dof_counter
                dof_counter += 1

        self.num_dofs = dof_counter

    def assemble_global_stiffness(self, type='linear'):
        """
        Assemble the global stiffness matrix by summing contributions from each element.
        
        Args:
            type (str): 'linear' for the elastic stiffness or another value for geometric stiffness.
            
        Returns:
            numpy.ndarray: The assembled global stiffness matrix.
        """
        K = np.zeros((self.num_dofs, self.num_dofs))
        # Loop over each element.
        for ele_id, ele in self.elements.items():
            # Get the global DOF indices for the element (node1 + node2).
            dofs_element = []
            for local_dof in range(6):  # DOFs for node1
                dofs_element.append(self.dof_map[(ele.node1.id, local_dof)])
            for local_dof in range(6):  # DOFs for node2
                dofs_element.append(self.dof_map[(ele.node2.id, local_dof)])
            # Get the element stiffness matrix in global coordinates.
            if type == 'linear':
                K_ele_global = ele.compute_global_stiffness()
            else:
                ele.assign_local_internal_force()
                K_ele_global = ele.compute_global_stiffness(type='geometry')
            # Scatter the element stiffness into the global stiffness matrix.
            for i in range(12):
                for j in range(12):
                    K[dofs_element[i], dofs_element[j]] += K_ele_global[i, j]
        return K
    
    def assemble_external_force(self):
        """
        Assemble the global external force vector from nodal loads.
        
        Returns:
            numpy.ndarray: The assembled load vector.
        """
        F = np.zeros(self.num_dofs)
        for node_id, node in self.nodes.items():
            for local_dof in range(6):
                global_dof = self.dof_map[(node_id, local_dof)]
                F[global_dof] += node.loads[local_dof]
        return F
    
    def apply_boundary_conditions(self, global_stiffness_matrix):
        """
        Apply boundary conditions by modifying the global stiffness matrix and load vector.
        Rows and columns corresponding to fixed DOFs are zeroed out and then set on the diagonal.
        
        Args:
            global_stiffness_matrix (numpy.ndarray): The original global stiffness matrix.
            
        Returns:
            tuple: Modified stiffness matrix and load vector.
        """
        K = global_stiffness_matrix.copy()
        F = self.Load.copy()
        for node_id, node in self.nodes.items():
            for local_dof, is_fixed in enumerate(node.fixed_dofs):
                if is_fixed:
                    global_dof = self.dof_map[(node_id, local_dof)]
                    # Zero out the row and column.
                    K[global_dof, :] = 0.0
                    K[:, global_dof] = 0.0
                    # Set the diagonal to 1.
                    K[global_dof, global_dof] = 1.0
                    # Set the corresponding load to 0.
                    F[global_dof] = 0.0
        return K, F
    
    def solve_u(self):
        """
        Solve the system of equations for nodal displacements.
        Updates the displacements in each node and returns the global displacement vector.
        
        Raises:
            ValueError: If the global stiffness matrix is singular or ill-conditioned.
            
        Returns:
            numpy.ndarray: The computed global displacement vector.
        """
        K, F = self.apply_boundary_conditions(self.Ke)
        try:
            u = np.linalg.solve(K, F)
        except np.linalg.LinAlgError as e:
            raise ValueError("Global stiffness matrix is singular or ill-conditioned. "
                             "Check boundary conditions.") from e
        # Store the computed displacements in each node.
        for node_id, node in self.nodes.items():
            for local_dof in range(6):
                global_dof = self.dof_map[(node_id, local_dof)]
                node.displacements[local_dof] = u[global_dof]
        return u
    
    def compute_reaction_force(self):
        """
        Compute the reaction forces at each node after solving for displacements.
        The reaction is calculated as the difference between total nodal force (K*u) and the applied load.
        Stores the reaction forces in each node.
        
        Returns:
            numpy.ndarray: Reaction forces at all DOFs.
        """
        F_total = self.Ke @ self.u  # Total nodal forces.
        reactions = F_total - self.Load
        for node_id, node in self.nodes.items():
            for local_dof in range(6):
                global_dof = self.dof_map[(node_id, local_dof)]
                node.reforces[local_dof] = reactions[global_dof]
        return reactions
    
    def linear_analyze(self):
        """
        High-level function to perform linear structural analysis.
        
        Steps:
          1. Assign DOF numbers.
          2. Assemble global stiffness matrix (Ke) and load vector.
          3. Solve for nodal displacements.
          4. Assemble global geometric stiffness matrix (Kg) for potential buckling analysis.
          5. Compute reaction forces.
        """
        self.assign_dof_numbers()
        self.Ke = self.assemble_global_stiffness()
        self.Load = self.assemble_external_force()
        self.u = self.solve_u()
        self.Kg = self.assemble_global_stiffness(type='geometry')
        self.reaction = self.compute_reaction_force()
    
    def compute_elastic_critical_load(self):
        """
        Compute the elastic critical load (buckling load) via eigenvalue analysis.
        The generalized eigenvalue problem is solved with the global elastic stiffness (Ke)
        and geometric stiffness (Kg) matrices. The smallest positive eigenvalue is selected
        as the critical load factor.
        
        Returns:
            tuple: (lambda_cr, normalized buckling mode shape)
        """
        Kg_bc, _ = self.apply_boundary_conditions(self.Kg)
        Ke_bc, _ = self.apply_boundary_conditions(self.Ke)
        eigenvalues, eigenvectors = eig(Ke_bc, -Kg_bc)
        eigenvalues_real = np.real(eigenvalues)
        # Sort eigenvalues and corresponding eigenvectors.
        idx_sorted = np.argsort(eigenvalues_real)
        sorted_evals = eigenvalues_real[idx_sorted]
        sorted_evecs = eigenvectors[:, idx_sorted]
        # Filter out near-zero or negative eigenvalues.
        lambda_cr_candidates = [val for val in sorted_evals if val > 1e-8]
        if len(lambda_cr_candidates) == 0:
            print("No positive eigenvalues found! Buckling might not occur in the expected range.")
            return 0
        else:
            lambda_cr = lambda_cr_candidates[0]
            mode_shape = np.real(sorted_evecs[:, np.where(sorted_evals == lambda_cr)[0][0]])
        geo_disp = mode_shape / np.max(np.abs(mode_shape))
        # Store normalized displacements (buckling mode shape) in each node.
        for node_id, node in self.nodes.items():
            for local_dof in range(6):
                global_dof = self.dof_map[(node_id, local_dof)]
                node.geo_disp[local_dof] = geo_disp[global_dof]
        return lambda_cr, mode_shape / np.max(np.abs(mode_shape))
    
    def post_processing(self):
        """
        Visualize the structure and its deformed configuration in 3D.
        
        For each element:
          - Plots the original (undeformed) line between node coordinates.
          - Interpolates displacements along the element using linear interpolation (axial) 
            and cubic Hermite interpolation (transverse) for smooth curves.
          - Transforms local displacements into global coordinates.
          - Plots the deformed shape.
        
        The final plot is saved as 'test1.png'.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for _, ele in self.elements.items():
            # print(f'---------- element {_}')
            start = ele.node1.coords
            end = ele.node2.coords
            # Plot the original undeformed element.
            ax.plot([start[0], end[0]], 
                    [start[1], end[1]], 
                    [start[2], end[2]],  
                    linestyle='--', markersize=6, color='black')
            
            # Compute local geometric displacements.
            geo_disp_local = ele.T @ (np.array(ele.node1.geo_disp + ele.node2.geo_disp))*5
            # print('local geometry displacement')
            # print(geo_disp_local)
            
            L = ele.L
            num_points = 20
            x = np.linspace(0, L, num_points)  # Interpolating points along the element.
            xi = x / L  # Normalized coordinates (0 <= xi <= 1)
            # Extract local displacements and rotations from the geometric displacement vector.
            u_x1, u_y1, u_z1, theta_x1, theta_y1, theta_z1 = geo_disp_local[0:6]
            u_x2, u_y2, u_z2, theta_x2, theta_y2, theta_z2 = geo_disp_local[6:12]
            
            # Axial displacement (linear interpolation).
            N1 = 1 - xi
            N2 = xi
            u = N1 * u_x1 + N2 * u_x2
            # print('u:',u)
            # Transverse displacement u_y (cubic Hermite interpolation).
            H1 = 1 - 3*xi**2 + 2*xi**3
            H2 = L * xi * (1 - xi)**2
            H3 = 3*xi**2 - 2*xi**3
            H4 = L * xi * (xi**2 - xi)
            v = H1 * u_y1 + H2 * theta_z1 + H3 * u_y2 + H4 * theta_z2
            # print('v:',v)
            # Transverse displacement u_z (cubic Hermite interpolation with scaling).
            G1 = 1 - 3*xi**2 + 2*xi**3
            G2 = L * xi * (1 - xi)**2
            G3 = 3*xi**2 - 2*xi**3
            G4 = L * xi * (xi**2 - xi)
            w = (G1 * u_z1 + G2 * theta_y1 + G3 * u_z2 + G4 * theta_y2) 
            # print('w:',w)
            # Stack the local displacement components.
            local_dis = np.vstack((u, v, w))
            # Transform local displacements to global coordinates.
            global_dis = np.linalg.inv(ele.T[:3, :3]) @ local_dis
            # print(global_dis)
            # Generate global coordinates along the original element line.
            global_coord = np.linspace(np.array(start), np.array(end), num_points).T
            deform_global_coord = global_coord + global_dis
            
            # Plot the deformed shape.
            ax.plot(deform_global_coord[0], deform_global_coord[1], deform_global_coord[2], color='darkred')
            
        plt.savefig('deformed_shape.png')
