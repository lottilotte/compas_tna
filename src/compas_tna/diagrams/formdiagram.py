from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import sys

from math import pi
from math import sin
from math import cos
from math import sqrt

import compas
import compas_tna

from compas.datastructures import Mesh
from compas.utilities import geometric_key
from compas.utilities import pairwise

from compas.datastructures.mesh.mesh import TPL

from compas.geometry import subtract_vectors_xy
from compas.geometry import add_vectors_xy
from compas.geometry import normalize_vector_xy
from compas.geometry import cross_vectors

if 'ironpython' in sys.version.lower():
    from compas.utilities import XFunc
    fd_numpy = XFunc('compas.numerical.fd.fd_numpy.fd_numpy', tmpdir=compas_tna.TEMP)

else:
    from compas.numerical.fd.fd_numpy import fd_numpy


__author__    = ['Tom Van Mele', ]
__copyright__ = 'Copyright 2014 - Block Research Group, ETH Zurich'
__license__   = 'MIT License'
__email__     = 'vanmelet@ethz.ch'


__all__ = ['FormDiagram', ]


class FormDiagram(Mesh):
    """"""

    def __init__(self):
        super(FormDiagram, self).__init__()
        self.default_vertex_attributes.update({
            'x'           : 0.0,
            'y'           : 0.0,
            'z'           : 0.0,
            'px'          : 0.0,
            'py'          : 0.0,
            'pz'          : 0.0,
            'sw'          : 0.0,
            't'           : 1.0,
            'is_anchor'   : False,
            'is_fixed'    : False,
            'is_external' : False,
            'rx'          : 0.0,
            'ry'          : 0.0,
            'rz'          : 0.0,
            'zT'          : 0.0
        })
        self.default_edge_attributes.update({
            'q'           : 1.0,
            'l'           : 0.0,
            'f'           : 0.0,
            'qmin'        : 1e-7,
            'qmax'        : 1e+7,
            'lmin'        : 1e-7,
            'lmax'        : 1e+7,
            'fmin'        : 1e-7,
            'fmax'        : 1e+7,
            'a'           : 0.0,
            'is_ind'      : False,
            'is_edge'     : True,
            'is_external' : False
        })
        self.default_face_attributes.update({
            'is_loaded': True
        })
        self.attributes.update({
            'name'                       : 'FormDiagram',
            'layer'                      : 'FormDiagram',
            'color.vertex'               : (255, 255, 255),
            'color.edge'                 : (0, 0, 0),
            'color.face'                 : (210, 210, 210),
            'color.vertex:is_anchor'     : (255, 0, 0),
            'color.vertex:is_fixed'      : (0, 0, 0),
            'color.vertex:is_supported'  : (255, 0, 0),
            'color.vertex:is_prescribed' : (0, 255, 0),
            'color.reaction'             : (0, 255, 0),
            'color.residual'             : (0, 255, 255),
            'color.load'                 : (0, 255, 0),
            'color.selfweight'           : (0, 0, 255),
            'color.force'                : (0, 0, 255),
            'scale.reaction'             : 1.0,
            'scale.residual'             : 1.0,
            'scale.load'                 : 1.0,
            'scale.force'                : 1.0,
            'scale.selfweight'           : 1.0,
            'tol.reaction'               : 1e-3,
            'tol.residual'               : 1e-3,
            'tol.load'                   : 1e-3,
            'tol.force'                  : 1e-3,
            'tol.selfweight'             : 1e-3,

            'density'                    : 1.0,

            'feet.scale'                 : 0.1,
            'feet.alpha'                 : 45,
            'feet.tol'                   : 0.1,
            'feet.mode'                  : 1,

            'AGS.k'                      : None,
            'AGS.m'                      : None,
        })

    @classmethod
    def from_lines(cls, lines, precision='3f'):
        """Construct a mesh object from a list of lines described by start and end point coordinates.

        Parameters
        ----------
        lines : list
            A list of pairs of point coordinates.
        delete_boundary_face : bool, optional
            The algorithm that finds the faces formed by the connected lines
            first finds the face *on the outside*. In most cases this face is not expected
            to be there. Therefore, there is the option to have it automatically deleted.
        precision: str, optional
            The precision of the geometric map that is used to connect the lines.

        Returns
        -------
        Mesh :
            A mesh object.

        See Also
        --------
        * :func:`compas.datastructures.network_find_faces`
        * :func:`compas.datastructures.FaceNetwork`
        * :meth:`from_vertices_and_faces`

        Examples
        --------
        >>> import compas
        >>> from compas.datastructures import Mesh
        >>> mesh = Mesh.from_obj(compas.get('bunny.ply'))

        """
        from compas.topology import network_find_faces
        from compas.datastructures import Network

        network = Network.from_lines(lines, precision=precision)

        mesh = cls()

        for key, attr in network.vertices(True):
            mesh.add_vertex(key, x=attr['x'], y=attr['y'], z=0)

        mesh.halfedge = network.halfedge

        network_find_faces(mesh, breakpoints=mesh.leaves())

        return mesh

    def __str__(self):
        """Compile a summary of the mesh."""
        numv = self.number_of_vertices()
        nume = len(list(self.edges_where({'is_edge': True})))
        numf = self.number_of_faces()

        vmin = self.vertex_min_degree()
        vmax = self.vertex_max_degree()
        fmin = self.face_min_degree()
        fmax = self.face_max_degree()

        return TPL.format(self.name, numv, nume, numf, vmin, vmax, fmin, fmax)

    def uv_index(self):
        """Returns a dictionary that maps edge keys (i.e. pairs of vertex keys)
        to the corresponding edge index in a list or array of edges.

        Returns
        -------
        dict
            A dictionary of uv-index pairs.

        See Also
        --------
        * :meth:`index_uv`

        """
        return {(u, v): index for index, (u, v) in enumerate(self.edges_where({'is_edge': True}))}

    def index_uv(self):
        """Returns a dictionary that maps edges in a list to the corresponding
        vertex key pairs.

        Returns
        -------
        dict
            A dictionary of index-uv pairs.

        See Also
        --------
        * :meth:`uv_index`

        """
        return dict(enumerate(self.edges_where({'is_edge': True})))

    # --------------------------------------------------------------------------
    # dual and reciprocal
    # --------------------------------------------------------------------------

    def dual(self, cls):
        # be more explicit with the boundary functions
        dual = cls()
        fkey_centroid = {fkey: self.face_centroid(fkey) for fkey in self.faces()}
        outer = self.vertices_on_boundary()
        inner = list(set(self.vertices()) - set(outer))
        vertices = {}
        faces = {}
        for key in inner:
            fkeys = self.vertex_faces(key, ordered=True)
            for fkey in fkeys:
                if fkey not in vertices:
                    vertices[fkey] = fkey_centroid[fkey]
            faces[key] = fkeys
        for key, (x, y, z) in vertices.items():
            dual.add_vertex(key, x=x, y=y, z=z)
        for fkey, vertices in faces.items():
            dual.add_face(vertices, fkey=fkey)
        return dual

    def find_faces(self):
        # add planarity check
        # rename finding faces function
        # add a mesh from lines function
        # add mesh from network
        # mesh from mesh
        # network from mesh
        # network from network
        from compas.topology import network_find_faces
        from compas.datastructures import Network
        network = Network.from_lines(lines, precision=precision)
        mesh = cls()
        for key, attr in network.vertices(True):
            mesh.add_vertex(key, x=attr['x'], y=attr['y'], z=0)
        mesh.halfedge = network.halfedge
        network_find_faces(mesh, breakpoints=mesh.leaves())

    # --------------------------------------------------------------------------
    # vertices
    # --------------------------------------------------------------------------

    def leaves(self):
        # consistent use of iterators and generators v lists
        return self.vertices_where({'vertex_degree': 1})

    def corners(self):
        # consistent use of iterators and generators v lists
        return self.vertices_where({'vertex_degree': 2})

    # --------------------------------------------------------------------------
    # edges
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # faces
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # helpers
    # --------------------------------------------------------------------------

    def anchors(self):
        # consistent use of iterators and generators v lists
        return [key for key, attr in self.vertices(True) if attr['is_anchor']]

    def fixed(self):
        # consistent use of iterators and generators v lists
        return [key for key, attr in self.vertices(True) if attr['is_fixed']]

    def residual(self):
        # there is a discrepancy between the norm of residuals calculated by the equilibrium functions`
        # and the result found here
        R = 0
        for key, attr in self.vertices_where({'is_anchor': False, 'is_fixed': False}, True):
            rx, ry, rz = attr['rx'], attr['ry'], attr['rz']
            R += sqrt(rx ** 2 + ry ** 2 + rz ** 2)
        return R

    def bbox(self):
        x, y, z = zip(* self.get_vertices_attributes('xyz'))
        return (min(x), min(y), min(z)), (max(x), max(y), max(z))

    # --------------------------------------------------------------------------
    # boundary
    # --------------------------------------------------------------------------

    def vertices_on_boundaries(self):
        """Find the vertices on the boundary.

        Parameters
        ----------
        ordered : bool, optional
            If ``True``, Return the vertices in the same order as they are found on the boundary.
            Default is ``False``.

        Returns
        -------
        list
            The vertices of the boundary.

        Warning
        -------
        If the vertices are requested in order, and the mesh has multiple borders,
        currently only the vertices of one of the borders will be returned.

        Examples
        --------
        >>>

        """
        vertices_set = set()
        for key, nbrs in iter(self.halfedge.items()):
            for nbr, face in iter(nbrs.items()):
                if face is None:
                    vertices_set.add(key)
                    vertices_set.add(nbr)
        vertices_all = list(vertices_set)

        boundaries = []
        key = sorted([(key, self.vertex_coordinates(key)) for key in vertices_all], key=lambda x: (x[1][1], x[1][0]))[0][0]

        while vertices_all:
            vertices = []
            start = key

            while 1:
                for nbr, fkey in iter(self.halfedge[key].items()):
                    if fkey is None:
                        vertices.append(nbr)
                        key = nbr
                        break

                if key == start:
                    boundaries.append(vertices)
                    vertices_all = [x for x in vertices_all if x not in vertices]
                    break

            if vertices_all:
                key = vertices_all[0]            

        return boundaries

    def is_boundary_convex(self, boundary):
        # return dict with key => True/False
        # return dict with key => cross_z
        pass

    # --------------------------------------------------------------------------
    # postprocess
    # --------------------------------------------------------------------------

    def collapse_small_edges(self, tol=1e-2):
        boundaries = self.vertices_on_boundaries()
        for boundary in boundaries:
            for u, v in pairwise(boundary):
                l = self.edge_length(u, v)
                if l < tol:
                    self.collapse_edge(v, u, t=0.5, allow_boundary=True)

    def relax(self, fixed):
        key_index = self.key_index()
        vertices = self.get_vertices_attributes('xyz')
        edges = list(self.edges_where({'is_edge': True}))
        edges = [(key_index[u], key_index[v]) for u, v in edges]
        fixed = list(fixed)
        fixed = [key_index[key] for key in fixed]
        qs = [self.get_edge_attribute(uv, 'q') for uv in edges]
        loads = self.get_vertices_attributes(('px', 'py', 'pz'), (0, 0, 0))
        xyz, q, f, l, r = fd_numpy(vertices, edges, fixed, qs, loads)
        for key, attr in self.vertices(True):
            index = key_index[key]
            attr['x'] = xyz[index][0]
            attr['y'] = xyz[index][1]
            attr['z'] = xyz[index][2]

    def smooth_interior(self):
        pass

    # --------------------------------------------------------------------------
    # boundary conditions
    # --------------------------------------------------------------------------

    def set_anchors(self, points=None, degree=0, keys=None):
        if points:
            xyz_key = self.key_xyz()
            for xyz in points:
                gkey = geometric_key(xyz)
                if gkey in xyz_key:
                    key = xyz_key[gkey]
                    self.set_vertex_attribute(key, 'is_anchor', True)
        if degree:
            for key in self.vertices():
                if self.vertex_degree(key) <= degree:
                    self.set_vertex_attribute(key, 'is_anchor', True)
        if keys:
            self.set_vertices_attribute('is_anchor', True, keys=keys)

    def update_boundaries(self, feet=1):
        boundaries = self.vertices_on_boundaries()
        exterior = boundaries[0]
        interior = boundaries[1:]
        self.update_exterior(exterior, feet=feet)
        self.update_interior(interior)

    def update_exterior(self, boundary, feet=2):
        """"""
        segments = self.split_boundary(boundary)
        if not feet:
            for vertices in segments:
                if len(vertices) > 2:
                    self.add_face(vertices, is_loaded=False)
                    u = vertices[-1]
                    v = vertices[0]
                    self.set_edge_attribute((u, v), 'is_edge', False)
                else:
                    u, v = vertices
                    self.set_edge_attribute((u, v), 'is_edge', False)
        else:
            self.add_feet(segments, feet=feet)

    def update_interior(self, boundaries):
        """"""
        for vertices in boundaries:
            self.add_face(vertices, is_loaded=False)

    def split_boundary(self, boundary):
        """"""
        segment = []
        segments = [segment]
        for key in boundary:
            segment.append(key)
            if self.vertex[key]['is_anchor']:
                segment = [key]
                segments.append(segment)
        segments[-1] += segments[0]
        del segments[0]
        return segments

    def add_feet(self, segments, feet=2):
        """"""
        def rotate(point, angle):
            x = cos(angle) * point[0] - sin(angle) * point[1]
            y = sin(angle) * point[0] + cos(angle) * point[1]
            return x, y, 0

        def cross_z(ab, ac):
            return ab[0] * ac[1] - ab[1] * ac[0]

        scale = self.attributes['feet.scale']
        alpha = self.attributes['feet.alpha'] * pi / 180
        tol   = self.attributes['feet.tol']

        key_foot = {}
        key_xy = {key: self.vertex_coordinates(key, 'xy') for key in self.vertices()}

        for i, vertices in enumerate(segments):
            key = vertices[0]
            after = vertices[1]
            before = segments[i - 1][-2]

            b = key_xy[before]
            o = key_xy[key]
            a = key_xy[after]

            ob = normalize_vector_xy(subtract_vectors_xy(b, o))
            oa = normalize_vector_xy(subtract_vectors_xy(a, o))

            z = cross_z(ob, oa)

            if z > +tol:
                r = normalize_vector_xy(add_vectors_xy(oa, ob))
                r = [-scale * axis for axis in r]

            elif z < -tol:
                r = normalize_vector_xy(add_vectors_xy(oa, ob))
                r = [+scale * axis for axis in r]

            else:
                ba = normalize_vector_xy(subtract_vectors_xy(a, b))
                r = cross_vectors([0, 0, 1], ba)
                r = [+scale * axis for axis in r]

            if feet == 1:
                x, y, z = add_vectors_xy(o, r)
                m = self.add_vertex(x=x, y=y, z=0, is_fixed=True, is_external=True)
                key_foot[key] = m

            elif feet == 2:
                lx, ly, lz = add_vectors_xy(o, rotate(r, +alpha))
                rx, ry, rz = add_vectors_xy(o, rotate(r, -alpha))
                l = self.add_vertex(x=lx, y=ly, z=0, is_fixed=True, is_external=True)
                r = self.add_vertex(x=rx, y=ry, z=0, is_fixed=True, is_external=True)
                key_foot[key] = l, r

            else:
                pass

        for vertices in segments:
            l = vertices[0]
            r = vertices[-1]

            if feet == 1:
                lm = key_foot[l]
                rm = key_foot[r]
                self.add_face([lm] + vertices + [rm], is_loaded=False)
                self.set_edge_attribute((l, lm), 'is_external', True)
                self.set_edge_attribute((rm, lm), 'is_edge', False)

            elif feet == 2:
                lb = key_foot[l][0]
                la = key_foot[l][1]
                rb = key_foot[r][0]
                self.add_face([lb, l, la], is_loaded=False)
                self.add_face([la] + vertices + [rb], is_loaded=False)
                self.set_edge_attribute((l, lb), 'is_external', True)
                self.set_edge_attribute((l, la), 'is_external', True)
                self.set_edge_attribute((lb, la), 'is_edge', False)
                self.set_edge_attribute((la, rb), 'is_edge', False)
            
            else:
                pass


# ==============================================================================
# Main
# ==============================================================================

if __name__ == '__main__':

    pass
