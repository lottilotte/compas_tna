from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import compas
import compas_rhino
import compas_tna

from compas.topology import mesh_flip_cycles

try:
    import rhinoscriptsyntax as rs
except ImportError:
    pass

from rhinoform import RhinoForm


__author__    = ['Tom Van Mele', ]
__copyright__ = 'Copyright 2016 - Block Research Group, ETH Zurich'
__license__   = 'MIT License'
__email__     = 'vanmelet@ethz.ch'


__all__ = []


class FormActions(object):

    # ==========================================================================
    # construct
    # ==========================================================================

    def form_from_obj(self):
        path = compas_rhino.select_file(folder=compas_tna.DATA, filter='OBJ files (*.obj)|*.obj||')
        if not path:
            return
        self.form = RhinoForm.from_obj(path)
        self.form.draw()

    def form_from_mesh(self):
        guid = compas_rhino.select_mesh()
        if not guid:
            return
        self.form = RhinoForm.from_mesh(guid)
        mesh_flip_cycles(self.form)
        self.form.draw()

    def form_from_surface(self):
        guid = compas_rhino.select_surface()
        if not guid:
            return
        self.form = RhinoForm.from_surface(guid)
        self.form.draw()

    def form_from_boundaries(self):
        pass

    def form_from_lines(self):
        guids = compas_rhino.select_lines()
        if not guids:
            return
        lines = compas_rhino.get_line_coordinates(guids)
        self.form = RhinoForm.from_lines(lines)
        self.form.draw()

    # ==========================================================================
    # update
    # ==========================================================================

    def form_redraw(self):
        self.form.draw()

    def form_update_attributes(self):
        compas_rhino.update_settings(self.form.attributes)

    def form_update_vertex_attributes(self):
        keys = self.form.select_vertices() or list(self.form.vertices())
        self.form.update_vertex_attributes(keys)

    def form_update_edge_attributes(self):
        keys = self.form.select_edges() or list(self.form.edges())
        self.form.update_edge_attributes(keys)

    def form_update_face_attributes(self):
        keys = self.form.select_faces() or list(self.form.faces())
        self.form.update_face_attributes(keys)

    def form_move_vertex(self):
        pass

    # ==========================================================================
    # select
    # ==========================================================================

    def form_select_parallel_edges(self):
        uv = self.form.select_edge()
        if not uv:
            return

        keys = self.form.get_parallel_edges(uv)
        self.form.highlight_edges(keys)

    def form_select_continuous_edges(self):
        uv = self.form.select_edge()
        if not uv:
            return

        keys = self.form.get_continuous_edges(uv)
        self.form.highlight_edges(keys)

    def form_select_boundary_vertices(self):
        keys = set(self.form.vertices_on_boundary())
        self.form.highlight_vertices(keys)

    def form_select_boundary_edges(self):
        keys = set(self.form.edges_on_boundary())
        self.form.highlight_edges(keys)

    def form_select_boundary_faces(self):
        keys = set(self.form.faces_on_boundary())
        self.form.highlight_faces(keys)

    # ==========================================================================
    # equilibrium
    # ==========================================================================

    def form_relax(self):
        keys = self.form.select_vertices()
        self.form.relax(keys)
        self.form.draw()

    def form_update_boundaries(self):
        boundaries = self.form.vertices_on_boundaries()
        exterior = boundaries[0]
        interior = boundaries[1:]
        self.form.update_exterior(exterior, feet=self.form.attributes['feet.mode'])
        self.form.update_interior(interior)


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    pass
