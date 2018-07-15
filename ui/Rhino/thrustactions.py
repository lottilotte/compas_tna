from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import compas
import compas_rhino
import compas_tna

from compas_tna.rhino import FormArtist


__author__    = ['Tom Van Mele', ]
__copyright__ = 'Copyright 2016 - Block Research Group, ETH Zurich'
__license__   = 'MIT License'
__email__     = 'vanmelet@ethz.ch'


__all__ = []


class ThrustActions(object):

    # ==========================================================================
    # visualise
    # ==========================================================================

    def form_show_reactions(self):
        artist = FormArtist(self.form, layer=self.form.attributes['layer'])
        artist.draw_reactions()
        artist.redraw()

    def form_hide_reactions(self):
        artist = FormArtist(self.form, layer=self.form.attributes['layer'])
        artist.clear_reactions()
        artist.redraw()

    def form_show_residuals(self):
        artist = FormArtist(self.form, layer=self.form.attributes['layer'])
        artist.draw_residuals()
        artist.redraw()

    def form_hide_residuals(self):
        artist = FormArtist(self.form, layer=self.form.attributes['layer'])
        artist.clear_residuals()
        artist.redraw()

    def form_show_loads(self):
        artist = FormArtist(self.form, layer=self.form.attributes['layer'])
        artist.draw_loads()
        artist.redraw()

    def form_hide_loads(self):
        artist = FormArtist(self.form, layer=self.form.attributes['layer'])
        artist.clear_loads()
        artist.redraw()

    def form_show_selfweight(self):
        artist = FormArtist(self.form, layer=self.form.attributes['layer'])
        artist.draw_selfweight()
        artist.redraw()

    def form_hide_selfweight(self):
        artist = FormArtist(self.form, layer=self.form.attributes['layer'])
        artist.clear_selfweight()
        artist.redraw()

    def form_show_forces(self):
        artist = FormArtist(self.form, layer=self.form.attributes['layer'])
        artist.draw_forces()
        artist.redraw()

    def form_hide_forces(self):
        artist = FormArtist(self.form, layer=self.form.attributes['layer'])
        artist.clear_forces()
        artist.redraw()


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    pass
