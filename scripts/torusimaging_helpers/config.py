import pathlib

import agama
import astropy.coordinates as coord
import astropy.units as u
import gala.potential as gp
import jax
from gala.units import galactic

jax.config.update("jax_enable_x64", True)

this_path = pathlib.Path(__file__).absolute().parent
base_path = this_path.parent.parent
data_path = base_path / "data"
cache_path = base_path / ".cache"

agama.setUnits(mass=1.0, length=1.0, time=1.0)


gala_pot = gp.CCompositePotential(
    disk=gp.MiyamotoNagaiPotential(m=6.91e10, a=3, b=0.25, units=galactic),
    halo=gp.NFWPotential(m=5.4e11, r_s=15.0, units=galactic),
)

agama_pot = agama.Potential(
    dict(
        type="miyamotonagai",
        mass=gala_pot["disk"].parameters["m"].value,
        scaleradius=gala_pot["disk"].parameters["a"].value,
        scaleheight=gala_pot["disk"].parameters["b"].value,
    ),
    dict(
        type="nfw",
        mass=gala_pot["halo"].parameters["m"].value,
        scaleradius=gala_pot["halo"].parameters["r_s"].value,
    ),
)

R0 = 8.275 * u.kpc
vc0 = 229 * u.km / u.s

galcen_frame = coord.Galactocentric(
    galcen_distance=R0, galcen_v_sun=[8.4, 251.8, 8.4] * u.km / u.s
)
