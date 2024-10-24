# Conversion of OpenMC Geometry Components to CAD

[![License](https://img.shields.io/badge/license-MIT-green)](https://opensource.org/licenses/MIT)


This project converts OpenMC models from XML files to a Cubit journal file for import as CAD geometry.

## Installation

To install this tool run:

```shell
python -m pip install git+https://github.com/openmc-dev/openmc_cad_adapter.git
```

This will make the `openmc_to_cad` script available for use. To convert an OpenMC model, run:

```shell
openmc_to_cad /path/to/openmc/model.xml
```

## Disclaimer

There has been no methodical V&V on this converter; use at your own risk!

<a name="deps"></a>
## Dependencies

  - [NumPy](https://numpy.org/)
  - [OpenMC](https://docs.openmc.org/en/stable/) (>= v0.14.0)

## Limitations

There are several known and many unknown limitations of this tool. It is in a
preliminary state and subject to considerable redesign, including the addition
of a backend for other CAD engines.

Specific Limitations:

  - general Cones are not handled
  - Torii are required to have a single minor radius, OpenMC allows for different minor radii orthogonal to the toroidal axis