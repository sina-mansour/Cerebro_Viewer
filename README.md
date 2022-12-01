# Cerebro_Viewer
A pythonic 3D viewer to visualize and plot brains and neuroimaging data

![alt text](https://github.com/sina-mansour/Cerebro_Viewer/blob/main/static/images/screen.png?raw=true)

---

Cerebro aims to provide a solution to advance the currently available methods for visualization of neuroimaging and brain connectivity data.

## Installation

[Cerebro](https://pypi.org/project/Cerebro-Viewer/) is provided as a python package and requires Python 3. To install the package, simply run the following command:

`pip install Cerebro-Viewer`

---

## Motivation

Were you ever stuck when trying to visualize your study findings? Did you want to visualize different data formats and notice there’s no software to visualize them ALL in one place? Have you thought about a different way to present your findings but couldn’t find a tool to do it? Do you find limitations in current neuroimaging visualization software? And finally, did you ever want to generate nice brain visualizations within your script (without launching third-party software) and found the existing packages incapable/slow? Cerebro aims to provide a solution that tries to answer these needs.

---

## Development plan

Cerebro is currently under active development. The plan is to further develop exciting features through open-source contributions and discussions during hackathons, brainhacks, and other relevant opportunities.


### Brainhack Global 2022

In Brainhack Global 2022 we aim to:

1. Brainstorm different visualization approaches to come up with ideas.
2. List different file formats that Cerebro should be able to generate.
3. Try a hands-on session of visualizing some brains with Cerebro.
4. Submitting issues for feature requests.
5. And finally, if you like, contributing to the script.

**Contributor:** Niousha-Dehestani, Deakin University/University of Melbourne

**Contributor:** Nazanin Sheykh Andalibi, Western Sydney University

**Contributor:** Sara, Western Sydney

**Contributor:** Arush Arun

---

To try the package, run the following code after installation:

```python
from cerebro import cerebro_brain_utils as cbu
from cerebro import cerebro_brain_viewer as cbv

my_brain_viewer = cbv.Cerebro_brain_viewer()

# render a surface
surface = 'pial'
surface_model = my_brain_viewer.load_template_GIFTI_cortical_surface_models(surface)

cifti_space = my_brain_viewer.visualize_cifti_space()

# render data over surface
dscalar_file = cbu.get_data_file(f'templates/HCP/dscalars/hcp.gradients.dscalar.nii')
dscalar_layer = my_brain_viewer.add_cifti_dscalar_layer(dscalar_file=dscalar_file,)

# enter interactive view
my_brain_viewer.show()

```
