# Machine-Learning assisted classification of diffraction images
## Overview

With the recent advent of short wavelength Free-Electron-Laser (FEL)
it is possible to obtain high-intensity x-ray pulses with femtosecond
duration. This allows for coherent diffraction imaging (CDI) experiments
on individual nanosized objects with a single x-ray laser shot.

At the “LINAC Coherent Light Source” (LCLS), for instance, with a
repetition rate of 120 Hz a typical hit-ratio of 20 % can be achieved [^1] [^2] .
The European XFEL facility will even top that with a maximum repetition
rate of 27 000 Hz [^3]. This may add up to several million diffraction pattern
in a single 12 hour shift. While storing such large amounts of data
is feasible, this mere data volume represents a severe problem
for the data analysis.

We here propose a workflow scheme to drastically reduce the amount
of work needed for the categorization of large data-sets of diffraction
patterns. First a classification and viewer is used for classifying
manually selected high quality diffraction pattern. These patterns are
then used as training data for a Residual Convolutional Neural Network
(RCNN). The RCNN is designed for the classification of data for
efficient indexing and subsequent analysis.

[^1]: *Emma, et al. First lasing and operation of an ångstrom-wavelength free-electron laser. Nat. Photonics, 4(9):641–647, 2010. ISSN 1749-4885. doi: 10.1038/nphoton.2010.176.*

[^2]: *Bostedt, et al. Linac Coherent Light Source: The first five years. Rev. Mod. Phys., 88(1):015007, 2016. ISSN 0034-6861. doi: 10.1103/RevModPhys.88.015007.*

[^3]: *Schneidmiller. Photon beam properties at the European XFEL. Technical report, XFEL, Hamburg, 2011.*

[^4]: *He, et al. Deep Residual Learning for Image Recognition. 2015.*

[^5]: *Szegedy, et al. Inception-v4, Inception-airynet and the Impact of Residual Connections on Learning. 2016.*

[^6]: *He, et al. Identity Mappings in Deep Residual Networks. 2016.*
