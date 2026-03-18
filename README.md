<!--
SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>

SPDX-License-Identifier: Apache-2.0
-->

# Thermo Raw Reader NeXus Tools

This repo uses
[RawFileReader](https://github.com/thermofisherlsms/RawFileReader/) to read
mass spectrometry data and convert it into the
[NeXus](https://www.nexusformat.org/) format. 

## NeXus details
Nexus is a general metadata structure, and so can host any data shape.
For the RFI we will be storing our data as one large 4 dimensional block. 
So far the four dimensions are layers, image_width, image_height, and spectrum.

