#!/bin/bash

jupyter nbconvert --to pdf --execute /tva/tva.ipynb --no-input
mv /tva/tva.pdf input/tva_report.pdf
