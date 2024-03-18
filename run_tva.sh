#!/bin/bash

jupyter nbconvert --to html --execute /tva/tva.ipynb --no-input
mv /tva/tva.pdf input/tva_report.pdf
