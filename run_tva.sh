#!/bin/bash

jupyter nbconvert --to html --execute /tva/tva.ipynb --no-input
mv /tva/tva.html input/tva_report.html
