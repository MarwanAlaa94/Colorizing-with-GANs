#!/bin/sh
echo 'starting job'
timestamp=$(date "+%d:%m:%y:%H:%M:%S")
cd /home/marwan/Colorizing-with-GANs/src
zip -r model_$timestamp.zip weights_places365_lab*
zip -r photoes_$timestamp.zip results/*
gdrive upload model_$timestamp.zip
gdrive upload photoes_$timestamp.zip
rm model_$timestamp.zip photoes_$timestamp.zip
echo 'finishing job'
