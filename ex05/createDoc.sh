#!/bin/sh

dirName=$(basename $(pwd))

texi2pdf --build=tidy --batch --quiet --output=doc.pdf doc.tex \
	&& mv doc.pdf SchroederVobruba_${dirName}.pdf

