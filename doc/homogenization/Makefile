all: view

view: homogenization.pdf
	skim-load homogenization.pdf

homogenization.pdf: homogenization.tex homogenization.bbl sequentialLaminates.m
	pdflatex -shell-escape homogenization

homogenization.bbl: references.bib
	pdflatex -shell-escape homogenization
	bibtex homogenization
