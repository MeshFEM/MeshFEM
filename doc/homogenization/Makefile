all: view

view: homogenization.pdf
	skim-load homogenization.pdf

homogenization.pdf: homogenization.tex homogenization.bbl
	pdflatex homogenization

homogenization.bbl: references.bib
	pdflatex homogenization
	bibtex homogenization
