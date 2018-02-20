all: view

view: shape_opt.pdf
	skim-load shape_opt.pdf

shape_opt.pdf: shape_opt.tex shape_opt.bbl
	pdflatex shape_opt

shape_opt.bbl: references.bib
	pdflatex shape_opt
	bibtex shape_opt
