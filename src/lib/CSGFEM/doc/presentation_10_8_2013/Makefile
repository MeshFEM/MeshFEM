EXAMPLES=
SNIPPETS=snippets/buffer_create.c snippets/context_apple.c snippets/run_vis.c snippets/kernel.cl
all: view slides-handout.pdf

view: slides.pdf
	skim-load slides.pdf

# RobustnessPrediction.bbl: references.bib
# 	pdflatex -shell-escape RobustnessPrediction
# 	bibtex   RobustnessPrediction
# 	pdflatex -shell-escape RobustnessPrediction

slides.pdf: slides.tex $(SNIPPETS)
	pdflatex -shell-escape  "\documentclass{beamer}\input{slides}" && mv beamer.pdf slides.pdf
slides-handout.pdf: slides.tex $(SNIPPETS)
	 pdflatex -shell-escape -job-name=slides-handout "\documentclass[handout]{beamer}\input{slides}" && mv beamer.pdf slides-handout.pdf

