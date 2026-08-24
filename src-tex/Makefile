.PHONY: all figures pdf audit

all: pdf audit

figures:
	cd .. && SOURCE_DATE_EPOCH=0 uv run --group dev python src-tex/generate_figures.py

pdf: figures report.tex references.bib sn-jnl.cls sn-vancouver-num.bst assets/tu-logo.png
	mkdir -p build
	SOURCE_DATE_EPOCH=0 tectonic -X compile --keep-logs --keep-intermediates --outdir build report.tex
	cp build/report.pdf report.pdf

audit: pdf
	cd .. && uv run python src-tex/check_report.py
