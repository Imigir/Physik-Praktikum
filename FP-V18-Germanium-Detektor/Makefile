ifeq (,$(shell sh -c 'cygpath --version 2> /dev/null'))
  # Unix
  pwd := $$(pwd)
  translate = $1
else
  # Windows mit MSys2/Cygwin
  pwd := $$(cygpath -m "$$(pwd)")
  translate = $(shell echo '$1' | sed 's/:/;/g')
endif


all: python tex

python: scripts/*w.py matplotlibrc header-matplotlib.tex | build
	TEXINPUTS="$(call translate,$(pwd):)" python scripts/*w.py

tex: build/*w.pdf

build/*w.pdf: header.tex *w.tex lit.bib | build

build/*w.pdf: FORCE
	TEXINPUTS="$(call translate,build:)" \
	BIBINPUTS="$(call translate,$(pwd):)" \
	latexmk \
	 --lualatex \
	 --output-directory=build \
	 --interaction=nonstopmode \
	--halt-on-error \
	*w.tex

build:
	mkdir -p build

clean:
	rm -rf build

edittex:
	gnome-open *w.tex & ./makeloop.sh

FORCE:

.PHONY: build all FORCE edittex python tex
