# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = mlens
SOURCEDIR     = 
BUILDDIR      = build


# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)


.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Install necessary modules
# TODO: should we load these as mockups?
install:
			pip install -U sphinx-gallery numpydoc matplotlib seaborn

# Populate parent dir
arange:
	cp -r build/html/ ../
	rm -r build
	-rm -rf tutorials

clean-build:
	rm -rf tutorials

# Make docs
docs:
	make html
	make arange
	make clean-build

html-noplot:
	$(SPHINXBUILD) -D plot_gallery=0 -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

# Clear all in docs except source
clean-cache:
	make clean
	rm -rf tutorials
	bash clean-html.sh
