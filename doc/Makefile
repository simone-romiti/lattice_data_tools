# Makefile for the `Hermes` documentation (in `quarto` markdown)
#
# Targets:
#   render   : build the HTML book in doc/_book/
#   preview  : build and open a live-reloading browser preview
#   clean    : remove the build output directory
#   help     : print this message (default target)

.PHONY: help render preview clean

# Default target: print available targets
help:
	@echo ""
	@echo "  Hermes documentation - available targets"
	@echo "  --------------------------------------------"
	@echo "  make render    Build the HTML book  →  _book/"
	@echo "  make preview   Build + live-reload browser preview"
	@echo "  make clean     Remove _book/ build output"
	@echo "  make help      Print this message (default)"
	@echo ""

render:
	quarto render

preview:
	quarto preview

clean:
	rm -rf _book
