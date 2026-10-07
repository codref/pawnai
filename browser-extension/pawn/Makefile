# Package the Pawn Capture MV3 extension (plain JS — no compile step).
#
# Usage:
#   make build                  # verify runtime files + icons
#   make dist                   # pawn-capture.zip for load-unpacked / share
#   make icons                  # regenerate icons/icon-48.png and icon-128.png
#
# From the repo root:
#   make browser-extension
#   make browser-extension-dist

EXT_ID          := pawn-capture
DIST_ZIP        := pawn-capture.zip
DIST_FILES      := manifest.json background.js sidepanel.html sidepanel.css sidepanel.js
DIST_DIRS       := content icons

.PHONY: help build icons check-dist package dist clean

help:
	@echo "Targets:"
	@echo "  build       Verify extension files and icons"
	@echo "  icons       Regenerate icons/icon-48.png and icons/icon-128.png"
	@echo "  package     Zip runtime files into $(DIST_ZIP)"
	@echo "  dist        build + package $(DIST_ZIP)"
	@echo "  clean       Remove $(DIST_ZIP) and .dist/"
	@echo ""
	@echo "Deploy: unzip $(DIST_ZIP), then Load unpacked → $(EXT_ID)/"
	@echo "Docs: ../../docs/BROWSER_CAPTURE.md"

icons:
	python3 scripts/gen_icons.py

check-dist:
	@for f in $(DIST_FILES); do \
		test -f $$f || { echo "missing $$f"; exit 1; }; \
	done
	@for d in $(DIST_DIRS); do \
		test -d $$d || { echo "missing directory $$d"; exit 1; }; \
	done
	@test -f icons/icon-48.png || { echo "missing icons/icon-48.png — run make icons"; exit 1; }
	@test -f icons/icon-128.png || { echo "missing icons/icon-128.png — run make icons"; exit 1; }
	@test -f content/region.js || { echo "missing content/region.js"; exit 1; }
	@python3 -c "import json; json.load(open('manifest.json')); print('manifest.json ok')"

build: check-dist
	@echo "Pawn Capture extension ready (no compile step)."

# pawn-capture.zip contains pawn-capture/{manifest.json, …}.
# Unzip, then Chrome/Edge → Load unpacked → select the pawn-capture/ folder.
package: check-dist
	rm -rf .dist $(DIST_ZIP)
	mkdir -p .dist/$(EXT_ID)
	cp $(DIST_FILES) .dist/$(EXT_ID)/
	cp -R content icons .dist/$(EXT_ID)/
	cd .dist && zip -r ../$(DIST_ZIP) $(EXT_ID)
	rm -rf .dist
	@echo "Wrote $(DIST_ZIP)"
	@echo "Unzip, then Load unpacked → $(EXT_ID)/  (see README.md)."

dist: build package

clean:
	rm -rf .dist $(DIST_ZIP)
