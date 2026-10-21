# Package the Pawn Capture MV3 extension (plain JS — no compile step).
#
# Usage:
#   make build                  # verify runtime files + icons
#   make dist                   # Chrome + Firefox zips
#   make package-chrome         # pawn-capture.zip (service_worker)
#   make package-firefox        # pawn-capture-firefox.zip (background.scripts)
#   make icons                  # regenerate icons
#
# From the repo root:
#   make browser-extension
#   make browser-extension-dist

EXT_ID          := pawn-capture
DIST_ZIP        := pawn-capture.zip
DIST_ZIP_FF     := pawn-capture-firefox.zip
DIST_FILES      := background.js sidepanel.html sidepanel.css sidepanel.js
DIST_DIRS       := content icons

.PHONY: help build icons check-dist package package-chrome package-firefox dist clean

help:
	@echo "Targets:"
	@echo "  build            Verify extension files and icons"
	@echo "  icons            Regenerate icons/icon-{16,48,128}.png"
	@echo "  package-chrome   Zip Chrome/Edge build → $(DIST_ZIP)"
	@echo "  package-firefox  Zip Firefox build → $(DIST_ZIP_FF)"
	@echo "  dist             build + both packages"
	@echo "  clean            Remove zips and .dist/"
	@echo ""
	@echo "Chrome: unzip $(DIST_ZIP), Load unpacked → $(EXT_ID)/"
	@echo "Firefox: unzip $(DIST_ZIP_FF), about:debugging → $(EXT_ID)/manifest.json"
	@echo "Docs: ../../docs/BROWSER_CAPTURE.md"

icons:
	python3 scripts/gen_icons.py

check-dist:
	@test -f manifest.json || { echo "missing manifest.json"; exit 1; }
	@test -f manifest.firefox.json || { echo "missing manifest.firefox.json"; exit 1; }
	@for f in $(DIST_FILES); do \
		test -f $$f || { echo "missing $$f"; exit 1; }; \
	done
	@for d in $(DIST_DIRS); do \
		test -d $$d || { echo "missing directory $$d"; exit 1; }; \
	done
	@test -f icons/icon-16.png || { echo "missing icons/icon-16.png — run make icons"; exit 1; }
	@test -f icons/icon-48.png || { echo "missing icons/icon-48.png — run make icons"; exit 1; }
	@test -f icons/icon-128.png || { echo "missing icons/icon-128.png — run make icons"; exit 1; }
	@test -f content/region.js || { echo "missing content/region.js"; exit 1; }
	@python3 -c "import json; json.load(open('manifest.json')); json.load(open('manifest.firefox.json')); print('manifests ok')"

build: check-dist
	@echo "Pawn Capture extension ready (no compile step)."

_stage = \
	rm -rf .dist/$(1); \
	mkdir -p .dist/$(1); \
	cp $(DIST_FILES) .dist/$(1)/; \
	cp -R content icons .dist/$(1)/; \
	cp $(2) .dist/$(1)/manifest.json

package-chrome: check-dist
	rm -rf .dist/$(EXT_ID) $(DIST_ZIP)
	mkdir -p .dist/$(EXT_ID)
	cp $(DIST_FILES) .dist/$(EXT_ID)/
	cp -R content icons .dist/$(EXT_ID)/
	cp manifest.json .dist/$(EXT_ID)/manifest.json
	cd .dist && zip -r ../$(DIST_ZIP) $(EXT_ID)
	rm -rf .dist/$(EXT_ID)
	@echo "Wrote $(DIST_ZIP)  (Chrome/Edge — Load unpacked → $(EXT_ID)/)"

package-firefox: check-dist
	rm -rf .dist/$(EXT_ID) $(DIST_ZIP_FF)
	mkdir -p .dist/$(EXT_ID)
	cp $(DIST_FILES) .dist/$(EXT_ID)/
	cp -R content icons .dist/$(EXT_ID)/
	cp manifest.firefox.json .dist/$(EXT_ID)/manifest.json
	cd .dist && zip -r ../$(DIST_ZIP_FF) $(EXT_ID)
	rm -rf .dist/$(EXT_ID)
	@echo "Wrote $(DIST_ZIP_FF)  (Firefox — Load Temporary Add-on → $(EXT_ID)/manifest.json)"

package: package-chrome package-firefox

dist: build package
	rm -rf .dist
	@echo "Chrome:  $(DIST_ZIP)"
	@echo "Firefox: $(DIST_ZIP_FF)"

clean:
	rm -rf .dist $(DIST_ZIP) $(DIST_ZIP_FF)
