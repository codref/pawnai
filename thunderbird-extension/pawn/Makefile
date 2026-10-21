# Package the Pawn Capture Thunderbird MailExtension (MV3 — no compile step).
#
# Usage:
#   make build                  # verify runtime files + icons
#   make dist                   # pawn-capture-thunderbird.xpi
#
# From the repo root:
#   make thunderbird-extension
#   make thunderbird-extension-dist

EXT_ID          := pawn-capture-tb
DIST_XPI        := pawn-capture-thunderbird.xpi
DIST_FILES      := background.js panel.html panel.css panel.js manifest.json
DIST_DIRS       := icons

.PHONY: help build check-dist package dist clean

help:
	@echo "Targets:"
	@echo "  build     Verify extension files and icons"
	@echo "  dist      build + $(DIST_XPI)"
	@echo "  package   Write $(DIST_XPI)"
	@echo "  clean     Remove xpi and .dist/"
	@echo ""
	@echo "Thunderbird: Add-ons Manager → gear → Debug Add-ons →"
	@echo "  Load Temporary Add-on → $(EXT_ID)/manifest.json"
	@echo "  or Install Add-on From File → $(DIST_XPI)"
	@echo "Docs: ../../docs/THUNDERBIRD_CAPTURE.md"

check-dist:
	@test -f manifest.json || { echo "missing manifest.json"; exit 1; }
	@for f in $(DIST_FILES); do \
		test -f $$f || { echo "missing $$f"; exit 1; }; \
	done
	@for d in $(DIST_DIRS); do \
		test -d $$d || { echo "missing directory $$d"; exit 1; }; \
	done
	@test -f icons/icon-16.png || { echo "missing icons/icon-16.png"; exit 1; }
	@test -f icons/icon-48.png || { echo "missing icons/icon-48.png"; exit 1; }
	@test -f icons/icon-128.png || { echo "missing icons/icon-128.png"; exit 1; }
	@python3 -c "import json; m=json.load(open('manifest.json')); assert m.get('manifest_version')==3; print('manifest ok', m.get('version'))"

build: check-dist
	@echo "Pawn Capture Thunderbird add-on ready (no compile step)."

package: check-dist
	rm -rf .dist/$(EXT_ID) $(DIST_XPI)
	mkdir -p .dist/$(EXT_ID)
	cp $(DIST_FILES) .dist/$(EXT_ID)/
	cp -R icons .dist/$(EXT_ID)/
	cd .dist/$(EXT_ID) && zip -r ../../$(DIST_XPI) .
	rm -rf .dist/$(EXT_ID)
	@echo "Wrote $(DIST_XPI)  (manifest.json at zip root)"

dist: build package
	rm -rf .dist
	@echo "Thunderbird: $(DIST_XPI)"

clean:
	rm -rf .dist $(DIST_XPI)
