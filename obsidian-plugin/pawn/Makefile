# Build the Obsidian Pawn plugin and push the three runtime files to an
# Android device over ADB (avoids syncing node_modules / src).
#
# Usage:
#   make build
#   make dist                                   # pawn.zip for GitHub / manual install
#   make adb-list-vaults
#   make adb-push ADB_VAULT=/sdcard/Documents/MyVault
#
# Optional:
#   ADB_SERIAL=...          # pass to adb -s when multiple devices
#   ADB_CONFIG=.obsidian    # override config folder name
#   ADB_RESTART=1           # force-stop Obsidian after push (default)

PLUGIN_ID       := pawn
DIST_FILES      := main.js manifest.json styles.css
DIST_ZIP        := pawn.zip

ADB             ?= adb
ADB_SERIAL      ?=
ADB_CONFIG      ?= .obsidian
ADB_RESTART     ?= 1
OBSIDIAN_PKG    := md.obsidian

# Device vault root, e.g. /sdcard/Documents/MyVault
ADB_VAULT       ?=

ifneq ($(ADB_SERIAL),)
ADB_CMD := $(ADB) -s $(ADB_SERIAL)
else
ADB_CMD := $(ADB)
endif

ADB_PLUGIN = $(ADB_VAULT)/$(ADB_CONFIG)/plugins/$(PLUGIN_ID)

.PHONY: help build install-deps check-dist package dist \
	adb-devices adb-list-vaults adb-push adb-restart

help:
	@echo "Targets:"
	@echo "  build             Bundle main.js (npm run build)"
	@echo "  package           Zip the three runtime files into $(DIST_ZIP)"
	@echo "  dist              Build + package $(DIST_ZIP)"
	@echo "  adb-list-vaults   Find vaults with a .obsidian folder on the phone"
	@echo "  adb-push          Build + push main.js/manifest.json/styles.css"
	@echo ""
	@echo "Required for adb-push:"
	@echo "  ADB_VAULT=/sdcard/Documents/YourVault"
	@echo ""
	@echo "Optional:"
	@echo "  ADB_SERIAL=<id>   ADB_CONFIG=.obsidian   ADB_RESTART=0|1"

install-deps:
	npm install

build: install-deps
	npm run build

check-dist: build
	@for f in $(DIST_FILES); do \
		test -f $$f || { echo "missing $$f — build failed?"; exit 1; }; \
	done

# pawn.zip contains pawn/{main.js,manifest.json,styles.css}. Unzip it into
# <vault>/.obsidian/plugins/ and Obsidian sees the plugin folder.
package:
	@for f in $(DIST_FILES); do \
		test -f $$f || { echo "missing $$f — run 'make build' first"; exit 1; }; \
	done
	rm -rf .dist $(DIST_ZIP)
	mkdir -p .dist/$(PLUGIN_ID)
	cp $(DIST_FILES) .dist/$(PLUGIN_ID)/
	cd .dist && zip -r ../$(DIST_ZIP) $(PLUGIN_ID)
	rm -rf .dist
	@echo "Wrote $(DIST_ZIP)"
	@echo "Unzip into <vault>/.obsidian/plugins/ (creates $(PLUGIN_ID)/)."

dist: check-dist package

adb-devices:
	$(ADB_CMD) devices -l

adb-list-vaults: adb-devices
	@echo "Looking for */$(ADB_CONFIG) on device (may take a few seconds)…"
	@$(ADB_CMD) shell 'for root in /sdcard /storage/emulated/0; do \
		[ -d "$$root" ] || continue; \
		find "$$root" -type d -name "$(ADB_CONFIG)" 2>/dev/null; \
	done' | sed 's|/$(ADB_CONFIG)$$||' | sort -u \
		| { grep . || echo "(none found — open a vault in Obsidian once, or set ADB_VAULT manually)"; }

adb-push:
	@test -n "$(ADB_VAULT)" || { \
		echo "Set ADB_VAULT to the vault root on the phone, e.g.:"; \
		echo "  make adb-push ADB_VAULT=/sdcard/Documents/MyVault"; \
		echo "Run 'make adb-list-vaults' to discover candidates."; \
		exit 1; \
	}
	@$(MAKE) --no-print-directory check-dist
	@echo "Pushing $(PLUGIN_ID) → $(ADB_PLUGIN)"
	$(ADB_CMD) shell mkdir -p "$(ADB_PLUGIN)"
	$(ADB_CMD) push main.js "$(ADB_PLUGIN)/main.js"
	$(ADB_CMD) push manifest.json "$(ADB_PLUGIN)/manifest.json"
	$(ADB_CMD) push styles.css "$(ADB_PLUGIN)/styles.css"
	@if [ "$(ADB_RESTART)" = "1" ]; then $(MAKE) --no-print-directory adb-restart; fi
	@echo "Done. Enable Community plugins → Pawn if this is the first install, then reopen Obsidian."

adb-restart:
	$(ADB_CMD) shell am force-stop $(OBSIDIAN_PKG)
	@echo "Stopped $(OBSIDIAN_PKG) — open Obsidian on the phone to reload plugins."
