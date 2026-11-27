# Pawn Portal (Vite + Vue 3 + Quasar). Root targets: make portal / make portal-dev.
NPM ?= npm

.PHONY: install build dev test clean

node_modules: package.json
	$(NPM) install --no-audit --no-fund
	@touch node_modules

install: node_modules

# vue-tsc typecheck + production build into dist/ (served by pawn-server at /portal).
build: node_modules
	$(NPM) run build

# Dev server on :5173/portal/ proxying /v1 to PAWN_API_URL (default http://localhost:8000).
# VITE_PAWN_MOCK=1 runs against the in-browser mock instead of a server.
dev: node_modules
	$(NPM) run dev

# Unit tests (vitest + jsdom) for the anchoring logic.
test: node_modules
	$(NPM) run test

clean:
	rm -rf dist
