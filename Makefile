IMAGE           ?= pawn-litellm-proxy
TAG             ?= latest
DOCKERFILE       = docker/litellm/Dockerfile
MLFLOW_HOST     ?= 127.0.0.1
MLFLOW_PORT     ?= 5000
# URL of the pawn-agent server reachable from inside the container.
# On Linux Docker Engine use host.docker.internal (requires --add-host below)
# or the machine's LAN IP.  On Docker Desktop it resolves automatically.
PAWN_AGENT_URL  ?= http://host.docker.internal:8000

# Obsidian Android ADB push (forwarded to obsidian-plugin/pawn/Makefile).
ADB_VAULT       ?=
ADB_SERIAL      ?=
ADB_CONFIG      ?= .obsidian
ADB_RESTART     ?= 1

.PHONY: build push run mlflow clean obsidian-plugin obsidian-plugin-adb obsidian-plugin-list-vaults

build:
	docker build -f $(DOCKERFILE) -t $(IMAGE):$(TAG) .

# Obsidian Pawn plugin → Android via ADB (see obsidian-plugin/pawn/Makefile).
# Example: make obsidian-plugin-adb ADB_VAULT=/sdcard/Documents/MyVault
obsidian-plugin:
	$(MAKE) -C obsidian-plugin/pawn build

obsidian-plugin-list-vaults:
	$(MAKE) -C obsidian-plugin/pawn adb-list-vaults ADB_SERIAL="$(ADB_SERIAL)" ADB_CONFIG="$(ADB_CONFIG)"

obsidian-plugin-adb:
	$(MAKE) -C obsidian-plugin/pawn adb-push \
		ADB_VAULT="$(ADB_VAULT)" \
		ADB_SERIAL="$(ADB_SERIAL)" \
		ADB_CONFIG="$(ADB_CONFIG)" \
		ADB_RESTART="$(ADB_RESTART)"

push:
	docker push $(IMAGE):$(TAG)

run:
	docker run --rm -p 4000:4000 \
		--add-host=host.docker.internal:host-gateway \
		-e LITELLM_MASTER_KEY=$(LITELLM_MASTER_KEY) \
		-e PAWN_AGENT_URL=$(PAWN_AGENT_URL) \
		$(IMAGE):$(TAG)

mlflow:
	mlflow server --host $(MLFLOW_HOST) --port $(MLFLOW_PORT)

clean:
	docker rmi $(IMAGE):$(TAG)
