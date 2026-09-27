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

# Self-signed TLS for direct pawn-server exposure (api.ssl_certfile / ssl_keyfile).
CERT_DIR        ?= certs
CERT_DAYS       ?= 825
CERT_CN         ?= localhost
# Comma-separated SANs for openssl -addext (OpenSSL 1.1.1+).
CERT_SAN        ?= DNS:localhost,IP:127.0.0.1

.PHONY: build push run mlflow clean ssl-cert \
	obsidian-plugin obsidian-plugin-adb obsidian-plugin-list-vaults

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

# Self-signed certificate for api.ssl_certfile / api.ssl_keyfile.
# Example with a LAN IP: make ssl-cert CERT_CN=pawn.local \
#   CERT_SAN='DNS:pawn.local,DNS:localhost,IP:192.168.1.10,IP:127.0.0.1'
ssl-cert:
	@mkdir -p "$(CERT_DIR)"
	openssl req -x509 -newkey rsa:4096 -sha256 -days $(CERT_DAYS) \
		-nodes \
		-keyout "$(CERT_DIR)/key.pem" \
		-out "$(CERT_DIR)/cert.pem" \
		-subj "/CN=$(CERT_CN)" \
		-addext "subjectAltName=$(CERT_SAN)"
	@chmod 600 "$(CERT_DIR)/key.pem"
	@echo "Wrote $(CERT_DIR)/cert.pem and $(CERT_DIR)/key.pem"
	@echo "Set in pawnai.yaml:"
	@echo "  api.ssl_certfile: $(CERT_DIR)/cert.pem"
	@echo "  api.ssl_keyfile:  $(CERT_DIR)/key.pem"

clean:
	docker rmi $(IMAGE):$(TAG)
