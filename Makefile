IMAGE           ?= pawn-litellm-proxy
TAG             ?= latest
DOCKERFILE       = docker/litellm/Dockerfile
# URL of the pawn-agent server reachable from inside the container.
# On Linux Docker Engine use host.docker.internal (requires --add-host below)
# or the machine's LAN IP.  On Docker Desktop it resolves automatically.
PAWN_AGENT_URL  ?= http://host.docker.internal:8000

.PHONY: build push run clean

build:
	docker build -f $(DOCKERFILE) -t $(IMAGE):$(TAG) .

push:
	docker push $(IMAGE):$(TAG)

run:
	docker run --rm -p 4000:4000 \
		--add-host=host.docker.internal:host-gateway \
		-e LITELLM_MASTER_KEY=$(LITELLM_MASTER_KEY) \
		-e PAWN_AGENT_URL=$(PAWN_AGENT_URL) \
		$(IMAGE):$(TAG)

clean:
	docker rmi $(IMAGE):$(TAG)
