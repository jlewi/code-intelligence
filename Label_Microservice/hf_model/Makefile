# TODO(jlewi): Need to sort out root vs project level Makefiles
CONTEXT=label-bot-dev

# Update the Tekton tasks in the dev namespace
apply-dev:
	kustomize build ../../tekton/installs/dev | kubectl --context=$(CONTEXT) apply -f - 

build-image:    	
	skaffold build --kube-context=${CONTEXT} -v info --file-output=hf-trainer.image.json

set-image:
	kpt cfg set ../../tekton hf-trainer-image $(shell yq r hf-trainer.image.json builds[0].tag )

update-image: build-image update-image

update-dev: update-image apply-dev