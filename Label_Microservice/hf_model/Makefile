# TODO(jlewi): Need to sort out root vs project level Makefiles
CONTEXT=label-bot-dev

# Update the Tekton tasks in the dev namespace
apply-dev:
	kustomize build ../../tekton/installs/dev | kubectl --context=$(CONTEXT) apply -f - 

update-image:    	
	skaffold build --kube-context=${CONTEXT} -v info --file-output=hf-trainer.image.json

set-diff-image:
	cd auto-update/base && kustomize edit set image gcr.io/issue-label-bot-dev/labelbot-diff=$(shell yq r labelbot-diff.image.json builds[0].tag )

update-diff-image: build-diff-image set-diff-image
	
	
