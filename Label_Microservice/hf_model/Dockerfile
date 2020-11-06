# https://cloud.google.com/ai-platform/deep-learning-containers/docs/choosing-container
FROM gcr.io/deeplearning-platform-release/pytorch-cpu.1-4

# Downgrade sentencepiece to 0.9.1 to avoid segfault
# see https://github.com/huggingface/transformers/issues/5359
RUN pip install \
    fire==0.3.1 \
    sentencepiece==0.1.91 \
    simpletransformers==0.48.15

#*************************************************************************************************************
# Install VSCode
#************************************************************************************************************

# Install coder
# download the coder binary, untar it, and allow it to be executed
RUN export VERSION=code-server-3.6.2-linux-amd64 && \
 curl -LO https://github.com/cdr/code-server/releases/download/v3.6.2/${VERSION}.tar.gz \
 && tar -xzvf ${VERSION}.tar.gz && \
 mv ${VERSION} /usr/local/ && \
 ln -sf /usr/local/${VERSION}/code-server /usr/local/bin/code-server && \
 chmod +x /usr/local/${VERSION}/code-server
