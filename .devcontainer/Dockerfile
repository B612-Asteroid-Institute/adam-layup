FROM mcr.microsoft.com/devcontainers/python:3.11

WORKDIR /Workspace
RUN chmod a+w /Workspace 
USER vscode

RUN curl -sSL https://pdm-project.org/install-pdm.py | python3 -
ENV PATH="${PATH}:/home/vscode/.local/bin"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/workspaces/adam-layup/.venv/lib/python3.11/site-packages"

RUN pdm config install.cache on

RUN pdm completion bash > /home/vscode/.bash_completion

# Cannot run pdm install until the container is run with directory mounted
# pdm install

