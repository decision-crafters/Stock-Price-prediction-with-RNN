# You could use `gitpod/workspace-full` as well.
FROM gitpod/workspace-full:latest
RUN pyenv install 3.11 \
    && pyenv global 3.11
