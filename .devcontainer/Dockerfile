FROM ecr-oy9xkyms-nmhhht2.registry.cucloud.cn/ai/miniconda3:25.1.1-2

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp

# [Optional] Uncomment to install a different version of Python than the default
# RUN conda install -y python=3.6 \
#     && pip install --no-cache-dir pipx \
#     && pipx reinstall-all

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>

# Create a non-root user
ARG USERNAME=vscode
ARG USER_UID=2008
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

# Set bash as default shell for the user
RUN chsh -s /bin/bash $USERNAME

# Set the working directory
WORKDIR /workspaces/vllm

# Switch pip and conda to Tsinghua source
RUN /opt/conda/bin/conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ \
    && /opt/conda/bin/conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ \
    && /opt/conda/bin/conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/ \
    && /opt/conda/bin/conda config --set show_channel_urls yes \
    && echo -e "[global]\nindex-url = https://pypi.tuna.tsinghua.edu.cn/simple" > /opt/conda/etc/pip.conf

# Install uv
RUN /opt/conda/bin/pip install --no-cache-dir uv