FROM ccr-registry.baidubce.com/aihc/aibox-pytorch:v1.0-torch2.11.0-cu12.6

LABEL org.opencontainers.image.title="task6-phasea-fullft-runtime"
LABEL org.opencontainers.image.description="Task 6 Phase A/F1 full-finetuning runtime on AIHC A800"

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements-task6.txt /tmp/requirements-task6.txt

RUN python -m pip install --no-cache-dir -r /tmp/requirements-task6.txt \
    && python -m pip check \
    && python -c "import torch, transformers, datasets, pyarrow, yaml, numpy, matplotlib, tokenizers, safetensors, sentencepiece; assert torch.__version__.startswith('2.11.0'), torch.__version__; assert torch.version.cuda == '12.6', torch.version.cuda; print('task6 image dependency check passed')" \
    && rm -f /tmp/requirements-task6.txt

WORKDIR /workspace

CMD ["/bin/bash"]
