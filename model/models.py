from transformers import pipeline

pipe = pipeline("text-generation", model="codeparrot/codeparrot-small")

# Run the environment locally from the space
# docker run -d -p 8000:8000 registry.hf.space/openenv-echo-env:latest