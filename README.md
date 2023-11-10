# Cog-SDXL for LCM and Replicate LoRAs

[![Replicate demo and cloud API](https://replicate.com/stability-ai/sdxl/badge)](https://replicate.com/stability-ai/sdxl)

This is an implementation of Stability AI's [SDXL](https://github.com/Stability-AI/generative-models) as a [Cog](https://github.com/replicate/cog) model.

## Development

Follow the [model pushing guide](https://replicate.com/docs/guides/push-a-model) to push your own fork of SDXL to [Replicate](https://replicate.com).

## Basic Usage

for prediction,

```bash
cog predict -i prompt="A TOK emoji of a man" -i replicate_weights="https://pbxt.replicate.delivery/DUxxgRlwU5q3DNhaaEPnH70H6afeUh18iIFTZkbioqVWeoEjA/trained_model.tar" -i disable_safety_checker=True -i seed=23969
```
