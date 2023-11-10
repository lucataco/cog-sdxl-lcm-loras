import replicate
output = replicate.run(
    "lucataco/sdxl-lcm-loras:48bdebc9f383f0e9f9e321e40c1ec7f08ac7b4dd49cf777646a6498c94605051",
    input={"prompt": "A TOK emoji of a man", "replicate_weights": "https://pbxt.replicate.delivery/DUxxgRlwU5q3DNhaaEPnH70H6afeUh18iIFTZkbioqVWeoEjA/trained_model.tar", "seed": 23969},
)