# PLENA model frontend

This directory owns a physical source copy of the graph frontend needed by
`plena-compile-model`. It does not import frontend Python source from the ETRI
repository and contains no symbolic links.

The frontend path is:

1. local Hugging Face model construction with fake checkpoint tensors for
   graph capture only;
2. `torch.export` capture of the actual model `forward`;
3. official torch-mlir FX import to Torch MLIR;
4. fail-closed dense-Llama graph, parameter-ABI, and target-capability
   certification.

`plena_static_frontend.py` and `plena_decode_generator.py` are retained as the
model/config capability normalizer used by the certificate. Their old ETRI
dialect renderers are compatibility code and are not the PLENA Program-v5
lowering path.

External dependencies remain normal toolchain/runtime dependencies: Python,
PyTorch, Transformers, torch-mlir, and the user-selected Hugging Face model
checkpoint. They are not source-code links.
