# Weights format

Weights are one flat binary file, `.jkpw`: a header, a table of tensors and the raw data. It needs no parsing
library and loads equally with `fread`, `mmap` or `fetch()` into an `ArrayBuffer`. The model id in the file
selects the network, so one loader serves every model. BatchNorm is already folded into the convolutions.

The same container, with other tensors, holds the command-line tool's output (`.jkpt`: `xy`, `score`, `desc`).

## Layout

All integers are little-endian.

```text
char[4]  magic        "JKPW"
u32      version      1
u32      n_meta
u32      n_tensors
n_meta    x { u16 key_len, key bytes, u16 value_len, value bytes }      (UTF-8 strings)
n_tensors x { u16 name_len, name bytes, u8 dtype, u8 ndim, u32 dims[ndim], u64 offset, u64 nbytes }
padding, then the raw tensor data; each tensor starts at `offset` (from the start of the file, 64-byte aligned)
```

`dtype`: 0 = float32, 1 = float16, 2 = int32, 3 = uint8. Tensor data is row-major (C order).

Metadata keys of weights files: `kind`, `model`, `c1`, `c2`, `c3`, `c4`, `dim`, `K`, `M`, `source`,
`source_commit`, `license`.

## Shipped files

| File | Model | Tensors | Size | License |
|---|---|---|---|---|
| `aliked-n16.jkpw` | ALIKED, `aliked-n16` (128-dimensional descriptors) | 44 tensors, 676,876 values | 2.71 MB | BSD-3-Clause |
| `raco.jkpw` | RaCo detector | 93 tensors, 496,662 values | 1.99 MB | Apache-2.0 |

One FP32 file per model; FP16 mode converts the weights when the context is created.
