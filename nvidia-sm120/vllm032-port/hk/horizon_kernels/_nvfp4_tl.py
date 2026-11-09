# Device-side NVFP4 helpers for horizon_kernels, vendored from upstream TileLang
# examples/dequantize_gemm/quantize/nvfp4.py (MIT, (c) Tile-AI) so horizon_kernels does not import upstream examples.
import tilelang.language as T

CHUNK_ROWS = 128          # SM120 blockscaled_chunk_kmajor: 128-row scale chunks
CHUNK_WORDS = 4           # 4 uint32 scale words (= 16 scales = 256 K values) per chunk row
FP4_E2M1_MAX = 6.0
_BLOCKSCALED_CHUNK_ROWS = CHUNK_ROWS
_UE4M3_MIN_NORMAL = 2.0**-6
_UE4M3_MAX = 1.75 * (2.0**8)


def _tl_encode_ue4m3_scale_byte_ceil(x):
    x = T.alloc_var(T.float32, init=T.max(T.min(x, _UE4M3_MAX), 0.0))
    sub_mantissa = T.cast(T.ceil(x[0] * 512.0), T.int32)
    sub_code = T.min(T.max(sub_mantissa, 0), 7)
    sub_code = T.if_then_else(sub_mantissa > 7, 0x08, sub_code)

    bits = T.reinterpret(x[0], T.uint32)
    exponent_unbiased = T.cast((bits >> 23) & 0xFF, T.int32) - 127
    exponent_unbiased = T.min(T.max(exponent_unbiased, -6), 8)
    mantissa_bits = T.cast(bits & 0x7FFFFF, T.int32)
    mantissa = (mantissa_bits >> 20) + T.if_then_else((mantissa_bits & 0xFFFFF) != 0, 1, 0)
    biased_exponent = exponent_unbiased + 7
    carry = mantissa >= 8
    biased_exponent = T.if_then_else(carry, biased_exponent + 1, biased_exponent)
    biased_exponent = T.min(T.max(biased_exponent, 1), 15)
    mantissa = T.if_then_else(carry, 0, mantissa)
    mantissa = T.min(T.max(mantissa, 0), 7)
    normal_code = (biased_exponent << 3) | mantissa

    code = T.if_then_else(x[0] < _UE4M3_MIN_NORMAL, sub_code, normal_code)
    return T.if_then_else(x[0] <= 0.0, 0, code)


def _tl_decode_ue4m3_scale_byte(code):
    exponent = (code >> 3) & 0x0F
    mantissa = code & 0x07
    mantissa_f32 = T.cast(mantissa, T.float32)
    normal_bits = T.cast(((exponent + 120) << 23) | (mantissa << 20), T.uint32)
    normal = T.reinterpret(normal_bits, T.float32)
    subnormal = mantissa_f32 * 0.125 * _UE4M3_MIN_NORMAL
    return T.if_then_else(exponent == 0, subnormal, normal)


def _tl_ue4m3_scale_byte_inverse(code):
    exponent = (code >> 3) & 0x0F
    mantissa = code & 0x07

    normal_factor = T.if_then_else(
        mantissa == 0,
        1.0,
        T.if_then_else(
            mantissa == 1,
            0.8888888955116272,
            T.if_then_else(
                mantissa == 2,
                0.800000011920929,
                T.if_then_else(
                    mantissa == 3,
                    0.7272727489471436,
                    T.if_then_else(
                        mantissa == 4,
                        0.6666666865348816,
                        T.if_then_else(
                            mantissa == 5,
                            0.6153846383094788,
                            T.if_then_else(mantissa == 6, 0.5714285969734192, 0.5333333611488342),
                        ),
                    ),
                ),
            ),
        ),
    )
    normal_bits = T.cast((134 - exponent) << 23, T.uint32)
    normal_inv = T.reinterpret(normal_bits, T.float32) * normal_factor

    subnormal_inv = T.if_then_else(
        mantissa == 0,
        0.0,
        T.if_then_else(
            mantissa == 1,
            512.0,
            T.if_then_else(
                mantissa == 2,
                256.0,
                T.if_then_else(
                    mantissa == 3,
                    170.6666717529297,
                    T.if_then_else(
                        mantissa == 4,
                        128.0,
                        T.if_then_else(
                            mantissa == 5,
                            102.4000015258789,
                            T.if_then_else(mantissa == 6, 85.33333587646484, 73.14286041259766),
                        ),
                    ),
                ),
            ),
        ),
    )
    return T.if_then_else(exponent == 0, subnormal_inv, normal_inv)


def _tl_scale_source_coords(logical_row, k64_word, _scale_layout_id: int, scale_cols):
    row_in_chunk = logical_row % _BLOCKSCALED_CHUNK_ROWS
    flat_word = (
        (logical_row // _BLOCKSCALED_CHUNK_ROWS) * _BLOCKSCALED_CHUNK_ROWS * scale_cols
        + k64_word * _BLOCKSCALED_CHUNK_ROWS
        + (row_in_chunk % 32) * 4
        + (row_in_chunk // 32)
    )
    physical_row = flat_word // scale_cols
    physical_word = flat_word % scale_cols
    return physical_row, physical_word


encode_ue4m3_ceil = _tl_encode_ue4m3_scale_byte_ceil
decode_ue4m3 = _tl_decode_ue4m3_scale_byte
ue4m3_inverse = _tl_ue4m3_scale_byte_inverse


def scale_source_coords(logical_row, k64_word, scale_cols):
    return _tl_scale_source_coords(logical_row, k64_word, 0, scale_cols)
