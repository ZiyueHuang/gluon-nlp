import numpy as np
from numpy.testing import assert_allclose
import mxnet as mx
import os

if (os.name=='posix'):
    path = os.path.abspath('libdiag_mm_lib.so')
    mx.library.load(path)

from mxnet.gluon import HybridBlock
from gluonnlp.attention_cell import masked_softmax, MultiHeadAttentionCell
mx.npx.set_np()


def multi_head_sliding_window_dot_attn(F, query, key, value, dilation, w, batch_size, seq_length,
                                       symmetric=True, dtype=np.float32):
    # 1. Calculate the attention weights
    # scores shape  (batch_size, seq_length, num_heads, w + w + 1) if symmetric else
    #               (batch_size, seq_length, num_heads, w + 1)
    scores = F.sw_atten_score(query.as_nd_ndarray(), key.as_nd_ndarray(),
                              dilation.as_nd_ndarray(), w=w,
                              symmetric=symmetric).as_np_ndarray()
    # mask shape  (batch_size, seq_length, num_heads, seq_length)
    valid_len = np.zeros((batch_size,))
    valid_len[:] = seq_length
    valid_len = mx.np.array(valid_len, dtype=np.int64, ctx=mx.gpu(0))
    mask = F.mask_like(scores.as_nd_ndarray(), dilation.as_nd_ndarray(), valid_len.as_nd_ndarray(),
                       w=w, symmetric=symmetric).as_np_ndarray()

    attn_weights = masked_softmax(F, scores, mask, dtype=dtype)
    # 2. Calculate the context vector
    # (batch_size, seq_length, num_heads, num_head_units)
    context_vec = F.sw_atten_context(attn_weights.as_nd_ndarray(), value.as_nd_ndarray(),
                                     dilation.as_nd_ndarray(), w=w,
                                     symmetric=symmetric).as_np_ndarray()
    # (batch_size, seq_length, num_units)
    context_vec = F.npx.reshape(context_vec, (-2, -2, -1))

    return context_vec, [scores, attn_weights]


class MultiHeadSlidingWindowAttentionCell(HybridBlock):
    def __init__(self, w, batch_size, seq_length, symmetric=True, query_units=None, num_heads=None,
                 attention_dropout=0.0, scaled: bool = True, normalized: bool = False,
                 eps: float = 1E-6, dtype='float32', layout='NTK'):
        super().__init__()
        self._query_units = query_units
        self._w = w
        self._batch_size = batch_size
        self._seq_length = seq_length
        self._symmetric = symmetric
        self._num_heads = num_heads
        self._attention_dropout = attention_dropout
        self._scaled = scaled
        self._normalized = normalized
        self._eps = eps
        self._dtype = dtype
        assert(layout == 'NTK')
        self._layout = layout
        if self._query_units is not None:
            assert self._num_heads is not None
            assert self._query_units % self._num_heads == 0,\
                'The units must be divisible by the number of heads.'
            self._query_head_units = self._query_units // self._num_heads
        else:
            self._query_head_units = None

    @property
    def layout(self):
        return self._layout

    def hybrid_forward(self, F, query, key, value, dilation):
        return multi_head_sliding_window_dot_attn(F, query=query, key=key, value=value,
                    dilation=dilation, w=self._w, batch_size=self._batch_size,
                    seq_length=self._seq_length, symmetric=self._symmetric, dtype=self._dtype)


def gen_sliding_window_mask_full(batch_size, seq_length, w, w_right):
    """Generate sliding_window attention mask for the full attention matrix ( seq_len^2 ).
    """
    mask_np = np.ones((batch_size, seq_length, seq_length))
    for i in range(seq_length):
        if i > w:
            mask_np[:, i, :(i-w)] = 0
        if i < (seq_length-w_right-1):
            mask_np[:, i, (i+w_right+1):] = 0
    return mask_np


def gen_sliding_window_mask_diag(batch_size, seq_length, w, w_right):
    """Generate sliding_window attention mask for
         the compact attention matrix ( seq_len x (w + w_right + 1) ).
    """
    mask_np = np.ones((batch_size, seq_length, w + w_right + 1))
    for i in range(0, min(w, seq_length)):
        mask_np[:, i, :(w-i)] = 0
    for i in range(seq_length - w_right, seq_length):
        mask_np[:, i, -(w-(seq_length-i)+1):] = 0
    return mask_np


def test_multi_head_sliding_window_dot_attention_cell(batch_size, seq_length, num_heads,
                                                      num_head_units, w, symmetric):
    """Ground truth is computed by the existing MultiHeadAttentionCell using a sliding window mask
    """
    w_right = w if symmetric else 0
    attn_cell = MultiHeadAttentionCell(scaled=False, layout='NTK')
    sw_attn_cell = MultiHeadSlidingWindowAttentionCell(w=w, batch_size=batch_size, seq_length=seq_length,
                       symmetric=symmetric, scaled=False, layout='NTK')
    # Generate the data
    query_np = np.random.normal(0, 1, (batch_size, seq_length, num_heads, num_head_units))
    key_np = np.random.normal(0, 1, (batch_size, seq_length, num_heads, num_head_units))
    value_np = np.random.normal(0, 1, (batch_size, seq_length, num_heads, num_head_units))
    mask_np = gen_sliding_window_mask_full(batch_size, seq_length, w, w_right)
    mask = mx.np.array(mask_np, ctx=mx.gpu(0), dtype=np.float32)

    query = mx.np.array(query_np, ctx=mx.gpu(0), dtype=np.float32)
    key = mx.np.array(key_np, ctx=mx.gpu(0), dtype=np.float32)
    value = mx.np.array(value_np, ctx=mx.gpu(0), dtype=np.float32)

    query.attach_grad()
    key.attach_grad()
    value.attach_grad()

    with mx.autograd.record():
        out, [score, attn_weights] = attn_cell(query, key, value, mask)
        out.backward()

    out_np = out.asnumpy()
    grad_query = query.grad.asnumpy()
    grad_key = key.grad.asnumpy()
    grad_value = value.grad.asnumpy()

    query.grad[:] = 0
    key.grad[:] = 0
    value.grad[:] = 0

    dilation = mx.np.ones((num_heads,), dtype=np.int64, ctx=mx.gpu(0))

    with mx.autograd.record():
        sw_out, [score, attn_weights] = sw_attn_cell(query, key, value, dilation)
        sw_out.backward()

    sw_out_np = sw_out.asnumpy()
    sw_grad_query = query.grad.asnumpy()
    sw_grad_key = key.grad.asnumpy()
    sw_grad_value = value.grad.asnumpy()

    assert_allclose(sw_out_np, out_np, 1E-5, 1E-5)
    #print(np.max(np.abs( sw_out_np - out_np )))
    assert_allclose(sw_grad_key, grad_key, 1E-5, 1E-5)
    #print(np.max(np.abs( sw_grad_key - grad_key )))
    assert_allclose(sw_grad_value, grad_value, 1E-5, 1E-5)
    #print(np.max(np.abs( sw_grad_value - grad_value )))
    assert_allclose(sw_grad_query, grad_query, 1E-5, 1E-5)
    #print(np.max(np.abs( sw_grad_query - grad_query )))
    print('PASS')


# Large
test_multi_head_sliding_window_dot_attention_cell(32, 1024, 8, 64, 32, True)
test_multi_head_sliding_window_dot_attention_cell(32, 1024, 8, 64, 32, False)
test_multi_head_sliding_window_dot_attention_cell(16, 1024, 8, 64, 128, True)
test_multi_head_sliding_window_dot_attention_cell(16, 1024, 8, 64, 128, False)
# Small
test_multi_head_sliding_window_dot_attention_cell(2, 8, 2, 3, 10, True)
test_multi_head_sliding_window_dot_attention_cell(2, 8, 2, 3, 10, False)
test_multi_head_sliding_window_dot_attention_cell(2, 8, 2, 3, 2, True)
test_multi_head_sliding_window_dot_attention_cell(2, 8, 2, 3, 2, False)




