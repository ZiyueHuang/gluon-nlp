import mxnet as mx
import os
import time
import numpy as np
from numpy.testing import assert_allclose

#load library
if (os.name=='posix'):
    path = os.path.abspath('libdiag_mm_lib.so')
    mx.library.load(path)


def diag_to_full(diag, batch_size, seq_length, num_heads, w, w_right):
    """ e.g.  w = 2, w_right = 2, seq_length = 6
    diag : sliding window attention score (in the compact form)
      y y x x x
      y x x x x
      x x x x x
      x x x x x
      x x x x y
      x x x y y
    output : full attention score (seq_length^2)
      x x x 0 0 0
      x x x x 0 0
      x x x x x 0
      0 x x x x x
      0 0 x x x x
      0 0 0 x x x
    """
    w_len = w + w_right + 1
    full = np.zeros((batch_size, seq_length, num_heads, seq_length + w + w_right))
    for i in range(seq_length):
        full[:, i, :, i : (i + w_len)] = diag[:, i, :, 0 : w_len]
    full = full[:, :, :, w : (seq_length + w)]
    return full


def out_diag_set_zero(full, seq_length, w, w_right):
    """
    Set the elements outside the sliding window to 0.
    e.g.  w = 2, w_right = 2, seq_length = 6
    full : full attention score (seq_length^2)
      x x x y y y
      x x x x y y
      x x x x x y
      y x x x x x
      y y x x x x
      y y y x x x
    output :
      x x x 0 0 0
      x x x x 0 0
      x x x x x 0
      0 x x x x x
      0 0 x x x x
      0 0 0 x x x
    """
    for i in range(seq_length):
        if i > w:
            full[:, i, :, :(i-w)] = 0
        if i < (seq_length-w_right-1):
            full[:, i, :, (i+w_right+1):] = 0


def test(batch_size, seq_length, num_heads, num_hidden, w, symmetric):
    """Ground truth is computed by numpy (using full attention matrix)
    A : matrix of shape (m, n); B : matrix of shape (n, k).
    C = np.matmul(A, B)
    Given the gradients w.r.t. C, the gradients w.r.t. A and B are computed by
    dA = np.matmul(dC, B.T)
    dB = np.matmul(A.T, dC)
    """
    w_right = w if symmetric else 0
    w_len = w + w_right + 1

    print('TEST for query * key = score')

    query_sym = mx.sym.Variable('query')
    key_sym = mx.sym.Variable('key')
    dilation_sym = mx.sym.Variable('dilation')
    score_sym = mx.sym.my_diag_mm(query_sym, key_sym, dilation_sym, w=w,
                                  diagonal_lhs=False, transpose_lhs=False, symmetric=symmetric)
    # the buffer for gradients w.r.t inputs
    in_grads = [mx.nd.ones((batch_size, seq_length, num_heads, num_hidden), ctx=mx.gpu(0)),
                mx.nd.ones((batch_size, seq_length, num_heads, num_hidden), ctx=mx.gpu(0)),
                mx.nd.ones((num_heads,), dtype=np.int64, ctx=mx.gpu(0))]

    query_np = np.random.randn(batch_size, seq_length, num_heads, num_hidden)
    query = mx.nd.array(query_np, ctx=mx.gpu(0))
    key_np = np.random.randn(batch_size, seq_length, num_heads, num_hidden)
    key = mx.nd.array(key_np, ctx=mx.gpu(0))
    dilation = mx.nd.ones((num_heads,), dtype=np.int64, ctx=mx.gpu(0))

    score_exe = score_sym._bind(ctx=mx.gpu(0), args={'query':query, 'key':key, 'dilation':dilation},
                                args_grad=in_grads)
    # batch_size, seq_length, num_heads, w_len
    score = score_exe.forward()[0]
    # batch_size, seq_length, num_heads, seq_length
    score_np = diag_to_full(score.asnumpy(), batch_size, seq_length, num_heads, w, w_right)

    # ground truth is computed by numpy (using full attention matrix)
    query_np_T = np.transpose(query_np, (0, 2, 1, 3))
    key_np_T = np.transpose(key_np, (0, 2, 3, 1))
    score_gt = np.matmul(query_np_T, key_np_T)
    score_gt = np.transpose(score_gt, (0, 2, 1, 3))
    out_diag_set_zero(score_gt, seq_length, w, w_right)
    #print('score diff  ', np.max(np.abs( score_np - score_gt )))
    assert_allclose(score_np, score_gt, 1E-3, 1E-3)

    out_grad_np = np.random.randn(batch_size, seq_length, num_heads, w_len)
    out_grad = mx.nd.array(out_grad_np, ctx=mx.gpu(0))
    score_exe.backward([out_grad])

    # ground truth is computed by numpy (using full attention matrix)
    # batch_size, seq_length, num_heads, seq_length
    out_grad_np = diag_to_full(out_grad_np, batch_size, seq_length, num_heads, w, w_right)
    out_diag_set_zero(out_grad_np, seq_length, w, w_right)
    # batch_size, num_heads, seq_length, seq_length
    out_grad_np = np.transpose(out_grad_np, (0, 2, 1, 3))
    lhs_grad_np = np.matmul(out_grad_np, key_np.transpose((0, 2, 1, 3)))
    #print('lhs grad diff  ', np.max(np.abs( in_grads[0].asnumpy() - lhs_grad_np.transpose((0, 2, 1, 3)) )))
    assert_allclose(in_grads[0].asnumpy(), lhs_grad_np.transpose((0, 2, 1, 3)), 1E-3, 1E-3)
    # batch_size, num_heads, seq_length, seq_length (Transpose)
    out_grad_np = np.transpose(out_grad_np, (0, 1, 3, 2))
    rhs_grad_np = np.matmul(out_grad_np, query_np.transpose((0, 2, 1, 3)))
    #print('rhs grad diff  ', np.max(np.abs( in_grads[1].asnumpy() - rhs_grad_np.transpose((0, 2, 1, 3)) )))
    assert_allclose(in_grads[1].asnumpy(), rhs_grad_np.transpose((0, 2, 1, 3)), 1E-3, 1E-3)
    print('PASS')

    mx.nd.waitall()


    print('TEST for score * value = out')

    value_np = np.random.randn(batch_size, seq_length, num_heads, num_hidden)
    value = mx.nd.array(value_np, ctx=mx.gpu(0))

    score_sym = mx.sym.Variable('score')
    value_sym = mx.sym.Variable('value')
    dilation_sym = mx.sym.Variable('dilation')
    out_sym = mx.sym.my_diag_mm(score_sym, value_sym, dilation_sym, w=w,
                                diagonal_lhs=True, transpose_lhs=False, symmetric=symmetric)
    in_grads = [mx.nd.ones(score.shape, ctx=mx.gpu(0)),
                mx.nd.ones(value.shape, ctx=mx.gpu(0)),
                mx.nd.ones((num_heads,), dtype=np.int64, ctx=mx.gpu(0))]
    out_exe = out_sym._bind(ctx=mx.gpu(0), args={'score':score, 'value':value, 'dilation':dilation},
                            args_grad=in_grads)
    # batch_size, seq_length, num_heads, num_hidden
    out = out_exe.forward()[0]

    # ground truth is computed by numpy (using full attention matrix)
    score_np = diag_to_full(score.asnumpy(), batch_size, seq_length, num_heads, w, w_right)
    score_np = np.transpose(score_np, (0, 2, 1, 3)) # batch_size, num_heads, seq_length, seq_length
    value_np_T = np.transpose(value_np, (0, 2, 1, 3)) # batch_size, num_heads, seq_length, num_hidden
    out_gt = np.matmul(score_np, value_np_T)
    #print('out diff  ', np.max(np.abs( out.asnumpy() - out_gt.transpose((0, 2, 1, 3)) )))
    assert_allclose(out.asnumpy(), out_gt.transpose((0, 2, 1, 3)), 1E-3, 1E-3)

    out_grad_np = np.random.randn(batch_size, seq_length, num_heads, num_hidden)
    out_grad = mx.nd.array(out_grad_np, ctx=mx.gpu(0))
    out_exe.backward([out_grad])

    # ground truth is computed by numpy (using full attention matrix)
    # batch_size, num_heads, seq_length, num_hidden
    out_grad_np_T = np.transpose(out_grad_np, (0, 2, 1, 3))
    lhs_grad_np = np.matmul(out_grad_np_T, value_np_T.transpose((0, 1, 3, 2)))
    # batch_size, num_heads, seq_length, seq_length
    lhs_grad_np = np.transpose(lhs_grad_np, (0, 2, 1, 3))
    out_diag_set_zero(lhs_grad_np, seq_length, w, w_right)
    lhs_grad = diag_to_full(in_grads[0].asnumpy(), batch_size, seq_length, num_heads, w, w_right)
    #print('lhs grad diff  ', np.max(np.abs( lhs_grad - lhs_grad_np )))
    assert_allclose(lhs_grad, lhs_grad_np, 1E-3, 1E-3)
    score_np_T = np.transpose(score_np, (0, 1, 3, 2)) # Transpose
    rhs_grad_np = np.matmul(score_np_T, out_grad_np_T)
    #print('rhs grad diff  ', np.max(np.abs( in_grads[1].asnumpy() - rhs_grad_np.transpose((0, 2, 1, 3)) )))
    assert_allclose(in_grads[1].asnumpy(), rhs_grad_np.transpose((0, 2, 1, 3)), 1E-3, 1E-3)
    print('PASS')

    mx.nd.waitall()




def diag_to_full_d(diag, batch_size, seq_length, num_heads, w, w_right, dilation):
    w_len = dilation * (w + w_right) + 1
    full = np.zeros((batch_size, seq_length, num_heads, seq_length + dilation * (w + w_right)))
    for i in range(seq_length):
        full[:, i, :, i : (i + w_len) : dilation] = diag[:, i, :, :]
    full = full[:, :, :, dilation * w : (seq_length + dilation * w)]
    return full


def extract_diag_d(full, batch_size, seq_length, num_heads, w, w_right, dilation):
    w_len = dilation * (w + w_right) + 1
    full = np.concatenate([np.zeros((batch_size, seq_length, num_heads, w * dilation)),
                           full,
                           np.zeros((batch_size, seq_length, num_heads, w_right * dilation))],
                          axis=-1)
    diag = np.zeros((batch_size, seq_length, num_heads, w + w_right + 1))
    for i in range(seq_length):
        diag[:, i, :, :] = full[:, i, :, i : (i + w_len) : dilation]
    return diag



def test_d(batch_size, seq_length, num_heads, num_hidden, w, symmetric, d):
    w_right = w if symmetric else 0
    # query * key = score
    query_np = np.random.randn(batch_size, seq_length, num_heads, num_hidden)
    query = mx.nd.array(query_np, ctx=mx.gpu(0))
    key_np = np.random.randn(batch_size, seq_length, num_heads, num_hidden)
    key = mx.nd.array(key_np, ctx=mx.gpu(0))
    dilation = mx.nd.ones((num_heads,), dtype=np.int64, ctx=mx.gpu(0))
    dilation[:] = d

    score = mx.nd.my_diag_mm(query, key, dilation, w=w, diagonal_lhs=False,
                             transpose_lhs=False, symmetric=symmetric)

    # ground truth is computed by numpy (using full attention matrix)
    query_np_T = np.transpose(query_np, (0, 2, 1, 3))
    key_np_T = np.transpose(key_np, (0, 2, 3, 1))
    score_gt = np.matmul(query_np_T, key_np_T)
    score_gt = np.transpose(score_gt, (0, 2, 1, 3))
    score_gt = extract_diag_d(score_gt, batch_size, seq_length, num_heads, w, w_right, d)

    #print(np.max( np.abs(score_gt - score.asnumpy()) ))
    assert_allclose(score.asnumpy(), score_gt, 1E-3, 1E-3)

    # score * value = out
    value_np = np.random.randn(batch_size, seq_length, num_heads, num_hidden)
    value = mx.nd.array(value_np, ctx=mx.gpu(0))
    out = mx.nd.my_diag_mm(score, value, dilation, w=w,
                           diagonal_lhs=True, transpose_lhs=False, symmetric=symmetric)

    # ground truth is computed by numpy (using full attention matrix)
    score_full = diag_to_full_d(score_gt, batch_size, seq_length, num_heads, w, w_right, d)
    score_full = np.transpose(score_full, (0, 2, 1, 3)) # batch_size, num_heads, seq_length, seq_length
    value_np_T = np.transpose(value_np, (0, 2, 1, 3)) # batch_size, num_heads, seq_length, num_hidden
    out_gt = np.matmul(score_full, value_np_T).transpose((0, 2, 1, 3))

    #print(np.max( np.abs(out_gt - out.asnumpy()) ))
    assert_allclose(out_gt, out.asnumpy(), 1E-3, 1E-3)
    print('PASS')

    mx.nd.waitall()




# Small
test(2, 8, 2, 3, 2, True)
test(2, 8, 2, 3, 2, False)
# Large
test(32, 2048, 8, 64, 64, True)
test(32, 2048, 8, 64, 64, False)

# Small
test_d(2, 10, 2, 3, 2, True, 2)
test_d(2, 10, 2, 3, 2, False, 2)
# Large
test_d(32, 2048, 8, 64, 64, True, 4)
test_d(32, 2048, 8, 64, 64, False, 4)



