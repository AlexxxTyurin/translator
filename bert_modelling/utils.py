import tensorflow as tf


def swap_axes(x, axes):
    leading = tf.range(tf.rank(x) - len(axes))  # [0, 1]
    trailing = axes + tf.rank(x)  # [3, 2]
    new_order = tf.concat([leading, trailing], axis=0)  # [0, 1, 3, 2]
    res = tf.transpose(x, new_order)

    return res


def prune_linear_layer(layer: tf.keras.layers.Dense, index: tf.int64, dim: int = 0) -> tf.keras.layers.Dense:

    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer
