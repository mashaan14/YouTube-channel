* Don't use `optax.softmax_cross_entropy_with_integer_labels`
  ```python
  # loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch['label']).mean()
  one_hot_labels = jax.nn.one_hot(batch['label'], logits.shape[1])
  loss = optax.softmax_cross_entropy(logits, one_hot_labels).mean()
  ```
* Don't import cifar-10 using `tensorflow_datasets`, use `torchvision` instead.
