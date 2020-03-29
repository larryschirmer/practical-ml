input data: a linear dataset is generated with a known `m` and `b`

two tensor variables are generated with a random float and uses a gradient tape to refine the a guess for `m` and `b` such that the mean squared difference the predicted `Y` and the actual `Y` is minimized
