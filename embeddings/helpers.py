import io
import matplotlib.pyplot as plt
import tensorflow as tf


def pad_sequences(dataset):
    padded_shapes = ([None], ())
    shuffle_buffer_size = 1000
    padded_batch_size = 10

    return dataset.shuffle(shuffle_buffer_size).padded_batch(padded_batch_size, padded_shapes=padded_shapes)


def plot_history(history, filename='embeddings/history.png'):
    plt.close('all')
    fig = plt.figure()

    loss = fig.add_subplot(121)
    loss.set_title('loss')
    loss.plot(history['loss'], '.', label="training loss")
    loss.plot(history['val_loss'], 'b', label="calidation loss")

    accuracy = fig.add_subplot(122)
    accuracy.set_title('accuracy')
    accuracy.plot(history['accuracy'], '.', label="training accuracy")
    accuracy.plot(history['val_accuracy'],
                  'b', label="validation accuracy")

    plt.tight_layout()
    fig.savefig(filename)


def save(model, filename='embeddings/saved_models/model.h5'):
    model.save(filename)


def load(filename='embeddings/saved_models/model.h5'):
    return tf.keras.models.load_model(filename)


def saveEmbedding(embedding_layer, encoder):
    encoding_weights = embedding_layer.get_weights()[0]

    vector_file_instance = io.open(
        'embeddings/embedding/vectors.tsv', 'w', encoding='utf-8')
    metadata_file_instance = io.open(
        'embeddings/embedding/metadata.tsv', 'w', encoding='utf-8')

    for num, word in enumerate(encoder.subwords):
        vec = encoding_weights[num + 1]  # skip 0, it's padding.
        vector_file_instance.write('\t'.join([str(x) for x in vec]) + "\n")
        metadata_file_instance.write(word + "\n")
    vector_file_instance.close()
    metadata_file_instance.close()
