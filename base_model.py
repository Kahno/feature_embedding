import tensorflow as tf

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Embedding, Activation
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.layers.experimental.preprocessing import Hashing
from tensorflow.keras.backend import pow, sum
from tensorflow.keras.metrics import AUC

from tensorflow_addons.optimizers import LazyAdam


def BaseModel(
    num_feats, num_bins, num_factors=-1,
    gbdt_enhance=None,
    int_modules=[], lin_modules=[], both_modules=[],
    optimizer=LazyAdam(0.003), loss="binary_crossentropy", additional_metrics=[],
):
    if gbdt_enhance:
        num_feats += gbdt_enhance["num_trees"]

    inputs = Input((num_feats,), dtype=tf.int32)    
    hashs = Hashing(num_bins=num_bins)(inputs)
    
    # Basic feature interaction embeddings
    if num_factors > 1:
        int_embs = Embedding(num_bins, num_factors, input_length=num_feats)(hashs)
    
    # Basic linear feature embeddings
    lin_embs = Embedding(num_bins, 1, input_length=num_feats)(hashs)
    
    # Interaction embedding enrichment modules
    if int_modules and num_factors > 1:
        for layer in int_modules:
            int_embs = layer(int_embs)
    
    # Linear embedding enrichment modules
    if lin_modules:
        for layer in lin_modules:
            lin_embs = layer(lin_embs)
    
    # Interaction + Linear embedding enrichment modules
    if both_modules and num_factors > 1:
        for layer in both_modules:
            int_embs, lin_embs = layer([int_embs, lin_embs])
    
    # Bias term for classic FM
    bias_term = tf.Variable(GlorotUniform()(shape=(), dtype=tf.float32), True)
    
    # Linear term for classic FM
    linear_term = tf.math.reduce_sum(lin_embs, axis=1)
    
    # Interaction term for classic FM
    # (if ignored, model becomes classic LR)
    interaction_term = tf.zeros((), tf.float32)
    if num_factors > 1:
        a = pow(tf.math.reduce_sum(int_embs, axis=1), 2)
        b = tf.math.reduce_sum(pow(int_embs, 2), axis=1)
        interaction_term += sum(a - b, 1, keepdims=True) * 0.5
    
    outputs = Activation("sigmoid")(bias_term + linear_term + interaction_term)
    
    modules = int_modules + lin_modules + both_modules
    combined_modules_name = "_".join([m.title for m in modules])
    
    model_name = f"FM{num_factors}" if num_factors > 1 else "LR"
    model_name += f"_{num_bins}"
    if modules:
        model_name += f"__{combined_modules_name}"
    if gbdt_enhance:
        model_name += f"_GBDT{gbdt_enhance['num_trees']}_{gbdt_enhance['num_leaves']}"
    
    model = Model(inputs=inputs, outputs=outputs, name=model_name)
    model.compile(loss=loss, optimizer=optimizer, metrics=[AUC()] + additional_metrics)
    return model


