import tensorflow as tf
import time
import numpy as np

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps
        
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def loss_function(real, pred):
    """
    Вычисление функции потерь с учетом padding токенов.
    
    Args:
        real: реальные значения
        pred: предсказанные значения
    """
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = tf.keras.losses.sparse_categorical_crossentropy(real, pred, from_logits=True)
    
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

def train_step(transformer, optimizer, inp, tar):
    """
    Один шаг обучения.
    
    Args:
        transformer: модель трансформера
        optimizer: оптимизатор
        inp: входные данные
        tar: целевые данные
    """
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    
    with tf.GradientTape() as tape:
        predictions, _ = transformer([inp, tar_inp], training=True)
        loss = loss_function(tar_real, predictions)
        
    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    
    return loss

def train_model(transformer, dataset, epochs, optimizer):
    """
    Обучение модели.
    
    Args:
        transformer: модель трансформера
        dataset: набор данных для обучения
        epochs: количество эпох
        optimizer: оптимизатор
    """
    for epoch in range(epochs):
        start = time.time()
        total_loss = 0
        steps = 0
        
        for (batch, (inp, tar)) in enumerate(dataset):
            batch_loss = train_step(transformer, optimizer, inp, tar)
            total_loss += batch_loss
            steps += 1
            
            if batch % 5 == 0:
                print(f'Epoch {epoch + 1} Batch {batch} Loss {batch_loss:.4f}')
        
        avg_loss = total_loss / steps
        print(f'Epoch {epoch + 1} Loss {avg_loss:.4f}')
        print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n') 