"""
Utility functions for fast ai mooc
"""
import keras.backend as K



def limit_mem():
    """ Limits memory use on GPUs """
    K.get_session().close()
    config = K.tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=config))
