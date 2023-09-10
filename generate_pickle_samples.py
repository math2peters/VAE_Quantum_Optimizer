import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import tensorflow as tf
from VAE_generator import VAEDataGeneratorKeras
import platform
import sys
import pickle
from VAEGO_keras import VAE



if __name__ == '__main__':
    from multiprocessing import Manager
    
    total_train_samples = 512*2000
    batch_size = 64
    number_epochs = 1
    input_size = 64
    latent_dim = 8


    # Create a manager and a shared list
    manager = Manager()
    shared_data_list = manager.list()
    # create generators that make artificial data and run them through the qutip hamiltonian
    training_generator = VAEDataGeneratorKeras(array_size=input_size, num_samples=total_train_samples, 
                                               batch_size=batch_size, shared_data_list=shared_data_list, save_data=True)
    training_generator.get_cost_value = True

    validation_generator = VAEDataGeneratorKeras(array_size=input_size, num_samples=512, batch_size=batch_size)
    validation_generator.get_cost_value = True
    
    print(f"Python Platform: {platform.platform()}")
    print(f"Tensor Flow Version: {tf.__version__}")
    print(f"Keras Version: {keras.__version__}")
    print()
    print(f"Python {sys.version}")
    gpu = len(tf.config.list_physical_devices('GPU'))>0
    print("GPU is", "available" if gpu else "NOT AVAILABLE")

    vae = VAE(input_size=input_size, latent_dim=latent_dim)

    lr_decay = keras.optimizers.schedules.ExponentialDecay(1e-2, batch_size*batch_size, 1e-5)
    optimizer = keras.optimizers.legacy.Adam(amsgrad=True, learning_rate=lr_decay)#, beta_1=0.5)

    vae.vae_model.compile(optimizer)


    vae.vae_model.fit(training_generator,
            batch_size=batch_size,
            epochs=number_epochs)#,
            # use_multiprocessing=True,
            # workers=8)
    

    shared_data_array = np.array(list(shared_data_list))
    shared_data_array = shared_data_array.reshape(-1, 65)
    print(shared_data_array.shape)
    reconstruction_train = shared_data_array[:, :-1]
    cost_train = shared_data_array[:, -1]
    print("Mean cost is: {:.4f}".format(np.mean(cost_train)))
    with open('data_v3.pkl', 'wb') as file:
        pickle.dump(shared_data_array, file)

    plt.figure(figsize=(12, 8))

    for i in range(10):
        plt.subplot(5, 2, i+1)
        train_data, cost_data = validation_generator.generate_data()
        pred_y = vae.vae_model.predict([train_data, cost_data])
        reconstruction = pred_y[0]
        x = np.linspace(0, validation_generator.max_time, len(train_data))
        plt.plot(x, train_data[i])
        plt.scatter(x, reconstruction[i])
    
    plt.tight_layout()
    plt.show()