import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import tensorflow as tf
from data_generator import VAEDataGeneratorKeras
import platform
import sys
import pickle
from VAE_optimizer_architecture import VAE

"""Code to generate a pickle file that contains samples and the associated populations
    """

if __name__ == '__main__':
    from multiprocessing import Manager
    
    total_train_samples = 10000#512*2000
    batch_size = 64
    number_epochs = 1
    input_size = 64
    latent_dim = 8
    predictor_data = True


    # Create a manager and a shared list
    manager = Manager()
    shared_data_list = manager.list()
    # create generators that make artificial data and run them through the qutip hamiltonian
    training_generator = VAEDataGeneratorKeras(array_size=input_size, num_samples=total_train_samples, 
                                               batch_size=batch_size, shared_data_list=shared_data_list, save_data=True, system='three_level')
    training_generator.get_population_value = predictor_data
    
    print(f"Python Platform: {platform.platform()}")
    print(f"Tensor Flow Version: {tf.__version__}")
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
            epochs=number_epochs)
    

    shared_data_array = np.array(list(shared_data_list))
    shared_data_array = shared_data_array.reshape(-1, 65)
    print(shared_data_array.shape)
    reconstruction_train = shared_data_array[:, :-1]
    cost_train = shared_data_array[:, -1]
    print("Mean population in |1> is: {:.4f}".format(np.mean(cost_train)))
    print("Max population in |1> is: {:.4f}".format(np.max(cost_train)))
    
    if predictor_data:
        with open('predictor_data.pkl', 'wb') as file:
            pickle.dump(shared_data_array, file)
    else:
        with open('VAE_data.pkl', 'wb') as file:
            pickle.dump(shared_data_array, file)

    # plt.figure(figsize=(12, 8))
    
    # show_NN_predictions = False

    # for i in range(10):
    #     plt.subplot(5, 2, i+1)
    #     train_data, cost_data = validation_generator.generate_data()
    #     pred_y = vae.vae_model.predict([train_data, cost_data])
    #     reconstruction = pred_y[0]
    #     x = np.linspace(0, validation_generator.max_time, len(train_data))
    #     plt.plot(x, train_data[i])
    #     if show_NN_predictions:
    #         plt.scatter(x, reconstruction[i])
    
    # # plt.xlabel("Time (arb.)")
    # # plt.ylabel("Drive Amplitude (Arb.)")
    # plt.tight_layout()
    # plt.show()