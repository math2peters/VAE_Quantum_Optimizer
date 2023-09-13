import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import tensorflow as tf
from data_generator import VAEDataGeneratorKeras
import platform
import sys
import pickle
from VAE_optimizer_architecture import VAE, BetaScheduler, PredictorScheduler


np.random.seed(0)



if __name__ == '__main__':
    from multiprocessing import Manager
    
    total_train_samples = 512*10
    batch_size = 64
    number_epochs = 1
    input_size = 64
    latent_dim = 8
    use_saved = True
    

    if use_saved:
        with open('data_v3.pkl', 'rb') as file:
            # Use pickle.load() to load the data from the file
            pickle_data = pickle.load(file)
            print(pickle_data.shape)
            reconstruction_train = pickle_data[:len(pickle_data), :-1][::-1]
            population_train = pickle_data[:len(pickle_data), -1][::-1]
    else:
        # Create a manager and a shared list
        manager = Manager()
        shared_data_list = manager.list()
        # create generators that make artificial data and run them through the qutip hamiltonian
        training_generator = VAEDataGeneratorKeras(array_size=input_size, num_samples=total_train_samples, batch_size=batch_size, shared_data_list=shared_data_list)#, load_pickled="test_data_v0.pkl")
        training_generator.get_population_value = True

    validation_generator = VAEDataGeneratorKeras(array_size=input_size, num_samples=512, batch_size=batch_size)
    validation_generator.get_population_value = True
    
    print(f"Python Platform: {platform.platform()}")
    print(f"Tensor Flow Version: {tf.__version__}")
    print()
    print(f"Python {sys.version}")
    gpu = len(tf.config.list_physical_devices('GPU'))>0
    print("GPU is", "available" if gpu else "NOT AVAILABLE")

    vae = VAE(input_size=input_size, latent_dim=latent_dim, beta=0)
    vae.encoder.summary()
    vae.decoder.summary()    

    beta_scheduler = BetaScheduler(start_beta=0, end_beta=1e-3, turn_on_step=batch_size)
    predictor_scheduler = PredictorScheduler(total_steps=total_train_samples//batch_size)
    lr_decay = keras.optimizers.schedules.ExponentialDecay(1e-2, batch_size, .9825)
    optimizer = keras.optimizers.legacy.Adam(amsgrad=True, learning_rate=lr_decay)
    vae.set_non_trainable_layers(['population_predictor_network'])
    vae.vae_model.compile(optimizer)
    vae.vae_model.summary()


    vae.vae_model.fit(x=[reconstruction_train, population_train],
            y=reconstruction_train,
            validation_data=validation_generator,
            batch_size=batch_size,
            epochs=number_epochs,
            use_multiprocessing=True,
            workers=10,
            callbacks=[beta_scheduler])
            
    vae.vae_model.save_weights("VAEGO_no_population.h5")
    
    
    if not use_saved:
        print(np.array(shared_data_list).shape)
        with open('data_v0.pkl', 'wb') as file:
            pickle.dump(list(shared_data_list), file)

    plt.figure(figsize=(12, 8))

    for i in range(8):
        plt.subplot(4, 2, i+1)
        train_data, population_data = validation_generator.generate_data()
        pred_y = vae.vae_model.predict([train_data, population_data])
        reconstruction = pred_y[0]
        x = np.linspace(0, validation_generator.max_time, len(train_data))
        plt.plot(x, train_data[i])
        plt.scatter(x, reconstruction[i])
    
    plt.tight_layout()
    plt.show()