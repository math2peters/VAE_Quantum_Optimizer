import matplotlib.pyplot as plt
import numpy as np
from data_generator import VAEDataGeneratorKeras


if __name__ == '__main__':
    
    generator = VAEDataGeneratorKeras(array_size=64, num_samples=1, batch_size=1, system='three_level')
    
    x, p = generator.generate_data()
    x = np.ones(64)
    p_full = generator.get_population(x, return_full_values=True).expect
    t = np.linspace(0, generator.max_time, len(x))
    
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(t, x)
    plt.xlabel("Time (arb.)")
    plt.ylabel("Drive Amplitude (Arb.)")
    plt.subplot(2, 1, 2)
    [plt.plot(t, p_full[i], label="|{}>".format(i)) for i in range(len(p_full))]
    #plt.yscale('log')
    plt.legend()
    plt.xlabel("Time (arb.)")
    plt.ylabel("Population in state")

    plt.tight_layout()
    plt.show()