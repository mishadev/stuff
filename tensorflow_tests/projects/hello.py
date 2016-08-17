import tensorflow as tf
import numpy as np
import resource

def create_samples(n_clusters, n_samples_per_cluser, n_features, embiggen_factor, seed):
    slices=[]
    centroids=[]

    for cluster_idx in range(n_clusters):
        samples = tf.random_normal((n_samples_per_cluser, n_features),
                mean=0.0, stddev=5.0, dtype=tf.float32, seed=seed,
                name="cls_{}".format(cluster_idx))
        current_centroid = (np.random.random((1, n_features)) * embiggen_factor) - (embiggen_factor/2)
        centroids.append(current_centroid)
        samples += current_centroid
        slices.append(samples)

        samples = tf.concat(0, slices, name='samples')
        centroids = tf.concat(0, centroids, name='centroids')

    return centroids, samples

def plot_clusters(all_samples, centroids, n_samples_per_cluster):
    import matplotlib.pyplot as plt
    # Plot out the different clusters
    # Choose a different colour for each cluster
    colour = plt.cm.rainbow(np.linspace(0,1,len(centroids)))
    for i, centroid in enumerate(centroids):
        # Grab just the samples fpr the given cluster and plot them out with a new colour
        samples = all_samples[i*n_samples_per_cluster:(i+1)*n_samples_per_cluster]
        plt.scatter(samples[:,0], samples[:,1])
    # Also plot centroid
    plt.plot(centroid[0], centroid[1], markersize=15, marker="x", color='k', mew=10)
    plt.plot(centroid[0], centroid[1], markersize=10, marker="x", color='m', mew=5)
    plt.savefig("./plot.png")

def choose_random_centroids(samples, n_clusters):
    n_samples = tf.shape(samples)[0]
    random_indices = tf.random_shuffle(tf.range(0, n_samples))
    begin = [0]
    size = [n_clusters]
    centroid_indices = tf.slice(random_indices, begin, size)
    initial_centroids = tf.gather(samples, centroid_indices)
    return initial_centroids

def run():
    n_features = 2
    n_clusters = 3
    n_samples_per_cluster = 1000
    seed = 700
    embiggen_factor = 60

    np.random.seed(seed)

    centroids, samples = create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed)
    initial_centroids = choose_random_centroids(samples, n_clusters)
    nearest_indices = assign_to_nearest(samples, initial_centroids)
    updated_centroids = update_centroids(samples, nearest_indices, n_clusters)

    model = tf.initialize_all_variables()
    with tf.Session() as session:
        sample_values = session.run(samples)
        centroid_values = session.run(updated_centroids)

    plot_clusters(sample_values, centroid_values, n_samples_per_cluster)

def assign_to_nearest(samples, centroids):
    # START from http://esciencegroup.com/2016/01/05/an-encounter-with-googles-tensorflow/
    expanded_vectors = tf.expand_dims(samples, 0)
    expanded_centroids = tf.expand_dims(centroids, 1)
    distances = tf.reduce_sum( tf.square(
        tf.sub(expanded_vectors, expanded_centroids)), 2)
    mins = tf.argmin(distances, 0)
    # END from http://esciencegroup.com/2016/01/05/an-encounter-with-googles-tensorflow/
    nearest_indices = mins
    return nearest_indices

def update_centroids(samples, nearest_indices, n_clusters):
    # Updates the centroid to be the mean of all samples associated with it.
    nearest_indices = tf.to_int32(nearest_indices)
    partitions = tf.dynamic_partition(samples, nearest_indices, n_clusters)
    new_centroids = tf.concat(0, [tf.expand_dims(tf.reduce_mean(partition, 0), 0) for partition in partitions])
    return new_centroids

run()

print("{} Mb".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
