import openrouteservice as ors
import folium
import random

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

client = ors.Client(key='5b3ce3597851110001cf62481b8629b3b28a4c4f9322613051d25c40')


def example():
    m = folium.Map(location=[52.521861, 13.40744], tiles='cartodbpositron', zoom_start=13)

    # Some coordinates in Berlin
    coordinates = [[13.384116, 52.533558], [13.428726, 52.519355], [13.41774, 52.498929], [13.374825, 52.496369]]

    matrix = client.distance_matrix(
        locations=coordinates,
        profile='driving-car',
        metrics=['distance', 'duration'],
        validate=False,
    )

    for marker in coordinates:
        folium.Marker(location=list(reversed(marker))).add_to(m)

    print("Durations in secs: {}\n".format(matrix['durations']))
    print("Distances in m: {}".format(matrix['distances']))

    m.save("map.html")


def generate_coordinates(origins, offset, cluster_ratio, n_adresses, seed=0, draw=True):
    random.seed(seed)
    np.random.seed(seed)
    adresses = np.array([(random.choice(origins)[0] + random.uniform(-1, 1) * offset, random.choice(origins)[1] + random.uniform(-1, 1) * offset) for _ in range(cluster_ratio * n_adresses)])

    kmeans = KMeans(init='k-means++', n_clusters=n_adresses, n_init=10)
    kmeans.fit(adresses)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = offset * 0.01  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = adresses[:, 0].min() - offset, adresses[:, 0].max() + offset
    y_min, y_max = adresses[:, 1].min() - offset, adresses[:, 1].max() + offset
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    centroids = kmeans.cluster_centers_

    if draw:
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(1)
        plt.clf()
        plt.imshow(Z, interpolation='nearest',
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   cmap=plt.cm.tab20,
                   aspect='auto', origin='lower')

        plt.plot(adresses[:, 0], adresses[:, 1], 'k.', markersize=2)
        # Plot the centroids as a white X
        plt.scatter(centroids[:, 0], centroids[:, 1],
                    marker='x', s=150, linewidths=3,
                    color='w', zorder=10)
        np_origins = np.array(origins)
        plt.scatter(np_origins[:, 0], np_origins[:, 1],
                    marker='o', s=150, linewidths=1,
                    color='w', zorder=10)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        plt.show()

    return origins + list(map(list, centroids))


def get_distance_matrix(coordinates):
    return client.distance_matrix(
        locations=coordinates,
        profile='driving-car',
        metrics=['distance', 'duration'],
        validate=False,
    )


def generate_map_file(school_coordinates, capacity, offset=0.1, n_adresses=8, filename="generated_map.txt"):
    coordinates = generate_coordinates(school_coordinates, offset, 5, n_adresses)
    matrix = get_distance_matrix(coordinates)

    f = open(filename, "w+")
    lines = ["capacity " + str(capacity)]
    schools = "schools "
    for i in range(len(school_coordinates)):
        schools += str(i) + " "
    lines.append(schools[:-1])
    lines.append("nodes " + str(len(school_coordinates) + n_adresses))
    for line in matrix["durations"]:
        str_line = ""
        for duration in line:
            str_line += str(int(duration)) + " "
        lines.append(str_line)
    for coord in coordinates:
        lines.append("(" + "{:.6f}".format(coord[0]) + ", " + "{:.6f}".format(coord[1]) + ")")

    f.write('\n'.join(lines))


def main():
    generate_map_file([[13.384116, 52.533558], [13.428726, 52.519355]], 10, 0.1, 8)


if __name__ == "__main__":
    import sys

    main()
