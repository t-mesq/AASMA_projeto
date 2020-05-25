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


def distribute_addresses(distribution, n_schools):
    students_distribution = distribution.copy()
    schools_distribution = [[0] * len(distribution) for _ in range(n_schools)]

    for i, count in enumerate(distribution):
        for student in range(count):
            random.choice(schools_distribution)[i] += 1

    return schools_distribution


def generate_coordinates(origins, offset, n_points, n_addresses, seed=0, draw=True):
    random.seed(seed)
    np.random.seed(seed)
    addresses = np.array([(random.choice(origins)[0] + random.uniform(-1, 1) * offset, random.choice(origins)[1] + random.uniform(-1, 1) * offset) for _ in range(n_points)])

    kmeans = KMeans(init='k-means++', n_clusters=n_addresses, n_init=10)
    kmeans.fit(addresses)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = offset * 0.01  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = addresses[:, 0].min() - offset, addresses[:, 0].max() + offset
    y_min, y_max = addresses[:, 1].min() - offset, addresses[:, 1].max() + offset
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

        plt.plot(addresses[:, 0], addresses[:, 1], 'k.', markersize=2)
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

    return origins + list(map(list, centroids)), [list(kmeans.labels_).count(i) for i in range(n_addresses)]


def get_distance_matrix(coordinates):
    return client.distance_matrix(
        locations=coordinates,
        profile='driving-car',
        metrics=['distance', 'duration'],
        validate=False,
    )


def generate_map_file(school_coordinates, capacity, offset=0.1, n_points=80, n_addresses=8, iterations=10000, filename="generated_map.txt"):
    coordinates, addresses_distribution = generate_coordinates(school_coordinates, offset, n_points, n_addresses, seed=10)
    matrix = get_distance_matrix(coordinates)
    schools_distribution = distribute_addresses(addresses_distribution, len(school_coordinates))

    f = open(filename, "w+")
    lines = ["capacity " + str(capacity)]
    lines.append("iterations " + str(iterations))
    schools = "schools "
    for i in range(len(school_coordinates)):
        schools += str(i) + " "
    lines.append(schools[:-1])
    for school_distribution in schools_distribution:
        lines.append(' '.join(list(map(str, [0]*len(school_coordinates) + school_distribution))))
    lines.append("nodes " + str(len(school_coordinates) + n_addresses))
    for line in matrix["durations"]:
        str_line = ""
        for duration in line:
            str_line += str(int(duration)) + " "
        lines.append(str_line)
    for coord in coordinates:
        lines.append("(" + "{:.6f}".format(coord[0]) + ", " + "{:.6f}".format(coord[1]) + ")")

    f.write('\n'.join(lines))


def main():

    if len(arg) < 2:
        filename = "generated_map.txt"
    else:
        filename = arg[1]


    generate_map_file([[13.384116, 52.533558], [13.428726, 52.519355]], 10, 0.1, 30, 8, 10000000, filename)


if __name__ == "__main__":
    import sys

    main()
