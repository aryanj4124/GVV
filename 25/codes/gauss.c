#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define VECTOR_SIZE 10000  // Change this to the desired size of your vector
#define FILENAME "gaussian_data.dat"  // Name of the output DAT file

// Function to generate random numbers with a Gaussian distribution
double gaussianRandom() {
    double u1 = rand() / (double)RAND_MAX;
    double u2 = rand() / (double)RAND_MAX;
    return sqrt(-2 * log(u1)) * cos(2 * M_PI * u2);
}

int main() {
    int i;
    double vector[VECTOR_SIZE];

    // Seed the random number generator with the current time
    srand(time(NULL));

    // Generate the vector with Gaussian-distributed numbers
    for (i = 0; i < VECTOR_SIZE; i++) {
        vector[i] = gaussianRandom();
    }

    // Open the output DAT file for writing
    FILE* file = fopen(FILENAME, "w");
    if (file == NULL) {
        perror("Error opening file");
        return 1;
    }

    // Write the generated data to the DAT file
    for (i = 0; i < VECTOR_SIZE; i++) {
        fprintf(file, "%f\n", vector[i]);
    }

    // Close the file
    fclose(file);


    return 0;
}


