# Fast Monte Carlo Algorithm for Matrix Multiplication

This repository contains an implementation of the Approximate Matrix Multiplication Algorithm. It is a fork of the original repository [Fast-Monte-Carlo-Algorithm-for-Matrix-Multiplication](https://github.com/Kirk-Zhen/Fast-Monte-Carlo-Algorithm-for-Matrix-Multiplication), but with some additional optimizations.

## Optimizations

The following optimizations have been made in this implementation:

- Higher Speed: To improve the algorithm's performance, the usage of for loops has been minimized.

- Lower Error: The `replace` parameter in `np.random.choice` has been set to `False`, resulting in a reduced error rate.

Feel free to explore this optimized version of the algorithm for faster and more accurate matrix multiplication!

## Contributions

Contributions to this repository are welcome! If you have any suggestions for further optimizations or enhancements, feel free to open an issue or submit a pull request.

## Acknowledgments

We would like to thank the original author of the Monte Carlo Algorithm for Matrix Multiplication implementation, Kirk Zhen. Our optimizations have built upon his work, making the algorithm faster and more accurate.