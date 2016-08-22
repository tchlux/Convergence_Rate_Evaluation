Before running any code, sample results are provided in the ./Plots/
directory. These visualizations could guide you as to what results you
would like to achieve. They are the raw results of the tests
documented in the accompanying research paper.

In order to run the optimization test suite locally, first make
desired modifications to "test_algorithms.py" such as the FOLDERNAME
to save results into, the optimization algorithms to test, and the
objective functions to test those algorithms on. When ready, type:

   $ python3 test_algorithms.py

It will execute the code to generate comparative test results.

Once tests have finished running (status updates are given along the
way), the code in "plot_results.py" can be used to visualize the
output of tests run. The symmetrical modifications with respect to
optimization algorithms and objective functions need to be made to the
lists in this file as well. There are notes throughout, but
particularly comments can be used to plot data profiles or rank
probabilities alternatively. (the reason for not doing both at the
same time is just avoiding the use of multiple figures in pylab)

Note that this code is actively in development and may change.
