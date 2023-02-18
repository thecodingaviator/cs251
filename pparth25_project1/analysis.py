'''analysis.py
Run statistical analyses and plot Numpy ndarray data
YOUR NAME HERE
CS 251 Data Analysis Visualization
Spring 2023
'''
import numpy as np
import matplotlib.pyplot as plt


class Analysis:
    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        self.data = data

        # Make plot font sizes legible
        plt.rcParams.update({'font.size': 18})

    def set_data(self, data):
        '''Method that re-assigns the instance variable `data` with the parameter.
        Convenience method to change the data used in an analysis without having to create a new
        Analysis object.

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''

        self.data = data

    def min(self, headers, rows=[]):
        '''Computes the minimum of each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.
        (i.e. the minimum value in each of the selected columns)

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of min over, or over all indices
            if rows=[]

        Returns
        -----------
        mins: ndarray. shape=(len(headers),)
            Minimum values for each of the selected header variables

        NOTE: There should be no loops in this method!
        '''

        # Get the data from the data object
        data = self.data.select_data(headers, rows)

        # Compute the minimum of each column
        mins = np.min(data, axis=0)

        return mins

    def max(self, headers, rows=[]):
        '''Computes the maximum of each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of max over, or over all indices
            if rows=[]

        Returns
        -----------
        maxs: ndarray. shape=(len(headers),)
            Maximum values for each of the selected header variables

        NOTE: There should be no loops in this method!
        '''

        # Get the data from the data object
        data = self.data.select_data(headers, rows)

        # Compute the maximum of each column
        maxs = np.max(data, axis=0)

        return maxs

    def range(self, headers, rows=[]):
        '''Computes the range [min, max] for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of min/max over, or over all indices
            if rows=[]

        Returns
        -----------
        mins: ndarray. shape=(len(headers),)
            Minimum values for each of the selected header variables
        maxes: ndarray. shape=(len(headers),)
            Maximum values for each of the selected header variables

        NOTE: There should be no loops in this method!
        '''

        return self.min(headers, rows), self.max(headers, rows)

    def mean(self, headers, rows=[]):
        '''Computes the mean for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`).

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of mean over, or over all indices
            if rows=[]

        Returns
        -----------
        means: ndarray. shape=(len(headers),)
            Mean values for each of the selected header variables

        NOTE: You CANNOT use np.mean here!
        NOTE: There should be no loops in this method!
        '''

        # Get the data from the data object
        data = self.data.select_data(headers, rows)

        # Compute the mean of each column
        means = np.sum(data, axis=0) / data.shape[0]

        return means

    def var(self, headers, rows=[]):
        '''Computes the variance for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of variance over, or over all indices
            if rows=[]

        Returns
        -----------
        vars: ndarray. shape=(len(headers),)
            Variance values for each of the selected header variables

        NOTE: You CANNOT use np.var or np.mean here!
        NOTE: There should be no loops in this method!
        '''

        # Get the data from the data object
        data = self.data.select_data(headers, rows)

        # Compute the mean of each column
        means = self.mean(headers, rows)

        # Compute the variance of each column
        vars = np.sum((data - means)**2, axis=0) / (data.shape[0] - 1)

        return vars

    def std(self, headers, rows=[]):
        '''Computes the standard deviation for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of standard deviation over,
            or over all indices if rows=[]

        Returns
        -----------
        vars: ndarray. shape=(len(headers),)
            Standard deviation values for each of the selected header variables

        NOTE: You CANNOT use np.var, np.std, or np.mean here!
        NOTE: There should be no loops in this method!
        '''

        # Get the data from the data object
        data = self.data.select_data(headers, rows)

        # Compute the standard deviation of each column
        vars = np.sqrt(self.var(headers, rows))

        return vars

    def show(self):
        '''Simple wrapper function for matplotlib's show function.

        (Does not require modification)
        '''
        plt.show()

    def scatter(self, ind_var, dep_var, title):
        '''Creates a simple scatter plot with "x" variable in the dataset `ind_var` and
        "y" variable in the dataset `dep_var`. Both `ind_var` and `dep_var` should be strings
        in `self.headers`.

        Parameters:
        -----------
        ind_var: str.
            Name of variable that is plotted along the x axis
        dep_var: str.
            Name of variable that is plotted along the y axis
        title: str.
            Title of the scatter plot

        Returns:
        -----------
        x. ndarray. shape=(num_data_samps,)
            The x values that appear in the scatter plot
        y. ndarray. shape=(num_data_samps,)
            The y values that appear in the scatter plot

        NOTE: Do not call plt.show() here.
        '''

        # Get the data from the data object
        x = self.data.select_data([ind_var])
        y = self.data.select_data([dep_var])

        # Create the scatter plot
        plt.scatter(x, y)
        plt.title(title)
        plt.xlabel(ind_var)
        plt.ylabel(dep_var)

        return x, y

    def pair_plot(self, data_vars, fig_sz=(12, 12), title=''):
        '''Create a pair plot: grid of scatter plots showing all combinations of variables in
        `data_vars` in the x and y axes.

        Parameters:
        -----------
        data_vars: Python list of str.
            Variables to place on either the x or y axis of the scatter plots
        fig_sz: tuple of 2 ints.
            The width and height of the figure of subplots. Pass as a paramter to plt.subplots.
        title. str. Title for entire figure (not the individual subplots)

        Returns:
        -----------
        fig. The matplotlib figure.
            1st item returned by plt.subplots
        axes. ndarray of AxesSubplot objects. shape=(len(data_vars), len(data_vars))
            2nd item returned by plt.subplots

        TODO:
        - Make the len(data_vars) x len(data_vars) grid of scatterplots
        - The y axis of the first column should be labeled with the appropriate variable being
        plotted there.
        - The x axis of the last row should be labeled with the appropriate variable being plotted
        there.
        - There should be no other axis or tick labels (it looks too cluttered otherwise!)

        Tip: Check out the sharex and sharey keyword arguments of plt.subplots.
        Because variables may have different ranges, pair plot columns usually share the same
        x axis and rows usually share the same y axis.
        '''

        # Make the len(data_vars) x len(data_vars) grid of scatterplots
        fig, axes = plt.subplots(
            len(data_vars), len(data_vars), figsize=fig_sz)

        # Set the title of the figure
        fig.suptitle(title)

        # The y axis of the first column should be labeled with the appropriate variable being plotted there.
        for i in range(len(data_vars)):
            axes[i, 0].set_ylabel(data_vars[i])

        # The x axis of the last row should be labeled with the appropriate variable being plotted there.
        for i in range(len(data_vars)):
            axes[-1, i].set_xlabel(data_vars[i])

        # Remove the axis and tick labels
        for i in range(len(data_vars)):
            for j in range(len(data_vars)):
                axes[i, j].set_xticklabels([])
                axes[i, j].set_yticklabels([])
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])

        # Create the scatter plots
        for i in range(len(data_vars)):
            for j in range(len(data_vars)):
                axes[i, j].scatter(self.data.select_data(
                    [data_vars[j]]), self.data.select_data([data_vars[i]]))

        return fig, axes

    # Extension - Regression
    def regress(self, ind_var, dep_var, rows=[]):
        '''Computes the regression
        y = m * x + b
        where x is the independent variable and y is the dependent variable.

        Parameters:
        -----------
        ind_var: str.
            Name of variable that is plotted along the x axis
        dep_var: str.
            Name of variable that is plotted along the y axis
        rows: Python list of int.
            Indices of data samples to restrict computation of regression line over.

        Returns:
        -----------
        m. float.
            The slope of the regression line
        b. float.
            The intercept of the regression line
        r. float.
            The correlation coefficient of the regression line
        '''

        # Get the data from the data object
        x = self.data.select_data([ind_var], rows)
        y = self.data.select_data([dep_var], rows)

        # Compute values needed for regression line
        sigma_x = np.sum(x)
        sigma_y = np.sum(y)

        sigma_x2 = np.sum(x**2)
        sigma_y2 = np.sum(y**2)

        sigma_xy = np.sum(x*y)

        n = len(x)

        # Compute the regression line
        m = (n*sigma_xy - sigma_x*sigma_y)/(n*sigma_x2 - sigma_x**2)
        b = (sigma_y - m*sigma_x)/n

        # Compute r
        r = (n*sigma_xy - sigma_x*sigma_y)/(np.sqrt(n*sigma_x2 - sigma_x**2)*np.sqrt(n*sigma_y2 - sigma_y**2))

        # Return values
        return [m, b, r]
    
    def regress_plot(self, ind_var, dep_var, rows=[]):
        '''
        Creates a scatter plot with the regression line overlayed on top of it.

        Parameters:
        -----------
        ind_var: str.
            Name of variable that is plotted along the x axis
        dep_var: str.
            Name of variable that is plotted along the y axis
        rows: Python list of ints.
            Rows to include in the regression. If empty, all rows are included.
        '''

        # Get the data from the data object
        x = self.data.select_data([ind_var], rows)
        y = self.data.select_data([dep_var], rows)

        # Compute the regression line
        m, b, r = self.regress(ind_var, dep_var)

        # Create the scatter plot
        plt.scatter(x, y)
        plt.title(f'Regression Plot ({ind_var} vs. {dep_var})')

        # Create the regression line
        plt.plot(x, m*x + b, color='red')

        # Add the regression line equation to the plot
        plt.text(0.05, 0.95, f'y = {m:.2f}x + {b:.2f}', transform=plt.gca().transAxes, size = 8)

        # Add the correlation coefficient to the plot
        plt.text(0.05, 0.90, f'r = {r:.2f}', transform=plt.gca().transAxes, size = 8)

        plt.xlabel(ind_var)
        plt.ylabel(dep_var)

        plt.show()

    # Extension - SST, SSE, SSR
    def sst(self, dep_var, rows=[]):
        '''Computes the total sum of squares (SST) of the dependent variable.

        Parameters:
        -----------
        dep_var: str.
            Name of variable that is plotted along the y axis
        rows: Python list of ints.
            Rows to include in the regression. If empty, all rows are included.

        Returns:
        -----------
        sst. float.
            The total sum of squares of the dependent variable
        '''

        # Get the data from the data object
        y = self.data.select_data([dep_var], rows)

        # Compute the mean of the dependent variable
        y_bar = np.mean(y)

        # Compute the total sum of squares
        sst = np.sum((y - y_bar)**2)

        return sst
    
    def sse(self, ind_var, dep_var, rows=[]):
        '''Computes the error sum of squares (SSE) of the dependent variable.

        Parameters:
        -----------
        ind_var: str.
            Name of variable that is plotted along the x axis
        dep_var: str.
            Name of variable that is plotted along the y axis
        rows: Python list of ints.
            Rows to include in the regression. If empty, all rows are included.

        Returns:
        -----------
        sse. float.
            The error sum of squares of the dependent variable
        '''

        # Get the data from the data object
        x = self.data.select_data([ind_var], rows)
        y = self.data.select_data([dep_var], rows)

        # Compute the regression line
        m, b, r = self.regress(ind_var, dep_var)

        # Compute the error sum of squares
        sse = np.sum((y - (m*x + b))**2)

        return sse
    
    def ssr(self, ind_var, dep_var, rows=[]):
        '''Computes the regression sum of squares (SSR) of the dependent variable.

        Parameters:
        -----------
        ind_var: str.
            Name of variable that is plotted along the x axis
        dep_var: str.
            Name of variable that is plotted along the y axis
        rows: Python list of ints.
            Rows to include in the regression. If empty, all rows are included.

        Returns:
        -----------
        ssr. float.
            The regression sum of squares of the dependent variable
        '''

        # Get the data from the data object
        x = self.data.select_data([ind_var], rows)
        y = self.data.select_data([dep_var], rows)

        # Compute the regression line
        m, b, r = self.regress(ind_var, dep_var)

        # Compute the regression sum of squares
        ssr = np.sum((m*x + b - np.mean(y))**2)

        return ssr
