'''data.py
Reads CSV files, stores data, access/filter data by variable name
Parth Parth
CS 251 Data Analysis and Visualization
Spring 2023
'''

import numpy as np

class Data:
    filepath = None
    headers = None
    data = None
    header2col = None

    def __init__(self, filepath=None, headers=None, data=None, header2col=None):
        '''Data object constructor

        Parameters:
        -----------
        filepath: str or None. Path to data .csv file
        headers: Python list of strings or None. List of strings that explain the name of each
            column of data.
        data: ndarray or None. shape=(N, M).
            N is the number of data samples (rows) in the dataset and M is the number of variables
            (cols) in the dataset.
            2D numpy array of the dataset’s values, all formatted as floats.
            NOTE: In Week 1, don't worry working with ndarrays yet. Assume it will be passed in
                  as None for now.
        header2col: Python dictionary or None.
                Maps header (var str name) to column index (int).
                Example: "sepal_length" -> 0

        TODO:
        - Declare/initialize the following instance variables:
            - filepath
            - headers
            - data
            - header2col
            - Any others you find helpful in your implementation
        - If `filepath` isn't None, call the `read` method.
        '''
        self.filepath = filepath

        if(self.filepath is not None):
            self.read(filepath)
        # else if data is not none
        elif(data is not None):
            self.data = data

        # If headers is not none
        if(headers is not None):
            self.headers = headers

        # If header2col is not none
        if(header2col is not None):
            self.header2col = header2col

    def read(self, filepath):
        '''Read in the .csv file `filepath` in 2D tabular format. Convert to numpy ndarray called
        `self.data` at the end (think of this as 2D array or table).

        Format of `self.data`:
            Rows should correspond to i-th data sample.
            Cols should correspond to j-th variable / feature.

        Parameters:
        -----------
        filepath: str or None. Path to data .csv file

        Returns:
        -----------
        None. (No return value).
            NOTE: In the future, the Returns section will be omitted from docstrings if
            there should be nothing returned

        TODO:
        - Read in the .csv file `filepath` to set `self.data`. Parse the file to only store
        numeric columns of data in a 2D tabular format (ignore non-numeric ones). Make sure
        everything that you add is a float.
        - Represent `self.data` (after parsing your CSV file) as an numpy ndarray. To do this:
            - At the top of this file write: import numpy as np
            - Add this code before this method ends: self.data = np.array(self.data)
        - Be sure to fill in the fields: `self.headers`, `self.data`, `self.header2col`.

        NOTE: You may wish to leverage Python's built-in csv module. Check out the documentation here:
        https://docs.python.org/3/library/csv.html

        NOTE: In any CS251 project, you are welcome to create as many helper methods as you'd like.
        The crucial thing is to make sure that the provided method signatures work as advertised.

        NOTE: You should only use the basic Python library to do your parsing.
        (i.e. no Numpy or imports other than csv).
        Points will be taken off otherwise.

        TIPS:
        - If you're unsure of the data format, open up one of the provided CSV files in a text editor
        or check the project website for some guidelines.
        - Check out the test scripts for the desired outputs.
        '''

        # Start reading csv
        import csv

        # For local testing only. Sorry if I turned the next line in
        # self.filepath = "/home/parth/cs251/project1/" + filepath

        self.filepath = filepath

        # Open file
        with open(self.filepath, 'r') as file:
            reader = csv.reader(file)

            # Read in headers
            self.headers = next(reader)

            # Trim all headers
            for i in range(len(self.headers)):
                self.headers[i] = self.headers[i].strip()

            # Create a list to track which columns are numeric
            data = list(range(0, len(self.headers)))

            # Read in types
            types = next(reader)

            # Trim all types
            for i in range(len(types)):
                types[i] = types[i].strip()

            # Make sure that each element of types is numeric, string, enum, or date
            for i in range(len(types)):
                if(types[i] != "numeric" and types[i] != "string" and types[i] != "enum" and types[i] != "date"):
                    print("Error: Invalid type or types missing.")
                    return

            maxHeaders = len(self.headers)
 
            iterator = 0

            # Remove non-numeric columns
            while iterator < maxHeaders:
                if(types[iterator] != "numeric"):
                    self.headers.pop(iterator)
                    data.pop(iterator)
                    types.pop(iterator)
                    maxHeaders -= 1
                else:
                    iterator += 1

            # Read in data rowwise
            self.data = []
            # Read only if i is in data
            for row in reader:
                rows = [row[i] for i in data]
                
                # Convert to float
                for i in range(len(rows)):
                    rows[i] = float(rows[i])

                self.data.append(rows)

        # Store in dictionary
        self.header2col = {}
        for i in range(len(self.headers)):
            self.header2col[self.headers[i]] = i

        # Convert to numpy array
        self.data = np.array(self.data)

    def __str__(self):
        '''toString method

        (For those who don't know, __str__ works like toString in Java...In this case, it's what's
        called to determine what gets shown when a `Data` object is printed.)

        Returns:
        -----------
        str. A nicely formatted string representation of the data in this Data object.
            Only show, at most, the 1st 5 rows of data
            See the test code for an example output.
        '''

        strng = ""

        strng += "-------------------------------\n"
        strng += self.filepath + " (" + str(len(self.data)) + "x" + str(len(self.headers)) + ")\n"
        strng += "Headers: \n" + ' '.join(self.headers) + "\n"
        strng += "-------------------------------\n"
        if len(self.data) >= 5:
            strng += "Showing first 5/" + str(len(self.data)) + " rows\n"

        for i in range(min(5, len(self.data))):
            strng += ' '.join([str(x) for x in self.data[i]]) + "\n"

        strng += "-------------------------------\n"
        
        return strng

    def get_headers(self):
        '''Get method for headers

        Returns:
        -----------
        Python list of str.
        '''

        return self.headers

    def get_mappings(self):
        '''Get method for mapping between variable name and column index

        Returns:
        -----------
        Python dictionary. str -> int
        '''

        return self.header2col

    def get_num_dims(self):
        '''Get method for number of dimensions in each data sample

        Returns:
        -----------
        int. Number of dimensions in each data sample. Same thing as number of variables.
        '''

        return self.data.shape[1]

    def get_num_samples(self):
        '''Get method for number of data points (samples) in the dataset

        Returns:
        -----------
        int. Number of data samples in dataset.
        '''
        
        return self.data.shape[0]

    def get_sample(self, rowInd):
        '''Gets the data sample at index `rowInd` (the `rowInd`-th sample)

        Returns:
        -----------
        ndarray. shape=(num_vars,) The data sample at index `rowInd`
        '''
        
        return self.data[rowInd]

    def get_header_indices(self, headers):
        '''Gets the variable (column) indices of the str variable names in `headers`.

        Parameters:
        -----------
        headers: Python list of str. Header names to take from self.data

        Returns:
        -----------
        Python list of nonnegative ints. shape=len(headers). The indices of the headers in `headers`
            list.
        '''
        
        headInd = []

        for header in headers:
            headInd.append(self.header2col[header])

        return headInd

    def get_all_data(self):
        '''Gets a copy of the entire dataset

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(num_data_samps, num_vars). A copy of the entire dataset.
            NOTE: This should be a COPY, not the data stored here itself.
            This can be accomplished with numpy's copy function.
        '''

        # if data is None print error
        if self.data is None:
            print("Error: Data is empty.")
            return
        
        return np.copy(self.data)

    def head(self):
        '''Return the 1st five data samples (all variables)

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(5, num_vars). 1st five data samples.
        '''

        reqRows = min(5, len(self.data))

        # Create ndarray
        dataReq = np.ndarray(shape=(reqRows, len(self.headers)))

        # Fill in data
        for i in range(reqRows):
            for j in range(len(self.headers)):
                dataReq[i][j] = self.data[i][j]

        return dataReq

    def tail(self):
        '''Return the last five data samples (all variables)

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(5, num_vars). Last five data samples.
        '''

        reqRows = min(5, len(self.data))

        # Create ndarray
        dataReq = np.ndarray(shape=(reqRows, len(self.headers)))

        # Fill in data from the last reqRows rows
        for i in range(reqRows):
            for j in range(len(self.headers)):
                dataReq[i][j] = self.data[len(self.data) - reqRows + i][j]

        return dataReq


    def limit_samples(self, start_row, end_row):
        '''Update the data so that this `Data` object only stores samples in the contiguous range:
            `start_row` (inclusive), end_row (exclusive)
        Samples outside the specified range are no longer stored.

        (Week 2)

        '''

        newData = []

        # Loop between start_row and end_row
        for i in range(start_row, end_row):
            newData.append(self.data[i])

        self.data = newData

    def select_data(self, headers, rows=[]):
        '''Return data samples corresponding to the variable names in `headers`.
        If `rows` is empty, return all samples, otherwise return samples at the indices specified
        by the `rows` list.

        (Week 2)

        For example, if self.headers = ['a', 'b', 'c'] and we pass in header = 'b', we return
        column #2 of self.data. If rows is not [] (say =[0, 2, 5]), then we do the same thing,
        but only return rows 0, 2, and 5 of column #2.

        Parameters:
        -----------
            headers: Python list of str. Header names to take from self.data
            rows: Python list of int. Indices of subset of data samples to select.
                Empty list [] means take all rows

        Returns:
        -----------
        ndarray. shape=(num_data_samps, len(headers)) if rows=[]
                 shape=(len(rows), len(headers)) otherwise
            Subset of data from the variables `headers` that have row indices `rows`.

        Hint: For selecting a subset of rows from the data ndarray, check out np.ix_
        '''
        
        rowsToLoop = []

        # If rows is not empty, rowsToLoop is rows
        if rows != []:
            rowsToLoop = rows
        # Else, rowsToLoop is all rows
        else:
            rowsToLoop = range(self.data.shape[0])

        # Create ndarray
        dataReq = np.ndarray(shape=(len(rowsToLoop), len(headers)))

        # Fill in data
        for i in range(len(rowsToLoop)):
            for j in range(len(headers)):
                dataReq[i][j] = self.data[rowsToLoop[i]][self.header2col[headers[j]]]

        return dataReq
