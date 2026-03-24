import numpy as np
from astropy.table import Table
from astropy.io import fits
import matplotlib.pyplot as plt

class ThreePointDataClass:
    def __init__(self, name, bin_type, kernel1=None, kernel2=None, kernel3=None, sortz=True):
        """
        Initialize the 3pt data object.

        Args:
            name (str)     : The name of the data.
            bin_type (str) : The type of the bin. 
                             Should be 'SSS', 'SAS', or 'Multipole'.
            kernel1 (str)  : The kernel type for bin1.
            kernel2 (str)  : The kernel type for bin2.
            kernel3 (str)  : The kernel type for bin3.
            sortz (bool)   : Whether to identify the permutations of (z1, z2, z3)
        
        Description:
            This method initializes the 3pt data object with the provided 
            values. The values of name, bin_type, kernel1, kernel2, and kernel3 
            are stored as attributes of the object. The method also calls the 
            set_empty method to initialize the arrays.

        Note on bin_type:
            If bin_type is 'SSS', the 3pt data is stored in the form of 
            (z1, z2, z3, theta1, theta2, theta3, signal).
            If bin_type is 'SAS', the 3pt data is stored in the form of 
            (z1, z2, z3, theta1, theta2, phi, signal).
            If bin_type is 'Multipole', the 3pt data is stored in the form of 
            (z1, z2, z3, theta1, theta2, M, signal).

            Talking about the weak lensing analysis, 'SAS' bin_type is useful
            for shear 3PCF, 'Multipole' bin_type is useful for 3PCF multipole, 
            and 'SSS' bin_type is useful for aperture mass statistics.
        """
        self.name = name
        self.bin_type = bin_type
        self.kernel1 = kernel1 or 'nz_source'
        self.kernel2 = kernel2 or 'nz_source'
        self.kernel3 = kernel3 or 'nz_source'
        self.sortz   = sortz
        self.set_empty()

    def set_empty(self):
        """
        Initialize the arrays of the 3pt data.
        """
        # redshift bins
        self.z1 = np.array([], dtype=int)
        self.z2 = np.array([], dtype=int)
        self.z3 = np.array([], dtype=int)
        # triangle bins
        if self.bin_type == 'SSS':
            self.theta1 = np.array([])
            self.theta2 = np.array([])
            self.theta3 = np.array([])        
        elif self.bin_type == 'SAS':
            self.theta1 = np.array([])
            self.theta2 = np.array([])
            self.phi = np.array([])
        elif self.bin_type == 'Multipole':
            self.theta1 = np.array([])
            self.theta2 = np.array([])
            self.M      = np.array([], dtype=int)
        # signal
        self.signal = np.array([])
        # size
        self.size = 0

    def set_value(self, z1, z2, z3, b1, b2, b3, signal, where=None):
        """
        Set the value of the 3pt data.

        Args:
            z1 (float, array): The number of redshift bin.
            z2 (float, array): The number of redshift bin.
            z3 (float, array): The number of redshift bin.
            b1 (float, array): The value of bin1.
            b2 (float, array): The value of bin2.
            b3 (float, array): The value of bin3.
            signal (float, array): The signal value.
            where (array): where to put the values. If None, append.

        Description:
            This method sets the value of the 3pt data by appending the provided 
            values to the respective arrays. The values of z1, z2, and z3 are 
            appended to the corresponding arrays. Depending on the bin type, the 
            values of b1, b2, and b3 are appended to the respective arrays.
            Finally, the signal value is appended to the signal array.

            If bin_type is 'SSS', the values of b1, b2, and b3 are appended to 
            the arrays theta1, theta2, and theta3 respectively.
            If bin_type is 'SAS', the values of b1, b2, and b3 are appended to
            the arrays theta1, theta2, and phi respectively.
            If bin_type is 'Multipole', the values of b1, b2, and b3 are appended
            to the arrays theta1, theta2, and M respectively.
        """
        z1 = np.atleast_1d(z1)
        z2 = np.atleast_1d(z2)
        z3 = np.atleast_1d(z3)
        b1 = np.atleast_1d(b1)
        b2 = np.atleast_1d(b2)
        b3 = np.atleast_1d(b3)
        signal = np.atleast_1d(signal)
        if z1.size == 1 and b1.size > 1:
            z1 = np.repeat(z1, b1.size)
            z2 = np.repeat(z2, b2.size)
            z3 = np.repeat(z3, b3.size)
        assert z1.size == z2.size == z3.size == b1.size == b2.size == b3.size == signal.size, \
            f'All the arrays should have the same size. The sizes are {z1.size}, {z1.size}, {z2.size}, ' \
            f'{z3.size}, {b1.size}, {b2.size}, {b3.size}, {signal.size}'
        # Identify the 
        if self.sortz:
            z1, z2, z3 = np.sort([z1, z2, z3], axis=0)
        # Set the values
        def put(z, y, sel):
            if sel is None:
                z = np.append(z, y)
            else:
                z[sel] = y
            return z
        self.z1 = put(self.z1, z1, where)
        self.z2 = put(self.z2, z2, where)
        self.z3 = put(self.z3, z3, where)
        if self.bin_type == 'SSS':
            self.theta1 = put(self.theta1, b1, where)
            self.theta2 = put(self.theta2, b2, where)
            self.theta3 = put(self.theta3, b3, where)
        elif self.bin_type == 'SAS':
            self.theta1 = put(self.theta1, b1, where)
            self.theta2 = put(self.theta2, b2, where)
            self.phi = put(self.phi, b3, where)
        elif self.bin_type == 'Multipole':
            self.theta1 = put(self.theta1, b1, where)
            self.theta2 = put(self.theta2, b2, where)
            self.M = put(self.M, b3, where)
        self.signal = put(self.signal, signal, where)
        self.size = self.z1.size

    def where_to_set(self, z1, z2, z3, b1, b2, b3):
        z1 = np.atleast_1d(z1)
        z2 = np.atleast_1d(z2)
        z3 = np.atleast_1d(z3)
        b1 = np.atleast_1d(b1)
        b2 = np.atleast_1d(b2)
        b3 = np.atleast_1d(b3)
        if z1.size == 1 and b1.size > 1:
            z1 = np.repeat(z1, b1.size)
            z2 = np.repeat(z2, b2.size)
            z3 = np.repeat(z3, b3.size)
        assert z1.size == z2.size == z3.size == b1.size == b2.size == b3.size, \
            f'All the arrays should have the same size. The sizes are {z1.size}, {z1.size}, {z2.size}, ' \
            f'{z3.size}, {b1.size}, {b2.size}, {b3.size}'
        # Identify the 
        if self.sortz:
            z1, z2, z3 = np.sort([z1, z2, z3], axis=0)
        # determine
        where = []
        for _z1, _z2, _z3, _b1, _b2, _b3 in zip(z1, z2, z3, b1, b2, b3):
            sel = (self.z1==_z1) & (self.z2==_z2) & (self.z3==_z3)
            if self.bin_type == 'SSS':
                sel &= (self.theta1==_b1) & (self.theta2==_b2) & (self.theta3==_b3)
            elif self.bin_type == 'SAS':
                sel &= (self.theta1==_b1) & (self.theta2==_b2) & (self.phi==_b3)
            elif self.bin_type == 'Multipole':
                sel &= (self.theta1==_b1) & (self.theta2==_b2) & (self.M==_b3)
            where.append(np.arange(self.size)[sel][0])
        return np.array(where)

    def set_covariance(self, cov, nsim=0):
        """
        Set the covariance matrix.

        Args:
            cov (array): The covariance matrix.
            nsim (int) : The number of simulations used to 
                         estimate the covariance matrix. This is used to 
                         estimate the Hartlap factor.

        Description:
            This method sets the covariance matrix of the 3pt data. The 
            covariance matrix is stored as an attribute of the object.
        """
        self.cov     = cov
        self.nsim4cov= nsim

    def to_fits(self, filename=None):
        """
        Write the 3pt data to the fits file.

        Args:
            filename (str): The name of the fits file.

        Returns:
            hdul (HDUList): The HDUList object.
        """
        ## HDUList
        primary = fits.PrimaryHDU()
        hdul = fits.HDUList([primary])
        ## DATA VECTOR
        # create table
        if self.bin_type == 'SSS':
            data = [self.z1, self.z2, self.z3, self.theta1, self.theta2, self.theta3, self.signal]
            names= ['BIN1', 'BIN2', 'BIN3', 'THETA1', 'THETA2', 'THETA3', 'VALUE']
        elif self.bin_type == 'SAS':
            data = [self.z1, self.z2, self.z3, self.theta1, self.theta2, self.phi, self.signal]
            names= ['BIN1', 'BIN2', 'BIN3', 'THETA1', 'THETA2', 'PHI', 'VALUE']
        elif self.bin_type == 'Multipole':
            data = [self.z1, self.z2, self.z3, self.theta1, self.theta2, self.M, self.signal]
            names= ['BIN1', 'BIN2', 'BIN3', 'THETA1', 'THETA2', 'M', 'VALUE']
        table = Table(data, names=names)
        # create header
        header = fits.Header()
        header['BIN_TYPE'] = self.bin_type
        header['EXTNAME']  = self.name
        header['KERNEL_1'] = self.kernel1
        header['KERNEL_2'] = self.kernel2
        header['KERNEL_3'] = self.kernel3
        header['SORTZ']    = self.sortz
        header['3PT_DATA'] = True
        # create hdu
        hdu = fits.BinTableHDU(table, header=header)
        hdul.append(hdu)
        ## COVARIANCE MATRIX
        if hasattr(self, 'cov'):
            # create header
            header = fits.Header()
            header['EXTNAME']  = 'COVMAT'
            header['NSIM']     = self.nsim4cov
            header['3PT_DATA'] = True
            header['STRT_0'] = 0
            header['name_0'] = 'map3'
            # create hdu
            hdu = fits.ImageHDU(self.cov, header=header)
            hdul.append(hdu)
        # Write to file
        if filename:
            hdul.writeto(filename, overwrite=True)
        return hdul

    @classmethod
    def from_fits(cls, filename_or_hdul):
        """
        Read the 3pt data from the fits file.

        Args:
            filename_or_hdul (str, HDUList): The name of the fits file or HDUList
        
        Returns:
            obj (ThreePointDataClass): The 3pt data object.
        """
        if isinstance(filename_or_hdul, str):
            hdul = fits.open(filename_or_hdul)
        else:
            hdul = filename_or_hdul
        # signal
        hdu = hdul[1]
        # read header
        header = hdu.header
        name = header['EXTNAME']
        bin_type = header['BIN_TYPE']
        kernel1 = header['KERNEL_1']
        kernel2 = header['KERNEL_2']
        kernel3 = header['KERNEL_3']
        sortz   = header['SORTZ']
        obj = cls(name, bin_type, kernel1, kernel2, kernel3, sortz)
        # assign values
        data = hdu.data
        if bin_type == 'SSS':
            b1 = data['THETA1']
            b2 = data['THETA2']
            b3 = data['THETA3']
        elif bin_type == 'SAS':
            b1 = data['THETA1']
            b2 = data['THETA2']
            b3 = data['PHI']
        elif bin_type == 'Multipole':
            b1 = data['THETA1']
            b2 = data['THETA2']
            b3 = data['M']
        obj.set_value(data['BIN1'], data['BIN2'], data['BIN3'], b1, b2, b3, data['VALUE'])
        # covariance
        if len(hdul) > 2:
            hdu = hdul[2]
            obj.set_covariance(hdu.data, hdu.header['NSIM'])
        return obj

    def _parse_selection(self, val, which, condition, helper):
        """
        Parse the selection array for the given bin.

        Args:
            val (float, array)    : The bin value to select.
            which (str, array)    : The bin to select. 
            condition (str, array): The equality condition.
            helper (func)         : The helper function to get 
                                    the selection array.
        
        Returns:
            sel (array): The selection array.
        """
        # cast
        val = np.atleast_1d(val)
        which = np.atleast_1d(which)
        condition = np.atleast_1d(condition)
        n1 = val.size
        n2 = which.size
        n3 = condition.size
        n  = max([n1, n2, n3])
        if n>1:
            assert n1==1 or n1==n, 'val should be scalar or array of size n'
            assert n2==1 or n2==n, 'which should be scalar or array of size n'
            assert n3==1 or n3==n, 'condition should be scalar or array of size n'
        if n>1 and n1==1:
            val = np.repeat(val, n)
        if n>1 and n2==1:
            which = np.repeat(which, n)
        if n>1 and n3==1:
            condition = np.repeat(condition, n)

        # ``and'' selection
        sel = np.ones(self.size, dtype=bool)
        for v, w, c in zip(val, which, condition):
            sel &= helper(v, w, c)
        return sel        

    def selection_z_bin(self, z_val, which, condition='=='):
        """
        Get selection array for the given redshift bin.
        
        Args:
            z_val (int, array)    : The redshift bin to select.
            which (str, array)    : The redshift bin to select. 
                                    Should be 'z1', 'z2', or 'z3'.
            condition (str, array): The equality condition.
        
        Example:
            To select all the data points with z1 = 1, use:
            >>> sel = data.selection_z_bin(1, 'z1')
            >>> data_z1 = data[sel]
            To select all the data points with z = 2, use:
            >>> sel = data.selection_z_bin(2, 'z')
            >>> data_z = data[sel]
            To select all the data points with z1 >= 3, use:
            >>> sel = data.selection_z_bin(3, 'z1', '>=')
            >>> data_z1 = data[sel]
            Special case: to get (z1, z2, z3) == (1,2,3) without caring the
            order of z1, z2, z3, then use:
            >>> sel = data.selection_z_bin([1,2,3], 'z123')
            >>> data_z123 = data[sel]

        Returns:
            sel (array): The selection array.
        """
        def helper(z_val, which, condition):
            sel = np.ones(self.size, dtype=bool)
            if which == 'z1' or which == 'z':
                sel &= compare(self.z1, z_val, condition)
            elif which == 'z2' or which == 'z':
                sel &= compare(self.z2, z_val, condition)
            elif which == 'z3' or which == 'z':
                sel &= compare(self.z3, z_val, condition)
            else:
                raise ValueError('which should be z1, z2 or z3')
            return sel

        if which == 'z123':
            which = ['z1', 'z2', 'z3']
            z_val = np.sort(z_val)
        sel = self._parse_selection(z_val, which, condition, helper)
        return sel

    def selection_SAS_bin(self, b_val, which, condition='=='):
        """
        Get selection array for the given bin. This is applicable 
        for SAS bin type.

        Args:
            b_val (float, array)  : The bin value to select.
            which (str, array)    : The bin to select. 
                                    Should be 'theta1', 'theta2', or 'phi'.
            condition (str, array): The equality condition.

        Example:
            To select all the data points with theta1 = 0.1, use:
            >>> sel = data.selection_SASbin(0.1, 'theta1')
            >>> data_theta1 = data[sel]
            To select all the data points with theta2 = 0.1, use:
            >>> sel = data.selection_SASbin(0.1, 'theta2')
            >>> data_theta2 = data[sel]
            To select theta1 and theta2, use:
            >>> sel = data.selection_SASbin(0.1, 'theta')
            >>> data_theta = data[sel]
            To select all the data points with phi = 0.1, use:
            >>> sel = data.selection_SASbin(0.1, 'phi')
            >>> data_phi = data[sel]
            To select all the data points with theta1 >= 0.1, use:
            >>> sel = data.selection_SASbin(0.1, 'theta1', '>=')
            >>> data_theta1 = data[sel]
        
        Returns:
            sel (array): The selection array.
        """
        def helper(b_val, which, condition):
            sel = np.ones(self.size, dtype=bool)
            if which == 'theta1' or which == 'theta':
                sel &= compare(self.theta1, b_val, condition)
            elif which == 'theta2' or which == 'theta':
                sel &= compare(self.theta2, b_val, condition)
            elif which == 'phi':
                sel &= compare(self.phi, b_val, condition)
            else:
                raise ValueError('which must be one of \
                    theta1, theta2, theta, or phi')
            return sel
        
        assert self.bin_type == 'SAS', \
            'This method is only applicable for SAS bin type.'
        sel = self._parse_selection(b_val, which, condition, helper)
        return sel
        
    def selection_SSS_bin(self, b_val, which, condition='=='):
        """
        Get selection array for the given bin. This is applicable
        for SSS bin type.

        Args:
            b_val (float, array)  : The bin value to select.
            which (str, array)    : The bin to select. 
                                    Should be 'theta1', 'theta2', or 'theta3'.
            condition (str, array): The equality condition.
        
        Example:
            To select all the data points with theta1 = 0.1, use:
            >>> sel = data.selection_SSSbin(0.1, 'theta1')
            >>> data_theta1 = data[sel]
            To select all the data points with theta2 = 0.1, use:
            >>> sel = data.selection_SSSbin(0.1, 'theta2')
            >>> data_theta2 = data[sel]
            To select all the data points with theta3 = 0.1, use:
            >>> sel = data.selection_SSSbin(0.1, 'theta3')
            >>> data_theta3 = data[sel]
            To select all the data points with theta1 >= 0.1, use:
            >>> sel = data.selection_SSSbin(0.1, 'theta1', '>=')
            >>> data_theta1 = data[sel]
        
        Returns:
            sel (array): The selection array.
        """
        def helper(b_val, which, condition):
            sel = np.ones(self.size, dtype=bool)
            if which == 'theta1' or which == 'theta':
                sel &= compare(self.theta1, b_val, condition)
            elif which == 'theta2' or which == 'theta':
                sel &= compare(self.theta2, b_val, condition)
            elif which == 'theta3' or which == 'theta':
                sel &= compare(self.theta3, b_val, condition)
            else:
                raise ValueError('which must be one of \
                    theta1, theta2, theta3 or theta')
            return sel

        assert self.bin_type == 'SSS', \
            'This method is only applicable for SSS bin type.'
        sel = self._parse_selection(b_val, which, condition, helper)
        return sel

    def selection_Multipole_bin(self, b_val, which, condition='=='):
        """
        Get selection array for the given bin. This is applicable
        for Multipole bin type.

        Args:
            b_val (float, array)  : The bin value to select.
            which (str, array)    : The bin to select. 
                                    Should be 'theta1', 'theta2', or 'M'.
            condition (str, array): The equality condition.
        
        Example:
            To select all the data points with theta1 = 0.1, use:
            >>> sel = data.selection_Multipole_bin(0.1, 'theta1')
            >>> data_theta1 = data[sel]
            To select all the data points with theta2 = 0.1, use:
            >>> sel = data.selection_Multipole_bin(0.1, 'theta2')
            >>> data_theta2 = data[sel]
            To select all the data points with M = 0, use:
            >>> sel = data.selection_Multipole_bin(0, 'M')
            >>> data_M = data[sel]
            To select all the data points with theta1 >= 0.1, use:
            >>> sel = data.selection_Multipole_bin(0.1, 'theta1', '>=')
            >>> data_theta1 = data[sel]
        
        Returns:
            sel (array): The selection array.
        """
        def helper(b_val, which, condition):
            sel = np.ones(self.size, dtype=bool)
            if which == 'theta1' or which == 'theta':
                sel &= compare(self.theta1, b_val, condition)
            elif which == 'theta2' or which == 'theta':
                sel &= compare(self.theta2, b_val, condition)
            elif which == 'M':
                sel &= compare(self.M, b_val, condition)
            else:
                raise ValueError('which must be one of \
                    theta1, theta2, theta or M')
            return sel

        assert self.bin_type == 'Multipole', \
            'This method is only applicable for Multipole bin type.'
        sel = self._parse_selection(b_val, which, condition, helper)
        return sel
    
    def get_signal(self, sel=None):
        """
        Get the signal array.

        Args:
            sel (array): The selection array.
        
        Returns:
            signal (array): The signal array.
        """
        if sel is None:
            return self.signal
        else:
            return self.signal[sel]

    def get_covariance(self, sel=None):
        """
        Get the covariance matrix.

        Args:
            sel (array): The selection array.
        
        Returns:
            cov (array): The covariance matrix.
        """
        if sel is None:
            return self.cov
        else:
            return self.cov[np.ix_(sel, sel)]
    
    def get_std(self, sel=None):
        cov = self.get_covariance(sel)
        std = np.sqrt(np.diag(cov))
        return std
    
    def get_inverse_covariance(self, sel=None, Hartlap=True):
        """
        Get the inverse covariance matrix.

        Args:
            sel (array): The selection array.
            Hartlap (bool): Whether to apply Hartlap factor.
        
        Returns:
            icov (array): The inverse covariance matrix.
        """
        cov = self.get_covariance(sel)
        if Hartlap:
            nsim = self.nsim4cov
            n = cov.shape[0]
            assert nsim > n, 'nsim should be greater than n'
            n = cov.shape[0]
            f = (nsim - n - 2) / (nsim - 1)
            cov /= f
        icov = np.linalg.inv(cov)
        return icov

    def get_rcc(self, sel=None):
        """
        Get the correlation coefficient matrix.

        Args:
            sel (array): The selection array.
        
        Returns:
            rcc (array): The reduced covariance matrix.
        """
        cov = self.get_covariance(sel)
        diag= np.diag(cov)
        rcc = cov / np.sqrt(np.outer(diag, diag))
        return rcc

    def get_z_bin(self, sel=None, unique=False):
        """
        Get the redshift bin arrays.

        Args:
            sel (array): The selection array.
            unique (bool): Whether to return unique redshift bins.
        
        Returns:
            z1, z2, z3 (array): The redshift bin arrays.
        """
        if sel is None:
            z1, z2, z3 = self.z1, self.z2, self.z3
        else:
            z1, z2, z3 = self.z1[sel], self.z2[sel], self.z3[sel]
        if unique:
            z1, z2, z3 = np.unique([z1, z2, z3], axis=1)
        return np.array([z1, z2, z3])
    
    def get_t_bin(self, sel=None):
        """
        Get the triangle bin arrays.

        Args:
            sel (array): The selection array.
        
        Returns:
            b1, b2, b3 (array): The bin arrays.
        """
        if self.bin_type == 'SSS':
            if sel is None:
                b1, b2, b3 = self.theta1, self.theta2, self.theta3
            else:
                b1, b2, b3 = self.theta1[sel], self.theta2[sel], self.theta3[sel]
        elif self.bin_type == 'SAS':
            if sel is None:
                b1, b2, b3 = self.theta1, self.theta2, self.phi
            else:
                b1, b2, b3 = self.theta1[sel], self.theta2[sel], self.phi[sel]
        elif self.bin_type == 'Multipole':
            if sel is None:
                b1, b2, b3 = self.theta1, self.theta2, self.M
            else:
                b1, b2, b3 = self.theta1[sel], self.theta2[sel], self.M[sel]
        return np.array([b1, b2, b3])

    def get_snr(self, sel=None):
        icov = self.get_inverse_covariance(sel)
        s    = self.get_signal(sel)
        snr = np.matmul(s, np.matmul(icov, s))**0.5
        return snr

    def copy(self):
        """
        Create a copy of the 3pt data object.

        Returns:
            obj (ThreePointDataClass): The 3pt data object.
        """
        hdul = self.to_fits()
        obj = ThreePointDataClass.from_fits(hdul)
        return obj

    def replace(self, sel):
        """
        Replace the 3pt data object using a given selection/sort array.
        Note that this is a **destructive** method.

        Args:
            sel (array): The selection array.
        """
        self.z1 = self.z1[sel]
        self.z2 = self.z2[sel]
        self.z3 = self.z3[sel]
        self.theta1 = self.theta1[sel]
        self.theta2 = self.theta2[sel]
        if self.bin_type == 'SSS':
            self.theta3 = self.theta3[sel]
        elif self.bin_type == 'SAS':
            self.phi = self.phi[sel]
        elif self.bin_type == 'Multipole':
            self.M = self.M[sel]
        self.signal = self.signal[sel]
        self.size = self.z1.size
        if hasattr(self, 'cov'):
            self.cov = self.cov[np.ix_(sel, sel)]

    def sort(self, reverse=False, priority='z'):
        """
        Sort the order of entries.

        Args:
            reverse (bool): Whether to sort in reverse order.
            priority (str): The priority of sorting. 
                            Should be 'z' or 't', where 'z' sorts
                            the redshift bins and 't' sorts the triangle bins
                            as the primary sorting parameter and the other
                            as the secondary sorting parameter.

        Returns:
            obj (ThreePointDataClass): The 3pt data object.
        """
        # collect bins
        zbins = (self.z3, self.z2, self.z1)
        if self.bin_type == 'SSS':
            tbins = (self.theta3, self.theta2, self.theta1)
        elif self.bin_type == 'SAS':
            tbins = (self.phi, self.theta2, self.theta1)
        elif self.bin_type == 'Multipole':
            tbins = (self.M, self.theta2, self.theta1)
        # sort
        if priority == 'z':
            sel = np.lexsort(tbins+zbins)
        elif priority == 't':
            sel = np.lexsort(zbins+tbins)
        # reverse if needed
        if reverse:
            sel = sel[::-1]
        # return as a new object
        obj = self.copy()
        obj.replace(sel)
        return obj

    def reduce_by_z_bin_selection(self, scombs, verbose=False):
        """
        scombs must be a list of followings:
        - all
        - auto
        - cross
        - #,#,#  (e.g. 1.1.1   or 1.2.3)
        """
        if len(scombs) == 0:
            scombs = ['all']
        
        n = np.max(self.get_z_bin(unique=True)) # number of zbins
        scombs2 = []
        for scomb in scombs:
            if scomb == 'all':
                for i in range(1,n+1):
                    for j in range(i, n+1):
                        for k in range(j, n+1):
                            scombs2.append([i,j,k])
            if scomb == 'auto':
                for i in range(1,n+1):
                    scombs2.append([i,i,i])
            if scomb == 'cross':
                for i in range(1,n+1):
                    for j in range(i, n+1):
                        for k in range(j, n+1):
                            if not (i==j==k):
                                scombs2.append([i,j,k])
            if ',' in scomb:
                scombs2.append([int(i) for i in scomb.split(',')])
        scombs = np.array(scombs2)
        del scombs2

        if verbose:
            print('Preselection on sample_combination', self.size) 
        sel = np.zeros(self.size, dtype=bool)
        for scomb in scombs:
            sel |= self.selection_z_bin(scomb, 'z123', condition='==')
        assert np.sum(sel) > 0, 'No data after selection!'
        self.replace(sel)
        if verbose:
            print('Postselection on sample_combination', self.size)

    def _plot_z_bin(self, ax, colors, sel=None):
        """
        Plot the redshift bins.

        Args:
            ax (axis): The axis object.
            colors (list): The colors of the plot.
        """
        z1, z2, z3 = self.get_z_bin(sel=sel)
        ax.plot(z1, color=colors[0], label=r'$z_1$')
        ax.plot(z2, color=colors[1], label=r'$z_2$')
        ax.plot(z3, color=colors[2], label=r'$z_3$')
        ax.set_ylabel(r'redshift bin')
        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        return ax
    
    def _plot_t_bin(self, ax, colors, sel=None):
        """
        Plot the triangle bins.

        Args:
            ax (axis): The axis object.
            colors (list): The colors of the plot.
        """
        t1, t2, t3 = self.get_t_bin(sel=sel)
        if self.bin_type == 'SSS':
            ax.plot(t1, color=colors[0], label=r'$\theta_1$')
            ax.plot(t2, color=colors[1], label=r'$\theta_2$')
            ax.plot(t3, color=colors[2], label=r'$\theta_3$')
        elif self.bin_type == 'SAS':
            ax.plot(t1, color=colors[0], label=r'$\theta_1$')
            ax.plot(t2, color=colors[1], label=r'$\theta_2$')
            ax.plot(np.nan, color=colors[2], label=r'$\phi$')
            _ = ax.twinx()
            _.plot(t3, color=colors[2], label=r'$\phi$')
        elif self.bin_type == 'Multipole':
            ax.plot(t1, color=colors[0], label=r'$\theta_1$')
            ax.plot(t2, color=colors[1], label=r'$\theta_2$')
            ax.plot(np.nan, color=colors[2], label=r'$M$')
            _ = ax.twinx()
            _.plot(t3, color=colors[2], label=r'$\phi$')
        ax.set_ylabel(r'triangle bin')
        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        return ax
    
    def _plot_signal(self, ax, color, s=None, errorbar=True, yscale='linear', sel=None, nt=0, xshift=0, ls='-'):
        """
        Plot the signal.

        Args:
            ax (axis): The axis object.
            color (str): The color of the plot.
            s (array): The signal array, default is None.
            errorbar (bool): Whether to plot the error bars.
            yscale (str): The yscale of the plot.
            sel (array): The selection array.
            nt (int): The power of t to multiply the signal.
        """
        s = self.get_signal(sel=sel) if s is None else s
        resc = self.get_t_bin(sel=sel)[0,:]**nt
        ax.set_yscale(yscale)
        if errorbar and hasattr(self, 'cov'):
            std = self.get_std(sel=sel)
            ax.errorbar(np.arange(s.size)+xshift, s*resc, yerr=std*resc, fmt='.', color=color, ls=ls)
        else:
            ax.plot(np.arange(s.size)+xshift, s, color=color, ls=ls)
        ax.set_ylabel(r'signal')
        return ax

    def _plot_residual(self, ax, color, s, errorbar=True, norm=True):
        """
        Plot the residual signal.

        Args:
            ax (axis): The axis object.
            color (str): The color of the plot.
            s (array): The signal array.
            errorbar (bool): Whether to plot the error bars.
            norm (bool): Whether to normalize the residual.
        """
        res = self.get_signal()-s
        if norm:
            res /= np.sqrt(np.diag(self.cov))
        if errorbar and hasattr(self, 'cov'):
            err = np.ones(res.size) if norm else np.sqrt(np.diag(self.cov)) 
            ax.errorbar(np.arange(res.size), res, yerr=err, fmt='.', color=color)
        else:
            ax.plot(res, color=color)
        ax.set_ylabel(r'residual')
        ax.axhline(0.0, color='gray')
        ax.grid()
        return ax

    def plot(self, figsize=(10,6), signal_color=None, bin_colors=None, errorbar=True, yscale='linear', sel=None, nt=0):
        """
        Plot the 3pt data.

        Args:
            figsize (tuple): The figure size.
            signal_color (str): The color of the signal.
            bin_colors (list): The colors of the bins.
            errorbar (bool): Whether to plot the error bars.
            yscale (str): The yscale of the plot.
        """
        # set defaults
        if bin_colors is None:
            bin_colors = [None, None, None]
        bin_colors = [c or 'C%d'%i for i, c in enumerate(bin_colors)]
        # Data ordering before sort
        fig, axes = plt.subplots(3,1, figsize=(10, 6), sharex=True)
        plt.subplots_adjust(hspace=0.02)
        # redshift bin
        self._plot_z_bin(axes[0], bin_colors, sel=sel)
        # triangle bin
        self._plot_t_bin(axes[1], bin_colors, sel=sel)
        # signal
        self._plot_signal(axes[2], signal_color, errorbar=errorbar, yscale=yscale, sel=sel, nt=nt)
        # set x label
        axes[2].set_xlabel('Data index')
        return fig

    def plot_residual(self, s, figsize=(10,6), signal_colors=None, bin_colors=None, errorbar=True, yscale='linear'):
        """
        Plot the 3pt data.

        Args:
            figsize (tuple): The figure size.
            signal_colors (str): The colors of the signal.
            bin_colors (list): The colors of the bins.
            errorbar (bool): Whether to plot the error bars.
            yscale (str): The yscale of the plot.
        """
        # set defaults
        if bin_colors is None:
            bin_colors = [None, None, None]
        bin_colors = [c or 'C%d'%i for i, c in enumerate(bin_colors)]
        if signal_colors is None:
            signal_colors = ['k', 'r']
        # Data ordering before sort
        fig, axes = plt.subplots(4,1, figsize=(10, 6), sharex=True)
        plt.subplots_adjust(hspace=0.02)
        # redshift bin
        self._plot_z_bin(axes[0], bin_colors)
        # triangle bin
        self._plot_t_bin(axes[1], bin_colors)
        # signal in this class
        self._plot_signal(axes[2], signal_colors[0], errorbar=errorbar, yscale=yscale, ls='-')
        # external signal
        self._plot_signal(axes[2], signal_colors[1], s=s, errorbar=errorbar, yscale=yscale, ls='--', xshift=0.1)
        # plot residual
        self._plot_residual(axes[3], signal_colors[0], s, errorbar=errorbar)
        # set x label
        axes[2].set_xlabel('Data index')
        return fig

    def plot_covarivance(self, figsize=(7,7), log=True, cmap='bwr', sel=None):
        """
        Plot the covariance matrix.

        Args:
            figsize (tuple): The figure size.
            log (bool): Whether to plot the log of the covariance matrix.
            cmap (str): The color map.
            sel (array): The selection array.
        """
        if hasattr(self, 'cov'):
            z = self.get_covariance(sel=sel)
            if log:
                z = np.log10(np.abs(z)) * np.sign(z)
            vmax = np.max(np.abs(z))
            fig, ax = plt.subplots(figsize=figsize)
            ax.imshow(z, cmap=cmap, vmin=-vmax, vmax=vmax, origin='lower')
            return fig
        else:
            print('Covariance matrix is not available.')

    def plot_rcc(self, figsize=(7,7), cmap='bwr', sel=None):
        """
        Plot the correlation coefficient matrix.

        Args:
            figsize (tuple): The figure size.
            cmap (str): The color map.
            sel (array): The selection array.
        """
        if hasattr(self, 'cov'):
            z = self.get_rcc(sel=sel)
            fig, ax = plt.subplots(figsize=figsize)
            ax.imshow(z, cmap=cmap, vmin=-1, vmax=1, origin='lower')
        else:
            print('Covariance matrix is not available.')

    def plot2d(self, nx=5, ny=4, figaxes=None, ebar=True, fs=20, figsize=(16, 20), scale=1e-8, rescale_ebar=1, t_sidx=2, zbins=None, title=None, yscale='linear', ylim=(-10, 70), **kwargs):
        if figaxes is None:
            fig, axes = plt.subplots(nx, ny, figsize=figsize, sharex=True, sharey=True)
        else:
            fig, axes = figaxes
        plt.subplots_adjust(hspace=0.1, wspace=0.1)

        if title is not None:
            fig.suptitle(title, fontsize=fs, y=0.9)
        zbins = self.get_z_bin(unique=True).T
        for i, ax in enumerate(axes.flatten()):
            sel = self.selection_z_bin(zbins[i], 'z123')
            sig = self.get_signal(sel)
            std = self.get_std(sel)/2**0.5
            t = self.get_t_bin(sel)[0]

            z1, z2, z3 = zbins[i]

            ax.set_yscale(yscale)
            ax.set_ylim(ylim)
            ax.axhline(0, color='gray', ls='--')
            if ebar:
                ax.errorbar(t, t**t_sidx*sig/scale, yerr=t**t_sidx*std/scale*rescale_ebar, fmt='.', **kwargs)
            else:
                ax.plot(t, t**t_sidx*sig/scale, **kwargs)
            ax.text(0.65, 0.85, '({}, {}, {})'.format(z1, z2, z3), transform=ax.transAxes, fontsize=fs)

            ax.grid()
            if i % ny == 0:
                if t_sidx == 1:
                    ax.set_ylabel(r'$\theta \times \langle \mathcal{M}_{{\rm ap}}^3\rangle(\theta)$', fontsize=fs)
                else:
                    ax.set_ylabel(r'$10^{{7}}\theta^{} \times \langle \mathcal{{M}}_{{\rm ap}}^3\rangle(\theta)$'.format(t_sidx), fontsize=fs)
            if i >= ny*(nx-1):
                ax.set_xlabel(r'$\theta$ [arcmin]', fontsize=fs)

        return fig, axes

def compare(array, val, condition):
    if condition == '==':
        return val == array
    elif condition == '>':
        return val > array
    elif condition == '<':
        return val < array
    elif condition == '>=':
        return val >= array
    elif condition == '<=':
        return val <= array
    else:
        raise ValueError('Condition "{}" not recognized.'.format(condition))
