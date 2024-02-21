"""
Functions for reading:
    * Bruker binary (ser/fid) files.
    * Bruker JCAMP-DX parameter (acqus) files.
    * Bruker pulse program (pulseprogram) files.

Copied from NMRGlue script:
https://github.com/jjhelmus/nmrglue/blob/master/nmrglue/fileio/bruker.py
"""

__developer_info__ = """
Bruker file format information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Bruker binary files (ser/fid) store data as an array of numbers whose
endianness is determined by the parameter BYTORDA (1 = big endian, 0 = little
endian), and whose data type is determined by the parameter DTYPA (0 = int32,
2 = float64). Typically the direct dimension is digitally filtered. The exact
method of removing this filter is unknown but an approximation is available.

Bruker JCAMP-DX files (acqus, etc) are text file which are described by the
`JCAMP-DX standard <http://www.jcamp-dx.org/>`_.  Bruker parameters are
prefixed with a '$'.

Bruker pulseprogram files are text files described in various Bruker manuals.
Of special important are lines which describe external variable assignments
(surrounded by "'s), loops (begin with lo), phases (contain ip of dp) or
increments (contain id, dd, ipu or dpu).  These lines are parsed when reading
the file with nmrglue.

"""

import os
import numpy as np
import locale
from warnings import warn

def open_towrite(filename, overwrite=False, mode='wb'):
    """
    Open filename for writing and return file object

    Function checks if file exists (and raises IOError if overwrite=False) and
    creates necessary directories as needed.
    """
    # check if file exists and overwrite if False
    if os.path.exists(filename) and (overwrite is False):
        raise OSError("File exists, recall with overwrite=True")

    p, fn = os.path.split(filename)  # split into filename and path
    # create directories if needed
    if p != '' and os.path.exists(p) is False:
        os.makedirs(p)

    return open(filename, mode)

# Global read function and related utilities

def read(dir=".", bin_file=None, acqus_files=None, pprog_file=None, shape=None,
         cplex=None, big=None, isfloat=None, read_pulseprogram=True,
         read_acqus=True, procs_files=None, read_procs=True):
    """
    Read Bruker files from a directory.

    Parameters
    ----------
    dir : str
        Directory to read from.
    bin_file : str, optional
        Filename of binary file in directory. None uses standard files.
    acqus_files : list, optional
        List of filename(s) of acqus parameter files in directory. None uses
        standard files.
    pprog_file : str, optional
        Filename of pulse program in directory. None uses standard files.
    shape : tuple, optional
        Shape of resulting data.  None will guess the shape from the spectral
        parameters.
    cplex : bool, optional
        True is direct dimension is complex, False otherwise. None will guess
        quadrature from spectral parameters.
    big : bool or None, optional
        Endianness of binary file. True for big-endian, False for
        little-endian, None to determine endianness from acqus file(s).
    isfloat : bool or None, optional
        Data type of binary file. True for float64, False for int32. None to
        determine data type from acqus file(s).
    read_pulseprogram : bool, optional
        True to read pulse program, False prevents reading.
    read_acqus : bool, optional
        True to read acqus files(s), False prevents reading.
    procs_files : list, optional
        List of filename(s) of procs parameter files in directory. None uses
        standard files.
    read_procs : bool, optional
        True to read procs files(s), False prevents reading.

    Returns
    -------
    dic : dict
        Dictionary of Bruker parameters.
    data : ndarray
        Array of NMR data.

    See Also
    --------
    read_pdata : Read Bruker processed files.
    read_lowmem : Low memory reading of Bruker files.
    write : Write Bruker files.

    """
    if os.path.isdir(dir) is not True:
        raise OSError("directory %s does not exist" % (dir))

    # Take a shot at reading the procs file
    if read_procs:
        dic = read_procs_file(dir, procs_files)
    else:
        # create an empty dictionary
        dic = dict()

    # determine parameter automatically
    if bin_file is None:
        if os.path.isfile(os.path.join(dir, "fid")):
            bin_file = "fid"
        elif os.path.isfile(os.path.join(dir, "ser")):
            bin_file = "ser"

        # Look two directory levels lower.
        elif os.path.isdir(os.path.dirname(os.path.dirname(dir))):

            # ! change the dir
            dir = os.path.dirname(os.path.dirname(dir))

            if os.path.isfile(os.path.join(dir, "fid")):
                bin_file = "fid"
            elif os.path.isfile(os.path.join(dir, "ser")):
                bin_file = "ser"
            else:
                mesg = "No Bruker binary file could be found in %s"
                raise OSError(mesg % (dir))
        else:
            mesg = "No Bruker binary file could be found in %s"
            raise OSError(mesg % (dir))

    if read_acqus:
        # read the acqus_files and add to the dictionary
        acqus_dic = read_acqus_file(dir, acqus_files)
        dic = _merge_dict(dic, acqus_dic)

    if pprog_file is None:
        pprog_file = "pulseprogram"

    # read the pulse program and add to the dictionary
    if read_pulseprogram:
        try:
            dic["pprog"] = read_pprog(os.path.join(dir, pprog_file))
        except:
            warn('Error reading the pulse program')

    # determine file size and add to the dictionary
    dic["FILE_SIZE"] = os.stat(os.path.join(dir, bin_file)).st_size

    # determine shape and complexity for direct dim if needed
    if shape is None or cplex is None:
        gshape, gcplex = guess_shape(dic)
        if gcplex is True:    # divide last dim by 2 if complex
            t = list(gshape)
            t[-1] = t[-1] // 2
            gshape = tuple(t)
    if shape is None:
        shape = gshape
    if cplex is None:
        cplex = gcplex

    # determine endianness (assume little-endian unless BYTORDA is 1)
    if big is None:
        big = False     # default value
        if "acqus" in dic and "BYTORDA" in dic["acqus"]:
            if dic["acqus"]["BYTORDA"] == 1:
                big = True
            else:
                big = False

    # determine data type (assume int32 unless DTYPA is 2)
    if isfloat is None:
        isfloat = False     # default value
        if "acqus" in dic and "DTYPA" in dic["acqus"]:
            if dic["acqus"]["DTYPA"] == 2:
                isfloat = True
            else:
                isfloat = False

    # read the binary file
    f = os.path.join(dir, bin_file)
    null, data = read_binary(f, shape=shape, cplex=cplex, big=big,
                             isfloat=isfloat)
    return dic, data


def read_acqus_file(dir='.', acqus_files=None):
    """
    Read Bruker acquisition files from a directory.

    Parameters
    ----------
    dir : str
        Directory to read from.
    acqus_files : list, optional
        List of filename(s) of acqus parameter files in directory. None uses
        standard files. If filename(s) contains a full absolute path, dir is not used.

    Returns
    -------
    dic : dict
        Dictionary of Bruker parameters.
    """
    if acqus_files is None:
        acqus_files = []
        for f in ["acqus", "acqu2s", "acqu3s", "acqu4s"]:
            fp = os.path.join(dir, f)
            if os.path.isfile(fp):
                acqus_files.append(fp)

    # create an empty dictionary
    dic = dict()

    # read the acqus_files and add to the dictionary
    for f in acqus_files:
        if not os.path.isfile(f):
            f = os.path.join(dir, f)
        acqu = os.path.basename(f)
        dic[acqu] = read_jcamp(f)

    return dic


def read_procs_file(dir='.', procs_files=None):
    """
    Read Bruker processing files from a directory.

    Parameters
    ----------
    dir : str
        Directory to read from.
    procs_files : list, optional
        List of filename(s) of procs parameter files in directory. None uses
        standard files. If filename(s) contains a full absolute path, dir is not used.

    Returns
    -------
    dic : dict
        Dictionary of Bruker parameters.
    """

    if procs_files is None:

        # Reading standard procs files
        procs_files = []

        pdata_path = dir
        for f in ["procs", "proc2s", "proc3s", "proc4s"]:
            pf = os.path.join(pdata_path, f)
            if os.path.isfile(pf):
                procs_files.append(pf)

        if not procs_files:
            # procs not found in the given dir, try look adding pdata to the dir path

            if os.path.isdir(os.path.join(dir, 'pdata')):
                pdata_folders = [folder for folder in
                                 os.walk(os.path.join(dir, 'pdata'))][0][1]
                if '1' in pdata_folders:
                    pdata_path = os.path.join(dir, 'pdata', '1')
                else:
                    pdata_path = os.path.join(dir, 'pdata', pdata_folders[0])

            for f in ["procs", "proc2s", "proc3s", "proc4s"]:
                pf = os.path.join(pdata_path, f)
                if os.path.isfile(pf):
                    procs_files.append(pf)

    else:
        # proc paths were explicitly given
        # just check if they exists

        for i, f in enumerate(procs_files):
            pdata_path, f = os.path.split(f)
            if not pdata_path:
                pdata_path = dir

            pf = os.path.join(pdata_path, f)
            if not os.path.isfile(pf):
                mesg = "The file `%s` could not be found "
                warn(mesg % pf)
            else:
                procs_files[i] = pf

    # create an empty dictionary
    dic = dict()

    # read the acqus_files and add to the dictionary
    for f in procs_files:
        pdata_path = os.path.basename(f)
        dic[pdata_path] = read_jcamp(f)
    return dic


def write_pdata(dir, dic, data, roll=False, shape=None, submatrix_shape=None,
                scale_data=False, bin_file=None, procs_files=None,
                write_procs=False, pdata_folder=False, overwrite=False,
                big=None, isfloat=None, restrict_access=True):
    """
    Write processed Bruker files to disk.

    Parameters
    ----------
    dir : str
        Directory to write files to.
    dic : dict
        Dictionary of Bruker parameters.
    data : array_like
        Array of NMR data
    roll : int
        Number of points by which a circular shift needs to be applied to the data
        True will apply a circular shift of 1 data point
    shape : tuple, optional
        Shape of data, if file is to be written with a shape
        different than data.shape
    submatrix_shape : tuple, optional
        Shape of the submatrix used to store data (using Bruker specifications)
        If this is not given, the submatrix shape will be guessed from dic
    scale_data : Bool
        Apply a reverse scaling using the scaling factor defined in procs file
        By default, the array to be written will not be scaled using the value
        in procs but will be e scaled so  that the max intensity in that array
        will have a value between 2**28 and 2**29. scale_data is to be used when
        the array is itself a processed  bruker file that was read into nmrglue
    bin_file : str, optional
        Filename of binary file in directory. None uses standard files.
    procs_file : list, optional
        List of filename(s) of procs parameter files (to write out). None uses a
        list of standard files
    write_procs : Bool
        True to write out the procs files
    pdata_folder : int, optional
        Makes a folder and a subfolder ('pdata/pdata_folder') inside the given
        directory where pdata_folder is an integer. All files (procs and data) are
        stored inside pdata_folder. pdata_folder=False (or =0) does not make the
        pdata folder and pdata_folder=True makes folder '1'.
    overwrite : bool, optional
        Set True to overwrite files, False will raise a Warning if files
        exist.
    big : bool or None, optional
        Endianness of binary file. True for big-endian, False for
        little-endian, None to determine endianness from Bruker dictionary.
    isfloat : bool or None, optional
        Data type of binary file. True for float64, False for int32. None to
        determine data type from Bruker dictionary.
    restrict_access : not implemented

    """

    # see that data consists of only real elements
    data = np.roll(data.real, int(roll))

    # either apply a reverse scaling to the data or scale processed data
    # so that the max value is between 2**28 and 2**29 and cast to integers
    if scale_data:
        data = scale_pdata(dic, data, reverse=True)
    else:
        data = array_to_int(data)

    # see if the dimensionality is given
    # else, set it to the dimensions of data
    if shape is None:
        shape = data.shape

    # guess data dimensionality
    ndim = len(shape)

    # update PARMODE in dictionary
    # This is required when writing back 1D slices from a 2D, 2D planes of 3D, etc
    dic['procs']['PPARMOD'] = ndim - 1

    # reorder the submatrix according
    if submatrix_shape is None:
        submatrix_shape = guess_shape_and_submatrix_shape(dic)[1]

    data = reorder_submatrix(data, shape, submatrix_shape, reverse=True)

    # see if pdata_folder needs to make and set write path
    if pdata_folder is not False:
        try:
            procno = str(int(pdata_folder))
            pdata_path = os.path.join(dir, 'pdata', procno)
        except ValueError:
            raise ValueError('pdata_folder should be an integer')

        if not os.path.isdir(pdata_path):
            os.makedirs(pdata_path)
    else:
        pdata_path = dir

    # write out the procs files only for the desired dimensions
    if write_procs:
        if procs_files is None:
            proc = ['procs'] + [f'proc{i}s' for i in range(2, ndim+1)]
            procs_files = [f for f in proc if (f in dic)]

        for f in procs_files:
            write_jcamp(dic[f], os.path.join(pdata_path, f),
                        overwrite=overwrite)
            write_jcamp(dic[f], os.path.join(pdata_path, f[:-1]),
                        overwrite=overwrite)

    if bin_file is None:
        bin_file = str(ndim) + 'r'*ndim

    bin_full = os.path.join(pdata_path, bin_file)
    write_binary(bin_full, dic, data, big=big, isfloat=isfloat,
                 overwrite=overwrite)


def guess_shape(dic):
    """
    Determine data shape and complexity from Bruker dictionary.

    Returns
    -------
    shape : tuple
        Shape of data in Bruker binary file (R+I for all dimensions).
    cplex : bool
        True for complex data in last (direct) dimension, False otherwise.

    """
    # determine complexity of last (direct) dimension
    try:
        aq_mod = dic["acqus"]["AQ_mod"]
    except KeyError:
        aq_mod = 0

    if aq_mod in (0, 2):
        cplex = False
    elif aq_mod in (1, 3):
        cplex = True
    else:
        raise ValueError("Unknown Acquisition Mode")

    # file size
    try:
        fsize = dic["FILE_SIZE"]
    except KeyError:
        warn("cannot determine shape do to missing FILE_SIZE key")
        return (1,), True

    # extract td0,td1,td2,td3 from dictionaries
    try:
        td0 = float(dic["acqus"]["TD"])
    except KeyError:
        td0 = 1024   # default value

    try:
        td2 = int(dic["acqu2s"]["TD"])
    except KeyError:
        td2 = 0     # default value

    try:
        td1 = float(dic["acqu3s"]["TD"])
    except KeyError:
        td1 = int(td2)   # default value

    try:
        td3 = int(dic["acqu4s"]["TD"])
    except KeyError:
        td3 = int(td1)     # default value

    # From the acquisition reference manual (section on parameter NBL):
    #     ---
    #     If TD is not a multiple of 256 (1024 bytes), successive FIDs will
    #     still begin at 1024 byte memory boundaries. This is so for the FIDs
    #     in the acquisition memory as well as on disk. The size of the raw
    #     data file (ser) is therefore always a multiple of 1024 times NBL.
    #     ---
    # This seems to hold for 1D data sets as well. However, this paragraph
    # assumes that each data point is 4 bytes, hence the "multiple of 256".
    # For data in DTYPA=2 (float64), each point is 8 bytes, so while it always
    # allocates the fids in 1024-byte blocks, for float64 data it pads the data
    # (by points) out to multiples of 128, not 256. So we need to get the
    # data type before we guess the shape of the last dimension.

    # extract data type from dictionary
    try:
        dtypa = int(dic["acqus"]["DTYPA"])
    except KeyError:
        dtypa = 0   # default value, int32 data

    # last (direct) dimension is given by "TD" parameter in acqus file
    # rounded up to nearest (1024/(bytes per point))
    # next-to-last dimension may be given by "TD" in acqu2s. In 3D+ data
    # this is often the sum of the indirect dimensions
    if dtypa == 2:
        shape = [0, 0, td2, int(np.ceil(td0 / 128.) * 128.)]
    else:
        shape = [0, 0, td2, int(np.ceil(td0 / 256.) * 256.)]

    # additional dimension given by data size
    if shape[2] != 0 and shape[3] != 0:
        shape[1] = fsize // (shape[3] * shape[2] * 4)
        shape[0] = fsize // (shape[3] * shape[2] * shape[1] * 4)

    # if there in no pulse program parameters in dictionary return current
    # shape after removing zeros
    if "pprog" not in dic or "loop" not in dic["pprog"]:
        return tuple([int(i) for i in shape if i > 1]), cplex

    # if pulseprogram dictionary is missing loop or incr return current shape
    pprog = dic["pprog"]
    if "loop" not in pprog or "incr" not in pprog:
        return tuple([int(i) for i in shape if i > 1]), cplex

    # determine indirect dimension sizes from pulseprogram parameters
    loop = pprog["loop"]
    loopn = len(loop)       # number of loops
    li = [len(i) for i in pprog["incr"]]    # length of incr lists

    # replace td0,td1,td2,td3 in loop list
    rep = {'td0': td0, 'td1': td1, 'td2': td2, 'td3': td3}
    for i, v in enumerate(loop):
        if v in rep.keys():
            loop[i] = rep[v]

    # if the loop variables contains strings, return current shape
    # these variables could be resolved from the var key in the pprog dict
    # but this would require executing unknown code to perform the
    # arithmetic present in the string.
    if str in [type(e) for e in loop]:
        return tuple([int(i) for i in shape if i > 1]), cplex

    # size of indirect dimensions based on number of loops in pulse program
    # there are two kinds of loops, active and passive.
    # active loops are from indirect dimension increments, the corresponding
    # incr lists should have non-zero length and the size of the dimension
    # is twice that of the active loop size.
    # passive loops are from phase cycles and similar elements, these should
    # have zero length incr lists and should be of length 2.

    # The following checks for these and updates the indirect dimension
    # if the above is found.
    if loopn == 1:    # 2D with no leading passive loops
        if li[0] != 0:
            shape[2] = loop[0]
            shape = shape[-2:]

    elif loopn == 2:  # 2D with one leading passive loop
        if loop[0] == 2 and li[0] == 0 and li[1] != 0:
            shape[2] = 2 * loop[1]
            shape = shape[-2:]

    elif loopn == 3:  # 2D with two leading passive loops
        if loop[0] == loop[1] == 2 and li[0] == li[1] == 0 and li[2] != 0:
            shape[2] = 2 * loop[2]
            shape = shape[-2:]

    elif loopn == 4:  # 3D with one leading passive loop for each indirect dim
        if loop[0] == 2 and li[0] == 0 and li[1] != 0:
            shape[2] = 2 * loop[1]
        if loop[2] == 2 and li[2] == 0 and li[3] != 0:
            shape[1] = 2 * loop[3]
            shape = shape[-3:]

    elif loopn == 5:  # 3D with two/one leading passive loops
        if loop[1] == 2 and li[0] == li[1] == 0 and li[2] != 0:
            shape[2] = 2 * loop[2]
        if loop[3] == 2 and li[0] == li[3] == 0 and li[4] != 0:
            shape[1] = 2 * loop[4]
            shape = shape[-3:]

    elif loopn == 6:  # 4D with one leading passive loop for each indirect dim
        if loop[0] == 2 and li[0] == 0 and li[1] != 0:
            shape[2] = 2 * loop[1]
        if loop[2] == 2 and li[2] == 0 and li[3] != 0:
            shape[1] = 2 * loop[3]
        if loop[4] == 2 and li[4] == 0 and li[5] != 0:
            shape[0] = 2 * loop[5]

    elif loopn == 7:
        if loop[1] == 2 and li[0] == li[1] == 0 and li[2] != 0:
            shape[2] = 2 * loop[2]
        if loop[3] == 2 and li[0] == li[3] == 0 and li[4] != 0:
            shape[1] = 2 * loop[4]
        if loop[5] == 2 and li[0] == li[5] == 0 and li[6] != 0:
            shape[0] = 2 * loop[6]

    return tuple([int(i) for i in shape if i >= 2]), cplex


# Bruker binary (fid/ser) reading and writing

def read_binary(filename, shape=(1), cplex=True, big=True, isfloat=False):
    """
    Read Bruker binary data from file and return dic,data pair.

    If data cannot be reshaped as described a 1D representation of the data
    will be returned after printing a warning message.

    Parameters
    ----------
    filename : str
        Filename of Bruker binary file.
    shape : tuple
        Tuple describing shape of resulting data.
    cplex : bool
        Flag indicating if direct dimension is complex.
    big : bool
        Endianness of binary file, True for big-endian, False for
        little-endian.
    isfloat : bool
        Data type of binary file. True for float64, False for int32.

    Returns
    -------
    dic : dict
        Dictionary containing "FILE_SIZE" key and value.
    data : ndarray
        Array of raw NMR data.

    See Also
    --------
    read_binary_lowmem : Read Bruker binary file using minimal memory.

    """
    # open the file and get the data
    with open(filename, 'rb') as f:
        data = get_data(f, big=big, isfloat=isfloat)

    # complexify if needed
    if cplex:
        data = complexify_data(data)

    # create dictionary
    dic = {"FILE_SIZE": os.stat(filename).st_size}

    # reshape if possible
    try:
        return dic, data.reshape(shape)

    except ValueError:
        warn(str(data.shape) + "cannot be shaped into" + str(shape))
        return dic, data


def write_binary(filename, dic, data, overwrite=False, big=True,
                 isfloat=False):
    """
    Write Bruker binary data to file.

    Parameters
    ----------
    filename : str
        Filename to write to.
    dic : dict
        Dictionary of Bruker parameters.
    data : ndarray
        Array of NMR data.
    overwrite : bool
        True to overwrite files, False will raise a Warning if file exists.
    big : bool
        Endianness to write binary data with True for big-endian, False for
        little-endian.
    isfloat : bool
        Data type of binary file. True for float64, False for int32.

    See Also
    --------
    write_binary_lowmem : Write Bruker binary data using minimal memory.

    """
    # open the file for writing
    f = open_towrite(filename, overwrite=overwrite)

    # convert object to an array if it is not already one...
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    if np.iscomplexobj(data):
        put_data(f, uncomplexify_data(data, isfloat), big, isfloat)
    else:
        put_data(f, data, big, isfloat)
    f.close()

# binary get/put functions


def get_data(f, big, isfloat):
    """
    Get binary data from file object with given endianness and data type.
    """
    if isfloat:
        if big:
            return np.frombuffer(f.read(), dtype='>f8')
        else:
            return np.frombuffer(f.read(), dtype='<f8')
    else:
        if big:
            return np.frombuffer(f.read(), dtype='>i4')
        else:
            return np.frombuffer(f.read(), dtype='<i4')
        
        
# data manipulation functions


def complexify_data(data):
    """
    Complexify data packed real, imag.
    """
    return data[..., ::2] + data[..., 1::2] * 1.j


def uncomplexify_data(data_in, isfloat):
    """
    Uncomplexify data (pack real,imag) into a int32 or float64 array,
    depending on isfloat.
    """
    size = list(data_in.shape)
    size[-1] = size[-1] * 2
    if isfloat:
        data_out = np.empty(size, dtype="float64")
    else:
        data_out = np.empty(size, dtype="int32")
    data_out[..., ::2] = data_in.real
    data_out[..., 1::2] = data_in.imag
    return data_out


# JCAMP-DX functions

def read_jcamp(filename, encoding=locale.getpreferredencoding()):
    """
    Read a Bruker JCAMP-DX file into a dictionary.

    Creates two special dictionary keys _coreheader and _comments Bruker
    parameter "$FOO" are extracted into strings, floats or lists and assigned
    to dic["FOO"]

    Parameters
    ----------
    filename : str
        Filename of Bruker JCAMP-DX file.
    encoding : str
        Encoding of Bruker JCAMP-DX file. Defaults to the system default locale

    Returns
    -------
    dic : dict
        Dictionary of parameters in file.

    See Also
    --------
    write_jcamp : Write a Bruker JCAMP-DX file.

    Notes
    -----
    This is not a fully functional JCAMP-DX reader, it is only intended
    to read Bruker acqus (and similar) files.

    """
    dic = {"_coreheader": [], "_comments": []}  # create empty dictionary

    with open(filename, 'r', encoding=encoding) as f:
        while True:     # loop until end of file is found

            line = f.readline().rstrip()    # read a line
            if line == '':      # end of file found
                break

            if line[:6] == "##END=":
                # print("End of file")
                break
            elif line[:2] == "$$":
                dic["_comments"].append(line)
            elif line[:2] == "##" and line[2] != "$":
                dic["_coreheader"].append(line)
            elif line[:3] == "##$":
                try:
                    key, value = parse_jcamp_line(line, f)
                    dic[key] = value
                except:
                    warn("Unable to correctly parse line:" + line)
            else:
                warn("Extraneous line:" + line)

    return dic


def parse_jcamp_line(line, f):
    """
    Parse a single JCAMP-DX line.

    Extract the Bruker parameter name and value from a line from a JCAMP-DX
    file.  This may entail reading additional lines from the fileobj f if the
    parameter value extends over multiple lines.

    """

    # extract key= text from line
    key = line[3:line.index("=")]
    text = line[line.index("=") + 1:].lstrip()

    if "<" in text:   # string
        while ">" not in text:      # grab additional text until ">" in string
            text = text + "\n" + f.readline().rstrip()
        value = text[1:-1]  # remove < and >

    elif "(" in text:   # array
        num = int(line[line.index("..") + 2:line.index(")")]) + 1
        value = []
        rline = line[line.index(")") + 1:]

        # extract value from remainder of line
        for t in rline.split():
            value.append(parse_jcamp_value(t))

        # parse additional lines as necessary
        while len(value) < num:
            nline = f.readline().rstrip()
            for t in nline.split():
                value.append(parse_jcamp_value(t))

    elif text == "yes":
        value = True

    elif text == "no":
        value = False

    else:   # simple value
        value = parse_jcamp_value(text)

    return key, value


def parse_jcamp_value(text):
    """
    Parse value text from Bruker JCAMP-DX file returning the value.
    """
    if text == '':
        return None
    elif text.startswith('<') and text.endswith('>'):
        return text[1:-1]  # remove < and >
    else:
        if "." in text or "e" in text or 'inf' in text:
            try:
                return float(text)
            except ValueError:
                return text
        else:
            try:
                return int(text)
            except ValueError:
                return text


def write_jcamp(dic, filename, overwrite=False):
    """
    Write a Bruker JCAMP-DX file from a dictionary.

    Written file will differ slightly from Bruker's JCAMP-DX files in that all
    multi-value parameters will be written on multiple lines. Bruker is
    inconsistent on what is written to a single line and what is not.
    In addition line breaks may be slightly different but will always be
    within JCAMP-DX specification.  Finally long floating point values
    may loose precision when writing.

    For example:

        ##$QS= (0..7)83 83 83 83 83 83 83 22

        will be written as

        ##$QS= (0..7)
        83 83 83 83 83 83 83 22

    Parameters
    ----------
    dic : dict
        Dictionary of parameters to write
    filename : str
        Filename of JCAMP-DX file to write
    overwrite : bool, optional
        True to overwrite an existing file, False will raise a Warning if the
        file already exists.

    See Also
    --------
    read_jcamp : Read a Bruker JCAMP-DX file.

    """

    # open the file for writing
    f = open_towrite(filename, overwrite=overwrite, mode='w')

    # create a copy of the dictionary
    d = dict(dic)

    # remove the comments and core header key from dictionary
    comments = d.pop("_comments")
    corehdr = d.pop("_coreheader")

    # write out the core headers
    for line in corehdr:
        f.write(line)
        f.write("\n")

    # write out the comments
    for line in comments:
        f.write(line)
        f.write("\n")

    keys = sorted([i for i in d.keys()])

    # write out each key,value pair
    for key in keys:
        write_jcamp_pair(f, key, d[key])

    # write ##END= and close the file
    f.write("##END=")
    f.close()


def write_jcamp_pair(f, key, value):
    """
    Write out a line of a JCAMP file.

    a line might actually be more than one line of text for arrays.
    """

    # the parameter name and such
    line = "##$" + key + "= "

    # need to be type not isinstance since isinstance(bool, int) == True
    if type(value) == float or type(value) == int:  # simple numbers
        line = line + repr(value)

    elif isinstance(value, str):    # string
        line = line + "<" + value + ">"

    elif type(value) == bool:   # yes or no
        if value:
            line = line + "yes"
        else:
            line = line + "no"

    elif isinstance(value, list):
        # write out the current line
        line = line + "(0.." + repr(len(value) - 1) + ")"
        f.write(line)
        f.write("\n")
        line = ""

        # loop over elements in value printing out lines when
        # they reach > 70 characters or the next value would cause
        # the line to go over 80 characters
        for v in value:
            if len(line) > 70:
                f.write(line)
                f.write("\n")
                line = ""

            if isinstance(v, str):
                to_add = "<" + v + ">"
            else:
                to_add = repr(v)

            if len(line + " " + to_add) > 80:
                f.write(line)
                f.write("\n")
                line = ""

            if line != "":
                line = line + to_add + " "
            else:
                line = to_add + " "

    # write out the line and a newline character
    f.write(line)
    f.write("\n")


# pulse program read/writing functions

def read_pprog(filename):
    """
    Read a Bruker pulse program (pulseprogram) file.

    Resultsing dictionary contains the following keys:

    ========    ===========================================================
    key         description
    ========    ===========================================================
    var         dictionary of variables assigned in pulseprogram
    incr        list of lists containing increment times
    loop        list of loop multipliers
    phase       list of lists containing phase elements
    ph_extra    list of lists containing comments at the end of phase lines
    ========    ===========================================================

    The incr,phase and ph_extra lists match up with loop list.  For example
    incr[0],phase[0] and ph_extra[0] are all increment and phase commands
    with comments which occur during loop 0 which has loop[0] steps.

    Parameters
    ----------
    filename : str
        Filename of pulseprogram file to read from,

    Returns
    -------
    dic : dict
        A dictionary with keys described above.

    See Also
    --------
    write_pprog : Write a Bruker pulse program to file.

    """

    # open the file
    f = open(filename, 'r')

    # initialize lists and dictionaries
    var = dict()
    loop = []
    incr = [[]]
    phase = [[]]
    ph_extra = [[]]

    # loop over lines in pulseprogram looking for loops, increment,
    # assignments and phase commands
    for line in f:

        # split line into comment and text and strip leading/trailing spaces
        if ";" in line:
            comment = line[line.index(";"):]
            text = line[:line.index(";")].strip()
        else:
            comment = ""
            text = line.strip()

        # remove label from text when first word is all digits or
        # has "," as the last element
        if len(text.split()) != 0:
            s = text.split()[0]
            if s.isdigit() or s[-1] == ",":
                text = text[len(s):].strip()

        # skip blank lines and include lines
        if text == "" or text[0] == "#":
            # print(line,"--Blank, Comment or Include")
            continue

        # see if we have quotes and have an assignment
        # syntax "foo=bar"
        # add foo:bar to var dictionary
        if "\"" in text:
            if "=" in line:
                # strip quotes, split on = and add to var dictionary
                text = text.strip("\"")
                t = text.split("=")
                if len(t) >= 2:
                    key, value = t[0], t[1]
                    var[key] = value
                    # print(line,"--Assignment")
                else:
                    pass
                    # print(line,"--Statement")
                continue
            else:
                # print(line,"--Statement")
                continue

        # loops begin with lo
        # syntax is: lo to N time M
        # add M to loop list
        if text[0:2] == "lo":
            loop.append(text.split()[4])
            incr.append([])
            phase.append([])
            ph_extra.append([])
            # print(line,"--Loop")
            continue

        tokens = text.split()
        if len(tokens) >= 2:
            token2 = tokens[1]
            # increment statement have id, dd, ipu or dpu
            # syntax foo {id/dd/ipu/dpu}N
            # store N to incr list
            if token2.startswith('id') or token2.startswith('dd'):
                incr[len(loop)].append(int(token2[2:]))
                # print(line,"--Increment")
                continue

            if token2.startswith("ipu") or token2.startswith("dpu"):
                incr[len(loop)].append(int(token2[3:]))
                # print(line,"--Increment")
                continue

            # phase statement have ip or dp
            # syntax fpp {ip/dp}N extra
            # store N to phase list and extra to ph_extra list
            if token2.startswith("ip") or token2.startswith("dp"):
                phase[len(loop)].append(int(token2[2:]))

                # find the first space after "ip" and read past there
                last = text.find(" ", text.index("ip"))
                if last == -1:
                    ph_extra[len(loop)].append("")
                else:
                    ph_extra[len(loop)].append(text[last:].strip())
                # print(line,"--Phase")
                continue

            # print(line,"--Unimportant")

    f.close()

    # remove the last empty incr, phase and ph_extra lists
    incr.pop()
    phase.pop()
    ph_extra.pop()

    # convert loop to numbers if possible
    for i, t in enumerate(loop):
        if t.isdigit():
            loop[i] = int(t)
        else:
            if (t in var) and var[t].isdigit():
                loop[i] = int(var[t])

    # create the output dictionary
    dic = {"var": var, "incr": incr, "loop": loop, "phase": phase,
           "ph_extra": ph_extra}
    return dic


def write_pprog(filename, dic, overwrite=False):
    """
    Write a minimal Bruker pulse program to file.

    **DO NOT TRY TO RUN THE RESULTING PULSE PROGRAM**

    This pulse program should return the same dictionary when read using
    read_pprog, nothing else.  The pulse program will be nonsense.

    Parameters
    ----------
    filename : str
        Filename of file to write pulse program to.
    dic : dict
        Dictionary of pulse program parameters.
    overwrite : bool, optional
        True to overwrite an existing file, False will raise a Warning if the
        file already exists.

    See Also
    --------
    read_pprog : Read a Bruker pulse program.

    """

    # open the file for writing
    f = open_towrite(filename, overwrite=overwrite, mode='w')

    # write a comment
    f.write("; Minimal Bruker pulseprogram created by write_pprog\n")

    # write our the variables
    for k, v in dic["var"].items():
        f.write("\"" + k + "=" + v + "\"\n")

    # write out each loop
    for i, steps in enumerate(dic["loop"]):

        # write our the increments
        for v in dic["incr"][i]:
            f.write("d01 id" + str(v) + "\n")

        # write out the phases
        for v, w in zip(dic["phase"][i], dic["ph_extra"][i]):
            f.write("d01 ip" + str(v) + " " + str(w) + "\n")

        f.write("lo to 0 times " + str(steps) + "\n")

    # close the file
    f.close()


def _merge_dict(a, b):
    c = a.copy()
    c.update(b)
    return c
